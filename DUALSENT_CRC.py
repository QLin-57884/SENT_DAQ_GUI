# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 12:15:23 2025

@author: QLin
Updated: Counter-based SENT timestamp capture & decode with CRC validation and chunk boundary handling
"""
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import nidaqmx
from nidaqmx.constants import AcquisitionType
import threading
import time
import csv
from queue import Queue, Full

# === Config === #
DEVICE_NAME = "Dev1"
LINE_NAMES = ["port0/line0", "port0/line1"]  # SENT signal lines for channel 0 & 1
TS_BUFFER_SIZE = 10000      # number of timestamps per read
QUEUE_MAX_CHUNKS = 200      # processing queue depth
MAX_DISPLAY_LINES = 1000

# === Globals === #
is_running = False
acq_thread = None
decode_thread = None
gui_thread = None

# === DAQ Task Setup === #
def create_counter_tasks():
    tasks = []
    for idx, line in enumerate(LINE_NAMES):
        task = nidaqmx.Task()
        task.ci_channels.add_ci_change_detection_chan(
            f"{DEVICE_NAME}/ctr{idx}",
            rising_edge_chan=None,
            falling_edge_chan=f"{DEVICE_NAME}/{line}"
        )
        task.timing.cfg_implicit_timing(
            sample_mode=AcquisitionType.CONTINUOUS,
            samps_per_chan=TS_BUFFER_SIZE
        )
        task.start()
        tasks.append(task)
    return tasks

# === CRC Helper === #
def compute_crc4(nibbles):
    """
    Compute 4-bit CRC per SENT (poly x^4 + x + 1, init=0xF).
    nibbles: iterable of integer nibble values (0–15).
    Returns CRC nibble (0–15).
    """
    reg = 0xF
    for nib in nibbles:
        reg ^= (nib & 0xF)
        for _ in range(4):
            if reg & 0x8:
                reg = ((reg << 1) ^ 0x3) & 0xF
            else:
                reg = (reg << 1) & 0xF
    return reg

# === GUI App === #
class SentTimestampApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SENT Timestamp & Decode GUI")
        self.root.geometry("800x600")

        # Control buttons
        self.start_button = ttk.Button(root, text="Start", command=self.start_acquisition)
        self.stop_button  = ttk.Button(root, text="Stop",  command=self.stop_acquisition, state='disabled')
        self.save_button  = ttk.Button(root, text="Save Data", command=self.save_data, state='disabled')
        self.exit_button  = ttk.Button(root, text="Exit", command=self.exit_program)
        self.start_button.grid(row=0, column=0, padx=5, pady=5)
        self.stop_button.grid(row=0, column=1, padx=5, pady=5)
        self.save_button.grid(row=0, column=2, padx=5, pady=5)
        self.exit_button.grid(row=0, column=3, padx=5, pady=5)

        # Decode text box
        self.decode_box = tk.Text(root, height=25, width=100)
        self.decode_box.grid(row=1, column=0, columnspan=4, padx=10, pady=10)

        # Timestamp tasks and queues
        self.ts_tasks = create_counter_tasks()
        self.data_queue = Queue(maxsize=QUEUE_MAX_CHUNKS)
        self.ui_queue   = Queue(maxsize=QUEUE_MAX_CHUNKS)

        # Decode state and leftovers
        self.decoding_active = False
        self.leftover_ticks0 = np.array([], dtype=int)
        self.leftover_ticks1 = np.array([], dtype=int)
        self.sync_index = 0

    def start_acquisition(self):
        global is_running, acq_thread, decode_thread, gui_thread
        is_running = True
        self.start_button.config(state='disabled')
        self.stop_button.config(state='normal')
        self.save_button.config(state='normal')
        acq_thread = threading.Thread(target=self.acquisition_loop, daemon=True)
        decode_thread = threading.Thread(target=self.decode_loop, daemon=True)
        gui_thread    = threading.Thread(target=self.gui_loop, daemon=True)
        acq_thread.start()
        decode_thread.start()
        gui_thread.start()

    def stop_acquisition(self):
        global is_running
        is_running = False
        self.start_button.config(state='normal')
        self.stop_button.config(state='disabled')

    def acquisition_loop(self):
        while is_running:
            try:
                ts0 = self.ts_tasks[0].read(number_of_samples_per_channel=0)
                ts1 = self.ts_tasks[1].read(number_of_samples_per_channel=0)
                if ts0 or ts1:
                    try:
                        self.data_queue.put((np.array(ts0), np.array(ts1)), timeout=0.01)
                    except Full:
                        print("⚠️ Processing queue full — dropping batch")
                time.sleep(0.001)
            except Exception as e:
                messagebox.showerror("Acquisition Error", str(e))
                break

    def decode_loop(self):
        tick_us = 3.0
        zero_ticks = 12
        while is_running:
            try:
                ts0, ts1 = self.data_queue.get(timeout=0.1)
            except:
                continue

            # Compute ticks for this batch
            dt0 = np.diff(ts0) * 1e6
            dt1 = np.diff(ts1) * 1e6
            ticks0 = np.round(dt0 / tick_us).astype(int)
            ticks1 = np.round(dt1 / tick_us).astype(int)

            # Prepend leftover ticks from previous batch
            ticks0 = np.concatenate([self.leftover_ticks0, ticks0])
            ticks1 = np.concatenate([self.leftover_ticks1, ticks1])

            # Decode frames
            i = 0
            # Find sync if not already aligned
            if not self.decoding_active:
                sync_idxs = np.where(ticks0 == 56)[0]
                if sync_idxs.size:
                    self.decoding_active = True
                    i = sync_idxs[0]

            while self.decoding_active and (i + 8) < len(ticks0):
                # Extract status, data and CRC
                status0 = ticks0[i+1] - zero_ticks
                status1 = ticks1[i+1] - zero_ticks
                data0 = [(ticks0[i+2+k] - zero_ticks) for k in range(6)]
                data1 = [(ticks1[i+2+k] - zero_ticks) for k in range(6)]
                crc0   = ticks0[i+8] - zero_ticks
                crc1   = ticks1[i+8] - zero_ticks

                # Sensor values
                s1 = (data0[0] << 8) | (data0[1] << 4) | data0[2]
                s2 = (data1[0] << 8) | (data1[1] << 4) | data1[2]

                # CRC validation
                exp0 = compute_crc4([status0] + data0)
                exp1 = compute_crc4([status1] + data1)
                ok0 = (crc0 == exp0)
                ok1 = (crc1 == exp1)

                line = (
                    f"[Live] S1={s1}, S2={s2}, "
                    f"CRC0={crc0:1X}({'OK' if ok0 else 'ERR'}), "
                    f"CRC1={crc1:1X}({'OK' if ok1 else 'ERR'})\n"
                )
                try:
                    self.ui_queue.put(line, timeout=0.01)
                except Full:
                    print("⚠️ UI queue full — dropping line")

                # Advance to the next frame
                i += 9

            # Save leftovers (< one frame) for next batch
            self.leftover_ticks0 = ticks0[i:]
            self.leftover_ticks1 = ticks1[i:]
            self.decoding_active = False

    def gui_loop(self):
        while is_running:
            try:
                line = self.ui_queue.get(timeout=0.1)
                self.decode_box.after(0, lambda l=line: self._insert_line(l))
            except:
                continue

    def _insert_line(self, line):
        self.decode_box.insert(tk.END, line)
        # limit display size
        total = int(self.decode_box.index('end-1c').split('.')[0])
        if total > MAX_DISPLAY_LINES:
            self.decode_box.delete('1.0', f"{total-MAX_DISPLAY_LINES+1}.0")

    def save_data(self):
        filename = filedialog.asksaveasfilename(defaultextension=".csv",
                                                filetypes=[("CSV files","*.csv")])
        if not filename:
            return
        try:
            lines = self.decode_box.get('1.0', tk.END).strip().splitlines()
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Decoded SENT frames with CRC"])
                for l in lines:
                    writer.writerow([l])
            messagebox.showinfo("Saved", f"Decoded data saved to {filename}")
        except Exception as e:
            messagebox.showerror("Save Error", str(e))

    def exit_program(self):
        global is_running
        is_running = False
        for task in self.ts_tasks:
            task.close()
        self.root.quit()
        self.root.destroy()

# === Launch === #
if __name__ == '__main__':
    root = tk.Tk()
    app = SentTimestampApp(root)
    root.mainloop()
