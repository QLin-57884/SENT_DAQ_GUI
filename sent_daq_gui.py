# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 12:15:23 2025

@author: QLin
"""
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import nidaqmx
from nidaqmx.constants import AcquisitionType, LineGrouping
import threading
import time
import csv
from queue import Queue, Full

# === Config === #
DEVICE_NAME = "Dev1"
LINE_NAME = "port0/line0"
SAMPLE_RATE = 2_000_000
CHUNK_SIZE = 20000  # adjusted for ~10ms resolution
BUFFER_DURATION_SEC = 2
BUFFER_SIZE = SAMPLE_RATE * BUFFER_DURATION_SEC

# === Globals === #
rolling_buffer = np.zeros(BUFFER_SIZE, dtype=int)
is_running = False
acq_thread = None
decode_thread = None

# === GUI App === #
class SentDAQApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SENT Signal Acquisition GUI")
        self.root.geometry("1200x800")

        self.start_button = ttk.Button(root, text="Start", command=self.start_acquisition)
        self.stop_button = ttk.Button(root, text="Stop", command=self.stop_acquisition, state='disabled')
        self.save_button = ttk.Button(root, text="Save Buffer", command=self.save_data, state='disabled')
        self.exit_button = ttk.Button(root, text="Exit", command=self.exit_program)

        self.start_button.grid(row=0, column=0, padx=10, pady=10)
        self.stop_button.grid(row=0, column=1, padx=10, pady=10)
        self.save_button.grid(row=0, column=2, padx=10, pady=10)
        self.exit_button.grid(row=0, column=3, padx=10, pady=10)

        self.fig, self.ax = plt.subplots(figsize=(12, 4))
        self.line, = self.ax.step(np.arange(BUFFER_SIZE) / SAMPLE_RATE * 1e3, rolling_buffer, where='post')
        self.ax.set_ylim(-0.2, 1.2)
        self.ax.set_xlim(0, BUFFER_SIZE / SAMPLE_RATE * 1e3)
        self.ax.set_xlabel("Time (ms)")
        self.ax.set_ylabel("Signal")
        self.ax.set_title("Live SENT Signal")
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().grid(row=1, column=0, columnspan=4, sticky="nsew")

        self.decode_box = tk.Text(root, height=15, width=100)
        self.decode_box.grid(row=2, column=0, columnspan=4, pady=(10, 10))

        self.prev_chunk = np.array([], dtype=int)
        self.decoding_active = False
        self.sync_index_in_ticks = 0
        self.data_queue = Queue(maxsize=200)

    def start_acquisition(self):
        global is_running, acq_thread, decode_thread
        is_running = True
        self.start_button.config(state='disabled')
        self.stop_button.config(state='normal')
        self.save_button.config(state='normal')
        acq_thread = threading.Thread(target=self.acquisition_loop, daemon=True)
        decode_thread = threading.Thread(target=self.decode_worker, daemon=True)
        acq_thread.start()
        decode_thread.start()
        self.update_plot()

    def stop_acquisition(self):
        global is_running
        is_running = False
        self.start_button.config(state='normal')
        self.stop_button.config(state='disabled')

    def acquisition_loop(self):
        global rolling_buffer
        try:
            with nidaqmx.Task() as task:
                task.di_channels.add_di_chan(f"{DEVICE_NAME}/{LINE_NAME}", line_grouping=LineGrouping.CHAN_PER_LINE)
                task.timing.cfg_samp_clk_timing(SAMPLE_RATE, sample_mode=AcquisitionType.CONTINUOUS)
                task.in_stream.input_buf_size = int(BUFFER_SIZE * 1.5)
                task.start()

                self.prev_chunk = np.array([], dtype=int)

                while is_running:
                    new_data = task.read(number_of_samples_per_channel=CHUNK_SIZE, timeout=10.0)
                    new_data = np.array(new_data, dtype=int)

                    combined_data = np.concatenate((self.prev_chunk, new_data))
                    self.prev_chunk = new_data[-1000:]

                    try:
                        self.data_queue.put(combined_data, timeout=0.01)
                    except Full:
                        print("⚠️ Queue full — dropped a chunk!")

                    rolling_buffer = np.roll(rolling_buffer, -len(new_data))
                    rolling_buffer[-len(new_data):] = new_data

        except Exception as e:
            messagebox.showerror("DAQ Error", str(e))

    def update_plot(self):
        if is_running:
            self.line.set_ydata(rolling_buffer)
            self.canvas.draw()
            self.root.after(50, self.update_plot)

    def decode_worker(self):
        while is_running:
            if not self.data_queue.empty():
                chunk = self.data_queue.get()
                self.decode_sent_chunk(chunk)

    def decode_sent_chunk(self, signal_chunk, sampling_rate=1/2_000_000):
        tick_time = 3.0e-6
        samples_per_tick = tick_time / sampling_rate
        sync_ticks = 56
        zero_ticks = 12
        threshold_voltage = 3.0

        binary_signal = (signal_chunk > threshold_voltage).astype(np.int32)
        falling_edges = np.where(np.diff(binary_signal) < 0)[0]
        ticks = np.diff(falling_edges) / samples_per_tick
        ticks = ticks.astype(np.int32)

        if not self.decoding_active:
            sync_indices = np.where(ticks == sync_ticks)[0]
            if len(sync_indices) > 0:
                self.decoding_active = True
                self.sync_index_in_ticks = sync_indices[0]
            else:
                return

        i = self.sync_index_in_ticks
        while i + 8 < len(ticks):
            try:
                s1 = ((ticks[i+2] - zero_ticks) << 8) + \
                     ((ticks[i+3] - zero_ticks) << 4) + \
                     (ticks[i+4] - zero_ticks)

                s2 = ((ticks[i+7] - zero_ticks) << 8) + \
                     ((ticks[i+6] - zero_ticks) << 4) + \
                     (ticks[i+5] - zero_ticks)

                self.decode_box.insert(tk.END, f"[Live] Sensor1 = {s1}, Sensor2 = {s2}\n")
                i += 9
            except:
                self.decoding_active = False
                break

    def save_data(self):
        filename = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if not filename:
            return
        timestamps = np.arange(BUFFER_SIZE) / SAMPLE_RATE
        try:
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Time (s)", "Signal"])
                writer.writerows(zip(timestamps, rolling_buffer))
            messagebox.showinfo("Saved", f"Data saved to {filename}")
        except Exception as e:
            messagebox.showerror("Save Error", str(e))

    def exit_program(self):
        global is_running
        is_running = False
        self.root.quit()
        self.root.destroy()

# === Launch === #
if __name__ == '__main__':
    root = tk.Tk()
    app = SentDAQApp(root)
    root.mainloop()
