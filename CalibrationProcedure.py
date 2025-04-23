# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 10:48:48 2025

BCI test system for controlling injected power into the DUT
Including a user interface
User can input the target frequency
Save the calibration results which are the RF generator power inputs in a table.
Use the table at DUT test

@author: QLin
"""

import pyvisa
import time
import csv
import numpy as np
import tkinter as tk
from tkinter import ttk
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import threading
import nidaqmx
from nidaqmx.constants import LineGrouping, AcquisitionType

# ------------------------------
# Global instrument handles
# ------------------------------
is_busy = False
stop_calibration = False
# ------------------------------
rm = None
rf_gen = None
power_meter = None
selected_row_item = None
selected_row_values = []
manual_rf_monitoring = False

# ------------------------------
# Tkinter UI
# ------------------------------

from tkinter import filedialog

def load_frequency_table(freq_table_tree, status_var):
    from tkinter import messagebox
    import csv

    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if not file_path:
        status_var.set("⚠️ Load cancelled.")
        return

    try:
        with open(file_path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            required_cols = ["Frequency (MHz)", "Target Current (mA)"]

            # Check column headers
            if not all(col in reader.fieldnames for col in required_cols):
                messagebox.showerror("Invalid CSV",
                                     f"CSV must contain columns: {', '.join(required_cols)}")
                status_var.set("❌ Invalid CSV structure.")
                return

            # Clear the old table
            freq_table_tree.delete(*freq_table_tree.get_children())

            row_count = 0
            skipped = 0
            
            for idx, row in enumerate(reader, start=1):
                freq_str = row.get("Frequency (MHz)", "").strip()
                target_str = row.get("Target Current (mA)", "").strip()
            
                if not freq_str or not target_str:
                    skipped += 1
                    continue
            
                try:
                    freq = float(freq_str)
                    target = float(target_str)
                    tag = 'odd' if idx % 2 else 'even'
                    freq_table_tree.insert("", "end", values=[
                        idx, freq, "", target, "", "", "", ""
                    ],tags = (tag,))

                    row_count += 1
                except ValueError:
                    skipped += 1
                    continue

            if row_count == 0:
                status_var.set("⚠️ No valid rows found.")
            else:
                status_var.set(f"✅ Loaded {row_count} rows from {file_path}. Skipped {skipped} invalid rows.")

    except FileNotFoundError:
        status_var.set("⚠️ File not found.")
    except csv.Error as e:
        status_var.set(f"❌ CSV parsing error: {e}")
    except Exception as e:
        status_var.set(f"❌ Unexpected error: {e}")

def start_dut_recording():
    import datetime
    import pandas as pd
    global single_button, start_recording_button, run_dut_button, exit_button
    status_var.set("📡 DUT test sequence started...")
    single_button.config(state='disabled')
    start_recording_button.config(state='disabled')
    run_dut_button.config(state='disabled')
    exit_button.config(state='disabled')


    try:
        sig_type = signal_type_var.get().lower()
        num_channels = int(channel_count_entry.get())
        duration = float(duration_entry.get())
        sample_rate = float(sample_rate_entry.get())
        
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'dut_log_{sig_type}_{timestamp}.csv'

        with nidaqmx.Task() as task:
            if sig_type in ['digital', 'both']:
                digital_lines = f'Dev1/port0/line0:{num_channels-1}'
                task.di_channels.add_di_chan(digital_lines, line_grouping=LineGrouping.CHAN_PER_LINE)
                task.timing.cfg_samp_clk_timing(
                    rate=float(sample_rate_entry.get()),
                    sample_mode=AcquisitionType.FINITE,
                    samps_per_chan=int(float(sample_rate_entry.get()) * duration)
                )


            if sig_type in ['analog', 'both']:
                for ch in range(num_channels):
                    task.ai_channels.add_ai_voltage_chan(f'Dev1/ai{ch}')

            task.timing.cfg_samp_clk_timing(sample_rate,
                                            sample_mode=AcquisitionType.FINITE,
                                            samps_per_chan=int(sample_rate * duration))

            data = task.read(number_of_samples_per_channel=int(sample_rate * duration))

            # Flatten and time tag
            time_vector = [i / sample_rate for i in range(len(data[0]) if isinstance(data[0], list) else len(data))]
            if sig_type == 'digital' and isinstance(data, list):
                data = list(zip(*data))

            df = pd.DataFrame(data)
            df.insert(0, "Time (s)", time_vector[:len(df)])
            df.to_csv(filename, index=False)

        status_var.set(f"✅ DUT signal recording complete. Saved: {filename}")
        single_button.config(state='normal')
        start_recording_button.config(state='normal')
        run_dut_button.config(state='normal')
        exit_button.config(state='normal')


    except Exception as e:
        status_var.set(f"❌ Error recording signals: {e}")

def connect_instruments():
    global rm, rf_gen, power_meter
    if is_busy:
        status_var.set("⚠️ Cannot reconnect while calibration or test is in progress.")
        return

    status_var.set("Status: Connecting instruments...")

    try:
        
        rm = pyvisa.ResourceManager()
        rf_gen = rm.open_resource('USB0::0x0957::0x2018::0115001087::INSTR')
        power_meter = rm.open_resource('GPIB0::13::INSTR')
    
        rf_gen.write('*RST')
        rf_gen.write('RFO:STAT OFF')
        rf_gen.write('MOD:STAT OFF')
        rf_gen.write('FREQ:CW 1 MHz')
        rf_gen.write('AMPL:CW 0 dBm')
        power_meter.write('UNIT:POW DBM')
        power_meter.write('CONFigure1:SCALar:POWer:AC')
        power_meter.write('CONFigure2:SCALar:POWer:AC')
    
        status_var.set("✅ Instruments connected and initialized.")
    except pyvisa.VisaIOError as e:
        status_var.set(f"❌ VISA error: {e}")
    except Exception as e:
        status_var.set(f"❌ General connection error: {e}")


def update_manual_current_loop(zt_ohms):
    global manual_freq_entry,manual_amp_entry,manual_measured_current_var,manual_target_current_var,manual_zt_entry
    
    def dBm_to_watts(dBm):
        return 10 ** ((dBm - 30) / 10)

    if not manual_rf_monitoring:
        return

    try:
        power_meter.write('INIT1')
        reading = power_meter.query('READ1?').strip()
        power_watts = dBm_to_watts(float(reading))
        v_rms = np.sqrt(power_watts * 50)
        current_mA = (v_rms / zt_ohms) * 1000
        manual_measured_current_var.set(f"{current_mA:.2f}")
    except Exception as e:
        manual_measured_current_var.set("--")
        status_var.set(f"⚠️ Current read error: {e}")

    # Repeat after 200ms
    root.after(200, lambda: update_manual_current_loop(zt_ohms))


def manual_rf_setup():
    global manual_freq_entry,manual_amp_entry,manual_measured_current_var,manual_target_current_var,manual_zt_entry
    global manual_rf_monitoring
    try:
        freq = float(manual_freq_entry.get())
        amp = float(manual_amp_entry.get())
        zt = float(zt_entry.get())  # Use shared Zt

        if rf_gen:
            rf_gen.write(f'FREQ:CW {freq} MHz')
            rf_gen.write(f'AMPL:CW {amp} dBm')
            rf_gen.write('RFO:STAT ON')

            manual_target_current_var.set(target_current_entry.get())
            manual_rf_monitoring = True
            update_manual_current_loop(zt)

            status_var.set(f"✅ RF ON @ {freq} MHz, {amp} dBm. Monitoring current...")

        else:
            status_var.set("❌ RF generator not connected.")

    except ValueError:
        status_var.set("❌ Invalid input.")
    except Exception as e:
        status_var.set(f"❌ Manual RF setup error: {e}")

def manual_rf_off():
    global manual_rf_monitoring
    try:
        manual_rf_monitoring = False
        if rf_gen:
            rf_gen.write('RFO:STAT OFF')
            status_var.set("🛑 RF generator turned OFF.")
        else:
            status_var.set("❌ RF generator not connected.")
    except Exception as e:
        status_var.set(f"⚠️ Error turning off RF: {e}")


def run_single_frequency_calibration():
    global stop_calibration, is_busy
    global selected_row_item, selected_row_values

    is_busy = True
    status_var.set("📡 Calibration in progress...")

    global single_freq_entry, target_current_entry, zt_entry
    global kp_entry, ki_entry, kd_entry,initial_power_entry

    try:
        stop_calibration = False

        # Get user inputs
        target_current_mA = float(target_current_entry.get())
        frequency = float(single_freq_entry.get())   # MHz
        kp = float(kp_entry.get())
        ki = float(ki_entry.get())
        kd = float(kd_entry.get())
        zt_lookup = float(zt_entry.get())

        # Helper functions
        #Convert power from dBm to watts
        def dBm_to_watts(dBm):
            return 10 ** ((dBm - 30) / 10)
       
        def measure_power_watts():
            power_meter.write('INIT1')
            reading = power_meter.query('READ1?').strip()
            return dBm_to_watts(float(reading))
        #Convert watts to voltage:
        def compute_current(power_watts, zt_ohms):
            v_rms = np.sqrt(power_watts * 50)
            return v_rms / zt_ohms

        # Prepare calibration loop
        rf_gen.write(f'FREQ:CW {frequency} MHz')
        rf_gen.write('RFO:STAT ON')

        current_mA = 0
        error_integral = 0
        last_error = 0
        
        step = 0
        max_power = 10
        min_power = -30

        try:
            set_power = float(initial_power_entry.get())
        except:
            set_power = -20  # fallback default

# calibration tolerance is +/-0.5dB; convert it into mA is about +/-5.9%
        while abs(current_mA - target_current_mA)/target_current_mA > 0.055 and step < 100 and not stop_calibration:
            # Re-fetch gains in case user changes them mid-loop
            kp = float(kp_entry.get())
            ki = float(ki_entry.get())
            kd = float(kd_entry.get())

            rf_gen.write(f'AMPL:CW {set_power} dBm')
            time.sleep(0.5)

            power_watts = measure_power_watts()
            current_mA = compute_current(power_watts, zt_lookup) * 1000

            error = target_current_mA - current_mA
            error_integral += error
            error_derivative = error - last_error

            adjustment = kp * error + ki * error_integral + kd * error_derivative
            set_power += adjustment
            set_power = max(min(set_power, max_power), min_power)

            last_error = error
            step += 1

        rf_gen.write('RFO:STAT OFF')

        # Final power readings
        power_meter.write('INIT1')
        measurecurrentP = float(power_meter.query('READ1?').strip())
        power_meter.write('INIT2')
        forwP = float(power_meter.query('READ2?').strip())

        import datetime
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'bci_single_freq_result_{int(frequency)}MHz_{timestamp}.csv'
#save original data and computed power based on current measurement probe +power meter
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Frequency (MHz)', 'Set Power (dBm)', 'Forward Power (dBm)',
                             'Reflected Power (dBm)', 'Injected Current (mA)','Zt(ohm)'])
            writer.writerow([frequency, set_power, forwP, measurecurrentP, current_mA, zt_lookup])

        # Update UI entries with calibrated values
        single_freq_entry.delete(0, tk.END)
        single_freq_entry.insert(0, str(frequency))
        target_current_entry.delete(0, tk.END)
        target_current_entry.insert(0, str(round(current_mA, 2)))
        zt_entry.delete(0, tk.END)
        zt_entry.insert(0, str(zt_lookup))

        # ✅ Update the selected row in the Treeview if one is selected
        if selected_row_item and selected_row_values:
            point_number = selected_row_values[0]  # "Point #" from row
            freq_table_tree.item(selected_row_item, values=[
                point_number,                          # Point #
                frequency,                             # Freq (MHz)
                round(set_power, 2),                   # Amplitude (dBm)
                target_current_mA,                     # Target (mA)
                zt_lookup,                             # Zt
                round(kp, 3), round(ki, 3), round(kd, 3)  # PID values
            ])

        status_var.set(f"✅ Calibration complete. Results saved to: {filename}")

    except Exception as e:
        status_var.set(f"❌ Calibration error: {e}")

    finally:
        is_busy = False       

def stop_calibration_process():
    global stop_calibration
    stop_calibration = True
    status_var.set("🛑 Calibration stop requested.")
#######DUT Test Mode############
# It assumes the RF generator and power meter are already connected and initialized.
# This mode will:
# Load the power calibration result from the calibrated table
# Set the RF generator to calibrated power
# Measure forward and reflected power
# Log DUT test data (maybe to bci_dut_test_result.csv)
#################    
def record_digital_signal_and_save(filename, duration, sample_rate, channels):
    import nidaqmx
    import numpy as np

    try:
        with nidaqmx.Task() as task:
            digital_lines = f'Dev1/port0/line0:{channels-1}'
            task.di_channels.add_di_chan(digital_lines, line_grouping=LineGrouping.CHAN_PER_LINE)

            task.timing.cfg_samp_clk_timing(
                rate=sample_rate,
                sample_mode=AcquisitionType.FINITE,
                samps_per_chan=int(sample_rate * duration)
            )

            data = task.read(number_of_samples_per_channel=int(sample_rate * duration))
            data = np.array(data).T
            time_col = np.arange(data.shape[0]) / sample_rate
            output = np.column_stack((time_col, data))

            header = ['Time (s)'] + [f'Digital_{i}' for i in range(channels)]
            np.savetxt(filename, output, delimiter=",", header=",".join(header), comments='', fmt='%s')

    except Exception as e:
        print(f"❌ Error saving digital signal: {e}")

def run_dut_test_sequence(dwell_time_sec):
    global start_recording_button, is_busy
    is_busy = True
    status_var.set("📡 DUT test sequence started...")
    start_recording_button.config(state='disabled')

    try:
        import datetime
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        summary_log = []

        for row_id in freq_table_tree.get_children():
            values = freq_table_tree.item(row_id)['values']
            try:
                freq = float(values[1])
                power = float(values[2])  # already calibrated
            except (IndexError, ValueError):
                status_var.set(f"⚠️ Skipping row with invalid values: {values}")
                continue
            freq_table_tree.selection_set(row_id)
            freq_table_tree.see(row_id)  # Scrolls into view

            status_var.set(f"📶 Testing {freq} MHz...")
       #====== RF setup ======
            rf_gen.write(f'FREQ:CW {freq} MHz')
            rf_gen.write(f'AMPL:CW {power} dBm')
            rf_gen.write('RFO:STAT ON')
            time.sleep(0.05)  # Short stabilization time. this setup can refer to the RF generator manual for the value

            signal_filename = f'sent_log_{int(freq)}MHz_{timestamp}.csv'
            thread = threading.Thread(
                target=record_digital_signal_and_save,
                args=(signal_filename, dwell_time_sec, 10_000_000, 2)
            )
            thread.start()
            thread.join()

            power_meter.write('INIT1')
            fwd = float(power_meter.query('READ1?').strip())
            power_meter.write('INIT2')
            refl = float(power_meter.query('READ2?').strip())

            rf_gen.write('RFO:STAT OFF')

            summary_log.append([freq, power, fwd, refl, signal_filename])

        summary_filename = f'dut_test_summary_{timestamp}.csv'
        with open(summary_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Frequency (MHz)', 'Set Power (dBm)', 'Forward Power (dBm)',
                             'Reflected Power (dBm)', 'Signal Log File'])
            writer.writerows(summary_log)

        status_var.set(f"✅ Full DUT test complete. Summary saved to: {summary_filename}")
        start_recording_button.config(state='normal')

    except Exception as e:
        status_var.set(f"❌ DUT test sequence error: {e}")

    finally:
        is_busy = False


def exit_program():
    try:
        if rf_gen:
            rf_gen.write('OUTP OFF')
            rf_gen.close()
        if power_meter:
            power_meter.close()
        if rm:
            rm.close()
        status_var.set("🛑 Instruments safely disconnected.")
    except Exception as e:
        status_var.set(f"⚠️ Error during exit: {e}")
    root.quit()
    root.destroy()
# ------------------------------
# ------------------------------
# Build UI
def main():
    import matplotlib
    matplotlib.use("TkAgg")  # Ensures plot stays in the GUI
    plt.ioff()  # Disable interactive plots

    global root, status_var
    global single_freq_entry, target_current_entry, zt_entry, kp_entry, ki_entry, kd_entry,initial_power_entry
    global start_freq_entry, end_freq_entry, step_freq_entry
    global signal_type_var, channel_count_entry, duration_entry, sample_rate_entry, dwell_time_entry
    global manual_freq_entry,manual_amp_entry,zt_entry,manual_measured_current_var,manual_target_current_var,manual_zt_entry
    global fig, ax, canvas
    global single_button, run_dut_button, exit_button
    root = tk.Tk()
    root.title("BCI Calibration Control Panel")

    status_var = tk.StringVar(value="Status: Waiting for action...")

    # === Main Frame ===
    frame = ttk.Frame(root, padding=10)
    frame.grid(row=0, column=0, sticky="nw")

    # === Status Label ===
    status_label = ttk.Label(frame, textvariable=status_var, foreground='blue')
    status_label.grid(column=0, row=0, columnspan=3, sticky=tk.W, pady=(0, 10))

    # === Controls and Calibration Side-by-Side ===
    top_frame = ttk.Frame(frame)
    top_frame.grid(row=1, column=0, columnspan=3, sticky="nw")

    # --- Controls ---
    control_frame = ttk.LabelFrame(top_frame, text="🛠 Controls", padding=10)
    control_frame.grid(row=0, column=0, padx=(0, 20), pady=10, sticky="nw")

    ttk.Button(control_frame, text="Connect Instruments", width=20, command=connect_instruments).grid(row=0, column=0, pady=5)
    ttk.Button(control_frame, text="Stop Calibration", width=20, command=stop_calibration_process).grid(row=1, column=0, pady=5)
    exit_button = ttk.Button(control_frame, text="Exit Program", width=20, command=exit_program)
    exit_button.grid(row=2, column=0, pady=5)

    # --- Single Frequency Calibration ---
    single_frame = ttk.LabelFrame(top_frame, text="📦 Single Frequency Calibration", padding=10)
    single_frame.grid(row=0, column=1, pady=10, sticky="nw")

    single_labels = ["Single Frequency (MHz)", "Target Current (mA)",'RF Init. Amplitude (dBm)', "Zt (Ω)", "Kp", "Ki", "Kd"]
    single_entries = []
    for i, label in enumerate(single_labels):
        ttk.Label(single_frame, text=label).grid(column=0, row=i, sticky=tk.W, pady=2)
        entry = ttk.Entry(single_frame)
        entry.grid(column=1, row=i, pady=2)
        single_entries.append(entry)
    single_freq_entry, target_current_entry, initial_power_entry,zt_entry, kp_entry, ki_entry, kd_entry = single_entries
    single_freq_entry.insert(0, "1")
    target_current_entry.insert(0, "100")
    initial_power_entry.insert(0,"-20")
    zt_entry.insert(0, "1.0")
    kp_entry.insert(0, "0.8")
    ki_entry.insert(0, "0.1")
    kd_entry.insert(0, "0")

    single_button = ttk.Button(single_frame, text="Run Single Frequency Calibration",
        command=lambda: threading.Thread(target=run_single_frequency_calibration, daemon=True).start())
    single_button.grid(column=0, row=len(single_labels), columnspan=2, pady=5)
    # ======
    style = ttk.Style()
    style.theme_use("default")
    
    style.configure("Treeview",
        rowheight=25,
        borderwidth=1,
        relief="solid"
    )
    
    style.map("Treeview", background=[("selected", "#cce5ff")])

    
    table_frame = ttk.LabelFrame(frame, text="📋 Imported Frequency Sweep Table", padding=10)
    table_frame.grid(row=4, column=0, columnspan=3, padx=10, pady=10, sticky="w")
    columns = ["Point #", "Freq (MHz)", "Amplitude (dBm)", "Target (mA)", "Zt (Ω)", "Kp", "Ki", "Kd"]
    global freq_table_tree
    freq_table_tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=6)
    style = ttk.Style()
    style.theme_use("default")
    
    # Simulated "bordered" look
    style.configure("Treeview",
        rowheight=28,
        borderwidth=1,
        relief="solid",
        font=('Segoe UI', 10),
    )
    
    style.configure("Treeview.Heading",
        font=('Segoe UI', 10, 'bold')
    )
    
    # Alternate row coloring
    freq_table_tree.tag_configure('odd', background='#f2f2f2')
    freq_table_tree.tag_configure('even', background='#ffffff')

    for col in columns:
        freq_table_tree.heading(col, text=col, anchor = "center")
        freq_table_tree.column(col, width=100, anchor = "center")
    
    freq_table_tree.grid(row=0, column=0, columnspan=3, sticky="nsew")
    
    def on_table_row_selected(event):
        global selected_row_item, selected_row_values
        selected_item = freq_table_tree.focus()
        
        if not selected_item:
            return
    
        values = freq_table_tree.item(selected_item, "values")
       
        if not values or len(values) < 4:
            status_var.set("❌ Incomplete row selected.")
            return
    
        try:
            # Safely convert Frequency and Target
            freq_str = values[1]  # Freq (MHz)
            target_str = values[3]  # Target (mA)
    
            if freq_str.strip() == "" or target_str.strip() == "":
                raise ValueError("Frequency or Target is empty")
    
            freq = float(freq_str)
            target = float(target_str)
    
            single_freq_entry.delete(0, tk.END)
            single_freq_entry.insert(0, str(freq))
    
            target_current_entry.delete(0, tk.END)
            target_current_entry.insert(0, str(target))
            selected_row_item = selected_item
            selected_row_values = values
            status_var.set(f"✅ Row {values[0]} selected: Frequency = {freq} MHz, Target = {target} mA")
    
        except ValueError:
            status_var.set("⚠️ Selected row contains non-numeric frequency or target.")

            
    freq_table_tree.bind("<<TreeviewSelect>>", on_table_row_selected)
    ttk.Button(table_frame, text="📂 Load Table",
               command=lambda: load_frequency_table(freq_table_tree, status_var)
    ).grid(row=1, column=0, pady=5)
 
    def run_selected_row_calibration():
        selected = freq_table_tree.focus()
        if not selected:
            status_var.set("⚠️ No row selected.")
            return
        values = freq_table_tree.item(selected)["values"]
        if not values:
            return
        try:
            _,freq, _, target, _, _, _, _ = values
            freq = float(freq)
            target = float(target)
    
            # Set values into existing entries
            single_freq_entry.delete(0, tk.END)
            single_freq_entry.insert(0, str(freq))
            target_current_entry.delete(0, tk.END)
            target_current_entry.insert(0, str(target))
    
            # Run calibration for selected row
            run_single_frequency_calibration()
    
        except ValueError as e:
            status_var.set(f"❌ Invalid input in row: {e}")
    # --- Manual RF Control ---
    manual_frame = ttk.LabelFrame(top_frame, text="🖐 Manual RF Generator Control", padding=10)
    manual_frame.grid(row=0, column=2, pady=10, padx=(20, 0), sticky="nw")
    
    ttk.Label(manual_frame, text="Frequency (MHz)").grid(row=0, column=0, sticky=tk.W)
    manual_freq_entry = ttk.Entry(manual_frame)
    manual_freq_entry.insert(0, "1")
    manual_freq_entry.grid(row=0, column=1)
    
    ttk.Label(manual_frame, text="Amplitude (dBm)").grid(row=1, column=0, sticky=tk.W)
    manual_amp_entry = ttk.Entry(manual_frame)
    manual_amp_entry.insert(0, "0")
    manual_amp_entry.grid(row=1, column=1)
    
    ttk.Label(manual_frame, text="Zt (Ω)").grid(row=2, column=0, sticky=tk.W)
    manual_zt_entry = ttk.Entry(manual_frame)
    manual_zt_entry.insert(0, "0")
    manual_zt_entry.grid(row=2, column=1)
    
    
    ttk.Button(manual_frame, text="Set RF",
               command=lambda: threading.Thread(target=manual_rf_setup, daemon=True).start()).grid(row=3, column=0, columnspan=2, pady=5)
    
    ttk.Button(manual_frame, text="Turn RF OFF",
               command=lambda: threading.Thread(target=manual_rf_off, daemon=True).start()).grid(row=4, column=0, columnspan=2, pady=5)
    
    # Current and Target Display
    ttk.Label(manual_frame, text="Measured Current (mA):").grid(row=5, column=0, sticky=tk.W)
    manual_measured_current_var = tk.StringVar(value="--")
    ttk.Label(manual_frame, textvariable=manual_measured_current_var).grid(row=5, column=1)
    
    ttk.Label(manual_frame, text="Target Current (mA):").grid(row=6, column=0, sticky=tk.W)
    manual_target_current_var = tk.StringVar(value="--")
    ttk.Label(manual_frame, textvariable=manual_target_current_var).grid(row=6, column=1)
        

    # === DUT Signal Recorder ===
    recorder_frame = ttk.LabelFrame(frame, text="📥 DUT Signal Recorder", padding=10)
    recorder_frame.grid(row=2, column=0, columnspan=3, padx=10, pady=5, sticky="w")

    ttk.Label(recorder_frame, text="Signal Type").grid(row=0, column=0, sticky=tk.W)
    signal_type_var = tk.StringVar()
    signal_type_dropdown = ttk.Combobox(recorder_frame, textvariable=signal_type_var,
                                        values=["Digital", "Analog", "Both"], width=10)
    signal_type_dropdown.grid(row=0, column=1, padx=5, pady=2)
    signal_type_dropdown.current(0)

    ttk.Label(recorder_frame, text="Channels").grid(row=0, column=2, sticky=tk.W)
    channel_count_entry = ttk.Entry(recorder_frame, width=5)
    channel_count_entry.insert(0, "2")
    channel_count_entry.grid(row=0, column=3, padx=5)

    ttk.Label(recorder_frame, text="Duration (s)").grid(row=0, column=4, sticky=tk.W)
    duration_entry = ttk.Entry(recorder_frame, width=5)
    duration_entry.insert(0, "4")
    duration_entry.grid(row=0, column=5, padx=5)

    ttk.Label(recorder_frame, text="Sample Rate (Hz)").grid(row=1, column=0, sticky=tk.W)
    sample_rate_entry = ttk.Entry(recorder_frame, width=10)
    sample_rate_entry.insert(0, "10000000")
    sample_rate_entry.grid(row=1, column=1, padx=5)

    ttk.Label(recorder_frame, text="Dwell Time (s)").grid(row=2, column=0, sticky=tk.W)
    dwell_time_entry = ttk.Entry(recorder_frame, width=5)
    dwell_time_entry.insert(0, "4")
    dwell_time_entry.grid(row=2, column=1, padx=5)

    # global start_recording_button
    # start_recording_button = ttk.Button(recorder_frame, text="Start Recording",
    #     command=lambda: threading.Thread(target=start_dut_recording, daemon=True).start()
    # )
    # start_recording_button.grid(row=0, column=6, padx=10)

    global run_dut_button
    run_dut_button = ttk.Button(recorder_frame, text="Run Full DUT Test",
        command=lambda: threading.Thread(
            target=lambda: run_dut_test_sequence(dwell_time_sec=float(dwell_time_entry.get())),
            daemon=True
        ).start()
    )
    run_dut_button.grid(row=2, column=6, padx=10)

    # === Plot Panel ===
    fig, ax = plt.subplots(figsize=(8, 4))
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().grid(row=3, column=0, columnspan=3, padx=10, pady=10)


if __name__ == "__main__":
    main()
    root.mainloop()
