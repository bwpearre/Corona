import datetime as dt
import tkinter as tk
from tkinter import filedialog, ttk, END, BooleanVar
import matplotlib
matplotlib.use("TkAgg")
matplotlib.interactive(True)
import matplotlib.pyplot as plt
import numpy as np
import pandas
from numpy import mat # matrix
from numpy.linalg import inv
from scipy import signal
from scipy.fft import fftshift
import math
import time
import pdb
from scipy.optimize import curve_fit
import traceback, sys, code
import scipy.stats
from scipy.fft import fft
import csv
from pathlib import Path
import re
import warnings


# 20311011 is good

def exponential(x, a, b, c):
        #return a + b * x + c * np.square(x)
        return a * np.exp(b*x) + c

class Events:
        def __init__(self):
                self.start_indices = []
                self.end_indices = []
                self.sizes = []

        def add(self, start, end, size):
                self.start_indices.append(start)
                self.end_indices.append(end)
                self.sizes.append(size)

                
class CoronaBrowser(tk.Frame):
        
        def __init__(self, master=None):
                tk.Frame.__init__(self, master)

                # matplotlib wants to size windows in inches. Find out the screen size in inches:
                self.screendims_inches = [self.winfo_screenmmwidth(), self.winfo_screenmmheight()]
                self.screendims_inches = [x / 25.4 for x in self.screendims_inches]

                # Define the weird date format
                self.date_format = '%m/%d/%y %I:%M:%S %p'
                self.date_format_lidar = '%Y/%m/%d %H:%M'
                
                # Helper vars:
                self.plotTemperatureWithPotential = BooleanVar()

                # Create root window:
                self.grid()
                self.createWidgets()

                # Temperature correction fit. This was computed from 20311010_450_expurgated.csv
                self.fit = mat([[-0.020992708021557917],
                               [5.272377975649473]])
                
                # Dictionary is exponential fit params indexed on sensor serial number:
                self.fits = {-1: (1.713, -0.07977, 4.493),
                             20121725: (1.3545883156167038, -0.0714642458651333, 4.482818353894192),
                             20310992: (0.28929039047675625, -0.08857734679045634, 4.447723485085831)}
                        
                # Exponential temperature fit parameters computed from 20121725_1.csv

                self.no_temperature_correction_check = True
                

        def event_detection_enabled(self, state):
                if state:
                        self.detectionVoltageLabel.grid()
                        self.detectionVoltageVoltsLabel.grid()
                        self.detectionVoltageBox.grid()
                        self.detectionVoltageSecondsLabel.grid()
                        self.detectionCountBox.grid()
                        self.rmplButton['text'] = 'Detect + plot'
                else:
                        self.detectionVoltageLabel.grid_remove()
                        self.detectionVoltageVoltsLabel.grid_remove()
                        self.detectionVoltageBox.grid_remove()
                        self.detectionVoltageSecondsLabel.grid_remove()
                        self.detectionCountBox.grid_remove()
                        self.rmplButton['text'] = 'Plot'
                
        # Set up the main window.
        def createWidgets(self):
                row = 0
                self.loadButton = tk.Button(self, text="Load", command=self.loadFile)
                self.loadButton.grid(row=row, column=0, columnspan=4)
                self.saveButton = tk.Button(self, text='Save', command=self.saveFile, state='disabled')
                self.saveButton.grid(row=row, column=6)
                row += 1
                tk.Label(self, text='Voltage scaling factor:').grid(row=row, column=0)
                self.voltageScalingFactor = 1
                self.voltageScalingFactorBox = tk.Entry(self, width=8)
                self.voltageScalingFactorBox.insert(0, self.voltageScalingFactor)
                self.voltageScalingFactorBox['state'] = 'readonly'
                self.voltageScalingFactorBox.grid(row=row, column=1, sticky='W')
                self.voltageScalingFactorBox.insert(0, self.voltageScalingFactor)
                self.voltageScalingButton = tk.Button(self, text="Apply", command=self.applyCorrections, state='disabled')
                self.voltageScalingButton.grid(row=row, column=2)
                row += 1
                self.detectionVoltageLabel = tk.Label(self, text='Detection thresholds:', font=('bold'))
                self.detectionVoltageLabel.grid(row=row, column=0)
                self.detectionVoltageVoltsLabel = tk.Label(self, text='Volts:')
                self.detectionVoltageVoltsLabel.grid(row=row, column=1, sticky='E')
                self.detectionVoltageBox = tk.Entry(self, width=5)
                self.detectionVoltageBox.grid(row=row, column=2, sticky='W');
                self.detectionVoltageSecondsLabel = tk.Label(self, text='Seconds:')
                self.detectionVoltageSecondsLabel.grid(row=row, column=3, sticky='E')
                self.detectionCountBox = tk.Entry(self, width=5)
                self.detectionCountBox.grid(row=row, column=4, sticky='W');
                row += 1
                self.rmplButton = tk.Button(self, text="Plot", command=self.plotEvents, state='disabled')
                self.rmplButton.grid(row=row, column=0)
                self.plotTemperatureWithPotentialCheck = tk.Checkbutton(self, text="with temperature if available.", variable=self.plotTemperatureWithPotential)
                self.plotTemperatureWithPotentialCheck.grid(row=row, column=1, sticky='W')
                self.regressButton = tk.Button(self, text='Plot potential vs temp', command=self.plot_temperature_corrections)
                self.useNewRegressionButton = tk.Button(self, text='Use new regression this session', command=self.use_new_correction)
                row += 1
                self.waitbar_label = tk.Label(self, text='Ready.')
                self.waitbar_label.grid(row=row, column=0, columnspan=5)
                row += 1
                self.waitbar = ttk.Progressbar(self, orient="horizontal", length=300, value=0, mode='determinate')
                self.waitbar.grid(row=row, column=0, columnspan=5)
                row += 1
                self.debugButton = tk.Button(self, text="Debug", command=self.debug)
                self.debugButton.grid(row=row, column=3)
                self.quitButton = tk.Button(self, text="Quit", command=self.close_and_quit)
                self.quitButton.grid(row=row, column=4)

                self.eventThreshold = {'volts': 0.02, 'count': 40}
                self.detectionVoltageBox.insert(0, self.eventThreshold['volts'])
                self.detectionCountBox.insert(0, self.eventThreshold['count'])
                self.event_detection_enabled(False)


        # Clean up matplotlib windows etc
        def close_and_quit(self):
                plt.close('all')
                sys.exit()
                

        # Set the waitbar up for a new task
        def waitbar_start(self, text, maxval):
                self.waitbar['mode'] = 'determinate'
                self.waitbar['maximum'] = maxval
                self.waitbar['value'] = 0
                self.waitbar_label['text'] = text
                self.waitbar_percent = 0
                self.waitbar_maxval = maxval

        # Set the waitbar up for a new task
        def waitbar_indeterminate_start(self, text):
                self.waitbar['mode'] = 'indeterminate'
                self.waitbar_label['text'] = text
                self.update()
                self.waitbar.start()

        # Set the waitbar up for a new task
        def waitbar_indeterminate_done(self):
                self.waitbar['mode'] = 'determinate'
                self.waitbar['value'] = 0
                self.waitbar_label['text'] = 'Ready.'
                self.waitbar.stop()

        # Update the waitbar
        def waitbar_update(self, val):
                # Update the waitbar 100 times over the course of the load:
                new_percent = int(val*100/self.waitbar_maxval)
                if new_percent > self.waitbar_percent:
                        self.waitbar_percent = new_percent
                        self.waitbar['value'] = val
                        self.waitbar.update()


        # Done with waitbar for now
        def waitbar_done(self):
                self.waitbar['value'] = 0
                self.waitbar_label['text'] = 'Ready.'

        def debug(self):
                pdb.set_trace()

        # Ask for a filename, load it, plot it.
        def loadFile(self):
                data_filename = filedialog.askopenfilename(filetypes=[('Comma-separated values', '*.csv')])
                if data_filename:
                        # Reset stuff
                        self.sensor_serial_number = -1
                        
                        self.regressButton.grid_forget()
                        self.useNewRegressionButton.grid_forget()
                        self.times, self.volts_raw, self.temps = self.loadFileBen(Path(data_filename))
                        self.rmplButton['state'] = 'normal'
                        self.useNewRegressionButton['state'] = 'disabled'
                        if self.temperature_present:
                                self.event_detection_enabled(False)
                        else:
                                self.event_detection_enabled(True)

                        self.applyCorrections()
                        self.saveButton['state'] = 'normal'

        def setVoltageScalingFactor(self, vsf):
                self.voltageScalingFactor = vsf
                self.voltageScalingFactorBox.delete(0, END)
                self.voltageScalingFactorBox.insert(0, self.voltageScalingFactor)
                        
        def getVoltageScalingFactor(self):
                try:
                        self.voltageScalingFactor = float(self.voltageScalingFactorBox.get())
                except:
                        print(f'Could not convert voltage scaling factor "{self.voltageScalingFactorBox.get()}" to number. Resetting to 1.')
                        self.voltageScalingFactor = 1.0
                self.voltageScalingFactorBox.delete(0, END)
                self.voltageScalingFactorBox.insert(0, self.voltageScalingFactor)


        def applyCorrections(self):
                # Voltage scaling
                self.getVoltageScalingFactor()
                self.volts_scaled = self.volts_raw * self.voltageScalingFactor

                # Correct for sensor temperature. Preserves mean value (i.e. not mean-0)
                self.volts = self.applyTemperatureCorrection()

                self.common_temp = scipy.stats.mode(self.volts)[0][0]
                if self.temperature_present:
                        t = ' after temperature correction'
                else:
                        t = ''
                print(f'Potential: mode is {self.common_temp} V{t} (fyi; not used)')
                
                self.plotEvents()


        def plotEvents(self):
                self.events = self.find_events(self.times, self.volts)
                self.plot_voltages_matplotlib(self.times, self.volts, self.temps, self.events)
                if self.no_temperature_correction_check == False:
                        self.plot_temperature_corrections()
                        

        # Correct the voltage using self.fit for temperature
        def applyTemperatureCorrection(self):
                if not self.temperature_present:
                        return self.volts_scaled

                length = len(self.temps)
                y = self.volts_scaled

                # Seems like a good place to sort out the temperature correction:
                if self.sensor_serial_number in self.fits:
                        self.fit_exp = self.fits[self.sensor_serial_number]
                else:
                        print(f'*** UPDATE ***: No temperature calibration specifically for serial number {self.sensor_serial_number}.')
                        self.fit_exp = self.fits[-1]
                
                volts = y - exponential(self.temps, *self.fit_exp) + np.mean(self.volts_scaled)
                return volts

        # Show default and new regressions from voltage-vs-temperature
        def plot_temperature_corrections(self):
                if not self.temperature_present:
                        plt.close(2)
                        return

                self.waitbar_indeterminate_start('Computing regression...')
                length = len(self.temps)
                x = mat(self.temps).reshape((length,1))
                x = np.hstack((x, np.ones((length, 1))))
                y = mat(self.volts_scaled).reshape((length,1))

                
                xn = np.reshape(self.temps, -1)
                yn = np.reshape(self.volts_scaled, -1)

                #fit_desc_old = f'V = {self.fit[0,0]} * T + {self.fit[1,0]}'
                #fit_desc_old_short = r'As applied: $V^* \approx ' + f'{self.fit[0,0]:.3g} \cdot T + {self.fit[1,0]:.3g}$'
                mse_old = np.sum(np.square(exponential(xn, *self.fit_exp) - yn))/xn.size
                print(f'\n  Saved fit is V = {self.fit_exp[0]} * exp({self.fit_exp[1]} * T) + {self.fit_exp[2]}\n       MSE = {mse_old:.8f}')
                fit_desc_old_short = r'(saved) $V \approx ' + f'{self.fit_exp[0]:.3g} \cdot \exp({self.fit_exp[1]:.3g} \cdot T) + {self.fit_exp[2]:.3g}$'

                # Compute a new linear least-squares fit:
                fit = (x.T*x).I*x.T*y
                x_linear_fit = x * fit
                mse_linear = np.sum(np.square(y - x_linear_fit[:,0]))/xn.size
                print(f'  Linear fit (this set) would be V = {fit[0,0]} * T + {fit[1,0]}\n       MSE = {mse_linear:.8f}')
                fit_desc_short = r'(this set) $V \approx ' + f'{fit[0,0]:.3g} \cdot T + {fit[1,0]:.3g}$'

                # Exponential fit:
                
                exp_pars, exp_cov = curve_fit(exponential, xdata=xn,
                                              ydata=yn,
                                              p0 = (0,0,-3),
                                              maxfev=10000)
                mse_exp_new = np.sum(np.square(exponential(xn, *exp_pars) - yn))/xn.size

                print(f'  Exponential fit (this set) would be V = {exp_pars[0]} * exp({exp_pars[1]} * T) + {exp_pars[2]}\n       MSE = {mse_exp_new:.8f}')
                fit_desc_exp_short = r'(this set) $V \approx ' + f'{exp_pars[0]:.3g} \cdot \exp({exp_pars[1]:.3g} \cdot T) + {exp_pars[2]:.3g}$'
                self.fit_exp_new = exp_pars
                self.useNewRegressionButton['state'] = 'normal'

                sampleX_nl = np.linspace(min(self.temps)-0.1, max(self.temps+0.1), num=100)
                
                sampleY_nl = exponential(sampleX_nl, *exp_pars)
                sampleX = [min(self.temps)-0.3, max(self.temps)+0.3]
                sampleX = mat(sampleX).reshape((2, 1))
                sampleX = np.hstack((sampleX, np.ones((2, 1))))
                sampleY_linear = sampleX * fit
                sampleY_old = sampleX * self.fit
                sampleY_old_exp = exponential(sampleX_nl, *self.fit_exp)

                self.waitbar_indeterminate_start('Plotting regressions...')
                plt.figure(2, figsize=(self.screendims_inches[0]*0.8, self.screendims_inches[1]*0.4))
                plt.clf();
                plt.subplot(1, 3, 1)
                plt.scatter(self.temps, self.volts_scaled, s=0.01, c='black')
                plt.plot(sampleX_nl, sampleY_old_exp, c='blue', label=fit_desc_old_short)
                plt.plot(sampleX_nl, sampleY_nl, c='cyan', label=fit_desc_exp_short)
                plt.plot(sampleX[:,0], sampleY_linear, c='red', linestyle="--", label=fit_desc_short)
                plt.xlabel('Temperature (°C)')
                plt.ylabel('Potential (V)')
                plt.title('Linear regressions')
                plt.legend()
                plt.get_current_fig_manager().toolbar.zoom()

                volts = (y - x * fit).reshape((1,length)).tolist()[0] + np.mean(self.volts_scaled)
                volts_nl = y - exponential(self.temps, *exp_pars) + np.mean(self.volts_scaled)

                plt.subplot(1, 3, (2, 3))
                ax1 = plt.gca()
                ax1.plot(self.times, self.volts_scaled, label='Raw', c='blue', linewidth=1)
                ax1.plot(self.times, volts, label='Potential new linear correction, if using this dataset', c='red', linewidth=1)
                ax1.plot(self.times, self.volts, label='Corrected, as applied', c='black', linewidth=1)
                ax1.plot(self.times, volts_nl, label='Potential new exponential correction', c='cyan', linewidth=1)
                ax1.set_xlabel('Time')

                ax2 = ax1.twinx()
                ax2.set_ylabel('Temperature (°C)', color='green')
                ax2.plot(self.times, self.temps, color='green', label='Temperature', linewidth=1)
                ax2.tick_params(axis='y', labelcolor='green')
                
                ax1.legend()
                plt.get_current_fig_manager().toolbar.zoom()
                plt.title(self.datafile.stem)
                plt.get_current_fig_manager().toolbar.zoom()
                plt.show()
                self.waitbar_indeterminate_done();

        # Copy new fit parameters to default fit param location
        def use_new_correction(self):
                self.fit_exp = self.fit_exp_new
                self.applyCorrections()


        def saveFile(self):
                fname = self.datafile.parent / f'{self.datafile.stem}_adjusted.csv'
                print('fname will be "{fname}"')

                # Get the number of lines in the file so we can do a perfect progress bar...
                print(f'----- Saving {fname} -----')

                if self.temperature_present:
                        self.datathing = zip(self.times, self.volts_raw.transpose().tolist()[0], self.volts.transpose().tolist()[0], self.temps.transpose().tolist()[0])
                        with open(fname, 'w', newline='') as f:
                                print('Header lines, 4', file = f);
                                print(f'Columns, datetime, original potential (V), potential after processing (V), temperature (C)', file = f)
                                print(f'Voltage scaling factor, {self.voltageScalingFactor}', file = f)
                                print(f"Exponential temperature fit parameters (v' = v - a*exp(bT)+c + |v|), {str(self.fit_exp)[1:-1]}", file = f)
                                writer = csv.writer(f)
                                writer.writerows(self.datathing)
                else:
                        self.datathing = zip(self.times, self.volts_raw.transpose().tolist()[0], self.volts.transpose().tolist()[0])
                        with open(fname, 'w', newline='') as f:
                                print('Header lines, 3', file = f);
                                print(f'Columns, datetime, original potential (V), potential after processing (V)', file = f)
                                print(f'Voltage scaling factor, {self.voltageScalingFactor}', file = f)
                                writer = csv.writer(f)
                                writer.writerows(self.datathing)

                print('                  ...done.')

                
        # Faster loading. Note that Python grows lists sensibly, so
        # repeated calls to append() aren't as inefficient as they
        # look.
        def loadFileBen(self, fname):
            times = []
            volts = []
            temps = []

            self.datafile = fname

            # Get the number of lines in the file so we can do a perfect progress bar...
            start = time.perf_counter()
            print(f'----- Loading {fname} -----')
            self.waitbar_indeterminate_start('Checking file size...')
            num_lines = sum(1 for line in open(fname))
            # print(f'{num_lines} lines. Time to determine file line count: {time.perf_counter()-start} seconds.')
            self.waitbar_start('Loading...', num_lines)
            self.temperature_present = 0
            date_column = -1
            voltage_column = -1
            scaled_column = -1
            temperature_in_freedom_units = False

            with fname.open() as csv_file:
                csv_reader = csv.reader(csv_file)
                line_count = 0
                for row in csv_reader:
                        if line_count % 100000 == 0 and False:
                                print(f'  Reading row {line_count}...')
                        if line_count % 1000 == 0:
                                self.waitbar_update(line_count)
                        line_count += 1
                        if line_count == 2:
                                # This is the line with all the header info. Figure out what we have...
                                for column in range(1, len(row)):
                                        serialnumberfound = row[column].find("SEN S/N:")
                                        if serialnumberfound != -1:
                                                sn = int(row[column][serialnumberfound:].split(":")[1].split(',')[0])
                                                if self.sensor_serial_number == -1:
                                                        print(f'      Found sensor serial number {sn}')
                                                if self.sensor_serial_number == -1 or sn == self.sensor_serial_number:
                                                        self.sensor_serial_number = sn
                                                else:
                                                        raise(f"Conflicting sensor serial numbers {sn} and {self.sensor_serial_number}")
                                        if row[column][0:4].lower() == "date":
                                                print(f'      Date found in column {column}.')
                                                date_column = column
                                                utco = (row[column].split('GMT')[1]).split(':')
                                                # Lots of stupid complexity here just in case we're in Newfoundland!
                                                timedelta_corona = dt.timedelta(hours=abs(int(utco[0])), minutes=int(utco[1]))
                                                if np.sign(int(utco[0])) == -1:
                                                        timedelta_corona = -timedelta_corona
                                                self.timezone_corona = dt.timezone(timedelta_corona)
                                                print(f'            Timezone is {self.timezone_corona}')
                                        if row[column][0:4].lower() == "temp":
                                                # 0-indexing makes this risky, but I'm assuming that the first column is never temperature
                                                self.temperature_present = column
                                                if "°F" in row[column]:
                                                        temperature_in_freedom_units = True
                                                        print(f'      Temperature (°F) found in column {column}.')
                                                else:
                                                        print(f'      Temperature (°C) found in column {column}.')
                                        if row[column][0:4].lower() == 'volt':
                                                voltage_column = column
                                                print(f'      Voltage found in column {column}.')
                                        if row[column][0:6].lower() == 'scaled':
                                                print(f'      Scaled Series found in column {column}.')
                                                scaled_column = column

                                # If there's no "voltage" column but there is a "scaled" column, I guess we just use that instead... and guess that the voltage_scaling_factor should probably be 2                  
                                if scaled_column >= 0:
                                        if voltage_column == -1:
                                                print('  Could not find "Volt" but did find "Scaled Series". Interpreting it as scaled voltage, and setting my internal scaling factor to 1. Please verify.')
                                                voltage_column = scaled_column
                                                self.setVoltageScalingFactor(1)
                                        else:
                                                print('  Found both "Volt" and "Scaled Series". Using "Volt", but setting my internal scaling factor to 2. Please verify.')
                                                self.setVoltageScalingFactor(2)
                                else:
                                        print('  Found Voltage, but not Scaled Series. Guessing that my internal scaling factor should be 1. Please verify.')
                                        self.setVoltageScalingFactor(1)
                                        


                        # Here's the meat. Read each line, check for completeness, parse the dates, and add.
                        if line_count > 2:
                            if len(row) <= max([date_column, voltage_column, self.temperature_present]):
                                    print(f'  ***** Line {line_count}: row is incomplete. Corrupt/incomplete file? *****')
                            elif row[date_column] and row[voltage_column] and ((not self.temperature_present) or row[self.temperature_present]):
                                    try:
                                            t = dt.datetime.strptime(row[date_column], self.date_format)
                                            t = times.append(t.replace(tzinfo=self.timezone_corona))
                                    except ValueError:
                                            print(f'Line {line_count}: could not parse date string "{row[1]}" with expected format "{self.date_format}".')
                                            continue;
                                    volts.append(float(row[voltage_column]))
                                    if self.temperature_present:
                                            temps.append(float(row[self.temperature_present]))
                            else:
                                    print(f'Line {line_count} "{row}" contains missing values. Ignoring the row.')



            self.waitbar_done()
            #register_matplotlib_converters()
            length = min(len(times), len(volts))
            if self.temperature_present:
                    length = min(length, len(temps))
                    temps = np.array(temps[0:length]).reshape((length, 1))
                    if temperature_in_freedom_units:
                            temps =(temps - 32) * 5/9
                            temperature_in_freedom_units = False # Unnecessary, but just in case...
                    self.regressButton.grid(row=3, column=4)
                    self.useNewRegressionButton.grid(row=3, column=5)
                    self.useNewRegressionButton['state'] = 'disabled'
            times = times[0:length]
            #volts = np.array(volts[0:length])
            volts = np.array(volts[0:length]).reshape((length, 1))

            # See if we can find some WHOI data...
            self.loadWHOI(times)

            return times, volts, temps


        def loadWHOI(self, times):
                day_i = -1

                print(f'Start time is {times[0]}, which is file {times[0].strftime("%Y_%j")}. Last year,day is {times[-1].strftime("%Y_%j")}')
                fname = ''
                lastfname = self.datafile.parent / 'whoi' / 'lidar' / f'asit.lidar.{times[-1].strftime("%Y_%j")}.sta'

                self.times_lidar = []
                self.zv = np.ndarray((0,0))
                errors = 0
                
                while fname != lastfname:
                        day_i += 1

                        fname = self.datafile.parent / 'whoi' / 'lidar' / f'asit.lidar.{(times[0] + dt.timedelta(days=day_i)).strftime("%Y_%j")}.sta'
                        print(f'Loading {fname}')

                        zwindind = []
                        zwindz = []
                        zwind_legend = []
                        zv = []

                        # WHOI's file format appears to be tab-delimited in the data section, and =delimited above...
                        if fname.is_file():
                                num_lines = sum(1 for line in open(fname, errors='replace'))

                                with open(fname, errors='replace') as f:
                                        line_count = 0
                                        for row in f:
                                                line_count += 1
                                                if line_count == 1:
                                                        if row[0:9] == 'HeaderSize'[0:9]:
                                                                headersize = int(row.split('=')[1])+1
                                                        else:
                                                                print('Could not get header size. Skipping.')
                                                                break

                                                if line_count == headersize + 1:
                                                        names = row.split('\t')
                                                        for i in range(len(names)):
                                                                if "Z-wind (m/s)" in names[i]:
                                                                        zwindind.append(i)
                                                                        zwind_legend.append(names[i])
                                                                        zwindz.append(int(names[i].split('m', 1)[0]))
                                                        zv = np.ndarray((num_lines - headersize - 1, len(zwindz)))
                                                        
                                                if line_count >= headersize + 2:
                                                        row = row.split('\t')
                                                        try:
                                                                self.times_lidar.append(dt.datetime.strptime(row[0], self.date_format_lidar))
                                                        except ValueError:
                                                                print(f'Line {line_count}: could not parse date string "{row[0]}" with expected format "{self.date_format_lidar}".')
                                                                continue;
                                                        for i in range(len(zwindind)):
                                                                zv[line_count - headersize - 2, i] = float(row[zwindind[i]])
                                                        #if line_count < headersize + 4:
                                                        #        print('self.times_lidar[-1]')

                                        # If this is not our first time, concatenate
                                        # the data arrays. This is more efficient than
                                        # doing it inline above as I do with
                                        # times_lidar
                                        if hasattr(self, 'zv') and self.zv.size:
                                                if self.zwind_z != zwindz:
                                                        print(zwindz)
                                                        print(self.zwind_z)
                                                        raise Exception('Incompatibility', f'File {fname}: change in z-wind heights; probable data incompatibility')
                                                        break
                                                self.zv = np.concatenate((self.zv, zv), axis=0)
                                        else:
                                                self.zwind_legend = zwind_legend
                                                self.zwind_z = zwindz
                                                self.zv = zv

                                # print(f'Loaded. Sizes are {len(self.times_lidar)} x {len(self.zwind_z)}; data size is {self.zv.shape}')
                        else:
                                print(f'File "{fname}" does not exist.')
                                errors += 1
                                if errors >= 3:
                                        print('Not finding WHOI data files. Giving up.')
                                        break



    
        def find_events(self, times, volts):
                events = Events()
                # Temperature data mean we're using the atmospheric voltage monitor. Don't hilight events.
                if self.temperature_present:
                        return events
                
                try:
                        self.eventThreshold['volts'] = float(self.detectionVoltageBox.get())
                except:
                        print(f'Could not convert voltage "{self.detectionVoltageBox.get()}" to number. Resetting to default.')
                        self.eventThreshold['volts'] = 0.02
                        
                try:
                        self.eventThreshold['count'] = int(float(self.detectionCountBox.get()))
                except:
                        print(f'Could not convert count "{self.detectionCountBox.get()}" to number. Resetting to default.')
                        self.eventThreshold['count'] = 40;
                self.detectionVoltageBox.delete(0, END)
                self.detectionVoltageBox.insert(0, self.eventThreshold['volts'])
                self.detectionCountBox.delete(0, END)
                self.detectionCountBox.insert(0, self.eventThreshold['count'])


                self.waitbar_start('Looking for events...', len(times))

                thresholdCounter = 0
                aboveThresholdStart = 0

                for i in range(len(times)):
                        self.waitbar_update(i)

                        if volts[i] > self.eventThreshold['volts']:
                                if thresholdCounter == 0:
                                        aboveThresholdStart = i
                                thresholdCounter += 1
                        else:
                                if thresholdCounter > 0: # Was above threshold at time index i-1
                                        # If just counting samples:
                                        #if thresholdCounter >= self.eventThreshold['count']:
                                        #        events.append(i)
                                        # If looking for time-above-threshold:
                                        aboveThresholdTime = times[i-1] - times[aboveThresholdStart]
                                        if aboveThresholdTime.seconds >= self.eventThreshold['count']:
                                                #print(f'Found an event of duration > {aboveThresholdTime.seconds} seconds.')
                                                events.add(aboveThresholdStart, i, aboveThresholdTime.seconds)
                                thresholdCounter = 0;
                self.waitbar_done()
                return events

        # Plot voltage vs time using Matplotlib
        def plot_voltages_matplotlib(self, times, volts, temps=[], events=[]):

                fig = plt.figure(1, figsize=(self.screendims_inches[0]*0.8, self.screendims_inches[1]*0.4))
                fig.clf()

                if hasattr(self, 'zv') and self.zv.size:
                        plt.subplot(4, 1, (1, 2))
                else:
                        plt.subplot(3, 1, (1, 2))
                ax = fig.gca()

                if self.temperature_present & self.plotTemperatureWithPotential.get():

                        color = 'black'
                        ax.set_xlabel('Time')
                        ax.set_ylabel('Potential (V)', color=color)
                        ax.plot(times, volts, color=color, label='Potential', linewidth=1)
                        ax.tick_params(axis='y', labelcolor=color)

                        ax2 = ax.twinx()

                        color = 'tab:green'
                        ax2.set_ylabel('Temperature (°C)', color=color)
                        ax2.plot(times, temps, color=color, label='Temperature', linewidth=1)
                        ax2.tick_params(axis='y', labelcolor=color)
                        
                        for i in range(len(events.start_indices)):
                                # No longer used; may be back eventually...
                                if i == 0:
                                        ax.plot(times[events.start_indices[i]:events.end_indices[i]],
                                                volts[events.start_indices[i]:events.end_indices[i]],
                                                c='red', linewidth=3, label='event?')
                                else:
                                        ax.plot(times[events.start_indices[i]:events.end_indices[i]],
                                                volts[events.start_indices[i]:events.end_indices[i]],
                                                c='red', linewidth=3)

                        fig.tight_layout()  # otherwise the right y-label is slightly clipped

                else:
                        ax.plot(times, volts, label='Potential', c='black', linewidth=0.5)

                        
                        for i in range(len(events.start_indices)):
                                if i == 0:
                                        plt.plot(times[events.start_indices[i]:events.end_indices[i]],
                                                 volts[events.start_indices[i]:events.end_indices[i]],
                                                 c='red', linewidth=3, label='event?')
                                else:
                                        plt.plot(times[events.start_indices[i]:events.end_indices[i]],
                                                 volts[events.start_indices[i]:events.end_indices[i]],
                                                 c='red', linewidth=3)
                                #plt.scatter([times[i] for i in events.indices], [volts[i] for i in events.indices],
                                #            s=events.sizes, c='red', label='Event?')
                        plt.ylabel('Potential (V)')



                # Ted wants lines at midnight. Can't easily do that using matplotlib, so do it manually
                dr = pandas.date_range(self.times[1], self.times[-1], normalize=True).to_pydatetime()[1:].tolist()
                vr = ax.get_ylim()
                for i in range(len(dr)):
                        ax.plot([dr[i], dr[i]], vr, color='black', alpha=0.2)
                ax.set_ylim(vr)

                if hasattr(self, 'zv') and self.zv.size:
                        ax3 = plt.subplot(4, 1, 3, sharex = ax)
                else:
                        ax3 = plt.subplot(3, 1, 3, sharex = ax)
                
                winlen = 128
                noverlap = int(winlen / 2)
                spectimes = times[noverlap::noverlap]
                f, t, Sxx = signal.spectrogram(x=volts.flatten(), fs=0.1, noverlap=noverlap, window=signal.windows.tukey(winlen), mode='magnitude')
                spectimes = spectimes[0:Sxx.shape[1]]

                # Scale the spectrogram data for better visibility
                #Sxx = np.log(Sxx)
                Sxx = np.sqrt(Sxx)

                ax3.pcolormesh(*np.meshgrid(spectimes, f), Sxx, shading='gouraud', cmap='hot')
                ax3.set_ylabel('Frequency [Hz]')
                ax3.set_xlabel('Time')

                # And wind z velocities, if available...
                if hasattr(self, 'zv') and self.zv.size:
                        ax4 = plt.subplot(4, 1, 4, sharex = ax)
                        ax4.set_ylabel('Vertical wind (m/s)')

                        z = ((l - self.zwind_z[0]) / (self.zwind_z[-1]-self.zwind_z[0]) for l in self.zwind_z)
                        zz = np.fromiter(z, dtype=float)
                        colours = plt.get_cmap('viridis')(X=zz)
                        for i in range(len(self.zwind_z)-1, -1, -1):
                                ax4.plot(self.times_lidar, self.zv[:,i] + 0.00*self.zwind_z[i],
                                         label=self.zwind_legend[i], color=colours[i])
                                #handles, labels = ax4.get_legend_handles_labels()
                        ax4.legend()
                        #zmesh = ax4.pcolormesh(*np.meshgrid(self.times_lidar, self.zwind_z), self.zv.T, shading='gouraud', cmap='BrBG')
                        #fig.colorbar(zmesh)
                        
                        ax4.set_xlim(self.times[0], self.times[-1])
                
                
                ax.set_title(self.datafile.stem)
                plt.get_current_fig_manager().toolbar.zoom()
                plt.show()
    

cor = CoronaBrowser()
cor.master.title('Corona browser')
#try:
cor.mainloop()
# except:
#         type, value, tb = sys.exc_info()
#         traceback.print_exc()
#         last_frame = lambda tb=tb: last_frame(tb.tb_next) if tb.tb_next else tb
#         frame = last_frame().tb_frame
#         ns = dict(frame.f_globals)
#         ns.update(frame.f_locals)
#         code.interact(local=ns)
