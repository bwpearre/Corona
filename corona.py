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
from scipy.stats import linregress
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
import pandas as pd
import pytz
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator


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

                # Actually print stuff when I ask:
                np.set_printoptions(threshold=np.inf)
                pd.set_option('display.max_rows', None)
                pd.set_option('display.max_columns', None)
                pd.set_option('display.width', None)
                pd.set_option('display.max_colwidth', None)
                
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

                self.plots = ('Z-wind (m/s)', 'Z-wind Dispersion (m/s)') # BUG if there's only one, so need 2 until fixed.

                #self.plots = ('Z-wind (m/s)', 'Z-wind Dispersion (m/s)', 'Wind Speed max (m/s)', 'Wind Direction', 'wind_speed_mean (m/s)')
                # self.plots = ('Z-wind (m/s)', 'Z-wind Dispersion (m/s)', 'Wind Speed max (m/s)', 'Wind Direction', 'pressure_mean (hPa)', 'pressure_median (hPa)', 'pressure_std (hPa)', 'temperature_mean (degC)', 'temperature_median (degC)', 'temperature_std (degC)', 'humidity_mean (%RH)', 'humidity_median (%RH)', 'humidity_std (%RH)', 'wind_speed_mean (m/s)', 'wind_speed_std (m/s)', 'wind_direction_mean (degrees)', 'wind_direction_std (degrees)')

                self.debug_seq()


        def debug_seq(self):
                self.no_temperature_correction_check = True
                #self.model = tf.keras.models.load_model('model')
                self.loadFile(filename='data/20310992-2021-05+06.csv')
                #self.loadFile(filename='data/trunc.csv')


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
                self.regressButton.grid(row=row, column=2)
                self.useNewRegressionButton = tk.Button(self, text='Use new regression this session', command=self.use_new_correction)
                self.useNewRegressionButton.grid(row=row, column=3)
                row += 1
                self.doStatisticsButton = tk.Button(self, text='Statistics', command=self.doStatistics, state='disabled')
                self.doStatisticsButton.grid(row=row, column=0)
                self.doTrainButton = tk.Button(self, text='Train z-wind predictor', command=self.trainPredictor, state='disabled')
                self.doTrainButton.grid(row=row, column=1)
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

        # Reset variables that should be tossed out if a new file is loaded or etc.
        def reset_defaults(self):
                self.temps = []
                self.volts_scaled = []
                self.whoi_graphs = 0
                self.sensor_serial_number = -1
                self.regressButton.grid_forget()
                self.useNewRegressionButton.grid_forget()

                
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
                pd.set_option('display.max_rows', None) # print e.g. all columns in self.whoi.dtypes
                pdb.set_trace()

        # Ask for a filename, load it, plot it.
        def loadFile(self, filename=False):
                if not filename:
                        filename = filedialog.askopenfilename(filetypes=[('Comma-separated values', '*.csv')])
                if filename:
                        self.reset_defaults()
                        self.times, self.volts_raw, self.temps = self.loadFileBen(Path(filename))
                        self.rmplButton['state'] = 'normal'
                        self.useNewRegressionButton['state'] = 'disabled'
                        if self.temperature_present:
                                self.event_detection_enabled(False)
                        else:
                                self.event_detection_enabled(True)

                        self.applyCorrections()
                        self.loadWHOI(self.times)
                        try:
                                self.model = keras.models.load_model('model')
                                print('Loaded last saved z-wind prediction model.')
                                self.runPredictor()
                        except:
                                print('No z-wind prediction model found...')
                
                        
                        self.saveButton['state'] = 'normal'
                        self.doStatisticsButton['state'] = 'normal'
                        
                        self.plotEvents()

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
                print(f'  Potential: mode is {self.common_temp} V{t} (fyi; not used)')
                

        def plotEvents(self):
                self.events = self.find_events(self.times, self.volts)
                self.plot_timeseries(self.times, self.volts, self.temps, self.events)
                if not hasattr(self, 'no_temperature_correction_check'):
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
                plt.figure(num='temperature correction', figsize=(self.screendims_inches[0]*0.8, self.screendims_inches[1]*0.4))
                plt.clf()
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
            self.waitbar_indeterminate_done()
            # print(f'{num_lines} lines. Time to determine file line count: {time.perf_counter()-start} seconds.')
            self.waitbar_start('Loading...', num_lines)
            self.temperature_present = 0
            timezone_utc = pytz.timezone("UTC")
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
                                                timezone_corona = dt.timezone(timedelta_corona)
                                                print(f'            Timezone is {timezone_corona}')
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
                                            t = dt.datetime.strptime(row[date_column], self.date_format) - timedelta_corona
                                            times.append(timezone_utc.localize(t))
                                    except ValueError:
                                            print(f'  * Line {line_count}: could not parse date string "{row[1]}" with expected format "{self.date_format}".')
                                            continue;
                                    volts.append(float(row[voltage_column]))
                                    if self.temperature_present:
                                            temps.append(float(row[self.temperature_present]))
                            else:
                                    print(f'  * Line {line_count} "{row}" contains missing values. Ignoring the row.')



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
            volts = np.array(volts[0:length]).reshape((length,1))

            return times, volts, temps


        def loadWHOI(self, times):

                firstfnum = times[0].strftime("%Y_%j")
                lastfnum = times[-1].strftime("%Y_%j")

                self.waitbar_start('  Loading WHOI data...', (times[-1]-times[0]).days+1)

                #print(f'  WHOI: start time is {times[0]}, which is file {firstfnum}. Last year,day is {lastfnum}')

                #lastfname = self.datafile.parent / 'whoi' / 'lidar' / f'asit.lidar.{times[-1].strftime("%Y_%j")}.sta'
                # {<directory>: [ <filename prefix>, <filename suffix>, <invoke special code for WHOI's LIDAR files> ]}
                #filesets = {'lidar': [ 'asit.lidar.', '.sta', True ],
                #             'met': ['met.Vaisala_', '.csv', False ],
                #             'wind': ['met.Anemo_', '.csv', False ]}

                # Here are the files. A hassle since a Dict isn't ordered.
                filesets = {'lidar': [ 'asit.lidar.', '.sta', True ],
                            'met': ['asit.mininode.CLRohn_', '.csv', False ],
                            'wind': ['asit.mininode.Sonic1_', '.csv', False ]}
                
                dataframes = []

                
                # Okay, this is fucking stupid: I am going to read in
                # everything, then purge duplicate rows, then read in
                # everything AGAIN to fill in all the purged
                # data. Thanks, Python.

                ### ROUND 1 ###
                
                day_i = -1
                fnum = ''
                errors = 0
                
                while fnum != lastfnum:
                        day_i += 1
                        self.waitbar_update(day_i)
                        fnum = (times[0] + dt.timedelta(days=day_i)).strftime("%Y_%j")
                        print(f' Loading WHOI files {fnum}...')

                        daily_data = []

                        for fileset, prepost in filesets.items():
                                fname = self.datafile.parent / 'whoi' / fileset / f'{prepost[0]}{fnum}{prepost[1]}'
                                #print(f'      Loading {fname}')

                                if fname.is_file():
                                        if fileset == 'lidar':
                                                # Special code for WHOI's LIDAR files:
                                                with open(fname, errors='replace') as f:
                                                        row = f.readline()
                                                        if row[0:9] == 'HeaderSize'[0:9]:
                                                                headersize = int(row.split('=')[1])
                                                        else:
                                                                print('Could not get header size. Assuming 0.')
                                                                headersize = 0
                                                        mdf = pd.read_csv(fname, sep='=', nrows=headersize-1, index_col = 0, header=None,
                                                                          encoding='cp1252')
                                                        whoi_lidar_timezone = mdf.loc['timezone', 1]
                                                        # I can't deal with the 19 different timezone and time offset systems in Python. Just check that it hasn't changed:
                                                if whoi_lidar_timezone != "UTC+0":
                                                        raise Exception('LIDAR time offset changed.')

                                                df = pd.read_csv(fname, sep='\t', header=headersize, parse_dates=[0], index_col=0,
                                                                 encoding='cp1252')

                                                if len(daily_data):
                                                        daily_data = daily_data.join(df, how='outer')
                                                else:
                                                        daily_data = df


                                        else:
                                                # No check for timezone/td is possible here. Just assume?! FIXME
                                                df = pd.read_csv(fname, sep=',', header=[0,1], parse_dates=[0], index_col=0,
                                                                 encoding='cp1252')
                                                newcolumns = []

                                                # The column header is 2 rows deep: measurement and units. Also, there's
                                                # column-name overlap for the mean and std temperatures in 'met' and 'wind'
                                                # filesets. Join and clean:
                                                for i in df.columns:
                                                        if i[0].startswith('temperature'):
                                                                newcolumns.append(f'{i[0]} ({fileset}) ({i[1]})')
                                                        else:
                                                                newcolumns.append(f'{i[0]} ({i[1]})')
                                                df.columns=newcolumns
                                                                

                                                if len(daily_data):
                                                        daily_data = daily_data.join(df, how='outer')
                                                else:
                                                        daily_data = df


                                else:
                                        print(f'  File "{fname}" does not exist.')
                                        errors += 1
                                        if errors >= 30:
                                                print('  [[ Not finding WHOI data files. Giving up. ]]')
                                                return

                        dataframes.append(daily_data)
                        
                self.whoi = pd.concat(dataframes, copy=False)
                print(f'Before dropping duplicates: dataset is {self.whoi.shape}')
                self.whoi.sort_index(inplace=True, kind='mergesort')
                self.whoi = self.whoi[~self.whoi.index.duplicated(keep='first')] # FIXME: This will toss some rows that contain data. Fixed in ROUND 2.
                self.whoi.dropna(inplace=True, axis='columns', how='all')


                ### ROUND 2 ###

                print(f'Before ROUND 2: dataset is {self.whoi.shape}')
                
                day_i = -1
                fnum = ''
                errors = 0
                
                while fnum != lastfnum:
                        day_i += 1
                        self.waitbar_update(day_i)
                        fnum = (times[0] + dt.timedelta(days=day_i)).strftime("%Y_%j")
                        print(f' Loading WHOI files {fnum}...')

                        for fileset, prepost in filesets.items():
                                fname = self.datafile.parent / 'whoi' / fileset / f'{prepost[0]}{fnum}{prepost[1]}'
                                #print(f'      Loading {fname}')

                                if fname.is_file():
                                        if fileset == 'lidar':
                                                # Special code for WHOI's LIDAR files:
                                                with open(fname, errors='replace') as f:
                                                        row = f.readline()
                                                        if row[0:9] == 'HeaderSize'[0:9]:
                                                                headersize = int(row.split('=')[1])
                                                        else:
                                                                print('Could not get header size. Assuming 0.')
                                                                headersize = 0
                                                                mdf = pd.read_csv(fname, sep='=', nrows=headersize-1, index_col = 0, header=None)
                                                                whoi_lidar_timezone = mdf.loc['timezone', 1]
                                                                # I can't deal with the 19 different timezone and time offset systems in Python. Just check that it hasn't changed:
                                                if whoi_lidar_timezone != "UTC+0":
                                                        raise Exception('LIDAR time offset changed.')

                                                df = pd.read_csv(fname, sep='\t', header=headersize, parse_dates=[0], index_col=0,
                                                                 encoding='cp1252')

                                                self.whoi.update(df)

                                        else:
                                                # No check for timezone/td is possible here. Just assume?! FIXME
                                                df = pd.read_csv(fname, sep=',', header=[0,1], parse_dates=[0], index_col=0)
                                                newcolumns = []

                                                # The column header is 2 rows deep: measurement and units. Also, there's
                                                # column-name overlap for the mean and std temperatures in 'met' and 'wind'
                                                # filesets. Join and clean:
                                                for i in df.columns:
                                                        if i[0].startswith('temperature'):
                                                                newcolumns.append(f'{i[0]} ({fileset}) ({i[1]})')
                                                        else:
                                                                newcolumns.append(f'{i[0]} ({i[1]})')
                                                df.columns=newcolumns
                                                                
                                                self.whoi.update(df)
                                                

                                else:
                                        print(f'  File "{fname}" does not exist.')
                                        errors += 1
                                        if errors >= 30:
                                                print('  [[ Not finding WHOI data files. Giving up. ]]')
                                                return

                print(f'After ROUND 2: dataset is {self.whoi.shape}')
                self.whoi = self.whoi.tz_localize('UTC') # see "just check that it hasn't changed" above
                self.legends = {p:[] for p in self.plots}
                self.z = {p:[] for p in self.plots}
                print('Interpolating 20-minute to 10-minute data...')
                self.whoi.interpolate(inplace=True, method='linear', limit=1, limit_area='inside')
                
                
                #self.legends = [[] for x in range(len(self.plots))]
                #self.z = [[] for x in range(len(self.plots))]

                for i,t in enumerate(self.whoi.columns):
                        #print(f' Looking at column {i} : {t}')
                        for toplot in self.plots:
                                #print(f'Looking for "{toplot}" in "{t}"')
                                if toplot in t:
                                        #print('   ...found')
                                        #pdb.set_trace()
                                        #print(self.legends)
                                        self.legends[toplot].append(t)
                                        #print(f'Adding: self.legends[{toplot}].append({t})')
                                        try:
                                                self.z[toplot].append(int(t.split('m', 1)[0]))
                                        except:
                                                None
                for toplot in self.plots:
                        if len(self.legends[toplot]):
                                self.whoi_graphs += 1

                self.waitbar_done()
                self.doTrainButton['state'] = 'normal'


        def slice_frame(self, pattern, df):
                r = []
                for i,t in enumerate(df.columns):
                        #print(f' Looking at column {i} : {t}')
                        if pattern in t:
                                #print('   ...found')
                                #pdb.set_trace()
                                r.append(t)
                return r
                

        def trainPredictor(self):
                print('Training predictor...')
                self.n_avm_samples = 360
                batch_size = 128

                # Stick Ted's data into a dataframe. This has already had the timezone sorted.
                df = pd.DataFrame(data={'AVM volts': self.volts.squeeze()}, index=pd.DatetimeIndex(self.times))
                # Upsample WHOI LIDAR z-wind:

                lidarz = self.whoi.loc[:, self.slice_frame('Z-wind (m/s)', self.whoi)]

                df2 = lidarz.max(axis='columns').rename('Z-wind')
                #df3 = df.join(df2, how='outer')
                df3 = pandas.merge_asof(df, df2, left_index = True, right_index = True, direction='nearest', tolerance=dt.timedelta(minutes=20))
                df3 = df3.interpolate(method='linear', limit_direction='both')

                #df3.fillna(MASK, inplace=True)
                generator = TimeseriesGenerator(df3.loc[:,'AVM volts'], df3.loc[:,'Z-wind'],
                                                length = self.n_avm_samples, shuffle = True, batch_size=batch_size)

                model = tf.keras.models.Sequential()
                model.add(tf.keras.layers.Reshape((self.n_avm_samples, 1)))
                model.add(tf.keras.layers.Conv1D(20, 5, activation='relu'))
                model.add(tf.keras.layers.MaxPooling1D(2))
                #model.add(tf.keras.layers.Dropout(0.5))
                model.add(tf.keras.layers.Conv1D(15, 5, activation='relu'))
                model.add(tf.keras.layers.MaxPooling1D(2))
                #model.add(tf.keras.layers.Dropout(0.5))
                model.add(tf.keras.layers.Conv1D(10, 5, activation='relu'))
                model.add(tf.keras.layers.MaxPooling1D(2))
                model.add(tf.keras.layers.Dropout(0.5))
                model.add(tf.keras.layers.Conv1D(10, 5, activation='relu'))
                model.add(tf.keras.layers.Dense(7, activation='sigmoid'))
                model.add(tf.keras.layers.Dropout(0.5))
                model.add(tf.keras.layers.Flatten())
                model.add(tf.keras.layers.Dense(1)) # linear output

                loss_fn = tf.keras.losses.MeanSquaredError()
                adam = tf.keras.optimizers.Adam(learning_rate=0.1, name='adam')
                model.compile(optimizer='adam',
                              loss=loss_fn,
                              metrics=['accuracy'])

                model.fit(generator, workers=6, epochs=100)

                self.model = model
                model.save('model')
                
                self.runPredictor()


        def runPredictor(self):
                if not hasattr(self, 'model'):
                        self.waitbar_label['text'] = 'No model found.'
                        return

                print('Running trained predictor...')
                
                self.n_avm_samples = self.model.layers[0].output_shape[1]

                # Stick Ted's data into a dataframe. This has already had the timezone sorted.
                df = pd.DataFrame(data={'AVM volts': self.volts.squeeze()}, index=pd.DatetimeIndex(self.times))
                # Upsample WHOI LIDAR z-wind:

                lidarz = self.whoi.loc[:, self.slice_frame('Z-wind (m/s)', self.whoi)]

                df2 = lidarz.max(axis='columns').rename('Z-wind')
                df3 = pandas.merge_asof(df, df2, left_index = True, right_index = True, direction='nearest', tolerance=dt.timedelta(minutes=20))
                df3 = df3.interpolate(method='linear', limit_direction='both')

                print(self.model.summary())
                
                gen2 = TimeseriesGenerator(df3.loc[:,'AVM volts'], df3.loc[:,'Z-wind'], length = self.n_avm_samples, batch_size = len(self.volts), shuffle = False)
                x, y = gen2[0]
                self.z_predicted = np.full((self.n_avm_samples, 1), np.NaN)
                self.z_predicted = np.append(self.z_predicted, self.model.predict(x))

                fig = plt.figure(num = 'timeseries')
                fig.axes[1].plot(self.times, self.z_predicted + 2, color='red')

                self.doStatistics()
                
                
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
        def plot_timeseries(self, times, volts, temps=[], events=[]):

                fig = plt.figure(num='timeseries', figsize=(self.screendims_inches[0]*0.7, self.screendims_inches[1]*0.7))
                fig.clf()

                nsubplots_base = 1
                nsubplots = nsubplots_base + self.whoi_graphs
                
                plt.subplot(nsubplots, 1, 1)

                axes = [fig.gca()]

                if self.temperature_present & self.plotTemperatureWithPotential.get():

                        color = 'black'
                        axes[0].set_xlabel('Time')
                        axes[0].set_ylabel('Potential (V)', color=color)
                        axes[0].plot(times, volts, color=color, label='Potential', linewidth=1)
                        axes[0].tick_params(axis='y', labelcolor=color)

                        ax00 = axes[0].twinx()

                        color = 'tab:green'
                        ax00.set_ylabel('Temperature (°C)', color=color)
                        ax00.plot(times, temps, color=color, label='Temperature', linewidth=1)
                        ax00.tick_params(axis='y', labelcolor=color)
                        
                        for i in range(len(events.start_indices)):
                                # No longer used; may be back eventually...
                                if i == 0:
                                        ax[0].plot(times[events.start_indices[i]:events.end_indices[i]],
                                                volts[events.start_indices[i]:events.end_indices[i]],
                                                c='red', linewidth=3, label='event?')
                                else:
                                        ax[0].plot(times[events.start_indices[i]:events.end_indices[i]],
                                                volts[events.start_indices[i]:events.end_indices[i]],
                                                c='red', linewidth=3)

                        fig.tight_layout()  # otherwise the right y-label is slightly clipped

                else:
                        axes[0].plot(times, volts, label='Potential', c='black', linewidth=0.5)

                        
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



                # Lines at midnight.
                dr = pandas.date_range(self.times[1], self.times[-1], normalize=True).to_pydatetime()[1:].tolist()
                vr = axes[0].get_ylim()
                for i in range(len(dr)):
                        axes[0].plot([dr[i], dr[i]], vr, color='black', alpha=0.2)
                axes[0].set_ylim(vr)

                # Spectrogram:
                if False:
                        axes.append(plt.subplot(nsubplots, 1, 2, sharex = axes[0]))

                        winlen = 128
                        noverlap = int(winlen / 2)
                        spectimes = times[noverlap::noverlap]
                        f, t, Sxx = signal.spectrogram(x=volts.flatten(), fs=0.1, noverlap=noverlap, window=signal.windows.tukey(winlen), mode='magnitude')
                        spectimes = spectimes[0:Sxx.shape[1]]

                        # Scale the spectrogram data for better visibility
                        #Sxx = np.log(Sxx)
                        Sxx = np.sqrt(Sxx)

                        axes[1].pcolormesh(*np.meshgrid(spectimes, f), Sxx, shading='gouraud', cmap='hot')
                        axes[1].set_ylabel('Frequency [Hz]')
                        axes[1].set_xlabel('Time')

                ###### WHOI data: ######
                if self.whoi_graphs:
                        n = nsubplots_base
                        for toplot in self.plots:
                                if len(self.legends[toplot]) == 0:
                                        continue
                                axes.append(plt.subplot(nsubplots, 1, n+1, sharex = axes[0]))
                                axes[n].set_ylabel(toplot, rotation=45, horizontalalignment='right')

                                if len(self.legends[toplot]) == 1:
                                        axes[n].plot(self.whoi.index, self.whoi[toplot], label=i)
                                else:
                                        order = np.argsort(self.z[toplot])
                                        z = ((l - self.z[toplot][order[0]]) / (self.z[toplot][order[-1]]-self.z[toplot][order[0]]) for l in self.z[toplot])
                                        zz = np.fromiter(z, dtype=float)
                                        colours = plt.get_cmap('viridis')(X=zz)
                                        colour = 0
                                        
                                        for i in [self.legends[toplot][x] for x in order]:
                                                #print(f' axes {n}, order {i}')
                                                axes[n].plot(self.whoi.index, self.whoi[i],
                                                             label=i, color=colours[colour])
                                                colour += 1
                                #axes[n].legend()
                                axes[n].set_xlim(self.times[0], self.times[-1])
                                n += 1
                
                axes[0].set_title(self.datafile.stem)
                plt.get_current_fig_manager().toolbar.zoom()
                plt.show()

                if not hasattr(self, 'model'):
                        self.trainPredictor()
                
                self.runPredictor()
    

        def doStatistics(self):
                corr_interesting_threshold = 0.35 # Show all correlations above a threshold
                corr_interesting_n = 1 # Show the n most interesting

                #key = 'AVM volts'
                key = 'Predicted Z-wind'
                
                # Stick Ted's data into a dataframe. This has already had the timezone sorted.
                #df = pd.DataFrame(data={key: self.volts.squeeze()}, index=pd.DatetimeIndex(self.times))
                df = pd.DataFrame(data={key: self.z_predicted.squeeze()}, index=pd.DatetimeIndex(self.times))
                # Downsample onto the WHOI data's timestamps:
                #df = df.groupby(self.whoi.index[self.whoi.index.searchsorted(df.index)-1]).std()
                df = df.groupby(self.whoi.index[self.whoi.index.searchsorted(df.index)-1]).max()
                #df = df.groupby(self.whoi.index[self.whoi.index.searchsorted(df.index)-1]).mean()
                
                #self.whoi = self.whoi.join(df)
                
                whoi_interp = self.whoi.interpolate(method='linear', limit_direction='both')

                # Easiest most braindead way to line up all the data?
                df = df.join(whoi_interp)
                cor = df.corr()
                corV = cor.loc[:,key]

                # Show the full correlation matrix
                if False:
                        plt.matshow(np.abs(cor))
                        plt.show()
                        plt.xticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=14, rotation=45)
                        plt.yticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=14)
                        cb = plt.colorbar()
                        cb.ax.tick_params(labelsize=14)
                        plt.title('Correlation Matrix', fontsize=16);

                if True:
                        # List interesting indices, in order of interestingness:
                        corVs = corV.sort_values(ascending = False, key = lambda x: abs(x))

                        # Show ones above the threshold?
                        interesting = corVs.index[np.abs(corVs) > corr_interesting_threshold]
                        interesting = interesting[1:] # Don't need to see self-correlation of 1
                        # ...or show top n? Comment out the line below to use above.
                        interesting = corVs.index[range(1, corr_interesting_n+1)]

                        # This is too buggy:
                        #df.plot(x = key, kind = 'scatter', subplots = True)

                        if interesting.size:
                                n = int(np.ceil(np.sqrt(interesting.size)))
                                m = int(np.ceil(interesting.size / n))
                                plt.figure(num='correlations', figsize=(self.screendims_inches[0]*0.3, self.screendims_inches[1]*0.3))
                                plt.clf()
                                #fig, axs = plt.subplots(n, m)
                                #axsf = axs.flat
                                counter = 1
                                for y in interesting:
                                        mask = ~np.isnan(df.loc[:,key]) & ~np.isnan(df.loc[:,y])
                                        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(df.loc[mask,key], df.loc[mask,y])
                                        ax = plt.subplot(n, m, counter)
                                        df.plot.scatter(x = key, y = y, ax = ax, s=5, c='black', label=f'corr = {corV.loc[y]:.2f}, p={p_value:.2g}, r={r_value:.2g}')
                                        #axsf[counter].legend()
                                        counter += 1
                        else:
                                print(f'No correlations found > {corr_interesting_threshold}. Greatest was {corVs.iloc[1]}.')


# Set growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
        try:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)
else:
        print('NO GPUS')


# Actually print stuff when I ask:
np.set_printoptions(threshold=np.inf)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

root = tk.Tk()
root.geometry('+0-0')
cor = CoronaBrowser(master = root)
cor.master.title('Atmospheric Voltage Browser')

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
