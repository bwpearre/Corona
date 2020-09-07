#import pandas as pd
import datetime
import tkinter as tk
from tkinter import filedialog, ttk, END, BooleanVar
import matplotlib
#matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
#from pandas.plotting import register_matplotlib_converters
import numpy as np
from numpy import mat # matrix
from numpy.linalg import inv
import math
import time
import pdb
from scipy.optimize import curve_fit
import traceback, sys, code

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
                
                # Helper vars:
                self.plotTemperatureWithPotential = BooleanVar()

                # Create root window:
                self.grid()
                self.createWidgets()

                # Temperature correction fit. This was computed from 20311010_450_expurgated.csv
                self.fit = mat([[-0.020992708021557917],
                               [5.272377975649473]])
                # Exponential temperature fit parameters computed from 20121725_1.csv
                self.fit_exp = (1.7137714544047866, -0.07977523230422238, 4.493155756244882)

        # Set up the main window.
        def createWidgets(self):
                row = 0
                self.loadButton = tk.Button(self, text="Load", command=self.loadFile)
                self.loadButton.grid(row=row, column=0, columnspan=5)
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
                tk.Label(self, text='Detection thresholds:', font=('bold')).grid(row=row, column=0)
                tk.Label(self, text='Volts:').grid(row=row, column=1, sticky='E')
                self.detectionVoltageBox = tk.Entry(self, width=5)
                self.detectionVoltageBox.grid(row=row, column=2, sticky='W');
                tk.Label(self, text='Seconds:').grid(row=row, column=3, sticky='E')
                self.detectionCountBox = tk.Entry(self, width=5)
                self.detectionCountBox.grid(row=row, column=4, sticky='W');
                row += 1
                self.rmplButton = tk.Button(self, text="Detect + plot", command=self.plotEvents)
                self.rmplButton.grid(row=row, column=0)
                self.plotTemperatureWithPotentialCheck = tk.Checkbutton(self, text="with temperature if available.", variable=self.plotTemperatureWithPotential)
                self.plotTemperatureWithPotentialCheck.grid(row=row, column=1, sticky='W')
                self.regressButton = tk.Button(self, text='Replot potential vs temp', command=self.temperature_show_old_and_new_corrections)
                row += 1
                self.waitbar_label = tk.Label(self, text='Ready.')
                self.waitbar_label.grid(row=row, column=0, columnspan=5)
                row += 1
                self.waitbar = ttk.Progressbar(self, orient="horizontal", length=300, value=0, mode='determinate')
                self.waitbar.grid(row=row, column=0, columnspan=5)
                row += 1
                self.debugButton = tk.Button(self, text="Debug", command=self.debug)
                self.debugButton.grid(row=row, column=3)
                self.quitButton = tk.Button(self, text="Quit", command=self.quit)
                self.quitButton.grid(row=row, column=4)

                self.eventThreshold = {'volts': 0.02, 'count': 40}
                self.detectionVoltageBox.insert(0, self.eventThreshold['volts'])
                self.detectionCountBox.insert(0, self.eventThreshold['count'])
                

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
                        self.regressButton.grid_forget()
                        self.times, self.volts_raw, self.temps = self.loadFileBen(data_filename)
                        self.filename = data_filename 
                        self.voltageScalingFactorBox['state'] = 'normal'
                        self.voltageScalingButton['state'] = 'normal'
                        self.applyCorrections()

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

                # Correct for sensor temperature
                self.volts = self.applyTemperatureCorrection()
                
                self.plotEvents()


        def plotEvents(self):
                self.events = self.find_events(self.times, self.volts)
                self.plot_voltages_matplotlib(self.times, self.volts, self.temps, self.events)
                        

        # Correct the voltage using self.fit for temperature
        def applyTemperatureCorrection(self):
                if not self.temperature_present:
                        return self.volts_scaled

                length = len(self.temps)
                x = np.mat(self.temps).reshape((length,1))
                x = np.hstack((x, np.ones((length, 1))))
                y = np.mat(self.volts_scaled).reshape((length,1))
                #volts = (y - x * self.fit).reshape((1,length)).tolist()[0] + np.mean(self.volts_scaled)
                volts = y - exponential(self.temps, *self.fit_exp) + np.mean(self.volts_scaled)
                return volts

        # Show default and new regressions from voltage-vs-temperature
        def temperature_show_old_and_new_corrections(self):
                if not self.temperature_present:
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
                fit_desc_old = r'V = ' + f'{self.fit_exp[0]} * exp({self.fit_exp[1]} * T) + {self.fit_exp[2]}\n       MSE = {mse_old}'
                fit_desc_old_short = r'$V \approx ' + f'{self.fit_exp[0]:.3g} \cdot \exp({self.fit_exp[1]:.3g} \cdot T) + {self.fit_exp[2]:.3g}$'
                print(f'\n  Saved regression is {fit_desc_old}')

                # Compute the new least-squares fit:
                fit = (x.T*x).I*x.T*y
                x_linear_fit = x * fit
                mse_linear = np.sum(np.square(y - x_linear_fit[:,0]))/xn.size
                fit_desc = f'V = {fit[0,0]} * T + {fit[1,0]}'
                fit_desc_short = r'From this set: $V^* \approx ' + f'{fit[0,0]:.3g} \cdot T + {fit[1,0]:.3g}$'
                print(f'  Linear regression using this dataset would be {fit_desc}\n       MSE = {mse_linear}')

                # Exponential fit:
                
                exp_pars, exp_cov = curve_fit(exponential, xdata=xn,
                                              ydata=yn,
                                              p0 = (0,0,-3),
                                              maxfev=10000)
                mse_exp_new = np.sum(np.square(exponential(xn, *exp_pars) - yn))/xn.size

                fit_desc_exp = r'V = ' + f'{exp_pars[0]} * exp({exp_pars[1]} * T) + {exp_pars[2]}\n       MSE = {mse_exp_new}'
                #fit_desc_exp = r'$V = ' + f'{exp_pars[0]} * exp({exp_pars[1]} * ( T - {exp+pa)$'
                fit_desc_exp_short = r'$V \approx ' + f'{exp_pars[0]:.3g} \cdot \exp({exp_pars[1]:.3g} \cdot T) + {exp_pars[2]:.3g}$'
                print(f'  Exponential regression is {fit_desc_exp}')

                sampleX_nl = np.linspace(min(self.temps)-0.1, max(self.temps+0.1), num=100)
                
                sampleY_nl = exponential(sampleX_nl, *exp_pars)
                sampleX = [min(self.temps)-0.3, max(self.temps)+0.3]
                sampleX = mat(sampleX).reshape((2, 1))
                sampleX = np.hstack((sampleX, np.ones((2, 1))))
                sampleY = sampleX * fit
                sampleY_old = sampleX * self.fit
                sampleY_old_exp = exponential(sampleX_nl, *self.fit_exp)

                self.waitbar_indeterminate_start('Plotting regressions...')
                plt.figure(1, figsize=(self.screendims_inches[0]*0.95, self.screendims_inches[1]*0.5))
                plt.subplot(1, 3, 1)
                plt.scatter(self.temps, self.volts_scaled, s=0.01, c='black')
                plt.plot(sampleX[:,0], sampleY, c='red', label=fit_desc_short)
                plt.plot(sampleX_nl, sampleY_old_exp, c='blue', label=fit_desc_old_short)
                plt.plot(sampleX_nl, sampleY_nl, c='cyan', label=fit_desc_exp_short)
                plt.xlabel('Temperature (°C)')
                plt.ylabel('Potential (V)')
                plt.title('Linear regressions')
                plt.legend()
                plt.get_current_fig_manager().toolbar.zoom()

                volts = (y - x * fit).reshape((1,length)).tolist()[0] + np.mean(self.volts_scaled)
                volts_nl = y - exponential(self.temps, *exp_pars) + np.mean(self.volts_scaled)

                plt.subplot(1, 3, (2, 3))
                ax1 = plt.gca()
                ax1.plot(self.times, self.volts_scaled, label='Raw', c='black', linewidth=1)
                ax1.plot(self.times, volts, label='Potential new linear correction, if using this dataset', c='red', linewidth=1)
                ax1.plot(self.times, self.volts, label='Corrected, as applied', c='blue', linewidth=1)
                ax1.plot(self.times, volts_nl, label='Potential new exponential correction', c='cyan', linewidth=1)
                ax1.set_xlabel('Time')

                ax2 = ax1.twinx()
                ax2.set_ylabel('Temperature (°C)', color='green')
                ax2.plot(self.times, self.temps, color='green', label='Temperature', linewidth=1)
                ax2.tick_params(axis='y', labelcolor='green')
                
                ax1.legend()
                plt.get_current_fig_manager().toolbar.zoom()
                plt.title(self.filename)
                plt.get_current_fig_manager().toolbar.zoom()
                plt.show()
                self.waitbar_indeterminate_done();


        # Faster loading. Note that Python grows lists sensibly, so
        # repeated calls to append() aren't as inefficient as they
        # look.
        def loadFileBen(self, fname):
            import csv

            times = []
            volts = []
            temps = []


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

            with open(fname) as csv_file:
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
                                        if row[column][0:4].lower() == "date":
                                                print(f'      Date found in column {column}.')
                                                date_column = column
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
                                            times.append(datetime.datetime.strptime(row[date_column], self.date_format))
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
            times = times[0:length]
            volts = np.array(volts[0:length]).reshape((length, 1))

            return times, volts, temps


        def find_events(self, times, volts):
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

                events = Events()
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

                if self.temperature_present & self.plotTemperatureWithPotential.get():
                        fig = plt.figure(1, figsize=(self.screendims_inches[0]*0.95, self.screendims_inches[1]*0.5))
                        ax1 = fig.gca()
                        
                        color = 'blue'
                        ax1.set_xlabel('Time')
                        ax1.set_ylabel('Potential (V)', color=color)
                        ax1.plot(times, volts, color=color, label='Potential', linewidth=1)
                        ax1.tick_params(axis='y', labelcolor=color)

                        ax2 = ax1.twinx()

                        color = 'tab:green'
                        ax2.set_ylabel('Temperature (°C)', color=color)
                        ax2.plot(times, temps, color=color, label='Temperature', linewidth=1)
                        ax2.tick_params(axis='y', labelcolor=color)
                        
                        for i in range(len(events.start_indices)):
                                if i == 0:
                                        ax1.plot(times[events.start_indices[i]:events.end_indices[i]],
                                                 volts[events.start_indices[i]:events.end_indices[i]],
                                                 c='red', linewidth=3, label='event?')
                                else:
                                        ax1.plot(times[events.start_indices[i]:events.end_indices[i]],
                                                 volts[events.start_indices[i]:events.end_indices[i]],
                                                 c='red', linewidth=3)

                        fig.tight_layout()  # otherwise the right y-label is slightly clipped

                else:
                        plt.figure(figsize=(self.screendims_inches[0]*0.9, self.screendims_inches[1]*0.4))

                        plt.plot(times, volts, label='Potential', c='blue', linewidth=1)
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
                        plt.xlabel('Time')
                        plt.ylabel('Potential (V)')
                        plt.legend()
                plt.title(self.filename)
                plt.get_current_fig_manager().toolbar.zoom()
                plt.show()
    

cor = CoronaBrowser()
cor.master.title('Corona browser')
try:
        cor.mainloop()
except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        last_frame = lambda tb=tb: last_frame(tb.tb_next) if tb.tb_next else tb
        frame = last_frame().tb_frame
        ns = dict(frame.f_globals)
        ns.update(frame.f_locals)
        code.interact(local=ns)
