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

import avm



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

                # Helper vars:
                self.plotTemperatureWithPotential = BooleanVar()

                # Create root window:
                self.grid()
                self.createWidgets()

                        
                # Exponential temperature fit parameters computed from 20121725_1.csv

                self.plots = ('Z-wind (m/s)', 'Z-wind Dispersion (m/s)') # BUG if there's only one, so need 2 until fixed.

                #self.plots = ('Z-wind (m/s)', 'Z-wind Dispersion (m/s)', 'Wind Speed max (m/s)', 'Wind Direction', 'wind_speed_mean (m/s)')
                # self.plots = ('Z-wind (m/s)', 'Z-wind Dispersion (m/s)', 'Wind Speed max (m/s)', 'Wind Direction', 'pressure_mean (hPa)', 'pressure_median (hPa)', 'pressure_std (hPa)', 'temperature_mean (degC)', 'temperature_median (degC)', 'temperature_std (degC)', 'humidity_mean (%RH)', 'humidity_median (%RH)', 'humidity_std (%RH)', 'wind_speed_mean (m/s)', 'wind_speed_std (m/s)', 'wind_direction_mean (degrees)', 'wind_direction_std (degrees)')

                self.debug_seq()


        def debug_seq(self):
                self.no_temperature_correction_check = True
                #self.model = tf.keras.models.load_model('model')
                #self.loadFile(filename='data/20310992-2021-09.csv')
                self.loadFile(filename='data/20311010-2021-10.csv')
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
                self.saveButton = tk.Button(self, text='Save', command=self.debug, state='disabled')
                self.saveButton.grid(row=row, column=6)
                row += 1
                tk.Label(self, text='Voltage scaling factor:').grid(row=row, column=0)
                self.voltageScalingFactor = 1
                self.voltageScalingFactorBox = tk.Entry(self, width=8)
                self.voltageScalingFactorBox.insert(0, self.voltageScalingFactor)
                self.voltageScalingFactorBox['state'] = 'readonly'
                self.voltageScalingFactorBox.grid(row=row, column=1, sticky='W')
                self.voltageScalingFactorBox.insert(0, self.voltageScalingFactor)
                #self.voltageScalingButton = tk.Button(self, text="Apply", command=self.d_test.applyCorrections, state='disabled')
                #self.voltageScalingButton.grid(row=row, column=2)
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
                self.rmplButton = tk.Button(self, text="Plot", command=self.plot_timeseries, state='disabled')
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
                self.doRunPredictorButton = tk.Button(self, text='Run z-wind predictor', command=self.runPredictor, state='disabled')
                self.doRunPredictorButton.grid(row=row, column=2)
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
                self.regressButton.grid_forget()
                self.useNewRegressionButton.grid_forget()
                if hasattr(self, 'z_predicted'):
                        del self.z_predicted

                
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
                        self.d_test = avm.dataset(self, filename)
                        self.prePlotWHOI(d = self.d_test)

                        self.rmplButton['state'] = 'normal'
                        self.useNewRegressionButton['state'] = 'disabled'

                        #try:
                        if True:
                                self.model = keras.models.load_model('model')
                                self.doRunPredictorButton['state'] = 'normal'
                                print('Loaded last saved z-wind prediction model.')

                        #except:
                        #        print('No z-wind prediction model found...')
                
                        
                        self.saveButton['state'] = 'normal'
                        self.doStatisticsButton['state'] = 'normal'

                        self.event_detection_enabled(self.d_test.v_mode < 1)

                        self.plot_timeseries(self.d_test)


        def prePlotWHOI(self, d):
                # Set up for plotting...
                self.legends = {p:[] for p in self.plots}
                self.z = {p:[] for p in self.plots}
                        
                # Build a list of things to plot:
                for i,t in enumerate(d.whoi.columns):
                        #print(f' Looking at column {i} : {t}')
                        for toplot in self.plots:
                                #print(f'Looking for "{toplot}" in "{t}"')
                                if toplot in t:
                                        #print('   ...found')
                                        #print(self.legends)
                                        self.legends[toplot].append(t)
                                        print(f'Adding: self.legends[{toplot}].append({t})')
                                        try:
                                                self.z[toplot].append(int(t.split('m', 1)[0]))
                                        except:
                                                None
                for toplot in self.plots:
                        if len(self.legends[toplot]):
                                self.whoi_graphs += 1

                        
        def plotEvents(self, d=0):
                if isinstance(d, int):
                        d = self.d_test
                if d.v_mode > 1:
                        return
                d.events = self.find_events(d)
                self.plot_timeseries(d, self.events)
                if not hasattr(self, 'no_temperature_correction_check'):
                        self.plot_temperature_corrections()
                        

        # Show default and new regressions from voltage-vs-temperature
        def plot_temperature_corrections(self, d):
                if not d.temperature_present:
                        plt.close(2)
                        return

                self.waitbar_indeterminate_start('Computing regression...')
                length = len(d.temps)
                x = mat(d.temps).reshape((length,1))
                x = np.hstack((x, np.ones((length, 1))))
                y = mat(d.volts_scaled).reshape((length,1))

                
                xn = np.reshape(d.temps, -1)
                yn = np.reshape(d.volts_scaled, -1)

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

                sampleX_nl = np.linspace(min(d.temps)-0.1, max(d.temps+0.1), num=100)
                
                sampleY_nl = exponential(sampleX_nl, *exp_pars)
                sampleX = [min(d.temps)-0.3, max(d.temps)+0.3]
                sampleX = mat(sampleX).reshape((2, 1))
                sampleX = np.hstack((sampleX, np.ones((2, 1))))
                sampleY_linear = sampleX * fit
                sampleY_old = sampleX * self.fit
                sampleY_old_exp = exponential(sampleX_nl, *self.fit_exp)

                self.waitbar_indeterminate_start('Plotting regressions...')
                plt.figure(num='temperature correction', figsize=(self.screendims_inches[0]*0.8, self.screendims_inches[1]*0.4))
                plt.clf()
                plt.subplot(1, 3, 1)
                plt.scatter(d.temps, d.volts_scaled, s=0.01, c='black')
                plt.plot(sampleX_nl, sampleY_old_exp, c='blue', label=fit_desc_old_short)
                plt.plot(sampleX_nl, sampleY_nl, c='cyan', label=fit_desc_exp_short)
                plt.plot(sampleX[:,0], sampleY_linear, c='red', linestyle="--", label=fit_desc_short)
                plt.xlabel('Temperature (°C)')
                plt.ylabel('Potential (V)')
                plt.title('Linear regressions')
                plt.legend()
                plt.get_current_fig_manager().toolbar.zoom()

                volts = (y - x * fit).reshape((1,length)).tolist()[0] + np.mean(d.volts_scaled)
                volts_nl = y - exponential(d.temps, *exp_pars) + np.mean(d.volts_scaled)

                plt.subplot(1, 3, (2, 3))
                ax1 = plt.gca()
                ax1.plot(d.times, d.volts_scaled, label='Raw', c='blue', linewidth=1)
                ax1.plot(d.times, volts, label='Potential new linear correction, if using this dataset', c='red', linewidth=1)
                ax1.plot(d.times, d.volts, label='Corrected, as applied', c='black', linewidth=1)
                ax1.plot(d.times, volts_nl, label='Potential new exponential correction', c='cyan', linewidth=1)
                ax1.set_xlabel('Time')

                ax2 = ax1.twinx()
                ax2.set_ylabel('Temperature (°C)', color='green')
                ax2.plot(d.times, d.temps, color='green', label='Temperature', linewidth=1)
                ax2.tick_params(axis='y', labelcolor='green')
                
                ax1.legend()
                plt.get_current_fig_manager().toolbar.zoom()
                plt.title(d.datafile.stem)
                plt.get_current_fig_manager().toolbar.zoom()
                plt.show()
                self.waitbar_indeterminate_done();

                
        # Copy new fit parameters to default fit param location
        def use_new_correction(self, d):
                self.fit_exp = self.fit_exp_new
                d.applyCorrections()



        def slice_frame(self, pattern, df):
                r = []
                for i,t in enumerate(df.columns):
                        #print(f' Looking at column {i} : {t}')
                        if pattern in t:
                                #print('   ...found')
                                #pdb.set_trace()
                                r.append(t)
                return r
                

        def trainPredictor(self, d_train=0):
                if not isinstance(d_train, avm.dataset):
                        d_train = self.d_test
                
                print(f'Training predictor on {d_train.filename} ...')

                # d_validation = avm.dataset(self, 'data/203101010-something.csv')
                
                self.n_avm_samples = 360
                batch_size = 128

                # Stick Ted's data into a dataframe. This has already had the timezone sorted.
                df = pd.DataFrame(data={'AVM volts': d_train.volts.squeeze()}, index=pd.DatetimeIndex(d_train.times))
                # Upsample WHOI LIDAR z-wind:

                lidarz = d_train.whoi.loc[:, self.slice_frame('Z-wind (m/s)', d_train.whoi)]

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
                model.add(tf.keras.layers.Dropout(0.5))
                model.add(tf.keras.layers.Conv1D(15, 5, activation='relu'))
                model.add(tf.keras.layers.MaxPooling1D(2))
                model.add(tf.keras.layers.Dropout(0.5))
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

                model.fit(generator, workers=12, epochs=20)
                self.doRunPredictorButton['state'] = 'normal'

                model.save('model')
                
                self.runPredictor(model, d)

                # Um... can be called via button, so I guess we need to do it this way too...
                self.model = model
                return model


        def runPredictor(self, model=0, d=0):
                if isinstance(model, int):
                        if hasattr(self, 'model'):
                                print('runPredictor(): no model parameter passed; running on saved model.')
                                model = self.model
                        else:
                                print('runPredictor: no model found.')
                if isinstance(d, int):
                        d = self.d_test
                
                print('Running trained predictor...')
                
                self.n_avm_samples = model.layers[0].output_shape[1]

                # Stick Ted's data into a dataframe. This has already had the timezone sorted.
                df = pd.DataFrame(data={'AVM volts': d.volts.squeeze()}, index=pd.DatetimeIndex(d.times))
                # Upsample WHOI LIDAR z-wind:

                lidarz = d.whoi.loc[:, self.slice_frame('Z-wind (m/s)', d.whoi)]

                df2 = lidarz.max(axis='columns').rename('Z-wind')
                df3 = pandas.merge_asof(df, df2, left_index = True, right_index = True, direction='nearest', tolerance=dt.timedelta(minutes=20))
                df3 = df3.interpolate(method='linear', limit_direction='both')

                print(model.summary())

                print('Generating timeseries...')
                gen2 = TimeseriesGenerator(df3.loc[:,'AVM volts'], df3.loc[:,'Z-wind'], length = self.n_avm_samples, batch_size = len(d.volts), shuffle = False)
                x, y = gen2[0]
                self.z_predicted = np.full((self.n_avm_samples, 1), np.NaN)
                print('Running the model...')
                self.z_predicted = np.append(self.z_predicted, model.predict(x))
 
                print('Plotting...')
                fig = plt.figure(num = 'timeseries')
                fig.axes[1].plot(d.times, self.z_predicted + 2, color='red')

                self.doStatistics(d)

                
        def find_events(self, d):
                events = Events()
                # Temperature data mean we're using the atmospheric voltage monitor. Don't hilight events.
                if d.temperature_present:
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


                self.waitbar_start('Looking for events...', len(d.times))

                thresholdCounter = 0
                aboveThresholdStart = 0

                for i in range(len(d.times)):
                        self.waitbar_update(i)

                        if d.volts[i] > self.eventThreshold['volts']:
                                if thresholdCounter == 0:
                                        aboveThresholdStart = i
                                thresholdCounter += 1
                        else:
                                if thresholdCounter > 0: # Was above threshold at time index i-1
                                        # If just counting samples:
                                        #if thresholdCounter >= self.eventThreshold['count']:
                                        #        events.append(i)
                                        # If looking for time-above-threshold:
                                        aboveThresholdTime = d.times[i-1] - d.times[aboveThresholdStart]
                                        if aboveThresholdTime.seconds >= self.eventThreshold['count']:
                                                #print(f'Found an event of duration > {aboveThresholdTime.seconds} seconds.')
                                                events.add(aboveThresholdStart, i, aboveThresholdTime.seconds)
                                thresholdCounter = 0;
                self.waitbar_done()
                return events

        # Plot voltage vs time using Matplotlib
        def plot_timeseries(self, d=0, events=[]):
                if isinstance(d, int):
                        d = self.d_test

                fig = plt.figure(num='timeseries', figsize=(self.screendims_inches[0]*0.7, self.screendims_inches[1]*0.7))
                fig.clf()

                nsubplots_base = 1
                nsubplots = nsubplots_base + self.whoi_graphs
                
                plt.subplot(nsubplots, 1, 1)

                axes = [fig.gca()]

                if d.temperature_present & self.plotTemperatureWithPotential.get():

                        color = 'black'
                        axes[0].set_xlabel('Time')
                        axes[0].set_ylabel('Potential (V)', color=color)
                        axes[0].plot(d.times, d.volts, color=color, label='Potential', linewidth=1)
                        axes[0].tick_params(axis='y', labelcolor=color)

                        ax00 = axes[0].twinx()

                        color = 'tab:green'
                        ax00.set_ylabel('Temperature (°C)', color=color)
                        ax00.plot(d.times, d.temps, color=color, label='Temperature', linewidth=1)
                        ax00.tick_params(axis='y', labelcolor=color)

                        if isinstance(events, Events):
                            for i in range(len(events.start_indices)):
                                # No longer used; may be back eventually...
                                if i == 0:
                                        ax[0].plot(d.times[events.start_indices[i]:events.end_indices[i]],
                                                d.volts[events.start_indices[i]:events.end_indices[i]],
                                                c='red', linewidth=3, label='event?')
                                else:
                                        ax[0].plot(d.times[events.start_indices[i]:events.end_indices[i]],
                                                d.volts[events.start_indices[i]:events.end_indices[i]],
                                                c='red', linewidth=3)

                        fig.tight_layout()  # otherwise the right y-label is slightly clipped

                else:
                        axes[0].plot(d.times, d.volts, label='Potential', c='black', linewidth=0.5)


                        if isinstance(events, Events):
                            for i in range(len(events.start_indices)):
                                if i == 0:
                                        plt.plot(d.times[events.start_indices[i]:events.end_indices[i]],
                                                 d.volts[events.start_indices[i]:events.end_indices[i]],
                                                 c='red', linewidth=3, label='event?')
                                else:
                                        plt.plot(d.times[events.start_indices[i]:events.end_indices[i]],
                                                 d.volts[events.start_indices[i]:events.end_indices[i]],
                                                 c='red', linewidth=3)
                                #plt.scatter([times[i] for i in events.indices], [d.volts[i] for i in events.indices],
                                #            s=events.sizes, c='red', label='Event?')
                        plt.ylabel('Potential (V)')



                # Lines at midnight.
                dr = pandas.date_range(d.times[1], d.times[-1], normalize=True).to_pydatetime()[1:].tolist()
                vr = axes[0].get_ylim()
                for i in range(len(dr)):
                        axes[0].plot([dr[i], dr[i]], vr, color='black', alpha=0.2)
                axes[0].set_ylim(vr)

                # Spectrogram:
                if False:
                        axes.append(plt.subplot(nsubplots, 1, 2, sharex = axes[0]))

                        winlen = 128
                        noverlap = int(winlen / 2)
                        spectimes = d.times[noverlap::noverlap]
                        f, t, Sxx = signal.spectrogram(x=d.volts.flatten(), fs=0.1, noverlap=noverlap, window=signal.windows.tukey(winlen), mode='magnitude')
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
                                                axes[n].plot(d.whoi.index, d.whoi[i],
                                                             label=i, color=colours[colour])
                                                colour += 1
                                #axes[n].legend()
                                axes[n].set_xlim(d.times[0], d.times[-1])
                                n += 1
                
                axes[0].set_title(d.datafile.stem)
                plt.get_current_fig_manager().toolbar.zoom()
                plt.show()

                self.doStatistics(d)

        def doStatistics(self, d=0):
                if isinstance(d, int):
                        d = self.d_test

                corr_interesting_threshold = 0.1 # Show all correlations above a threshold
                corr_interesting_n = 4 # Show the n most interesting

                key = 'AVM volts'
                key_z = 'Predicted Z-wind'

                
                # Stick Ted's data into a dataframe. This has already had the timezone sorted.
                df = pd.DataFrame(data={key: d.volts.squeeze()}, index=pd.DatetimeIndex(d.times))
                if hasattr(self, 'z_predicted'):
                        df[key_z] = self.z_predicted.squeeze()

                # Downsample onto the WHOI data's timestamps:
                #df = df.groupby(d.whoi.index[d.whoi.index.searchsorted(df.index)-1]).std()
                df = df.groupby(d.whoi.index[d.whoi.index.searchsorted(df.index)-1]).max()
                #df = df.groupby(d.whoi.index[d.whoi.index.searchsorted(df.index)-1]).mean()
                
                whoi_interp = d.whoi.interpolate(method='linear', limit_direction='both')

                # Easiest most braindead way to line up all the data?
                df = df.join(whoi_interp)
                cor = df.corr()
                corV = cor.loc[:,key].drop({key, key_z}, errors='ignore') # correlation with key; drop self-corr

                # List interesting indices, in order of interestingness:
                corVs = corV.sort_values(ascending = False, key = lambda x: abs(x))
                print(corVs)

                # Show ones above the threshold?
                interesting = corVs.index[np.abs(corVs) > corr_interesting_threshold]
                # ...or show top n? Comment out the line below to use above.
                interesting = corVs.index[range(0, np.min((corr_interesting_n, len(corVs.index))))]
                if hasattr(self, 'z_predicted'):
                        corVpred = cor.loc[:,key_z].drop({key, key_z}, errors='ignore') # correlation with key; drop self-corr
                        corVpreds = corVpred.sort_values(ascending = False, key = lambda x: abs(x))
                        print(corVpreds)

                if interesting.size or hasattr(self, 'z_predicted'):
                        # Figure out the layout
                        n = int(np.ceil(np.sqrt(interesting.size)))
                        m = int(np.ceil(interesting.size / n))

                        # If we run off the rectangle we'll just silently drop the least interesting ;)
                        plt.figure(num='correlations', figsize=(self.screendims_inches[0]*0.4, self.screendims_inches[1]*0.4))
                        plt.clf()
                        counter = 1
                        if hasattr(self, 'z_predicted'):
                                y = corVpreds.index[0]
                                mask = ~np.isnan(df.loc[:,key_z]) & ~np.isnan(df.loc[:,y])
                                slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(df.loc[mask,key_z], df.loc[mask,y])
                                ax = plt.subplot(n, m, counter)
                                df.plot.scatter(x = key_z, y = y, ax = ax, s=5, c='blue', label=f'corr = {corVpred.loc[y]:.2f}, p={p_value:.2g}, r={r_value:.2g}')
                                counter += 1
                                
                        for y in interesting:
                                mask = ~np.isnan(df.loc[:,key]) & ~np.isnan(df.loc[:,y])
                                slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(df.loc[mask,key], df.loc[mask,y])
                                try: # If z_predicted has pushed us outside the "interesting rectangle", just omit the last one
                                        ax = plt.subplot(n, m, counter)
                                except:
                                        print("One interesting graph was omitted for prettier layout; it didn't really seem all that interesting after all...")
                                        break
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
