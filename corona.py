import pandas as pd
import datetime
import tkinter as tk
from tkinter import filedialog, ttk, END
import matplotlib
#matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
from bokeh.plotting import figure, show
from bokeh.models.sources import ColumnDataSource
from bokeh.models import DatetimeTickFormatter, HoverTool, BoxZoomTool, PanTool, WheelPanTool, ResetTool, ZoomInTool, ZoomOutTool
from bokeh.models import LinearAxis, Range1d
import numpy as np
from numpy import mat # matrix
from numpy.linalg import inv
import math

import time

# 20311011 is good



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

                # Create root window:
                self.grid()
                self.createWidgets()

        # Set up the main window.
        def createWidgets(self):
                self.loadButton = tk.Button(self, text="Load", command=self.loadFile)
                self.loadButton.grid(row=0, column=0, columnspan=5)
                tk.Label(self, text='Detection thresholds:', font=('bold')).grid(row=1, column=0)
                tk.Label(self, text='Volts:').grid(row=1, column=1, sticky='E')
                self.detectionVoltageBox = tk.Entry(self, width=5)
                self.detectionVoltageBox.grid(row=1, column=2, sticky='W');
                tk.Label(self, text='Seconds:').grid(row=1, column=3, sticky='E')
                self.detectionCountBox = tk.Entry(self, width=5)
                self.detectionCountBox.grid(row=1, column=4, sticky='W');
                
                self.rbButton = tk.Button(self, text="Detect (Bokeh)", command=self.processFile)
                self.rbButton.grid(row=2, column=0)
                self.rmplButton = tk.Button(self, text="Detect (MatPlotLib)", command=self.processFileMPL)
                self.rmplButton.grid(row=2, column=1)
                self.regressButton = tk.Button(self, text='Potential vs Temp', command=self.regress)
                self.waitbar_label = tk.Label(self, text='Ready.')
                self.waitbar_label.grid(row=3, column=0, columnspan=5)
                self.waitbar = ttk.Progressbar(self, orient="horizontal", length=300, value=0, mode='determinate')
                self.waitbar.grid(row=4, column=0, columnspan=5)
                self.quitButton = tk.Button(self, text="Quit", command=self.quit)
                self.quitButton.grid(row=6, column=4)

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

        # Ask for a filename, load it, plot it.
        def loadFile(self):
                data_filename = filedialog.askopenfilename(filetypes=[('Comma-separated values', '*.csv')])
                if data_filename:
                        self.regressButton.grid_forget()
                        self.times, self.volts, self.temps = self.loadFileBen(data_filename)                        
                        self.filename = data_filename
                        self.processFileMPL()

        def processFile(self):
                self.events = self.find_events(self.times, self.volts)
                self.plot_voltages_bokeh(self.times, self.volts, self.temps, self.events)
                
        def processFileMPL(self):
                self.events = self.find_events(self.times, self.volts)
                self.plot_voltages_matplotlib(self.times, self.volts, self.temps, self.events)
                        

        # Load a file using Pandas. This is  easy, but a little slow.
        def loadFilePandas(self, data_filename):
                dateparser = lambda dates: [pd.datetime.strptime(d, self.date_format) for d in dates]

                d = pd.read_csv(data_filename, header = None, skiprows = 2,
                                names = ['index', 'date', 'volts', 'h', 's', 'e'],
                                usecols = ['date', 'volts'],
                                parse_dates=['date'], date_parser=dateparser)
                register_matplotlib_converters()
                d.head()
                return d['date'].tolist(), d['volts'].tolist()

        def regress(self):
                length = len(self.temps)
                if length == 0:
                        return 0

                self.waitbar_indeterminate_start('Plotting regression...')

                x = mat(self.temps).reshape((length,1))
                y = mat(self.volts).reshape((length,1))

                x = np.hstack((x, np.ones((length, 1))))
                
                fit = (x.T*x).I*x.T*y
                fit_desc = f'V = {fit[0,0]} * T + {fit[1,0]}'
                print(f'Least-squares linear regression is {fit_desc}')

                sampleX=mat([[min(self.temps)-0.3, 1],
                               [max(self.temps)+0.3, 1]])
                sampleY=sampleX*fit
                
                plt.figure(figsize=(self.screendims_inches[0]*0.5, self.screendims_inches[1]*0.6))
                plt.scatter(self.temps, self.volts, label='data', s=0.01, c='blue')
                plt.plot(sampleX[:,0], sampleY, c='green', label=fit_desc)
                plt.xlabel('Temperature (C)')
                plt.ylabel('Potential (V)')
                plt.title(self.filename)
                plt.legend()
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
            self.waitbar_indeterminate_start('Checking file size...')
            num_lines = sum(1 for line in open(fname))
            #print(f'{num_lines} lines. Time to determine file line count: {time.perf_counter()-start} seconds.')
            self.waitbar_start('Loading...', num_lines)
            
            with open(fname) as csv_file:
                csv_reader = csv.reader(csv_file)
                line_count = 0
                try:
                    for row in csv_reader:
                        self.waitbar_update(line_count)
                        if line_count < 2:
                            line_count += 1
                        else:
                            times.append(datetime.datetime.strptime(row[1], self.date_format))
                            volts.append(float(row[2]))
                            if len(row) > 3:
                                    temps.append(float(row[3]))
                            line_count += 1
                except ValueError:
                    print(f'Warning reading line {line_count}: could not parse date string "{row[1]}" with expected format "{self.date_format}". Using first {line_count-1} lines. Complete row was {row}.')
                except IndexError:
                    print(f'Warning reading line {line_count}: {row} looked incomplete. Using first {line_count-1} lines.')
                          
            self.waitbar_done()
            register_matplotlib_converters()
            lengths = min(len(times), len(volts))
            times=times[0:lengths-1]
            volts=volts[0:lengths-1]
            if len(temps) > 1:
                    temps = temps[0:lengths-1]
                    self.regressButton.grid(row=2, column=4)

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
                                                print(f'Found an event of duration > {aboveThresholdTime.seconds} seconds.')
                                                events.add(aboveThresholdStart, i, aboveThresholdTime.seconds)
                                thresholdCounter = 0;

                self.waitbar_done()
                return events

        # Plot voltage vs time using the Bokeh library.
        def plot_voltages_bokeh(self, times, volts, temps, events):
                self.waitbar_indeterminate_start('Plotting...')
                p = figure(plot_width=3, plot_height=1, x_axis_type='datetime',
                           tools=[BoxZoomTool(),
                                  BoxZoomTool(dimensions='width'),
                                  WheelPanTool(dimension='width'),
                                  ZoomOutTool(factor=0.5),
                                  ZoomOutTool(factor=0.5, dimensions='width'),
                                  ResetTool()],
                           title=self.filename)
                p.sizing_mode = 'scale_width'
                p.yaxis.axis_label = 'Potential (mV)'
                p.line(x = times, y = volts, legend='Potential', color='black')
                p.y_range = Range1d(start=min(volts), end=max(volts))
                if len(temps):
                        p.extra_y_ranges = {"foo": Range1d(start=min(temps), end=max(temps))}
                        p.add_layout(LinearAxis(y_range_name="foo", axis_label='Temperature (C)'), 'right')
                        p.line(x = times, y = temps, legend='Temperature', color='blue', y_range_name="foo")
                p.xaxis[0].formatter = DatetimeTickFormatter(days='%Y-%m-%d %H:%M', hours='%Y-%m-%d %H:%M', hourmin='%Y-%m-%d %H:%M',
                                                             minutes='%Y-%m-%d %H:%M', minsec='%Y-%m-%d %H:%M:%S',
                                                             seconds='%Y-%m-%d %H:%M:%S')
                p.circle([times[i] for i in events.end_indices], [volts[i] for i in events.end_indices],
                         size=[math.sqrt(i) for i in events.sizes], legend='event?', color='red', alpha=0.7)

                show(p)
                self.waitbar_indeterminate_done()


        # Plot voltage vs time using Matplotlib
        def plot_voltages_matplotlib(self, times, volts, temps=[], events=[]):

                if len(temps):
                        fig, ax1 = plt.subplots(figsize=(self.screendims_inches[0]*0.9,
                                                         self.screendims_inches[1]*0.4))
                        plot_v_to = ax1
                        

                        color = 'blue'
                        ax1.set_xlabel('Time')
                        ax1.set_ylabel('Potential (V)', color=color)
                        ax1.plot(times, volts, color=color, label='Potential', linewidth=1)
                        ax1.tick_params(axis='y', labelcolor=color)

                        ax2 = ax1.twinx()

                        color = 'tab:green'
                        ax2.set_ylabel('Temperature (C)', color=color)
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
                                #plt.scatter([times[i] for i in events.indices], [volts[i] for i in events.indices],
                                #            s=events.sizes, c='red', label='Event?')
                        #if len(events.start_indices):
                        #        plt.legend([l1, l2, l3],["Potential", "Temperature", "Event?"])
                        #else:
                        #        plt.legend([l1, l2], ["Potential", "Temperature"])
                                
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
                plt.get_current_fig_manager().toolbar.zoom()
                plt.title(self.filename)
                plt.show()
    
                

cor = CoronaBrowser()
cor.master.title('Corona browser')
cor.mainloop()
cor.destroy()

