import pandas as pd
from datetime import datetime
import tkinter as tk
from tkinter import filedialog
import matplotlib
#matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
import time

# 20311011 is good

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
                self.loadButton = tk.Button(self, text="Load", command=self.doFile)
                self.loadButton.grid()
                self.quitButton = tk.Button(self, text="Replot with Bokeh", command=self.processFile)
                self.quitButton.grid()
                self.quitButton = tk.Button(self, text="Replot with MatPlotLib", command=self.processFileMPL)
                self.quitButton.grid()
                self.quitButton = tk.Button(self, text="Quit", command=self.quit)
                self.quitButton.grid()

        # Ask for a filename, load it, plot it.
        def doFile(self):
                data_filename = filedialog.askopenfilename(filetypes=[('Comma-separated values', '*.csv')])
                if data_filename:
                        #start = time.time() 
                        self.times, self.volts = self.loadFileBen(data_filename)
                        self.filename = data_filename
                        self.processFile()

        def processFile(self):
                #print(time.time() - start) # how many seconds does it take to load the file?
                self.events = self.find_events(self.times, self.volts)
                #self.plot_voltages_bokeh(self.times, self.volts, self.events)
                self.plot_voltages_bokeh(self.times, self.volts, self.events)
                
        def processFileMPL(self):
                #print(time.time() - start) # how many seconds does it take to load the file?
                #self.events = self.find_events(self.times, self.volts)
                #self.plot_voltages_bokeh(self.times, self.volts, self.events)
                self.plot_voltages_matplotlib(self.times, self.volts)
                        

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


        # Faster loading. Note that Python grows lists sensibly, so
        # repeated calls to append() aren't as inefficient as they
        # look.
        def loadFileBen(self, fname):
            import csv

            times = []
            volts = []

            with open(fname) as csv_file:
                csv_reader = csv.reader(csv_file)
                line_count = 0
                try:
                    for row in csv_reader:
                        if line_count < 2:
                            line_count += 1
                        else:
                            times.append(datetime.strptime(row[1], self.date_format))
                            volts.append(float(row[2]))
                            line_count += 1
                except ValueError:
                    print(f'Warning reading line {line_count}: could not parse date string "{row[1]}" with expected format "{self.date_format}". Using first {line_count-1} lines.')
                except IndexError:
                    print(f'Warning reading line {line_count}: incomplete line. Using first {line_count-1} lines.')
            register_matplotlib_converters()
            lengths = min(len(times), len(volts))
            times=times[0:lengths-1]
            volts=volts[0:lengths-1]
            return times, volts


        def find_events(self, times, volts):
                eventThreshold = {'volts': 0.02, 'count': 5}

                events = []
                thresholdCounter = 0
                for i in range(len(times)):
                        if volts[i] > eventThreshold['volts']:
                                thresholdCounter += 1
                        else:
                                thresholdCounter = 0;
                        if thresholdCounter == eventThreshold['count']:
                                events.append(i)
                return events

        # Plot voltage vs time using the Bokeh library.
        def plot_voltages_bokeh(self, times, volts, events):
                from bokeh.plotting import figure, show
                from bokeh.models.sources import ColumnDataSource
                from bokeh.models import DatetimeTickFormatter, HoverTool, BoxZoomTool, WheelPanTool, ResetTool, ZoomOutTool

                p = figure(plot_width=3, plot_height=1, x_axis_type='datetime',
                           tools=[BoxZoomTool(),
                                  WheelPanTool(dimension='width'),
                                  ZoomOutTool(factor=0.5),
                                  ResetTool()],
                           title=self.filename)
                p.sizing_mode = 'scale_width'
                p.line(x = times, y = volts, legend='trace')
                p.xaxis[0].formatter = DatetimeTickFormatter(days='%Y-%m-%d %H:%M', hours='%Y-%m-%d %H:%M', hourmin='%Y-%m-%d %H:%M',
                                                             minutes='%Y-%m-%d %H:%M', minsec='%Y-%m-%d %H:%M:%S',
                                                             seconds='%Y-%m-%d %H:%M:%S')
                p.circle([times[i] for i in events], [volts[i] for i in events], size=8, legend='event?', color='red', alpha=0.6)

                show(p)


        # Plot voltage vs time using Matplotlib
        def plot_voltages_matplotlib(self, times, volts):
                plt.figure(figsize=(self.screendims_inches[0]*0.9, self.screendims_inches[1]*0.3))
                plt.plot(times, volts)
                plt.xlabel('time')
                plt.ylabel('volts')
                plt.title(self.filename)
                plt.show()
    
                

cor = CoronaBrowser()
cor.master.title('Corona browser')
cor.mainloop()
cor.destroy()

