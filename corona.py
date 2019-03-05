#!/Users/bwp/anaconda3/bin/python
import csv
from datetime import datetime
import tkinter as tk
from tkinter import filedialog
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


class CoronaBrowser(tk.Frame):
        
        def __init__(self, master=None):
                tk.Frame.__init__(self, master)
                self.grid()
                self.createWidgets()

        def createWidgets(self):
                self.loadButton = tk.Button(self, text="Load", command=self.loadFile)
                self.loadButton.grid()
                self.quitButton = tk.Button(self, text="Quit", command=self.quit)
                self.quitButton.grid()

        def loadFile(self):
                data_filename = filedialog.askopenfilename(filetypes=[('Comma-separated values', '*.csv')])

                self.times = []
                self.volts = []
                if data_filename:
                        with open(data_filename) as csv_file:
                                csv_reader = csv.reader(csv_file, delimiter=',')
                                line_count = 0
                                for row in csv_reader:
                                        if line_count == 0:
                                                line_count += 1
                                                continue

                                        elif line_count == 1:
                                                line_count += 1

                                        else:
                                                dt = datetime.strptime(row[1], '%m/%d/%y %I:%M:%S %p')
                                                self.times.append(dt)
                                                v = float(row[2])
                                                self.volts.append(v)
                                                line_count += 1

                print(f'Read {self.times[0]} -- {self.times[-1]} ({line_count} samples).')
                self.plot_voltages_mat()


        def plot_voltages_bokeh(self):
                from bokeh.plotting import figure, show
                from bokeh.models.sources import ColumnDataSource
                from bokeh.models import DatetimeTickFormatter, HoverTool, BoxZoomTool, WheelPanTool, ResetTool, ZoomOutTool

                p = figure(plot_width=3, plot_height=1, x_axis_type='datetime',
                           tools=[BoxZoomTool(),
                                  WheelPanTool(dimension='width'),
                                  ZoomOutTool(factor=0.5),
                                  ResetTool()])
                p.sizing_mode = 'scale_width'
                p.line(x = self.times, y = self.volts)
                p.xaxis[0].formatter = DatetimeTickFormatter(days='%Y-%m-%d %H:%M', hours='%Y-%m-%d %H:%M', hourmin='%Y-%m-%d %H:%M',
                                                             minutes='%Y-%m-%d %H:%M', minsec='%Y-%m-%d %H:%M:%S',
                                                             seconds='%Y-%m-%d %H:%M:%S')

                #output_notebook()
                show(p)


        def plot_voltages_mat(self):
                plt.figure()
                plt.plot(self.times, self.volts)
                plt.show()
    
                

cor = CoronaBrowser()
cor.master.title('Corona browser')
cor.mainloop()
cor.destroy()

