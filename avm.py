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

class dataset:
        def __init__(self, browser, filename):
                self.filename = filename
                self.browser = browser
                
                self.times, self.volts_raw, self.temps = self.loadFileBen(Path(filename))

        # Faster loading. Note that Python grows lists sensibly, so
        # repeated calls to append() aren't as inefficient as they
        # look.
        def loadFileBen(self, fname):
            times = []
            volts = []
            temps = []

            self.datafile = fname

            # Define the weird date format
            date_format = ['%m/%d/%y %I:%M:%S %p', '%m/%d/%y %H:%M']
                
            # Get the number of lines in the file so we can do a perfect progress bar...
            start = time.perf_counter()
            print(f'----- Loading {fname} -----')
            self.browser.waitbar_indeterminate_start('Checking file size...')
            num_lines = sum(1 for line in open(fname))
            self.browser.waitbar_indeterminate_done()
            # print(f'{num_lines} lines. Time to determine file line count: {time.perf_counter()-start} seconds.')
            self.browser.waitbar_start('Loading...', num_lines)
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
                                self.browser.waitbar_update(line_count)
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
                                            t = dt.datetime.strptime(row[date_column], date_format[0]) - timedelta_corona
                                            times.append(timezone_utc.localize(t))
                                    except ValueError:
                                            print(f"Date format '{date_format[0]}' doesn't work; trying '{date_format[1]}'")
                                            t = dt.datetime.strptime(row[date_column], date_format[1]) - timedelta_corona
                                            times.append(timezone_utc.localize(t))
                                    except ValueError:
                                            print(f'  * Line {line_count}: could not parse date string "{row[1]}" with expected format "{date_format}".')
                                            continue;
                                    volts.append(float(row[voltage_column]))
                                    if self.temperature_present:
                                            temps.append(float(row[self.temperature_present]))
                            else:
                                    print(f'  * Line {line_count} "{row}" contains missing values. Ignoring the row.')



            self.browser.waitbar_done()
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

                date_format_lidar = '%Y/%m/%d %H:%M'
                
                self.browser.waitbar_start('  Loading WHOI data...', (times[-1]-times[0]).days+1)

                #print(f'  WHOI: start time is {times[0]}, which is file {firstfnum}. Last year,day is {lastfnum}')

                #lastfname = self.datafile.parent / 'whoi' / 'lidar' / f'asit.lidar.{times[-1].strftime("%Y_%j")}.sta'
                # {<directory>: [ <filename prefix>, <filename suffix>, <invoke special code for WHOI's LIDAR files> ]}
                #filesets = {'lidar': [ 'asit.lidar.', '.sta', True ],
                #             'met': ['met.Vaisala_', '.csv', False ],
                #             'wind': ['met.Anemo_', '.csv', False ]}

                # Here are the files. A hassle since a Dict isn't ordered.
                #filesets = {'lidar': [[ 'asit.lidar.', '.sta', 1 ]],
                #            'met': [['asit.mininode.CLRohn_', '.csv', 0 ]],
                #            'wind': [['asit.mininode.Sonic1_', '.csv', 0 ]]}
                filesets = { 'lidar': [[ 'asit.ZXlidar.', '.CSV', 2], [ 'asit.lidar.', '.sta', 1 ]]}
                
                dataframes = []

                
                # Okay, this is fucking stupid: I am going to read in
                # everything, then purge duplicate rows, then read in
                # everything AGAIN to fill in all the purged data,
                # because Python/Pandas doesn't seem to really have
                # any mechanisms to do this more cleverly.

                ### ROUND 1 ###
                
                day_i = -1
                fnum = ''
                errors = 0
                
                while fnum != lastfnum:
                        day_i += 1
                        self.browser.waitbar_update(day_i)
                        fnum = (times[0] + dt.timedelta(days=day_i)).strftime("%Y_%j")
                        print(f' Loading WHOI files {fnum}...')

                        daily_data = []

                        for fileset, prepost in filesets.items():
                                fname = self.datafile.parent

                                # Due to WHOI changing the LIDAR
                                # around 2021-10-01, need to jump
                                # through hoops to find the correct
                                # file.
                                for i in range(len(prepost)):
                                        fname = self.datafile.parent / 'whoi' / fileset / f'{prepost[i][0]}{fnum}{prepost[i][1]}'
                                        if fname.is_file:
                                                # we've got a promising filename
                                                print(f'Confirmed file {fname}')
                                                break

                                print(f'      Loading {fname}')

                                if fname.is_file():
                                        if fileset == 'lidar':
                                                # Special code for WHOI's LIDAR files:
                                                with open(fname, errors='replace') as f:
                                                        row = f.readline()
                                                        if row[0:9] == 'HeaderSize'[0:9]:
                                                                print('Eeeew, the old LIDAR.')
                                                                headersize = int(row.split('=')[1])
                                                                mdf = pd.read_csv(fname, sep='=', nrows=headersize-1, index_col = 0, header=None,
                                                                                  encoding='cp1252')
                                                                whoi_lidar_timezone = mdf.loc['timezone', 1]
                                                                # I can't deal with the 19 different timezone and time offset systems in Python. Just check that it hasn't changed:
                                                                if whoi_lidar_timezone != "UTC+0":
                                                                        raise Exception('LIDAR time offset changed.')

                                                                df = pd.read_csv(fname, sep='\t', header=headersize, parse_dates=[0], index_col=0,
                                                                                 encoding='cp1252')
                                                        elif row[0:9] == 'CSV Converter'[0:9]:
                                                                print('Aaaah, the new LIDAR.')
                                                                headersize = 1
                                                                df = pd.read_csv(fname, index_col = 1, header = 1, encoding='cp1252')
                                                                pdb.set_trace()
                                                        else:
                                                                print('Could not get header size. Assuming 0.')
                                                                headersize = 0

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

                        if len(daily_data):
                                dataframes.append(daily_data)

                print(f'type of dataframes: {type(dataframes)}, length {len(dataframes)}')
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
                        self.browser.waitbar_update(day_i)
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

                #self.browser.waitbar_done()
                #self.doTrainButton['state'] = 'normal'

                
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
                
        def setVoltageScalingFactor(self, vsf):
                self.voltageScalingFactor = vsf
                self.browser.voltageScalingFactorBox.delete(0, END)
                self.browser.voltageScalingFactorBox.insert(0, self.voltageScalingFactor)
                        
        def getVoltageScalingFactor(self):
                try:
                        self.voltageScalingFactor = float(self.voltageScalingFactorBox.get())
                except:
                        print(f'Could not convert voltage scaling factor "{self.voltageScalingFactorBox.get()}" to number. Resetting to 1.')
                        self.voltageScalingFactor = 1.0
                self.browser.voltageScalingFactorBox.delete(0, END)
                self.browesr.voltageScalingFactorBox.insert(0, self.voltageScalingFactor)

