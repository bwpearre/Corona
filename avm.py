import datetime as dt
from tkinter import END
import numpy as np
from numpy import mat  # matrix
import time
import scipy.stats
import csv
from pathlib import Path
import pandas as pd
import pytz
import geopy.distance
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import pdb
import re


def exponential(x, a, b, c):
    # return a + b * x + c * np.square(x)
    return a * np.exp(b * x) + c


voltage_extreme = 0.5


class dataset:

    def __init__(self, browser, filename):
        self.filename = filename
        self.browser = browser
        self.sensor_serial_number = np.NaN

        self.loadSensors()

        # Temperature correction fit. This was computed from 20311010_450_expurgated.csv
        self.fit = mat([[-0.020992708021557917],
                        [5.272377975649473]])

        # Dictionary is exponential fit params indexed on sensor serial number:
        self.fits = {-1: (1.713, -0.07977, 4.493),
                     20121725: (1.3545883156167038, -0.0714642458651333, 4.482818353894192),
                     20310992: (0.28929039047675625, -0.08857734679045634, 4.447723485085831)}

        # Convert new to old column names:
        self.column_rename = {'Vertical Wind Speed (m/s) at 40m': '40m Z-wind (m/s)',
                              'Vertical Wind Speed (m/s) at 47m': '47m Z-wind (m/s)',
                              'Vertical Wind Speed (m/s) at 67m': '67m Z-wind (m/s)',
                              'Vertical Wind Speed (m/s) at 77m': '77m Z-wind (m/s)',
                              'Vertical Wind Speed (m/s) at 87m': '87m Z-wind (m/s)',
                              'Vertical Wind Speed (m/s) at 97m': '97m Z-wind (m/s)',
                              'Vertical Wind Speed (m/s) at 107m': '107m Z-wind (m/s)',
                              'Vertical Wind Speed (m/s) at 127m': '127m Z-wind (m/s)',
                              'Vertical Wind Speed (m/s) at 147m': '147m Z-wind (m/s)',
                              'Vertical Wind Speed (m/s) at 167m': '167m Z-wind (m/s)',
                              'Vertical Wind Speed (m/s) at 187m': '187m Z-wind (m/s)',
                              'Vertical Wind Speed (m/s) at 38m': '40m Z-wind (m/s)',
                              'Vertical Wind Speed (m/s) at 68m': '77m Z-wind (m/s)'}

        self.loadFileBen(Path(filename))
        self.loadWHOI(self.times)

    # Read the sensor list, which connects serial numbers with 4-D locations and gain settings etc. Unsorted, since faster to Stalinsort and then Quicksort.
    def loadSensors(self):
        filename = 'data/sensors.csv'
        self.sensors = pd.read_csv(filename, skipinitialspace=True, parse_dates=['Date'], index_col='Date')

    # Pull the first number out of a string. If none found, return NaN
    def leading_numeric(self, str):
        str = str.strip()
        if str is None:
            return np.NaN
        else:
            digits = re.search("[+-]?\d*(\.\d+)?", str).group()
            if digits is None:
                print(f'Could not get digits out of "{str}"')
                pdb.set_trace()
                return np.NaN
            if digits.isdigit():
                return float(digits)
            else:
                return np.NaN

    # Much faster loading than pandas (is it really worth it? It
    # was during the early stage of this project when loading was
    # by far the slowest thing). Note that Python grows lists
    # sensibly, so repeated calls to append() aren't as
    # inefficient as they look.
    def loadFileBen(self, filename):
        times = []
        volts = []
        temps = []

        self.datafile = filename

        # Define the legal date formats. Just error out if these fail.
        date_format = ['%m/%d/%Y %H:%M:%S', '%m/%d/%y %I:%M:%S %p']  # , '%m/%d/%y %H:%M' ,'%m/%d/%Y %H:%M' ]

        # Get the number of lines in the file so we can do a perfect progress bar...
        start = time.perf_counter()
        print(f'----- Loading {filename} -----')
        errors_incomplete = 0
        errors_missing = 0
        self.browser.waitbar_indeterminate_start('Checking file size...')
        num_lines = sum(1 for line in open(filename))
        self.browser.waitbar_indeterminate_done()
        # print(f'{num_lines} lines. Time to determine file line count: {time.perf_counter()-start} seconds.')
        self.browser.waitbar_start('Loading...', num_lines)
        self.temperature_present = 0
        timezone_utc = pytz.timezone("UTC")
        date_column = -1
        voltage_column = -1
        scaled_column = -1
        temperature_in_freedom_units = False
        still_in_header = True
        self.distance_max = 0

        with filename.open() as csv_file:
            csv_reader = csv.reader(csv_file)
            line_count = 0
            for row in csv_reader:
                if line_count % 100000 == 0 and False:
                    print(f'  Reading row {line_count}...')
                if line_count % 1000 == 0:
                    self.browser.waitbar_update(line_count)
                line_count += 1

                if still_in_header:
                    foo = row[0].split('=')
                    if foo[0][0] == '#':
                        for column in range(1, len(row)):
                            serialnumberfound = row[column].find("S/N:")
                            if serialnumberfound != -1:
                                sn = self.leading_numeric(row[column][serialnumberfound:].split(":")[1].split(',')[0])
                                sn = int(sn)
                                if np.isnan(self.sensor_serial_number):
                                    self.sensor_serial_number = sn
                                    self.sensor = self.sensors[self.sensors['LGR S/N'] == sn].sort_values(by='Date',
                                                                                                          ascending=False).iloc[
                                        0]
                                    print(
                                        f'      Location = {self.sensor["Station name"]} at {self.sensor["Latitude"]}, {self.sensor["Longitude"]}')
                                elif sn == self.sensor_serial_number:
                                    None
                                else:
                                    raise (f"Conflicting sensor serial numbers {sn} and {self.sensor_serial_number}")

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

                        # If there's no "voltage" column but there is a "scaled" column, I guess we just use that instead... and guess that the voltage_scaling_factor should probably be 1
                        if scaled_column >= 0:
                            if voltage_column == -1 or True:
                                print(
                                    '  Could not find "Volt" (ACTUALLY THIS MAY BE A LIE) but did find "Scaled Series". Interpreting it as scaled voltage, and setting my internal scaling factor to 1. Please verify.')
                                voltage_column = scaled_column
                                self.setVoltageScalingFactor(1)
                            else:
                                print(
                                    '  Found both "Volt" and "Scaled Series". Using "Volt", but setting my internal scaling factor to 2. Please verify.')
                                self.setVoltageScalingFactor(2)
                        else:
                            print(
                                '  Found Voltage, but not Scaled Series. Guessing that my internal scaling factor should be 1. Please verify.')
                            self.setVoltageScalingFactor(1)

                        still_in_header = False


                # Here's the meat. Read each line, check for completeness, parse the dates, and add.
                else:  # still_in_header = False
                    if len(row) <= max([date_column, voltage_column, self.temperature_present]):
                        errors_incomplete += 1
                        errors_incomplete_line = line_count
                        errors_incomplete_example = row
                    elif row[date_column] and row[voltage_column] and (
                            (not self.temperature_present) or row[self.temperature_present]):
                        for date_f in date_format:
                            # print(f'Date format "{date_f}"')
                            error_count = 0
                            try:
                                t = dt.datetime.strptime(row[date_column], date_f) - timedelta_corona
                                # print(f'success; date is {t}')
                                continue
                            except ValueError:
                                error_count += 1
                                if error_count == len(date_format):
                                    print(
                                        f'  * Line {line_count}: could not parse date string "{row[1]}" with expected format "{date_format}".')
                        times.append(timezone_utc.localize(t))
                        try:
                            volts.append(float(row[voltage_column]))
                        except:
                            print(f'  * Line {line_count}: could not parse volts string "{row[voltage_column]}".')
                            volts.append(np.NaN)
                        if self.temperature_present:
                            temps.append(float(row[self.temperature_present]))
                    else:
                        errors_missing += 1
                        errors_missing_line = line_count
                        errors_missing_example = row
            if errors_missing:
                print(
                    f'  ***** {errors_missing} errors like: Line {errors_missing_line} "{errors_missing_example}" is missing values. Ignoring the row.')
            if errors_incomplete:
                print(
                    f'  ***** {errors_incomplete} errors like: Line {errors_incomplete_line}: "{errors_incomplete_example}" is incomplete. Corrupt/incomplete file? *****')

        self.browser.waitbar_done()

        # register_matplotlib_converters()
        length = min(len(times), len(volts))
        if self.temperature_present:
            length = min(length, len(temps))
            temps = np.array(temps[0:length]).reshape((length, 1))
            if temperature_in_freedom_units:
                temps = (temps - 32) * 5 / 9
                temperature_in_freedom_units = False  # Unnecessary, but just in case...
                self.browser.regressButton.grid(row=3, column=4)
                self.browser.useNewRegressionButton.grid(row=3, column=5)
                self.browser.useNewRegressionButton['state'] = 'disabled'
                times = times[0:length]
                # volts = np.array(volts[0:length])
        volts = np.array(volts[0:length]).reshape((length, 1))

        self.times = times
        self.volts_raw = volts
        self.temps = temps
        self.applyCorrections()

        if len(temps):
            self.avmpd = pd.DataFrame({'AVM volts': self.volts.squeeze(), 'temps': self.temps}, index=self.times)
        else:
            self.avmpd = pd.DataFrame({'AVM volts': self.volts.squeeze()}, index=self.times)

    def loadWHOIRaw(self, times):
        headersize = 2
        # fname = 'data/Wind_1166@Y2022_M05_D01.CSV'
        # dateparse = lambda x: dt.datetime.strptime(x, '%d-%m-%Y %H:%M:%S')
        days = sorted({t.date() for t in times})
        daily_data = []

        for d in days:
            fname = self.datafile.parent / 'whoi' / 'lidar_raw' / f'Wind_1166@{d.strftime("Y%Y_M%m_D%d")}.CSV'
            if fname.exists():
                # print(f' && Loading file "{fname}"...')
                m = pd.read_csv(fname, index_col=1, parse_dates=[1], dayfirst=True, header=1, encoding='cp1252')
                m.drop(list(m.filter(regex='Checksum')), axis=1, inplace=True)
                # FUCK THOSE FUCKING RATFUCKERS
                m.replace(inplace=True, to_replace=9998, value=np.NaN)
                m.replace(inplace=True, to_replace=9999, value=np.NaN)
                df = m.filter(like='Vertical Wind Speed', axis=1)
                df.rename(inplace=True, columns=self.column_rename)
                # df = -df # Positive UP

                daily_data.append(df)

        if len(daily_data):
            self.whoi_raw = pd.concat(daily_data, copy=False)
            self.whoi_raw = self.whoi_raw.tz_localize('UTC')  # assume :}
        else:
            False  # self.whoi_raw = []

        # Build an all-up structure with raw data: combine,
        # interpolate, remove original raw LIDAR (rather than
        # raw AVM since that is spikier):
        all = pd.merge(self.avmpd, self.whoi_raw, left_index=True, right_index=True, how='outer')
        all.interpolate(method='linear', limit_direction='both', limit=2, inplace=True)
        self.all = all.loc[self.avmpd.index]
        self.all['41m Z-wind (m/s)'] = np.roll(self.all['40m Z-wind (m/s)'].to_numpy(), 261895)
        self.all['42m Z-wind (m/s)'] = np.roll(self.all['40m Z-wind (m/s)'].to_numpy(), -3300)
        # self.all['43m Z-wind (m/s)'] = np.roll(self.all['40m Z-wind (m/s)'].to_numpy(), 50871)

        # self.all['47m Z-wind (m/s)'] = np.roll(self.all['40m Z-wind (m/s)'].to_numpy(), -50871)

    def loadWHOI(self, times):

        try:
            print('Trying to load whoiraw')
            self.loadWHOIRaw(times)
            self.whoi_raw_available = True
            print('   raw: yep')
        except:
            self.whoi_raw_available = False
            print('   raw: nope')

        firstfnum = times[0].strftime("%Y_%j")
        lastfnum = times[-1].strftime("%Y_%j")

        # fname = 'data/Wind_1166@Y2022_M05_D01.CSV'
        print(f'Times[0] is {times[0]}, times[-1] is {times[-1]}')
        date_format_lidar = '%Y/%m/%d %H:%M'

        print(f'  WHOI: start time is {times[0]}, which is file {firstfnum}. Last year,day is {lastfnum}')

        # lastfname = self.datafile.parent / 'whoi' / 'lidar' / f'asit.lidar.{times[-1].strftime("%Y_%j")}.sta'
        # {<directory>: [ <filename prefix>, <filename suffix>, <invoke special code for WHOI's LIDAR files> ]}
        # filesets = {'lidar': [ 'asit.lidar.', '.sta', True ],
        #             'met': ['met.Vaisala_', '.csv', False ],
        #             'wind': ['met.Anemo_', '.csv', False ]}
        # filesets = {'lidar': [ 'asit.ZXlidar.', '.sta', 2 ],
        #             'met': ['met.Vaisala_', '.csv', False ],
        #             'wind': ['met.Anemo_', '.csv', False ]}

        # Here are the files. A hassle since a Dict isn't ordered.
        # filesets = {'lidar': [[ 'asit.lidar.', '.sta', 1 ]],
        #           'met': [['asit.mininode.CLRohn_', '.csv', 0 ]],
        #           'wind': [['asit.mininode.Sonic1_', '.csv', 0 ]]}

        # To compare new vs. old LIDAR:
        # filesets = { 'lidar': [[ 'asit.ZXlidar.', '.csv', 2], [ 'asit.lidar.', '.sta', 1 ]],
        #             'met': [['met.Vaisala_', '.csv', 0 ]],
        #             'wind': [['met.Anemo_', '.csv', 0 ]]}
        filesets = {'lidar': [['asit.ZXlidar.', '.csv', 2]],
                    'met': [['asit.Vaisala_', '.csv', 0]],
                    'wind': [['met.Anemo_', '.csv', 0]]}

        dataframes = []

        # Okay, this is fucking stupid: I am going to read in
        # everything, then purge duplicate rows, then read in
        # everything AGAIN to fill in all the purged data,
        # because Python/Pandas doesn't seem to really have
        # any mechanisms to do this more cleverly.

        n_days = (times[-1] - times[0]).days + 1
        self.browser.waitbar_start('  Loading WHOI data...', n_days)

        for round in {0, 1}:

            day_i = -1
            fnum = ''
            errors = 0
            lidarloc = np.empty((0, 2), float)

            while fnum != lastfnum:
                day_i += 1

                # DIFFERENT between round 0 and round : waitbar goes to halfway:
                self.browser.waitbar_update(day_i / 2 + round * n_days / 2)

                fnum = (times[0] + dt.timedelta(days=day_i)).strftime("%Y_%j")
                # print(f' Loading WHOI files {fnum} ({(times[0] + dt.timedelta(days=day_i)).strftime("%Y-%m-%d")}) (round {round})...')

                daily_data = []

                for fileset, prepost in filesets.items():
                    fname = self.datafile.parent

                    # Due to WHOI changing the LIDAR
                    # around 2021-10-01, need to jump
                    # through hoops to find the correct
                    # file.
                    for i in range(len(prepost)):
                        fname = self.datafile.parent / 'whoi' / fileset / f'{prepost[i][0]}{fnum}{prepost[i][1]}'
                        if fname.exists():
                            # we've got an extant filename
                            break

                    # print(f'      Loading {fname}')

                    if fname.is_file():
                        if fileset == 'lidar':
                            # Special code for WHOI's LIDAR files:
                            with open(fname, errors='replace') as f:
                                row = f.readline()
                                if row[0:9] == 'HeaderSize'[0:9]:
                                    # Old LIDAR
                                    # Manufacturer (via Ted) says "positive up", Eve Cinquino says "positive down". I think Eve was right.
                                    # Date index is "end of interval"
                                    headersize = int(row.split('=')[1])
                                    mdf = pd.read_csv(fname, sep='=', nrows=headersize - 1, index_col=0, header=None,
                                                      encoding='cp1252')
                                    timezone = mdf.loc['timezone', 1]
                                    location = mdf.loc['Location', 1].strip()
                                    latitude, longitude = self.whoi_lidar_old_latlon_parse(mdf.loc['GPS Location', 1])
                                    dist = geopy.distance.distance((self.sensor["Latitude"], self.sensor["Longitude"]),
                                                                   (latitude, longitude)).m
                                    # print(f'            Location: {location}, {int(np.round(dist))} m from AVM')
                                    if dist > 1000:
                                        print(
                                            f'            ***** {fname}: Distance between LIDAR and AVM is {dist / 1000:.1f} km *****')
                                    if dist > self.distance_max:
                                        self.distance_max = dist
                                    lidarloc = np.append(lidarloc, np.array([[latitude, longitude]]), axis=0)

                                    # I can't deal with the 19 different timezone and time offset systems in Python. Just check that it hasn't changed:
                                    if timezone != "UTC+0":
                                        raise Exception('LIDAR time offset changed.')

                                    df = pd.read_csv(fname, sep='\t', header=headersize, parse_dates=[0], index_col=0,
                                                     encoding='cp1252')

                                    # Align from END to MIDDLE of interval
                                    df.set_index(df.index.to_series() - dt.timedelta(minutes=5), inplace=True)
                                    df = df.filter(like='Z-wind (m/s)')
                                    df = -df  # Positive UP

                                    dfh = df.columns.tolist()
                                    heights = [int(i.split('m')[0]) for i in dfh]

                                    # self.column_rename = {}
                                    # for height in heights:
                                    #    self.column_rename[f'Vertical Wind Speed (m/s) at {height}m'] = f'{height}m Z-wind (m/s)'

                                    # Artisinal nearest-neighbour interpolation ;)
                                    # self.column_rename[f'Vertical Wind Speed (m/s) at 38m'] = f'40m Z-wind (m/s)'
                                    # print(self.column_rename)

                                elif row[0:9] == 'CSV Converter'[0:9]:
                                    # New LIDAR
                                    # Date index is "beginning of interval". This will be modified below.
                                    # Manual says "positive up" (ZephIR-Waltz-Manual pdf p. 55)
                                    headersize = 1
                                    df = pd.read_csv(fname, parse_dates=[1], dayfirst=True, index_col=1, header=1,
                                                     encoding='cp1252')
                                    # Align index from BEGINNING to MIDDLE of interval:
                                    df.set_index(df.index.to_series() + dt.timedelta(minutes=5), inplace=True)

                                    # For debugging: the Checksum column contains a huge thing that makes printing the df difficult
                                    df.drop(list(df.filter(like='Checksum')), axis=1, inplace=True)

                                    # Get heights
                                    dfh = df.filter(like='Vertical Wind Speed').columns.tolist()
                                    h = [i.split(' ')[5] for i in dfh]
                                    heights = [int(i.split('m')[0]) for i in h]

                                    # Actually, let's just get rid of everything but what we currently care about:
                                    latlon = df['GPS'].iat[0].split()
                                    latitude = float(latlon[0])
                                    longitude = float(latlon[1])
                                    dist = geopy.distance.distance((self.sensor["Latitude"], self.sensor["Longitude"]),
                                                                   (latitude, longitude)).m
                                    # print(f'            Location: {int(np.round(dist))} m from AVM')
                                    if dist > 1000:
                                        print(
                                            f'            ***** {fname}: Distance between LIDAR and AVM is {dist / 1000:.1f} km *****')
                                    if dist > self.distance_max:
                                        self.distance_max = dist
                                    lidarloc = np.append(lidarloc, np.array([[latitude, longitude]]), axis=0)

                                    # Rename columns to match old LIDAR
                                    df.rename(inplace=True, columns=self.column_rename)

                                    # Let's find timestamps with error codes and put them in some new columns:
                                    vws = set(self.column_rename.values())

                                    df['e8'] = 0
                                    df['e9'] = 0
                                    for col in vws:
                                        df['e8'] += (df[col] == 9998).astype(int)
                                        df['e9'] += (df[col] == 9999).astype(int)

                                    # FUCK THOSE FUCKING RATFUCKERS
                                    df.replace(inplace=True, to_replace=9998, value=np.NaN)
                                    df.replace(inplace=True, to_replace=9999, value=np.NaN)

                                    # dfh = df.columns.tolist()
                                    # heights = [int(i.split('m')[0]) for i in dfh]

                                    # For pre-rename
                                    # pdb.set_trace()

                                else:
                                    print('Could not get header size. Assuming 0.')
                                    headersize = 0

                            # DIFFERENT between round 1 and round 2: don't add (possibly redundant) indices; just fill in missing values
                            if round == 0:
                                if len(daily_data):  # Join lidar+met+wind
                                    daily_data = daily_data.join(df, how='outer')
                                else:
                                    daily_data = df
                            else:
                                self.whoi.update(df)


                        else:
                            # No check for timezone/td is possible here. Just assume?! FIXME
                            df = pd.read_csv(fname, sep=',', header=[0, 1], parse_dates=[0], index_col=0,
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
                            df.columns = newcolumns

                            if round == 0:
                                if len(daily_data):  # Join lidar+met+wind
                                    daily_data = daily_data.join(df, how='outer', rsuffix='r')
                                else:
                                    daily_data = df
                            else:
                                self.whoi.update(df)


                    else:
                        print(f'  File "{fname}" does not exist.')
                        errors += 1
                        if errors >= 30:
                            print('  [[ Not finding WHOI data files. Giving up. ]]')
                            return

                if round == 0:
                    if len(daily_data):
                        dataframes.append(daily_data)

            if round == 0:
                self.whoi = pd.concat(dataframes, copy=False)
                print(f'Before dropping duplicates: dataset is {self.whoi.shape}')
                self.whoi.sort_index(inplace=True, kind='mergesort')
                self.whoi = self.whoi[~self.whoi.index.duplicated(
                    keep='first')]  # FIXME: This will toss some rows that contain data. Fixed in ROUND 1.
                self.whoi.dropna(inplace=True, axis='columns', how='all')
            print(f'After ROUND {round}: dataset is {self.whoi.shape}')

        self.whoi = self.whoi.tz_localize('UTC')  # see "just check that it hasn't changed" above

        fig = plt.figure(num='map')
        fig.clf()

        # Map points of interest:
        map_poi = {'Edgartown Airfield': (41.3893, -70.6122)}
        # "locs" is used to compute the extent of the map points.
        locs = np.array([[self.sensor["Latitude"], self.sensor["Longitude"]]], ndmin=2)
        locs = np.append(locs, lidarloc, axis=0)
        for i in map_poi:
            locs = np.append(locs, np.array([map_poi[i][0], map_poi[i][1]], ndmin=2), axis=0)

        terrain = cimgt.GoogleTiles(style='satellite')
        self.map = fig.add_subplot(1, 1, 1, projection=terrain.crs)
        self.map.set_extent([locs.min(axis=0)[1] - 0.2, locs.max(axis=0)[1] + 0.2, locs.min(axis=0)[0] - 0.15,
                             locs.max(axis=0)[0] + 0.15], crs=ccrs.Geodetic())
        self.map.add_image(terrain, 13)
        gl = self.map.gridlines(draw_labels=True)
        gl.xlines = False
        gl.ylines = False

        self.map.plot(self.sensor["Longitude"], self.sensor["Latitude"], marker='o', color='yellow', markersize=12,
                      alpha=1, transform=ccrs.Geodetic())
        # Points Of Interest:
        for i in map_poi:
            self.map.plot(map_poi[i][1], map_poi[i][0], marker='o', color='orange', markersize=15,
                          alpha=1, transform=ccrs.Geodetic())
        for i in range(lidarloc.shape[0]):
            self.map.plot(lidarloc[i, 1], lidarloc[i, 0], marker='o', color='red', markersize=4,
                          alpha=1, transform=ccrs.Geodetic())

        # Manually add Edgartown Airfield

        # Get the final list of heights:
        dfh = self.whoi.filter(like='Z-wind (m/s)').columns.tolist()
        self.heights = [int(i.split('m')[0]) for i in dfh]
        print(f'Final LIDAR heights: {self.heights}')

        # print('*** NOT Interpolating 20-minute to 10-minute data...')
        self.whoi.interpolate(inplace=True, method='time', limit=1, limit_area='inside')
        # print('*** Interpolating wind_speed_w_mean (m/s)')
        self.whoi['wind_speed_w_mean (m/s)'] = self.whoi['wind_speed_w_mean (m/s)'].interpolate(method='time',
                                                                                                limit_direction='both')
        self.whoi['wind_speed_mean (m/s)'] = self.whoi['wind_speed_mean (m/s)'].interpolate(method='time',
                                                                                            limit_direction='both')

        self.browser.waitbar_done()
        self.browser.doTrainButton['state'] = 'normal'

        # Compute air density according to the formula Ted mailed:
        try:
            print('ATTEMPTING DENSITY')
            p = 'pressure_mean (hPa)'
            RH = 'humidity_mean (%RH)'
            t = 'temperature_mean (met) (degC)'
            self.whoi.loc[:, 'density'] = (0.34848 * self.whoi[p] - 0.009 * self.whoi[RH] * np.exp(
                0.061 * self.whoi[t])) / (273.15 + self.whoi[t])
        except:
            print('Could not compute air density.')

        print(self.whoi.dtypes)

    def applyCorrections(self):
        # Voltage scaling
        self.getVoltageScalingFactor()
        self.volts_scaled = self.volts_raw * self.voltageScalingFactor

        # Correct for sensor temperature. Preserves mean value (i.e. not mean-0)
        # self.volts = self.applyTemperatureCorrection()
        self.volts = self.volts_scaled

        # Approximate mode of data: usually 0 unless we quantise it
        self.v_mode = scipy.stats.mode((self.volts * 10).astype(int))[0][0][0] / 10
        self.volts = self.volts - self.v_mode
        if self.temperature_present:
            t = ' after temperature correction'
        else:
            t = ''
        print(f'  Voltage: mode is roughly {self.v_mode} V{t}')

        # Ted's idea: filter those voltage extrema:
        self.volts[self.volts > voltage_extreme] = np.NaN
        self.volts[self.volts < -voltage_extreme] = np.NaN
        # self.volts.all[self.volts < 0.01 and self.volts > -0.01] = np.NaN

    def setVoltageScalingFactor(self, vsf):
        self.voltageScalingFactor = vsf
        self.browser.voltageScalingFactorBox.delete(0, END)
        self.browser.voltageScalingFactorBox.insert(0, self.voltageScalingFactor)

    def getVoltageScalingFactor(self):
        try:
            self.voltageScalingFactor = float(self.browser.voltageScalingFactorBox.get())
        except:
            print(
                f'Could not convert voltage scaling factor "{self.browser.voltageScalingFactorBox.get()}" to number. Resetting to 1.')
            self.voltageScalingFactor = 1.0
        self.browser.voltageScalingFactorBox.delete(0, END)
        self.browser.voltageScalingFactorBox.insert(0, self.voltageScalingFactor)

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
            print(
                f'*** UPDATE ***: No temperature calibration specifically for serial number {self.sensor_serial_number}.')
            self.fit_exp = self.fits[-1]

        volts = y - exponential(self.temps, *self.fit_exp) + np.mean(self.volts_scaled)
        return volts

    def saveFile(self):
        fname = self.datafile.parent / f'{self.datafile.stem}_adjusted.csv'
        print('fname will be "{fname}"')

        # Get the number of lines in the file so we can do a perfect progress bar...
        print(f'----- Saving {fname} -----')

        if self.temperature_present:
            self.datathing = zip(self.times, self.volts_raw.transpose().tolist()[0], self.volts.transpose().tolist()[0],
                                 self.temps.transpose().tolist()[0])
            with open(fname, 'w', newline='') as f:
                print('Header lines, 4', file=f);
                print(f'Columns, datetime, original potential (V), potential after processing (V), temperature (C)',
                      file=f)
                print(f'Voltage scaling factor, {self.voltageScalingFactor}', file=f)
                print(f"Exponential temperature fit parameters (v' = v - a*exp(bT)+c + |v|), {str(self.fit_exp)[1:-1]}",
                      file=f)
                writer = csv.writer(f)
                writer.writerows(self.datathing)
        else:
            self.datathing = zip(self.times, self.volts_raw.transpose().tolist()[0], self.volts.transpose().tolist()[0])
            with open(fname, 'w', newline='') as f:
                print('Header lines, 3', file=f);
                print(f'Columns, datetime, original potential (V), potential after processing (V)', file=f)
                print(f'Voltage scaling factor, {self.voltageScalingFactor}', file=f)
                writer = csv.writer(f)
                writer.writerows(self.datathing)

        print('                  ...done.')

    def whoi_lidar_old_latlon_parse(self, latlon):
        ll = latlon.split(', ')
        lat = ll[0].split(':')[1]
        latd = lat[0:-1]
        latns = lat[-1]
        latitude = float(latd)
        if latns.lower() == 's':
            latitude = -latitude
        lon = ll[1].split(':')[1]
        lond = lon[0:-1]
        lonew = lon[-1]
        longitude = float(lond)
        if lonew.lower() == 'w':
            longitude = -longitude

        return latitude, longitude
