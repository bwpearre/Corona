import numpy as np
import csv
import pandas
from pathlib import Path
import pdb
import datetime as dt
import pytz
import time

def loadFile(filename):
    print(f'----- Loading {filename.name} -----')
    times = []
    volts = []
    temps = []

    # Define the weird date format, or perhaps the new date format that I'm hoping for...
    date_format = ['%m/%d/%y %I:%M:%S %p', '%m/%d/%y %H:%M']

    line_count = 0
    date_column = -1
    voltage_column = -1
    scaled_column = -1
    temperature_in_freedom_units = False
    still_in_header = True
    temperature_present = 0
    timezone_utc = pytz.timezone("UTC")
    
    with filename.open() as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            line_count += 1
            if still_in_header:
                foo = row[0].split('=')
                if 'location' in foo[0].lower():
                    location = foo[1].strip()
                elif 'latitude' in foo[0].lower():
                    latitude = float(foo[1])
                elif 'longitude' in foo[0].lower():
                    longitude = float(foo[1])

                elif foo[0][0]=='#':
                    # This is the line with all the header info. Figure out what we have...
                    print(f'   Location = {location} at {latitude}, {longitude}')
                    for column in range(1, len(row)):
                            serialnumberfound = row[column].find("SEN S/N:")
                            if serialnumberfound != -1:
                                    sn = int(row[column][serialnumberfound:].split(":")[1].split(',')[0])
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
                                    temperature_present = column
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
                        voltage_column = scaled_column
                    still_in_header = False


            # Here's the meat. Read each line, check for completeness, parse the dates, and add.
            else: # still_in_header = False
                if len(row) <= max([date_column, voltage_column, temperature_present]):
                        False
                elif row[date_column] and row[voltage_column] and ((not temperature_present) or row[temperature_present]):
                        try:
                                t = dt.datetime.strptime(row[date_column], date_format[0]) - timedelta_corona
                                times.append(timezone_utc.localize(t))
                        except ValueError:
                                t = dt.datetime.strptime(row[date_column], date_format[1]) - timedelta_corona
                                times.append(timezone_utc.localize(t))
                        except ValueError:
                                continue;
                        volts.append(float(row[voltage_column]))
                        if temperature_present:
                                temps.append(float(row[temperature_present]))



    return times, volts, temps, location


p = Path(r'data').glob('*.csv')
files = [x for x in p if x.is_file()]

date_format = ['%m/%d/%y %I:%M:%S %p', '%m/%d/%y %H:%M']
for f in files:
    try:
        times, volts, temps, location = loadFile(f)
        print(f'{f.name} ---> {location} {times[0].strftime("%Y-%m")} ({(times[-1]-times[0]).days}d).csv')
        f.rename(f'{location} {times[0].strftime("%Y-%m")} ({(times[-1]-times[0]).days}d).csv')
    except BaseException as e:
        print(f'Exception on "{f.name}": {e}')
