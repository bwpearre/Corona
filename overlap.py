import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
from pathlib import Path
import avm
import geopy.distance
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import pdb

def whoi_lidar_old_latlon_parse(latlon):
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


date_format_lidar = '%Y/%m/%d %H:%M'


# Convert new to old column names:
column_rename = {'Vertical Wind Speed (m/s) at 40m': '40m Z-wind (m/s)', 'Vertical Wind Speed (m/s) at 47m': '47m Z-wind (m/s)', 'Vertical Wind Speed (m/s) at 67m': '67m Z-wind (m/s)', 'Vertical Wind Speed (m/s) at 77m': '77m Z-wind (m/s)', 'Vertical Wind Speed (m/s) at 87m': '87m Z-wind (m/s)', 'Vertical Wind Speed (m/s) at 97m': '97m Z-wind (m/s)', 'Vertical Wind Speed (m/s) at 107m': '107m Z-wind (m/s)', 'Vertical Wind Speed (m/s) at 127m': '127m Z-wind (m/s)', 'Vertical Wind Speed (m/s) at 147m': '147m Z-wind (m/s)', 'Vertical Wind Speed (m/s) at 167m': '167m Z-wind (m/s)', 'Vertical Wind Speed (m/s) at 187m': '187m Z-wind (m/s)', 'Vertical Wind Speed (m/s) at 38m': '40m Z-wind (m/s)'}

filesets = { 'new lidar': [ 'asit.ZXlidar.', '.csv', 2],
             'old lidar': [ 'asit.lidar.', '.sta', 1 ]}

dataframes = []

firstfnum = 270
lastfnum = 278

parent = Path('.')
parent = parent / 'data' / 'whoi' / 'lidar'

for round in {0, 1}:

    day_i = -1
    fnum = firstfnum - 1
    errors = 0
    locs = np.empty((0,2), float)

    while fnum < lastfnum:
        fnum = fnum + 1

        print(f' Loading WHOI files {fnum} (round {round})...')

        daily_data = []

        for fileset, prepost in filesets.items():

            fname = parent / f'{prepost[0]}2021_{fnum}{prepost[1]}'


            if fname.is_file():

                print(f'      Loading {fname}')

                column_rename_more = {}
                
                with open(fname, errors='replace') as f:
                    row = f.readline()
                    if row[0:9] == 'HeaderSize'[0:9]:
                        
                        # Old LIDAR
                        headersize = int(row.split('=')[1])
                        mdf = pd.read_csv(fname, sep='=', nrows=headersize-1, index_col = 0, header=None,
                                          encoding='cp1252')
                        timezone = mdf.loc['timezone', 1]
                        location = mdf.loc['Location', 1].strip()
                        latitude, longitude = whoi_lidar_old_latlon_parse(mdf.loc['GPS Location', 1])
                        locs = np.append(locs, np.array([[latitude, longitude]]), axis=0)

                        if timezone != "UTC+0":
                                raise Exception('LIDAR time offset changed.')

                        df = pd.read_csv(fname, sep='\t', header=headersize, parse_dates = [0], index_col=0,
                                         encoding='cp1252')
                        print(f'          Date range {df.index[0]}   --   {df.index[-1]}')
                        df = df.filter(like='Z-wind (m/s)')
                        dfh = df.columns.tolist()
                        heights = [int(i.split('m')[0]) for i in dfh]

                        for cn in df.columns.tolist():
                            column_rename_more[cn] = f'{cn} old'
                        df.rename(inplace=True, columns=column_rename_more)
                        



                    elif row[0:9] == 'CSV Converter'[0:9]:
                        # New LIDAR
                        headersize = 1
                        df = pd.read_csv(fname, parse_dates = [1], dayfirst = True, index_col = 1, header = 1, encoding='cp1252')
                        print(f'          Date range {df.index[0]}  --   {df.index[-1]}')
                        #df = df[df.columns.drop(list(df.filter(like='Checksum')))] # Checksum column is annoying, and useless for now
                        # Actually, let's just get rid of everything but what we currently care about:
                        latlon = df['GPS'].iat[0].split()
                        latitude = float(latlon[0])
                        longitude = float(latlon[1])
                        locs = np.append(locs, np.array([[latitude, longitude]]), axis=0)

                        df = df.filter(like='Vertical Wind Speed')

                        dfh = df.columns.tolist()
                        h = [i.split(' ')[5] for i in dfh]
                        heights = [int(i.split('m')[0]) for i in h]

                        df.rename(inplace=True, columns=column_rename)

                        for cn in df.columns.tolist():
                            column_rename_more[cn] = f'{cn} new'
                        df.rename(inplace=True, columns=column_rename_more)
                            
                        # FUCK THOSE FUCKING RATFUCKERS
                        df.replace(inplace=True, to_replace=9998, value=np.NaN)
                        df.replace(inplace=True, to_replace=9999, value=np.NaN)


                    else:
                        print('Could not get header size. Assuming 0.')
                        headersize = 0

                # DIFFERENT between round 1 and round 2: don't add (possibly redundant) indices; just fill in missing values
                if round == 0:
                        if len(daily_data): # Join lidars
                                daily_data = daily_data.join(df, how='outer')
                        else:
                                daily_data = df
                else:
                        whoi.update(df)


            else:
                print(f'  File "{fname}" does not exist.')
                errors += 1
                if errors >= 30:
                    print('  [[ Not finding WHOI data files. Giving up. ]]')

        if round == 0:
            if len(daily_data):
                    dataframes.append(daily_data)

    if round == 0:
            whoi = pd.concat(dataframes, copy=False)
            print(f'Before dropping duplicates: dataset is {whoi.shape}')
            whoi.sort_index(inplace=True, kind='mergesort')
            whoi = whoi[~whoi.index.duplicated(keep='first')] # FIXME: This will toss some rows that contain data. Fixed in ROUND 1.
            whoi.dropna(inplace=True, axis='columns', how='all')
    print(f'After ROUND {round}: dataset is {whoi.shape}')

whoi = whoi.tz_localize('UTC') # see "just check that it hasn't changed" above

fig = plt.figure(num='map')
fig.clf()

terrain = cimgt.GoogleTiles(style='satellite')
map = fig.add_subplot(1, 1, 1, projection=terrain.crs)
map.set_extent([locs.min(axis=0)[1]-0.03, locs.max(axis=0)[1]+0.03, locs.min(axis=0)[0]-0.03, locs.max(axis=0)[0]+0.04], crs=ccrs.Geodetic())
map.add_image(terrain, 13)
gl = map.gridlines(draw_labels=True)
gl.xlines = False
gl.ylines = False

for i in range(locs.shape[0]):
        map.plot(locs[i,1], locs[i,0], marker='o', color='red', markersize=4,
                      alpha=1, transform=ccrs.Geodetic())

# Get the final list of heights:
dfh = whoi.columns.tolist()
print(dfh)
heights = [int(i.split('m')[0]) for i in dfh]
print(f'Final LIDAR heights: {heights}')


ax = whoi.filter(like='old').plot(colormap='Blues')
whoi.filter(like='new').plot(ax=ax, colormap='Reds')

plt.show()
