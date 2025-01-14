# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 08:46:21 2025

@author: lorinc
"""

import pandas as pd
import re

# Path to your Excel file
excel_file = r"p:\11209810-cmems-nws\Data\OSPAR_paper\stations_for_validation.from_COMPEAT_20062014_Test_Data.upper10m.xlsx"

# Read all sheets from the Excel file
sheets = pd.read_excel(excel_file, sheet_name=None)

# Initialize an empty list to store data
dataframes = []

# Loop through each sheet
for sheet_name, data in sheets.items():
    # Remove variable suffix (_NITR, _PHOS, _CHL, etc.) using regex
    station_name = re.sub(r'_(NITR|PHOS|CHL)$', '', sheet_name)  # Remove only the suffix
    
    # Add a new column for the station name
    data['Station'] = station_name
    
    try:
        data = data.drop(columns=['Hour', 'Minute', 'Second'])
    except:
        try:
            data = data.drop(columns=['Hour', 'Minute'])
        except:
            print("")
    
    data = data.dropna(axis=1, how='all')
            
    # Append the sheet's data to the list
    dataframes.append(data)

# Concatenate all data into a single DataFrame
final_dataframe = pd.concat(dataframes[4:], ignore_index=True)

final_dataframe = final_dataframe.drop(columns=['Assessment Unit'])

# Combine Year, Month, Day, Hour, and Minute into a single Time column
final_dataframe['Time'] = pd.to_datetime(final_dataframe[['Year', 'Month', 'Day']])

# Drop the original Year, Month, Day, Hour, and Minute columns if no longer needed
final_dataframe = final_dataframe.drop(columns=['Year', 'Month', 'Day'])

# Remove units from the column names
final_dataframe.columns = final_dataframe.columns.str.replace(r'\s*\[.*?\]', '', regex=True)

#Rename the stations
final_dataframe['Station'] = final_dataframe['Station'].replace({
    'add1': 'channel_area_france',
    'add2': 'north_northsea',
    'add3': 'nw_shetland',
    'add4': 'WES_Stn_104',
    '74E9_0040': '74E9_0040 (Liverpool_Bay)',
    'M7456': 'TERSLG235',
    'M3615': 'NOORDWK70',    
    'M4908': 'TERSLG50',    
    'M3440': 'NOORDWK10',  
})

# Get the first latitude and longitude for each station
first_lat_lon = final_dataframe.groupby('Station')[['Latitude', 'Longitude']].first()

# Merge the first latitude and longitude back into the original dataframe
final_dataframe = final_dataframe.merge(first_lat_lon, on='Station', suffixes=('', '_first'))

# Replace the Latitude and Longitude with the first ones for each station
final_dataframe['Latitude'] = final_dataframe['Latitude_first']
final_dataframe['Longitude'] = final_dataframe['Longitude_first']

# Drop the temporary columns
final_dataframe = final_dataframe.drop(columns=['Latitude_first', 'Longitude_first'])

final_dataframe = final_dataframe.rename(columns={
    'Nitrate': 'NO3',
    'Phosphate': 'PO4',
    'Chlorophyll a': 'CHL'
})

# Melting the dataframe to collapse the three columns
collapsed_df = final_dataframe.melt(id_vars=[col for col in final_dataframe.columns if col not in ['NO3', 'PO4', 'CHL']],
                                   value_vars=['NO3', 'PO4', 'CHL'],
                                   var_name='Variable',
                                   value_name='Value')

collapsed_df = collapsed_df.dropna(subset=['Value'])

# Save the combined DataFrame to a CSV file if needed
collapsed_df.to_csv(r"p:\11209810-cmems-nws\Data\OSPAR_paper\OSPAR_station_data.csv", index=False)
