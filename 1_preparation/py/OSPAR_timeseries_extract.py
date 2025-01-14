# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 11:03:14 2025

@author: lorinc
"""

import os

import numpy as np
import xarray as xr
import pandas as pd
import cartopy as cp

import datetime as dt
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.lines as mlines

import skill_metrics as sm

# dfm_tools 
import dfm_tools as dfmt
from dfm_tools.xarray_helpers import preprocess_hisnc


# In[2]:


# Funtions:

def find_nearest_location_from_2D_output(ds, lat, lon):
    '''
    ds needs to be a datarray: 1 variable, 1 depth and 1 timestamp.
    Also, 2D lat/lon needs to be named: latitude and longitude.
    '''
    # Find the index of the grid point nearest a specific lat/lon
    abslat = np.abs(ds.latitude-lat)
    abslon = np.abs(ds.longitude-lon)
    c = np.maximum(abslon, abslat)
    ([xloc], [yloc]) = np.where(c == np.min(c))
    # Then use that index location to get the values at the x/y diminsion
    point_ds = ds.sel(x=yloc, y=xloc)     ### Note: x and y inverted as in netCDF listed as y,x (in that order!)
    return point_ds

def years_in_order(start_year, end_year):
    return list(range(start_year, end_year + 1))

def var_mapping(var):
    if var == 'CHL':
        var_sat = 'CHL'
        var_IBI = 'chl'
        var_NWS = 'chl'
        var_DFM = 'Chlfa'
        var_obs = 'Chlfa'
    elif var == 'OXY':
        var_sat = ''
        var_IBI = 'o2'
        var_NWS = 'o2'
        var_DFM = 'OXY'
        var_obs = 'OXY'
    elif var == 'NO3':
        var_sat = ''
        var_IBI = 'no3'  
        var_NWS = 'no3'  
        var_DFM = 'NO3'
        var_obs = 'NO3'
    elif var == 'PH':
        var_sat = ''
        var_IBI = 'ph'  
        var_NWS = 'ph'  
        var_DFM = 'pH'
        var_obs = 'pH'
    elif var == 'PO4':
        var_sat = ''
        var_IBI = 'po4'  
        var_NWS = 'po4'  
        var_DFM = 'PO4'
        var_obs = 'PO4'
    elif var == 'PCO2':
        var_sat = ''
        var_IBI = 'spco2'  
        var_NWS = 'spco2'  
        var_DFM = 'pCO2water'
        var_obs = 'pCO2'
              
    return var_sat, var_IBI, var_NWS, var_DFM, var_obs

# Function to extract lat and lon from the 'geom' column
def extract_lat_lon(geom):
    # Split the string, assuming it's formatted as 'POINT (lon lat)'
    lon, lat = geom.replace('POINT ', '').strip('()').split()
    return float(lat), float(lon)

#%%
## Choose the year, model and output directory
start_year = 2009
end_year = 2014
model = 'rea'
selected_years = years_in_order(start_year, end_year)
slice_2d = 'surface' 

rootdir = r'P:\11209810-cmems-nws\model_output\combined_yearly' if os.name == 'nt' else r'/p/11209810-cmems-nws/model_output/combined_yearly'

outdir = fr'P:\11209810-cmems-nws\figures\timeseries\{start_year}_{end_year}'

#OSPAR Taylor stations
plot_locs = ['M5514', 'NOORDWK70', 'TERSLG50', 'TERSLG235', 'M10539', 'M15898',
        'Aa13', 'W03', 'TH1', 'Stonehaven', '45CV',
        '74E9_0040 (Liverpool_Bay)', 'NOORDWK10', 'WCO_L4_PML',
        'channel_area_france', 'north_northsea', 'nw_shetland',
        'WES_Stn_104']

## Extracting 

variables = ['chl', 'no3', 'po4']
variables_DFM = ['Chlfa', 'NO3', 'PO4']
   
obs_path = r"p:\11209810-cmems-nws\Data\OSPAR_paper\OSPAR_station_data.csv"    
obs = pd.read_csv(obs_path)

# Filter the dataframe based on the stations in the list
filtered_df = obs[obs['Station'].isin(plot_locs)][['Station', 'Latitude', 'Longitude']]
# Drop duplicate rows for each station, keeping only the first occurrence
coordinates_df = filtered_df[['Station', 'Latitude', 'Longitude']].drop_duplicates(subset=['Station'])

latitudes = coordinates_df['Latitude'].tolist()
longitudes = coordinates_df['Longitude'].tolist()

IBI_df_list = []
NWS_df_list = []
for loc in plot_locs:
    print(' ')
    print(fr'Extracting time series at {loc}')
    latitude = coordinates_df['Latitude'][coordinates_df['Station'] == loc].values
    longitude = coordinates_df['Longitude'][coordinates_df['Station'] == loc].values
    ### Models
    #NWS
    office = 'NWS'  
    
    NWS_xr_ds = []
    print(f'Opening {slice_2d}_{office}_{model}_{start_year}_{end_year}')
    for year in selected_years: 
        basedir = os.path.join(rootdir,fr'{slice_2d}_{office}_{model}_{year}.nc')
        NWS_xr_year = xr.open_dataset(basedir)
        NWS_crop = NWS_xr_year[variables]#[:,0,:,:]                                             # extract surface
        NWS_crop = NWS_crop.sel(longitude=longitude, latitude=latitude, method='nearest')    # Select point
        #waq_xr = waq_xr.rename({'latitude':'y', 'longitude':'x'})                            # select variable and depth
        NWS_xr_ds.append(NWS_crop)
        
    # Merge:
    print(f'Merging {slice_2d}_{office}_{model}_{start_year}_{end_year}')    
    NWS_xr = xr.concat(NWS_xr_ds, dim='time') 
    NWS_df = NWS_xr.to_dataframe().reset_index()
    NWS_df['Station'] = loc
    NWS_df_list.append(NWS_df)
    
    #IBI
    office = 'IBI'
        
    IBI_xr_ds = []
    print(f'Opening {slice_2d}_{office}_{model}_{start_year}_{end_year}')
    for year in selected_years: 
        basedir = os.path.join(rootdir,fr'{slice_2d}_{office}_{model}_{year}.nc')
        IBI_xr_year = xr.open_dataset(basedir)
        IBI_crop = IBI_xr_year[variables]#[:,0,:,:]                                                    # extract surface
        IBI_crop = find_nearest_location_from_2D_output(ds=IBI_crop, lat=latitude, lon=longitude)   # Select point
        #waq_xr = waq_xr.rename({'latitude':'y', 'longitude':'x'})                            # select variable and depth
        IBI_xr_ds.append(IBI_crop)
        
    # Merge:
    print(f'Merging {slice_2d}_{office}_{model}_{start_year}_{end_year}')    
    IBI_xr = xr.concat(IBI_xr_ds, dim='time') 
    
    IBI_df = IBI_xr.to_dataframe().reset_index()
    IBI_df['Station'] = loc
    IBI_df_list.append(IBI_df)
    
#     #DFM
#     office = 'DFM'
    
#     print(f'Opening {office}_{start_year}_{end_year}')
    
#     DFM_xr_ds = []
#     for year in selected_years: 
#         base_path = r'p:\archivedprojects\11206044-002ospareutrophication\final_results_SurfSara'
#         # base_path = r'p:\archivedprojects\11208067-003-ospar-model-results\current'
#         for folder in os.listdir(base_path):
#             if folder.startswith(f'current_{year}'):  # Matches folders starting with current_{year}
#                 DFM_model = os.path.join(base_path, folder)
#                 f = os.path.join(DFM_model, r"DCSM-FM_0_5nm_waq_0000_his.nc")  
#         DFM_xr = xr.open_mfdataset(f, preprocess=preprocess_hisnc)
        
#         DFM_xr = dfmt.rename_waqvars(DFM_xr)
#         DFM_crop = DFM_xr[variables_DFM].sel(stations=loc, laydim=-1)                               # select point and surface
                    
#         DFM_crop['NO3'] = DFM_crop['NO3']*1000/14 
#         DFM_crop['PO4'] = DFM_crop['PO4']*1000/30.97 
   
#         DFM_xr_ds.append(DFM_crop)        
    
#     print(f'Merging {office}_{start_year}_{end_year}')
#     DFM_xr = xr.concat(DFM_xr_ds, dim='time')
    
#     DFM_df = DFM_xr.to_dataframe().reset_index()
#     DFM_df['Station'] = loc
#     DFM_df_list.append(DFM_df)

#Concatenate, process, save tables - NWS
NWS_df_all = pd.concat(NWS_df_list, ignore_index=True) 
NWS_df_all = NWS_df_all.rename(columns={
    'no3': 'NO3',
    'po4': 'PO4',
    'chl': 'CHL'
})

# Melting the dataframe to collapse the three columns
NWS_df_all = NWS_df_all.melt(id_vars=[col for col in NWS_df_all.columns if col not in ['NO3', 'PO4', 'CHL']],
                                   value_vars=['NO3', 'PO4', 'CHL'],
                                   var_name='Variable',
                                   value_name='Value')

NWS_df_all = NWS_df_all.dropna(subset=['Value'])

# Save the combined DataFrame to a CSV file if needed
NWS_df_all.to_csv(fr"p:\11209810-cmems-nws\Data\OSPAR_paper\NWS_{start_year}_{end_year}.csv", index=False)

#Concatenate, process, save tables - IBI
IBI_df_all = pd.concat(IBI_df_list, ignore_index=True) 
IBI_df_all = IBI_df_all.rename(columns={
    'no3': 'NO3',
    'po4': 'PO4',
    'chl': 'CHL'
})

# Melting the dataframe to collapse the three columns
IBI_df_all = IBI_df_all.melt(id_vars=[col for col in IBI_df_all.columns if col not in ['NO3', 'PO4', 'CHL']],
                                   value_vars=['NO3', 'PO4', 'CHL'],
                                   var_name='Variable',
                                   value_name='Value')

IBI_df_all = IBI_df_all.dropna(subset=['Value'])

# Save the combined DataFrame to a CSV file if needed
IBI_df_all.to_csv(fr"p:\11209810-cmems-nws\Data\OSPAR_paper\IBI_{start_year}_{end_year}.csv", index=False)

# #Concatenate, process, save tables - DFM
# DFM_df_all = pd.concat(DFM_df_list, ignore_index=True) 
# DFM_df_all = DFM_df_all.rename(columns={
#     'no3': 'NO3',
#     'po4': 'PO4',
#     'chl': 'CHL'
# })

# # Melting the dataframe to collapse the three columns
# DFM_df_all = DFM_df_all.melt(id_vars=[col for col in DFM_df_all.columns if col not in ['NO3', 'PO4', 'CHL']],
#                                    value_vars=['NO3', 'PO4', 'CHL'],
#                                    var_name='Variable',
#                                    value_name='Value')

# DFM_df_all = DFM_df_all.dropna(subset=['Value'])

# # Save the combined DataFrame to a CSV file if needed
# DFM_df_all.to_csv(fr"p:\11209810-cmems-nws\Data\OSPAR_paper\DFM_{start_year}_{end_year}.csv", index=False)