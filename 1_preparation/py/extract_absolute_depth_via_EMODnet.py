#!/usr/bin/env python
# coding: utf-8

# Packages: 

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from shapely.wkt import loads

# Read downloaded EMODnet file:
bathy = xr.open_dataset(r'p:\11209810-cmems-nws\Data\EMODnet_bathymetry\bathymetry_2022_97da_6336_ec54.nc')
# plot the file to check elevation:
#bathy.elevation.plot(robust=True)  # extent looks correct.

start_year = 2021
end_year = 2022
gridded = False #True - yes, False - no
variables = ['Chlfa', 'OXY', 'pH', 'PO4', 'NO3']#, 'pCO2']
#variables = ['OXY']

#%%
#Loop
for variable in variables:
    
    print(' ')
    print(f'Running {variable} for {start_year}_{end_year}')

    # Import a list of NWDM stations, that ref from the bottom
    if gridded:
        obs_path=fr'p:\11209810-cmems-nws\Data\NWDM_observations\combined_years\{start_year}_{end_year}_{variable}_obs_gridded_bottom.csv'
    else:
        obs_path=fr'p:\11209810-cmems-nws\Data\NWDM_observations\combined_years\{start_year}_{end_year}_{variable}_obs_bottom.csv'
    
    # obs = pd.read_csv(r'p:\11209810-cmems-nws\Data\NWDM_observations\2003_OXY_obs_bottom.csv')  # with pandas not get locs in correct format
    obs = gpd.read_file(obs_path)
    if len(obs) == 0:
        print('There is no bottom observation. Skipping this variable.')
        continue
    else:
        # Convert the 'geom' column to shapely geometry objects if it's in WKT format
        obs['geom'] = obs['geom'].apply(loads)
        
        # Set the 'geom' column as the geometry column
        obs = obs.set_geometry("geom")
        obs.set_crs("EPSG:4326", inplace=True)
        
        name = []
        depth = []
        # For every station, we want the depth from EMDOnet:
        for loc in np.unique(obs.station):
            # Get lat and lon
            point = obs[obs.station == loc].iloc[0].geom
            depth_val = bathy.elevation.sel(latitude=point.y, longitude=point.x, method='nearest').values
            name.append(loc)
            depth.append(depth_val)
        
        # combine in df
        depth_per_station = pd.DataFrame({'station': name, 'water_depth': np.array([item.item() for item in depth])})
        
        
        # Add depth as column to obs:
        obs_with_depth = pd.merge(obs, depth_per_station, on='station')
        
        # Convert columns to numeric values
        obs_with_depth['water_depth'] = pd.to_numeric(obs_with_depth['water_depth'], errors='coerce')
        obs_with_depth['depth'] = pd.to_numeric(obs_with_depth['depth'], errors='coerce')
        obs_with_depth['absolute_depth'] = obs_with_depth['water_depth'] - obs_with_depth['depth']
        
        obs_with_depth['depth'] = obs_with_depth['absolute_depth'].copy()
        obs_bottom = obs_with_depth[['station','geom','datetime','depth', 'value']]
        
        # Save to .csv
        #obs_bottom.to_csv(obs_path, index=False)
        
        # Merge with 'surface'-referenced obs
        obs_surface = pd.read_csv(obs_path[:-11] + '_surface' + '.csv')  # with pandas not get locs in correct format
        obs_depth = pd.concat([obs_surface, obs_bottom], ignore_index=True)
        obs_depth['depth'] = np.where(obs_depth['depth'] > 0, -obs_depth['depth'], obs_depth['depth']) #convert positive depth values to negative
        
        # Save to .csv
        obs_depth.to_csv(obs_path[:-11] + '_depth' + '.csv', index=False)