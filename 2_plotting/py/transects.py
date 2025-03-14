#!/usr/bin/env python
# coding: utf-8


## Packages

import os
import pyproj

import numpy as np
import xarray as xr
import xugrid as xu
import pandas as pd
import matplotlib as mpl

import dfm_tools as dfmt
import matplotlib.pyplot as plt
import cmocean

import geopandas as gpd
from shapely.geometry import LineString, Point
from geopy.distance import geodesic
import warnings
import math


# Functions

def years_in_order(start_year, end_year):
    return list(range(start_year, end_year + 1))

def var_mapping(var):
    if var == 'CHL':
        var_sat = 'CHL'
        var_IBI = 'chl'
        var_NWS = 'chl'
        var_DFM = 'mesh2d_Chlfa'
        var_obs = 'Chlfa'
    elif var == 'OXY':
        var_sat = ''
        var_IBI = 'o2'
        var_NWS = 'o2'
        var_DFM = 'mesh2d_OXY'
        var_obs = 'OXY'
    elif var == 'NO3':
        var_sat = ''
        var_IBI = 'no3'  
        var_NWS = 'no3'  
        var_DFM = 'mesh2d_NO3'
        var_obs = 'NO3'
    elif var == 'PH':
        var_sat = ''
        var_IBI = 'ph'  
        var_NWS = 'ph'  
        var_DFM = 'mesh2d_pH'
        var_obs = 'pH'
    elif var == 'PO4':
        var_sat = ''
        var_IBI = 'po4'  
        var_NWS = 'po4'  
        var_DFM = 'mesh2d_PO4'
        var_obs = 'PO4'
    elif var == 'PCO2':
        var_sat = ''
        var_IBI = 'spco2'  
        var_NWS = 'spco2'  
        var_DFM = 'mesh2d_pCO2water'
        var_obs = 'pCO2'
        
    return var_sat, var_IBI, var_NWS, var_DFM, var_obs

# Function to calculate geodesic distance
def calculate_distance_to_start(station_geom):
    station_coords = (station_geom.y, station_geom.x)  # (lat, lon)
    start_coords_tuple = (start_point.y, start_point.x)  # (lat, lon)
    return geodesic(station_coords, start_coords_tuple).meters  # Distance in meters

trans_dict = {
    'NORWAY1': 
        {'x1': -0.64, 
         'y1': 60.8,
         'x2': 4.64, 
         'y2': 60.8},
    'NORWAY2': 
        {'x1': -2.24, 
         'y1': 59.36,
         'x2': 4.96, 
         'y2': 59.36},
    'DENMARK': 
        {'x1': -2.08, 
         'y1': 56.96,
         'x2': 8.32, 
         'y2': 56.96},
    'NOORDWK': 
        {'x1': 4.404777986040303, 
         'y1': 52.26059775635567,
         'x2': 3.5300248567287253, 
         'y2': 52.58532019508742},        
    'TERSLG': 
        {'x1': 5.1492051575453575, 
         'y1': 53.41453186743249,
         'x2': 3.156039626051446, 
         'y2': 55.17151139049696},
    'ROTTMPT': 
        {'x1': 6.562847418252058, 
         'y1': 53.56538591610043,
         'x2': 6.212820857916571, 
         'y2': 54.117344728657535},
    'WALCRN': 
        {'x1': 3.409497727805905, 
         'y1': 51.54806447247548,
         'x2': 2.677801866004242, 
         'y2': 51.95612410943133},
}
    
# Suppress specific warning types (e.g., DeprecationWarning, UserWarning)
warnings.filterwarnings("ignore", category=UserWarning)


#Directories
rootdir = r'P:\11209810-cmems-nws\model_output\combined_yearly' if os.name == 'nt' else r'/p/11209810-cmems-nws/model_output/combined_yearly'

## Choose the year, model, variables
start_year = 2003
end_year = 2017
model = 'rea'
slice_2d = 'transect'
NWDM_depth = 'depth' #'surface' 'depth' 
NWDM_gridded = False # True or False
selected_years = years_in_order(start_year, end_year)
# Create vertical bins of size X meters for the profile observations
bin_size = 5 #in meters
# Define a distance threshold (e.g., 0.05 degrees ~ approx 5-6 km) for the transect line to snap observations
distance_threshold = 0.08
print(f'distance threshold: {distance_threshold}')
unit = 'CMEMS' # CMEMS / DFM

outdir = fr'P:\11209810-cmems-nws\figures\transects\{start_year}_{end_year}' if os.name == 'nt' else fr'/p/11209810-cmems-nws/figures/transects/{start_year}_{end_year}'
if not os.path.exists(outdir):
    os.makedirs(outdir)

offices = ['IBI']#, 'NWS']
variables = ['PH', 'PO4', 'NO3', 'CHL', 'OXY']

# Choose transects
# transects = ['NORWAY1','NORWAY2','DENMARK','NOORDWK', 'TERSLG', 'ROTTMPT', 'WALCRN']
transects = ['NORWAY2']

for office in offices:
    for var in variables:
        print(' ')
        print(f'Running {var} for {office}')
        var_sat, var_IBI, var_NWS, var_DFM, var_obs = var_mapping(var) #get variable names for each model
        
        #Colorbar settings
        if var == 'CHL':
            vmin, vmax, step = 0,10, 20
            cbar_title = 'Chlorophyll [mg m-3]'
            cmap = cmocean.cm.algae
        elif var == 'NO3':
            if unit == 'CMEMS':
                vmin, vmax, step = 0,50, 20
            elif unit == 'DFM':
                vmin, vmax, step = 0,1.2, 20
            cbar_title = 'Nitrate [mmol m-3]'
            cmap = cmocean.cm.matter
        elif var == 'PO4':
            if unit == 'CMEMS':
                vmin, vmax, step = 0.0 ,1.0, 20
            elif unit == 'DFM':
                vmin, vmax, step = 0.0,0.08, 20
            cbar_title = 'Phosphate [mmol m-3]'
            cmap = cmocean.cm.matter
        elif var == 'OXY':
            if unit == 'CMEMS':
                vmin, vmax, step = 120, 330, 20
            elif unit == 'DFM':
                vmin, vmax, step = 0,12, 20
            cbar_title = 'Dissolved Oxygen [mmol m-3]'
            cmap = cmocean.cm.oxy
        elif var == 'PH':
            vmin, vmax, step = 7.0, 8.5, 20
            cbar_title = 'pH'
            cmap = cmocean.cm.thermal
              
        # read a database file
        if NWDM_gridded:
            try:
                obs_dir = fr'P:\11209810-cmems-nws\Data\NWDM_observations\combined_years\{start_year}_{end_year}_{var_obs}_obs_gridded_{NWDM_depth}.csv' if os.name == 'nt' else fr'/p/11209810-cmems-nws/Data/NWDM_observations/combined_years/{start_year}_{end_year}_{var_obs}_obs_gridded_{NWDM_depth}.csv'
                obs = pd.read_csv(obs_dir)
            except:
                try:
                    obs_dir = fr'P:\11209810-cmems-nws\Data\NWDM_observations\combined_years\2003_2017_{var_obs}_obs_gridded_{NWDM_depth}.csv' if os.name == 'nt' else fr'/p/11209810-cmems-nws/Data/NWDM_observations/combined_years/2003_2017_{var_obs}_obs_gridded_{NWDM_depth}.csv'
                    obs = pd.read_csv(obs_dir)
                except:
                    obs_dir = fr'P:\11209810-cmems-nws\Data\NWDM_observations\combined_years\2003_2017_{var_obs}_obs_gridded_surface.csv' if os.name == 'nt' else fr'/p/11209810-cmems-nws/Data/NWDM_observations/combined_years/2003_2017_{var_obs}_obs_gridded_surface.csv' 
                    obs = pd.read_csv(obs_dir)
                    
        else:
            try:
                obs_dir = fr'P:\11209810-cmems-nws\Data\NWDM_observations\combined_years\{start_year}_{end_year}_{var_obs}_obs_{NWDM_depth}.csv' if os.name == 'nt' else fr'/p/11209810-cmems-nws/Data/NWDM_observations/combined_years/{start_year}_{end_year}_{var_obs}_obs_{NWDM_depth}.csv'
                obs = pd.read_csv(obs_dir)
            except:
                try:
                    obs_dir = fr'P:\11209810-cmems-nws\Data\NWDM_observations\combined_years\2003_2017_{var_obs}_obs_{NWDM_depth}.csv' if os.name == 'nt' else fr'/p/11209810-cmems-nws/Data/NWDM_observations/combined_years/2003_2017_{var_obs}_obs_{NWDM_depth}.csv'
                    obs = pd.read_csv(obs_dir)
                except:
                    obs_dir = fr'P:\11209810-cmems-nws\Data\NWDM_observations\combined_years\2003_2017_{var_obs}_obs_surface.csv' if os.name == 'nt' else fr'/p/11209810-cmems-nws/Data/NWDM_observations/combined_years/2003_2017_{var_obs}_obs_surface.csv' 
                    obs = pd.read_csv(obs_dir)
                    obs['depth'] = np.where(obs['depth'] > 0, -obs['depth'], obs['depth']) #convert positive depth values to negative
          
        obs['datetime'] = pd.to_datetime(obs['datetime'], format='mixed', dayfirst = False)
        obs['depth'] = obs['depth'].fillna(-1)
        obs['station'] = obs['station'].fillna(obs['geom'])
        obs['station'] = obs['station'].astype('string')

        if NWDM_gridded:
            obs['depth'] = -1 #for gridded, assume a surface depth
        
        for trans in transects:
            print(' ')
            print(f'Runnning {trans}')

            obs_trans = obs
            # Convert DataFrame to GeoDataFrame
            df = gpd.GeoDataFrame(
                obs_trans,
                geometry=gpd.GeoSeries.from_wkt(obs_trans["geom"]),  # Convert WKT to geometry
                crs="EPSG:4326",  # WGS 84 Coordinate Reference System
            )
                           
            # Define the starting and ending coordinates of the transect line
            x1, y1 = trans_dict[trans]['x1'], trans_dict[trans]['y1']
            x2, y2 = trans_dict[trans]['x2'], trans_dict[trans]['y2']
            start_coords = (x1, y1)  # Example: (longitude, latitude)
            end_coords = (x2, y2)
            
            # Create a LineString geometry
            line = LineString([start_coords, end_coords])
            
            # Calculate distance of each station from the line
            df["distance_to_line"] = df.geometry.distance(line)
                       
            # Filter stations within the threshold distance
            close_stations = df[df["distance_to_line"] <= distance_threshold]
            obs_trans = close_stations.copy()
            
            # Define the starting point of the transect
            start_point = Point(start_coords)  # e.g., (-4.5, 48.2)
            
            # Calculate distance from the starting point for close_stations
            obs_trans["distance_from_start"] = obs_trans.geometry.apply(calculate_distance_to_start)
            try:
                obs_trans["distance_from_start"] = obs_trans["distance_from_start"].round(0).astype(int)
            except:
                print("No observations")
            #Station names  and distances to plot
            plot_locs = list(obs_trans['station'].unique()) #station names along transect
            
            # Group by 'station' and extract the first (or unique) 'distance_from_start'
            plot_dist = obs_trans.groupby("station")["distance_from_start"].first().reset_index()
                                                      
            #Filter obs to years
            obs = obs.loc[(obs.datetime>=f'{start_year}-01-01') & (obs.datetime<=f'{str(int(end_year)+1)}-01-01')]
            
            positions = []
            mean_obs = []
            
            for i in range(0,len(plot_locs)):
                obs_point = obs.loc[obs.station==plot_locs[i]]
                obs_point = obs_point.set_index('datetime')
                obs_point = obs_point[['value','depth']]
                obs_point.index = pd.to_datetime(obs_point.index)

                if var == 'CHL':
                    obs_season_list = []
                    obs_season = obs_point.loc[obs_point.index.month.isin([3,4,5,6,7,8,9]), ['value','depth']]
                    if len(obs_season) == 0:
                        obs_season_list.append([])
                    else:
                        # Handle the range of depths (negative and zero)
                        min_depth = math.floor(obs_season['depth'].min())  # Round down for most negative depth
                        max_depth = 0  # Since sea depths are zero or negative
                        
                        # Create bins starting from the minimum depth up to 0
                        bins = range(min_depth, max_depth + bin_size, bin_size)
                        obs_season['depth_bin'] = pd.cut(obs_season['depth'], bins=bins, right=False)
                        
                        # Calculate bin midpoints (average of bin edges)
                        obs_season['depth'] = obs_season['depth_bin'].apply(
                            lambda b: (b.left + b.right) / 2 if pd.notnull(b) else None  # Handle NaN bins safely
                        )
                        obs_season['depth'] =  obs_season['depth'].astype(float)                        
                        # Drop the depth_bin column (optional, for cleaner output)
                        obs_season.drop(columns=['depth_bin'], inplace=True)
                        
                        obs_season = obs_season.groupby('depth')['value'].mean().reset_index()
                        obs_season['depth'] = np.where(obs_season['depth'] > 0.0, -1.0, obs_season['depth']) #convert positive depth values to -1
                        obs_season_list.append(obs_season)
                elif var == 'NO3':
                    obs_season_list = []
                    obs_season = obs_point.loc[obs_point.index.month.isin([1,2,12]), ['value','depth']]
                    if len(obs_season) == 0:
                        obs_season_list.append([])
                    elif unit == 'CMEMS':
                        obs_season['value'] = obs_season['value']*1000/14.006720 
                    else:
                        obs_season['value'] = obs_season['value']
                        
                        # Handle the range of depths (negative and zero)
                        min_depth = math.floor(obs_season['depth'].min())  # Round down for most negative depth
                        max_depth = 0  # Since sea depths are zero or negative
                        
                        # Create bins starting from the minimum depth up to 0
                        bins = range(min_depth, max_depth + bin_size, bin_size)
                        obs_season['depth_bin'] = pd.cut(obs_season['depth'], bins=bins, right=False)
                        
                        # Calculate bin midpoints (average of bin edges)
                        obs_season['depth'] = obs_season['depth_bin'].apply(
                            lambda b: (b.left + b.right) / 2 if pd.notnull(b) else None  # Handle NaN bins safely
                        )
                        obs_season['depth'] =  obs_season['depth'].astype(float)
                        # Drop the depth_bin column (optional, for cleaner output)
                        obs_season.drop(columns=['depth_bin'], inplace=True)
                        
                        obs_season = obs_season.groupby('depth')['value'].mean().reset_index()
                        obs_season['depth'] = np.where(obs_season['depth'] > 0.0, -1.0, obs_season['depth']) #convert positive depth values to -1
                        obs_season_list.append(obs_season)
                elif var == 'PO4':
                    obs_season_list = []
                    obs_season = obs_point.loc[obs_point.index.month.isin([1,2,12]), ['value','depth']]
                    if len(obs_season) == 0:
                        obs_season_list.append([])
                    elif unit == 'CMEMS':
                        obs_season['value'] = obs_season['value']*1000/30.973762 
                    else:
                        obs_season['value'] = obs_season['value']
                        
                        # Handle the range of depths (negative and zero)
                        min_depth = math.floor(obs_season['depth'].min())  # Round down for most negative depth
                        max_depth = 0  # Since sea depths are zero or negative
                        
                        # Create bins starting from the minimum depth up to 0
                        bins = range(min_depth, max_depth + bin_size, bin_size)
                        obs_season['depth_bin'] = pd.cut(obs_season['depth'], bins=bins, right=False)
                        
                        # Calculate bin midpoints (average of bin edges)
                        obs_season['depth'] = obs_season['depth_bin'].apply(
                            lambda b: (b.left + b.right) / 2 if pd.notnull(b) else None  # Handle NaN bins safely
                        )
                        obs_season['depth'] =  obs_season['depth'].astype(float)
                        # Drop the depth_bin column (optional, for cleaner output)
                        obs_season.drop(columns=['depth_bin'], inplace=True)
                        
                        obs_season = obs_season.groupby('depth')['value'].mean().reset_index()
                        obs_season['depth'] = np.where(obs_season['depth'] > 0.0, -1.0, obs_season['depth']) #convert positive depth values to -1
                        obs_season_list.append(obs_season)
                elif var == 'OXY':
                    obs_season_list = []              
                    obs_season = obs_point.copy()
                    obs_season = obs_point.loc[obs_point.index.month.isin([6,7,8]), ['value','depth']]
                    if len(obs_season) == 0:
                        obs_season_list.append([])
                    elif unit == 'CMEMS':
                        obs_season['value'] = obs_season['value']*1000/31.998 
                    else:
                        obs_season['value'] = obs_season['value']

                        # Handle the range of depths (negative and zero)
                        min_depth = math.floor(obs_season['depth'].min())  # Round down for most negative depth
                        max_depth = 0  # Since sea depths are zero or negative
                        
                        # Create bins starting from the minimum depth up to 0
                        bins = range(min_depth, max_depth + bin_size, bin_size)
                        obs_season['depth_bin'] = pd.cut(obs_season['depth'], bins=bins, right=False)
                        
                        # Calculate bin midpoints (average of bin edges)
                        obs_season['depth'] = obs_season['depth_bin'].apply(
                            lambda b: (b.left + b.right) / 2 if pd.notnull(b) else None  # Handle NaN bins safely
                        )
                        obs_season['depth'] =  obs_season['depth'].astype(float)
                        # Drop the depth_bin column (optional, for cleaner output)
                        obs_season.drop(columns=['depth_bin'], inplace=True)
                        
                        obs_season = obs_season.groupby('depth')['value'].min().reset_index()
                        obs_season['depth'] = np.where(obs_season['depth'] > 0.0, -1.0, obs_season['depth']) #convert positive depth values to -1
                        obs_season_list.append(obs_season)
                elif var == 'PH':
                    obs_season_list = []
                    for s in [[4,5], [8,9], [12,1]]: 
                        obs_season = obs_point.loc[obs_point.index.month.isin(s), ['value','depth']]
                        if len(obs_season) == 0:
                            obs_season_list.append([])
                        else:
                            # Handle the range of depths (negative and zero)
                            min_depth = math.floor(obs_season['depth'].min())  # Round down for most negative depth
                            max_depth = 0  # Since sea depths are zero or negative
                            
                            # Create bins starting from the minimum depth up to 0
                            bins = range(min_depth, max_depth + bin_size, bin_size)
                            obs_season['depth_bin'] = pd.cut(obs_season['depth'], bins=bins, right=False)
                            
                            # Calculate bin midpoints (average of bin edges)
                            obs_season['depth'] = obs_season['depth_bin'].apply(
                                lambda b: (b.left + b.right) / 2 if pd.notnull(b) else None  # Handle NaN bins safely
                            )
                            obs_season['depth'] =  obs_season['depth'].astype(float)
                            # Drop the depth_bin column (optional, for cleaner output)
                            obs_season.drop(columns=['depth_bin'], inplace=True)                            
                            obs_season = obs_season.groupby('depth')['value'].mean().reset_index()
                            obs_season['depth'] = np.where(obs_season['depth'] > 0.0, -1.0, obs_season['depth']) #convert positive depth values to -1
                            obs_season_list.append(obs_season)      
                
                mean_obs.append(obs_season_list)
                positions.append(plot_dist["distance_from_start"][i])           
            
            if office == 'DFM':
                ## Read DFM
                if 2014 <= start_year <= 2017 and 2014 <= end_year <= 2017:  # add DFM only for reanalysis!
                    print(fr'Opening {slice_2d}_{trans}_{office}_{model}_{start_year}_{end_year}')
                    DFM_xr_ds = []
                    for year in selected_years:
                        DFM_model = os.path.join(rootdir, fr'{slice_2d}_{trans}_{office}_{model}_{year}.nc')
                        DFM_xr_year = xu.open_dataset(DFM_model)
                        DFM_xr_ds.append(DFM_xr_year)
                #Merge files 
                print(f'Merging {office}_{start_year}_{end_year}')
                DFM_xr = xu.concat(DFM_xr_ds, dim='time')
                
                waq_xr_crop_list = []
                statistics_list = []
                waq_xr = DFM_xr
                if var_DFM == 'mesh2d_Chlfa':
                    waq_xr_season = waq_xr.where(waq_xr.time.dt.month.isin([3,4,5,6,7,8,9]), drop=True)  # Growing season
                    waq_xr_crop = waq_xr_season.mean(dim='time', keep_attrs=True)
                    waq_xr_crop = waq_xr_crop[var_DFM]
                    waq_xr_crop_list.append(waq_xr_crop)
                    statistics = 'growing_season_mean'
                    statistics_list.append(statistics)
                    print(f'{var}, statistics: {statistics}.')
                elif var_DFM == 'mesh2d_NO3':
                    waq_xr_season = waq_xr.where(waq_xr.time.dt.month.isin([12,1,2]), drop=True)  # Winter
                    if unit == 'CMEMS':
                        waq_xr_season[var_DFM] = waq_xr_season[var_DFM]*1000/14.006720 # convert units!
                    waq_xr_crop = waq_xr_season.mean(dim='time', keep_attrs=True)
                    waq_xr_crop = waq_xr_crop[var_DFM]
                    waq_xr_crop_list.append(waq_xr_crop)
                    statistics = 'winter_mean'
                    statistics_list.append(statistics)
                    print(f'{var}, statistics: {statistics}.')
                elif var_DFM == 'mesh2d_PO4':
                    waq_xr_season = waq_xr.where(waq_xr.time.dt.month.isin([12,1,2]), drop=True)  # Winter
                    if unit == 'CMEMS':
                        waq_xr_season[var_DFM] = waq_xr_season[var_DFM]*1000/30.973762 # convert units!
                    waq_xr_crop = waq_xr_season.mean(dim='time', keep_attrs=True)
                    waq_xr_crop = waq_xr_crop[var_DFM]
                    waq_xr_crop_list.append(waq_xr_crop)
                    statistics = 'winter_mean'
                    statistics_list.append(statistics)
                    print(f'{var}, statistics: {statistics}.')
                elif var_DFM == 'mesh2d_OXY':
                    waq_xr_season = waq_xr.where(waq_xr.time.dt.month.isin([6,7,8]), drop=True)  # Summer
                    if unit == 'CMEMS':
                        waq_xr_season[var_DFM] = waq_xr_season[var_DFM]*1000/31.998 # convert units!
                    waq_xr_crop = waq_xr_season.min(dim='time', keep_attrs=True)
                    waq_xr_crop = waq_xr_crop[var_DFM]
                    waq_xr_crop_list.append(waq_xr_crop)
                    statistics = 'summer_min'
                    statistics_list.append(statistics)
                    print(f'{var}, statistics: {statistics}.')                        
                elif var_DFM == 'mesh2d_pH':
                    for s in [[4,5], [8,9], [12,1]]: 
                        waq_xr_season = waq_xr.where(waq_xr.time.dt.month.isin(s), drop=True)
                        waq_xr_crop = waq_xr_season.mean(dim='time', keep_attrs=True)
                        waq_xr_crop = waq_xr_crop[var_DFM]
                        waq_xr_crop_list.append(waq_xr_crop)
                        if s==[4,5]:
                            statistics='spring_mean'
                        elif s==[8,9]:
                            statistics = 'summer_mean'
                        elif s==[12,1]:
                            statistics = 'winter_mean'
                        statistics_list.append(statistics)
                        print(f'{var}, statistics: {statistics}.')
                    
                   
                #Plotting
                print(f'Plotting {var} for {office}')
                
                ## Plotting with obs
                i = 0
                for waq_xr_crop in waq_xr_crop_list:            
                    fig, ax = plt.subplots()
                    
                    levels = np.linspace(vmin, vmax, step)
                    plot = waq_xr_crop.ugrid.plot(cmap=cmap, levels=levels, extend='both')
                    plot.colorbar.set_label(cbar_title)
                    
                    # obs
                    step_size = levels[1] - levels[0]
                    obs_levels = np.append(np.insert(levels,0,(levels[0]-step_size)), (levels[-1]+step_size))
                    norm = mpl.colors.BoundaryNorm(obs_levels, cmap.N) 
                    for pt in range(0, len(mean_obs)):
                        try:
                            plt.scatter(x=np.repeat(positions[pt], len(mean_obs[pt][i].depth)), y=mean_obs[pt][i].depth, c=mean_obs[pt][i].value, cmap=cmap, norm=norm, edgecolors='k', alpha=0.9)
                        except: Exception
                        
                    plt.xlim(left=-1000)
                    plt.ylim(top=2)
                    plt.title(fr"{office}")
                    plt.ylabel('Depth [m]')
                    plt.xlabel('Distance [m]')
                    plt.ticklabel_format(style="scientific", axis="x", scilimits=(0, 0))
                    
                    fig.tight_layout()
                    if NWDM_gridded:
                        plt.savefig(os.path.join(outdir, f'with_obs_{trans}_{slice_2d}_{office}_{start_year}_{end_year}_{var}_{statistics_list[i]}_{NWDM_depth}_gridded.png'), dpi=400, bbox_inches='tight')
                    else:
                        plt.savefig(os.path.join(outdir, f'with_obs_{trans}_{slice_2d}_{office}_{start_year}_{end_year}_{var}_{statistics_list[i]}_{NWDM_depth}.png'), dpi=400, bbox_inches='tight')
                    i=i+1
        
        # ### NWS
        
            elif office == 'NWS':  
    
                NWS_xr_ds = []
                print(f'Opening {slice_2d}_{office}_{model}_{start_year}_{end_year}')
                for year in selected_years: 
                    basedir = os.path.join(rootdir,fr'{slice_2d}_{trans}_{office}_{model}_{year}.nc')
                    NWS_xr_year = xr.open_dataset(basedir)
                    NWS_crop = NWS_xr_year[var_NWS]
                    #waq_xr = waq_xr.rename({'latitude':'y', 'longitude':'x'})                            
                    NWS_xr_ds.append(NWS_crop)
                    
                # Merge:
                print(f'Merging {slice_2d}_{office}_{model}_{start_year}_{end_year}')    
                NWS_xr = xr.concat(NWS_xr_ds, dim='time')    

                # # add crs coordinate and extract transect
                # with_crs = seasonal_mean.metpy.parse_cf()  # keeping all the vars
        
                        
                ## Distance into meters:
                ## https://stackoverflow.com/questions/72805622/converting-lat-lon-coordinates-into-meters-kilometers-for-use-in-a-metpy-hrrr-cr
                
                geod = pyproj.Geod(ellps='sphere')
                _, _, dist = geod.inv(np.repeat(NWS_xr['longitude'][0].values, len(NWS_xr.index)), 
                                      np.repeat(NWS_xr['latitude'][0].values, len(NWS_xr.index)),
                                      NWS_xr['longitude'], NWS_xr['latitude'])
                dist
                        
                # Temporal aggregation:

                waq_xr_crop_list = []
                statistics_list = []
                waq_xr = NWS_xr
                if var == 'CHL':
                    waq_xr_season = waq_xr.where(waq_xr.time.dt.month.isin([3,4,5,6,7,8,9]), drop=True)  # Growing season
                    waq_xr_crop = waq_xr_season.mean(dim='time', keep_attrs=True)
                    waq_xr_crop_list.append(waq_xr_crop)
                    statistics = 'growing_season_mean'
                    statistics_list.append(statistics)
                    print(f'{var}, statistics: {statistics}.')
                elif var == 'NO3':
                    waq_xr_season = waq_xr.where(waq_xr.time.dt.month.isin([12,1,2]), drop=True)  # Winter
                    waq_xr_crop = waq_xr_season.mean(dim='time', keep_attrs=True)
                    waq_xr_crop_list.append(waq_xr_crop)
                    statistics = 'winter_mean'
                    statistics_list.append(statistics)
                    print(f'{var}, statistics: {statistics}.')
                elif var == 'PO4':
                    waq_xr_season = waq_xr.where(waq_xr.time.dt.month.isin([12,1,2]), drop=True)  # Winter
                    waq_xr_crop = waq_xr_season.mean(dim='time', keep_attrs=True)
                    waq_xr_crop_list.append(waq_xr_crop)
                    statistics = 'winter_mean'
                    statistics_list.append(statistics)
                    print(f'{var}, statistics: {statistics}.')
                elif var == 'OXY':
                    waq_xr_season = waq_xr.where(waq_xr.time.dt.month.isin([6,7,8]), drop=True)  # Summer
                    waq_xr_crop = waq_xr_season.min(dim='time', keep_attrs=True)
                    waq_xr_crop_list.append(waq_xr_crop)
                    statistics = 'summer_min'
                    statistics_list.append(statistics)
                    print(f'{var}, statistics: {statistics}.')                        
                elif var == 'PH':
                    for s in [[4,5], [8,9], [12,1]]: 
                        waq_xr_season = waq_xr.where(waq_xr.time.dt.month.isin(s), drop=True)
                        waq_xr_crop = waq_xr_season.mean(dim='time', keep_attrs=True)
                        waq_xr_crop_list.append(waq_xr_crop)
                        if s==[4,5]:
                            statistics='spring_mean'
                        elif s==[8,9]:
                            statistics = 'summer_mean'
                        elif s==[12,1]:
                            statistics = 'winter_mean'
                        statistics_list.append(statistics)
                        print(f'{var}, statistics: {statistics}.')
                
                ## Plotting
                print(f'Plotting {var} for {office}')
                
                ## Plotting discrete - with obs
                i = 0
                for waq_xr_crop in waq_xr_crop_list: 
                    fig, ax = plt.subplots()
                    
                    # Identify the deepest depth where data exists for each face
                    # `depth` values where the data is not NaN
                    valid_depth = waq_xr_crop["depth"].where(waq_xr_crop.notnull()).min().values
                    depth_index = (waq_xr_crop["depth"] == valid_depth).argmax(dim="depth").values + 1
                    
                    levels = np.linspace(vmin, vmax, step)
                    plt.contourf(dist, waq_xr_crop['depth'][0:depth_index], waq_xr_crop[0:depth_index], cmap=cmap, levels=levels, extend='both')
                    plt.colorbar(label=cbar_title)
                    
                    # obs
                    step_size = levels[1] - levels[0]
                    obs_levels = np.append(np.insert(levels,0,(levels[0]-step_size)), (levels[-1]+step_size))
                    norm = mpl.colors.BoundaryNorm(obs_levels, cmap.N) 
                    for pt in range(0, len(mean_obs)):
                        try:
                            plt.scatter(x=np.repeat(positions[pt], len(mean_obs[pt][i].depth)), y=mean_obs[pt][i].depth, c=mean_obs[pt][i].value, cmap=cmap, norm=norm, edgecolors='k', alpha=0.9)
                        except: Exception
                    
                    plt.xlim(left=-1000)
                    plt.ylim(top=2)
                    plt.title(fr"{office}")
                    plt.ylabel('Depth [m]')
                    plt.xlabel('Distance [m]')
                    plt.ticklabel_format(style="scientific", axis="x", scilimits=(0, 0))
                    
                    fig.tight_layout()
                    if NWDM_gridded:
                        plt.savefig(os.path.join(outdir, f'with_obs_{trans}_{slice_2d}_{office}_{start_year}_{end_year}_{var}_{statistics_list[i]}_{NWDM_depth}_gridded.png'), dpi=400, bbox_inches='tight')
                    else:
                        plt.savefig(os.path.join(outdir, f'with_obs_{trans}_{slice_2d}_{office}_{start_year}_{end_year}_{var}_{statistics_list[i]}_{NWDM_depth}.png'), dpi=400, bbox_inches='tight')

                    i=i+1
                    
        # ### IBI    
        
            elif office == 'IBI':
    
                print(fr'Opening {slice_2d}_{trans}_{office}_{model}_{start_year}_{end_year}')
                IBI_xr_ds = []
                for year in selected_years:
                    IBI_model = os.path.join(rootdir, fr'{slice_2d}_{trans}_{office}_{model}_{year}.nc')
                    IBI_xr_year = xu.open_dataset(IBI_model)
                    IBI_xr_ds.append(IBI_xr_year)
                #Merge files 
                print(f'Merging {office}_{start_year}_{end_year}')
                IBI_xr = xu.concat(IBI_xr_ds, dim='time')

                ## Get distance in meters
                
                geod = pyproj.Geod(ellps='sphere')
                _, _, dist = geod.inv(np.repeat(IBI_xr['longitude'][0].values, len(IBI_xr.mesh2d_nFaces)), 
                                      np.repeat(IBI_xr['latitude'][0].values, len(IBI_xr.mesh2d_nFaces)),
                                      IBI_xr['longitude'], IBI_xr['latitude'])
                            
                dist  # Why is this distance longer than for NWS? Only about 2m, but just due to inaccuracies?
                #dist = IBI_xr.mesh2d_nFaces      
                
                waq_xr_crop_list = []
                statistics_list = []
                waq_xr = IBI_xr
                if var == 'CHL':
                    waq_xr_season = waq_xr.where(waq_xr.time.dt.month.isin([3,4,5,6,7,8,9]), drop=True)  # Growing season
                    waq_xr_crop = waq_xr_season.mean(dim='time', keep_attrs=True)
                    waq_xr_crop_list.append(waq_xr_crop)
                    statistics = 'growing_season_mean'
                    statistics_list.append(statistics)
                    print(f'{var}, statistics: {statistics}.')
                elif var == 'NO3':
                    waq_xr_season = waq_xr.where(waq_xr.time.dt.month.isin([12,1,2]), drop=True)  # Winter
                    waq_xr_crop = waq_xr_season.mean(dim='time', keep_attrs=True)
                    waq_xr_crop_list.append(waq_xr_crop)
                    statistics = 'winter_mean'
                    statistics_list.append(statistics)
                    print(f'{var}, statistics: {statistics}.')
                elif var == 'PO4':
                    waq_xr_season = waq_xr.where(waq_xr.time.dt.month.isin([12,1,2]), drop=True)  # Winter
                    waq_xr_crop = waq_xr_season.mean(dim='time', keep_attrs=True)
                    waq_xr_crop_list.append(waq_xr_crop)
                    statistics = 'winter_mean'
                    statistics_list.append(statistics)
                    print(f'{var}, statistics: {statistics}.')
                elif var == 'OXY':
                    waq_xr_season = waq_xr.where(waq_xr.time.dt.month.isin([6,7,8]), drop=True)  # Summer
                    waq_xr_crop = waq_xr_season.min(dim='time', keep_attrs=True)
                    waq_xr_crop_list.append(waq_xr_crop)
                    statistics = 'summer_min'
                    statistics_list.append(statistics)
                    print(f'{var}, statistics: {statistics}.')                        
                elif var == 'PH':
                    for s in [[4,5], [8,9], [12,1]]: 
                        waq_xr_season = waq_xr.where(waq_xr.time.dt.month.isin(s), drop=True)
                        waq_xr_crop = waq_xr_season.mean(dim='time', keep_attrs=True)
                        waq_xr_crop_list.append(waq_xr_crop)
                        if s==[4,5]:
                            statistics='spring_mean'
                        elif s==[8,9]:
                            statistics = 'summer_mean'
                        elif s==[12,1]:
                            statistics = 'winter_mean'
                        statistics_list.append(statistics)
                        print(f'{var}, statistics: {statistics}.')
                       
             
                ## Plotting
                print(f'Plotting {var} for {office}')
                i = 0
                for waq_xr_crop in waq_xr_crop_list:                                             
                    fig, ax = plt.subplots()
                    
                    levels = np.linspace(vmin, vmax, step)
                    
                    # Identify the deepest depth where data exists for each face
                    # `depth` values where the data is not NaN
                    valid_depth = waq_xr_crop["depth"].where(waq_xr_crop[var_IBI].notnull()).min().values
                    depth_index = (waq_xr_crop["depth"] == valid_depth).argmax(dim="depth").values + 1
                                        
                    plt.contourf(dist, waq_xr_crop['depth'][0:depth_index], waq_xr_crop[var_IBI][0:depth_index], cmap=cmap, levels=levels, extend='both')
                    plt.colorbar(label=cbar_title)
                    
                    # obs
                    # obs_levels = np.array([-1,0,1,2,3,4,5,6,7,8])  # need additional start and end to match above
                    step_size = levels[1] - levels[0]
                    obs_levels = np.append(np.insert(levels,0,(levels[0]-step_size)), (levels[-1]+step_size))
                    norm = mpl.colors.BoundaryNorm(obs_levels, cmap.N) 
                    for pt in range(0, len(mean_obs)):
                        try:
                            plt.scatter(x=np.repeat(positions[pt], len(mean_obs[pt][i].depth)), y=mean_obs[pt][i].depth, c=mean_obs[pt][i].value, cmap=cmap, norm=norm, edgecolors='k', alpha=0.9)
                        except: Exception
                    
                    plt.xlim(left=-1000)
                    plt.ylim(top=2)
                    plt.ylabel('Depth [m]')
                    plt.xlabel('Distance [m]')
                    plt.ticklabel_format(style="scientific", axis="x", scilimits=(0, 0))
                    
                    fig.tight_layout()
                    plt.title(fr"{office}")
                    if NWDM_gridded:
                        plt.savefig(os.path.join(outdir, f'with_obs_{trans}_{slice_2d}_{office}_{start_year}_{end_year}_{var}_{statistics_list[i]}_{NWDM_depth}_gridded.png'), dpi=400, bbox_inches='tight')
                    else:
                        plt.savefig(os.path.join(outdir, f'with_obs_{trans}_{slice_2d}_{office}_{start_year}_{end_year}_{var}_{statistics_list[i]}_{NWDM_depth}.png'), dpi=400, bbox_inches='tight')
                    i=i+1
