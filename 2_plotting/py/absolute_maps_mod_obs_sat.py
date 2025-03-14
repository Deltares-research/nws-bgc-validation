#!/usr/bin/env python
# coding: utf-8
 
## Imports

import os
import re
import numpy as np
import xarray as xr
import pandas as pd
import datetime as dt
# import dfm_tools as dfmt
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import cartopy.feature as cfeature

import cmocean
from matplotlib import colors


def years_in_order(start_year, end_year):
    return list(range(start_year, end_year + 1))

def var_mapping(var):
    if var == 'CHL':
        var_sat = 'CHL'
        var_IBI = 'chl'
        var_NWS = 'chl'
        var_DFM = 'mesh2d_Chlfa'  # old DFM
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

## Select domain

# LAT_MIN = None
# LAT_MAX = None
# LON_MIN = None
# LON_MAX = None
# zooming = 'full'

LAT_MIN = 48  # zoom onto the southern North Sea + NL
LAT_MAX = 62
LON_MIN = -10
LON_MAX = 13
zooming = 'zoom'

# Input directory
basedir = r'P:\11209810-cmems-nws\model_output' if os.name == 'nt' else r'/p/11209810-cmems-nws/model_output'
#rootdir = os.path.join(basedir, 'regridded_onto_NWS')
 
## Choose and read the model map files:
offices = ['DFM']  #
model = 'rea' #'rea' 'nrt'
start_year = 2015
end_year = 2017
slice_2d = 'stratification' #surface or bottom (bottom only works for DFM currently) 
NWDM_gridded = True
variables = ['temperature'] 
#variables = ['NO3','PO4'] 
#variables = ['PCO2'] #PCO2 only available in gridded format 


# Points to plot (only for non-gridded)
fixed_stations = False # True or False

#Buffer for surface observations
buffer = 5.5 # (meters) take all measuerements from the top x meters


# Minimum amount of observations per year
# min_obs_count = 1  # Can make it dependent on the variable and nr months in its aggregation.

# Path to output folder for plots
outdir = fr'P:\11209810-cmems-nws\figures\maps_model_obs\{start_year}_{end_year}' if os.name == 'nt' else fr'/p/11209810-cmems-nws/figures/maps_model_obs/{start_year}_{end_year}'

if not os.path.exists(outdir):
    os.makedirs(outdir)


selected_years = years_in_order(start_year, end_year)
for office in offices:
    if office == 'DFM' and not (2015 <= start_year <= 2017 and 2015 <= end_year <= 2017):
        print('DFM model not available.')
        continue
        
    for var in variables:
        print(f'Running {var} for {office}, gridded = {NWDM_gridded}, fixed_stations = {fixed_stations}')
        
        var_sat, var_IBI, var_NWS, var_DFM, var_obs = var_mapping(var) # get variable names for each model
        
        if office == 'satellite':
            vari = var_sat
            if var_sat == "":
                print('There is no satellite data for this variable.')
                continue  # Skip empty strings
        elif office == 'IBI':
            vari = var_IBI  
        elif office == 'NWS':
            vari = var_NWS  
        elif office == 'DFM':
            vari = var_DFM
        else:
            vari = var
        
        waq_xr_ds = []
        for year in selected_years:
            if office == 'satellite':
                waq_xr = xr.open_dataset(os.path.join(basedir, 'regridded_onto_NWS', fr'regridded_{office}_{year}.nc'))
            elif office == 'NWS':
                waq_xr = xr.open_dataset(os.path.join(basedir, 'combined_yearly', fr'{slice_2d}_{office}_{model}_{year}.nc'))
            elif var == 'temperature' and slice_2d == 'stratification':
                waq_xr = xr.open_dataset(os.path.join(basedir, fr'{year}_{slice_2d}_temperature.nc'))  # Need to have a stratification pre-processed .nc file
            else: #IBI and DFM
                waq_xr = xr.open_dataset(os.path.join(basedir, 'regridded_onto_NWS', fr'regridded_{slice_2d}_{office}_{year}.nc'))
            # waq_xr = dfmt.open_partitioned_dataset(os.path.join(rootdir,fr'{office}_{model}_{year}_ugrid.nc'))  
            
            waq_xr = waq_xr[vari]
            waq_xr_ds.append(waq_xr)
            
        # Merge:
        print(f'Merging regridded_{slice_2d}_{office}_{start_year}_{end_year}') 
        waq_xr_ds_concat = xr.concat(waq_xr_ds, dim='time')  
        
        if var == 'CHL':  # no unit conversion necessary
            waq_xr_ds_mean_list = []
            statistics_list = []
            waq_xr_ds_merge = waq_xr_ds_concat.where(waq_xr_ds_concat.time.dt.month.isin([3,4,5,6,7,8,9]), drop=True)  
            waq_xr_ds_mean = waq_xr_ds_merge.mean(dim='time')
            waq_xr_ds_mean_list.append(waq_xr_ds_mean)
            statistics='growing_season_mean'
            statistics_list.append(statistics)
            print(f'{var}, statistics: {statistics}.')
        elif var == 'NO3':
            waq_xr_ds_mean_list = []
            statistics_list = []
            waq_xr_ds_merge = waq_xr_ds_concat.where(waq_xr_ds_concat.time.dt.month.isin([12,1,2]), drop=True)  
            waq_xr_ds_mean = waq_xr_ds_merge.mean(dim='time')
            if office == 'DFM' and unit == 'CMEMS':
                waq_xr_ds_mean = waq_xr_ds_mean*1000/14.00672
            waq_xr_ds_mean_list.append(waq_xr_ds_mean)
            statistics='winter_mean'
            statistics_list.append(statistics)
            print(f'{var}, statistics: {statistics}.')
        elif var == 'PO4':
            waq_xr_ds_mean_list = []
            statistics_list = []
            waq_xr_ds_merge = waq_xr_ds_concat.where(waq_xr_ds_concat.time.dt.month.isin([1,2,12]), drop=True) 
            waq_xr_ds_mean = waq_xr_ds_merge.mean(dim='time') 
            if office == 'DFM' and unit == 'CMEMS':
                waq_xr_ds_mean = waq_xr_ds_mean*1000/30.973762
            waq_xr_ds_mean_list.append(waq_xr_ds_mean)
            statistics='winter_mean'
            statistics_list.append(statistics)
            print(f'{var}, statistics: {statistics}.')
        elif var == 'OXY':
            waq_xr_ds_mean_list = []
            statistics_list = []
            waq_xr_ds_merge = waq_xr_ds_concat.where(waq_xr_ds_concat.time.dt.month.isin([6,7,8]), drop=True)
            waq_xr_ds_mean = waq_xr_ds_merge.min(dim='time')
            if office == 'DFM' and unit == 'CMEMS':
                waq_xr_ds_mean = waq_xr_ds_mean*1000/31.998
            waq_xr_ds_mean_list.append(waq_xr_ds_mean)
            statistics='summer_min'
            statistics_list.append(statistics)
            print(f'{var}, statistics: {statistics}.')
        elif var == 'PCO2':
            waq_xr_ds_mean_list = []
            statistics_list = []
            for s in [[4,5], [8,9], [12,1]]: 
                waq_xr_ds_merge = waq_xr_ds_concat.where(waq_xr_ds_concat.time.dt.month.isin(s), drop=True)
                waq_xr_ds_mean = waq_xr_ds_merge.mean(dim='time')
                if office == 'DFM' and unit == 'CMEMS':
                    waq_xr_ds_mean = waq_xr_ds_mean*0.101325
                waq_xr_ds_mean_list.append(waq_xr_ds_mean)
                if s==[4,5]:
                    statistics='spring_mean'
                elif s==[8,9]:
                    statistics = 'summer_mean'
                elif s==[12,1]:
                    statistics = 'winter_mean'
                statistics_list.append(statistics)
                print(f'{var}, statistics: {statistics}.')
        elif var == 'PH':
            waq_xr_ds_mean_list = []
            statistics_list = []
            for s in [[4,5], [8,9], [12,1]]: 
                waq_xr_ds_merge = waq_xr_ds_concat.where(waq_xr_ds_concat.time.dt.month.isin(s), drop=True)
                waq_xr_ds_mean = waq_xr_ds_merge.mean(dim='time')
                waq_xr_ds_mean_list.append(waq_xr_ds_mean)
                if s==[4,5]:
                    statistics='spring_mean'
                elif s==[8,9]:
                    statistics = 'summer_mean'
                elif s==[12,1]:
                    statistics = 'winter_mean'
                statistics_list.append(statistics)
                print(f'{var}, statistics: {statistics}.')
        else:  # Careful with unit conversions, depending on quantity!
            waq_xr_ds_mean_list = []
            statistics_list = []
            waq_xr_ds_merge = waq_xr_ds_concat
            waq_xr_ds_mean = waq_xr_ds_merge.mean(dim='time')
            waq_xr_ds_mean_list.append(waq_xr_ds_mean)
            statistics='annual_mean'
            statistics_list.append(statistics)
            print(f'{var}, statistics: {statistics}.')
        
        ### Observations  ###
        print(f'Processing NWDM data for {var} for {office}')        
        latitudes = []
        longitudes = []
        colours = []
        
        # Read NWDM obs points:
        if NWDM_gridded == True:
            try:
                obs_path = fr'P:\11209810-cmems-nws\Data\NWDM_observations\combined_years\{start_year}_{end_year}_{var_obs}_obs_gridded_{slice_2d}.csv'   
                obs = pd.read_csv(obs_path)
            except:
                obs_path = fr'P:\11209810-cmems-nws\Data\NWDM_observations\combined_years\2003_2017_{var_obs}_obs_gridded_{slice_2d}.csv'  
                obs = pd.read_csv(obs_path)     
            
        else:
            try:
                obs_path = fr'P:\11209810-cmems-nws\Data\NWDM_observations\combined_years\{start_year}_{end_year}_{var_obs}_obs_{slice_2d}.csv'   
                obs = pd.read_csv(obs_path)
            except:
                obs_path = fr'P:\11209810-cmems-nws\Data\NWDM_observations\combined_years\2003_2017_{var_obs}_obs_{slice_2d}.csv'  
                obs = pd.read_csv(obs_path)            
        
        obs['datetime'] = pd.to_datetime(obs['datetime'], format='mixed')
        obs['datetime'] = obs['datetime'].astype('datetime64[ns]')
        obs['station'] = obs['station'].astype('string')
        obs = obs.loc[(obs.datetime>=f'{start_year}-01-01') & (obs.datetime<=f'{str(int(end_year)+1)}-01-01')]
        
        #Select observations in the top layer using buffer
        if NWDM_gridded == False:                   # Gridded already cropped!
            obs = obs[abs(obs['depth']) < buffer]   # abs, therefore works for surface and bottom 
        
        if fixed_stations == False and NWDM_gridded == False:
            plot_locs = np.unique(np.unique(obs.geom.values.astype(str)))
        else:
            # Get a unique list of the locations:
            plot_locs = np.unique(np.unique(obs.station.values.astype(str)))

        # Loop over stations
        l=1
        for loc in plot_locs:  # loop over locations
            print(f'{l} / {len(plot_locs)}')
            if fixed_stations == False and NWDM_gridded == False:
                station = obs[obs['geom'].str.endswith(loc, na=False)]
            else:
                station = obs[obs['station'].str.endswith(loc, na=False)]

            # remove empty df's
            if len(station) == 0:
                continue
            try:
                if fixed_stations == False and NWDM_gridded == False:
                    series = station[station['geom'].str.endswith(loc)] # select 1 station
                else:
                    series = station[station['station'].str.endswith(loc)] # select 1 station
                series_crop = station.set_index('datetime')
                series_crop = series_crop.value
                series_crop.index = pd.to_datetime(series_crop.index)
                series_crop = series_crop.dropna()
                if var_obs == 'Chlfa':  # no unit conversion
                    obs_season_list = []
                    obs_season = series_crop.where(series_crop.index.month.isin([3,4,5,6,7,8,9]))   # only CHL
                    obs_season_list.append(obs_season)
                elif var_obs == 'pH':
                    obs_season_list = []
                    for s in [[4,5], [8,9], [12,1]]: 
                        obs_season = series_crop.where(series_crop.index.month.isin(s))
                        obs_season_list.append(obs_season)
                elif var_obs == 'pCO2':
                    obs_season_list = []
                    for s in [[4,5], [8,9], [12,1]]: 
                        obs_season = series_crop.where(series_crop.index.month.isin(s))
                        if unit == 'CMEMS':
                            obs_season = obs_season*0.101325
                        obs_season_list.append(obs_season)
                elif var_obs == 'NO3':
                    obs_season_list = []
                    obs_season = series_crop.where(series_crop.index.month.isin([1,2,12]))   # only NO3
                    if unit == 'CMEMS':
                        obs_season = obs_season*1000/14.00672
                    obs_season_list.append(obs_season)
                elif var_obs == 'PO4':
                    obs_season_list = []
                    obs_season = series_crop.where(series_crop.index.month.isin([1,2,12]))   # only PO4
                    if unit == 'CMEMS':
                        obs_season = obs_season*1000/30.973762
                    obs_season_list.append(obs_season)
                elif var_obs == 'OXY':
                    obs_season_list = []
                    obs_season = series_crop.where(series_crop.index.month.isin([6,7,8]))  
                    if unit == 'CMEMS':
                        obs_season = obs_season*1000/31.998 
                    obs_season_list.append(obs_season)
                else:
                    print('Observation variable not recognized.')
                
                latitude = []
                longitude = []
                colour = []
                for obs_season in obs_season_list:

                    obs_season = obs_season.dropna()
                    # Group the Series by year
                    grouped = obs_season.groupby(obs_season.index.year)

                    ## If have multiple min_obs_count, depending on variable: (otherwise comment this and use above-defined)
                    if var == 'Si' or var == 'PO4' or var == 'NO3' or var == 'OXY':
                        min_obs_count = 3
                    elif var == 'CHL' or var == 'temperature': # growing season = more months. Using 9 / 12 not give enough obs though.
                        min_obs_count = 6
                    else:
                        min_obs_count = 3

                    obs_season = grouped.filter(lambda x: len(x) >= min_obs_count)
                    if len(obs_season) == 0:
                        colour.append([])
                        latitude.append([])
                        longitude.append([])                       
                    else:
                        if var_obs == 'OXY':
                            obs_seasonal_mean = obs_season.mean()   # for mean!
                            # obs_seasonal_mean = obs_season.min()   # for minimum!   -- No, want bottom 25 percentile... Check timeseries code!
                        else:
                            obs_seasonal_mean = obs_season.mean()  
                            
                        colour.append(obs_seasonal_mean)
                        ## Lat/lon:
                        lat = float(series.geom.iloc[0].split(' ')[1:][1][:-1])
                        lon = float(series.geom.iloc[0].split(' ')[1:][0][1:])
                        latitude.append(lat)
                        longitude.append(lon)
                    
                #Append outer list
                colours.append(colour)
                latitudes.append(latitude)
                longitudes.append(longitude)
                l=l+1
            except:
                continue
        
        ##### Plotting  #####
        print(f'Plotting for {var} for {office}') 
        i=0
        for waq_xr_ds_mean in waq_xr_ds_mean_list:
            #read the correct list items
            statistics=statistics_list[i]
            longitude=[]
            latitude=[]
            colour=[]
            for r in range(0, len(colours)):
                longitude.append(longitudes[r][i])
                latitude.append(latitudes[r][i])
                colour.append(colours[r][i])
            
            # Crop domain:
            if 'y' in waq_xr_ds_mean.dims:
                try:
                    # Drop the 'latitude' and 'longitude' coordinates
                    waq_xr_ds_mean = waq_xr_ds_mean.drop_vars(['latitude', 'longitude'])
                except Exception:
                    print(" ")
                    
                # Rename the dimensions 'y' to 'latitude' and 'x' to 'longitude'
                waq_xr_ds_mean = waq_xr_ds_mean.rename({'y':'latitude', 'x':'longitude'})  # rename lat,lon to y,x
            
            zoom = waq_xr_ds_mean.sel(longitude=slice(LON_MIN,LON_MAX),latitude=slice(LAT_MIN,LAT_MAX))  
            
            
            ## Define the characteristics of the plot
            fig = plt.figure(figsize=(7,6))
            ax = plt.axes(projection=ccrs.PlateCarree())                     # create ax + select map projection
            gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True)      # add the longitude / latitude lines
            gl.right_labels = False                                          # remove latitude labels on the right
            gl.top_labels = False                                            # remove longitude labels on the top
            gl.xlines = False
            gl.ylines = False
            ax.add_feature(cfeature.LAND, zorder=98, edgecolor='grey', linewidth=0.3, facecolor='w')      # add land mask
            ax.add_feature(cfeature.BORDERS, zorder=99, edgecolor='grey', linewidth=0.3)
            
            # colorbar settings:
            text = f'{statistics}_{slice_2d}'
            if var == 'NO3':
                if unit == 'CMEMS':
                    vmin,vmax = 0,50  
                elif unit == 'DFM':
                    vmin, vmax = 0,1.2 
                cmap = cmocean.cm.matter
            elif var == 'PO4':
                if unit == 'CMEMS':
                    vmin,vmax = 0,1.0 
                elif unit == 'DFM':
                    vmin, vmax = 0,0.08  
                cmap = cmocean.cm.matter
            elif var == 'CHL': 
                vmin, vmax = 0,10
                cmap = cmocean.cm.algae
            elif var == 'OXY':
                if unit == 'CMEMS':
                    vmin,vmax = 120,330  
                elif unit == 'DFM':
                    vmin, vmax = 0,12
                cmap = cmocean.cm.oxy
            elif var == 'PH':
                vmin,vmax = 7.0,8.5  
                cmap = cmocean.cm.thermal
            elif var == 'PCO2':
                if unit == 'CMEMS':
                    vmin,vmax = 10,50
                elif unit == 'DFM':
                    vmin, vmax = 0.1,0.6  ## NOTE: TO TEST! 
                cmap = cmocean.cm.solar
            elif var == 'temperature' and slice_2d == 'surface':
                vmin,vmax = 6.0,20.0    
                cmap = cmocean.cm.thermal
            elif var == 'temperature' and slice_2d == 'stratification':
                vmin,vmax = -6,6   
                cmap = cmocean.cm.balance
            elif var == 'salinity':
                vmin,vmax = 10,40 
                cmap = cmocean.cm.haline
            else: 
                print('Variable not recognised. Default colorbar settings are used.')
                vmin, vmax = 0,10
                cmap = cmocean.cm.thermal
            
            boundaries = np.array([b for b in np.arange(vmin,vmax+1E-6,(vmax-vmin)/20)])
            
            ## Plotting model 
            mod = zoom.plot(ax=ax, linewidth=0.5, edgecolors='face', cmap=cmap, add_colorbar=False, levels=boundaries)   
            
            ## Plotting obs
            if np.round(boundaries[-1],1) != vmax: 
                boundaries_obs = np.append(boundaries,vmax)   
            else:
                boundaries_obs = boundaries
    
            norm = colors.BoundaryNorm(boundaries_obs, cmap.N)  
            for c in range(0, len(colour)):
                if var =='PCO2':
                    ob = plt.scatter(longitude[c], latitude[c], c=colour[c], cmap=cmap, zorder=100, s=3, linewidths=0.05, edgecolors='k', norm=norm)
                    
                else:
                    ob = plt.scatter(longitude[c], latitude[c], c=colour[c], cmap=cmap, zorder=100, s=3, linewidths=0.1, edgecolors='k', norm=norm)
            
            ## Colourbar
            cbar = plt.colorbar(mod, fraction=0.025, pad=0.02, extend='neither')                                    # add the colorbar
            cbar.set_label(f'{var} {text}', rotation=90,fontsize=10, labelpad=15)   # add the title of the colorbar
    
            title = plt.title('') # to overwrite the standard title
            
            # ## Save figure
            plt.xlim((LON_MIN, LON_MAX))
            plt.ylim((LAT_MIN, LAT_MAX))
            plt.tight_layout()   # so labels don't get cut off
            plt.title(fr"{office}")
            if NWDM_gridded == True:
                plt.savefig(os.path.join(outdir, f'{zooming}_map_{office}_obs_{start_year}_{end_year}_{var}_{slice_2d}_{statistics}_gridded.png'), dpi=400, bbox_inches='tight')
            elif fixed_stations == False and NWDM_gridded == False:
                plt.savefig(os.path.join(outdir, f'{zooming}_map_{office}_obs_{start_year}_{end_year}_{var}_{slice_2d}_{statistics}_non-fixed.png'), dpi=400, bbox_inches='tight')
            else:
                plt.savefig(os.path.join(outdir, f'{zooming}_map_{office}_obs_{start_year}_{end_year}_{var}_{slice_2d}_{statistics}_fixed.png'), dpi=400, bbox_inches='tight')
            # plt.close()
            print(f'Plot saved for {var} for {office}, {statistics}')
            print(' ')
            i=i+1
