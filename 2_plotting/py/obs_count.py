#!/usr/bin/env python
# coding: utf-8

#%%  
## Imports

import os
import numpy as np
import pandas as pd
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

#%%   
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

#Input directory
basedir = r'P:\11209810-cmems-nws\model_output' if os.name == 'nt' else r'/p/11209810-cmems-nws/model_output'
#rootdir = os.path.join(basedir, 'regridded_onto_NWS')

#%%  
## Choose and read the model map files:

start_year = 2021
end_year = 2022
slice_2d = 'bottom' #surface or bottom (bottom only works for DFM currently) 
NWDM_gridded = False
# variables = ['NO3', 'PO4', 'PH', 'CHL', 'OXY'] 
variables = ['OXY'] #PCO2 only available in gridded format 

#Points to plot (only for non-gridded)
fixed_stations = False # True or False

#Buffer for surface observations
buffer = 10.5 # (meters) take all measuerements from the top x meters

#Minimum amount of observations per year
min_obs_count = 1

# path to output folder for plots
outdir = fr'P:\11209810-cmems-nws\figures\maps_model_obs\{start_year}_{end_year}' if os.name == 'nt' else fr'/p/11209810-cmems-nws/figures/maps_model_obs/{start_year}_{end_year}'

#%% 
if not os.path.exists(outdir):
    os.makedirs(outdir)

selected_years = years_in_order(start_year, end_year)
        
for var in variables:
    print(f'Running {var}, gridded = {NWDM_gridded}, fixed_stations = {fixed_stations}')
    
    var_sat, var_IBI, var_NWS, var_DFM, var_obs = var_mapping(var) # get variable names for each model
    
    ### Observations  ###
    print(f'Processing NWDM data for {var}')        
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
    if slice_2d == 'surface' and NWDM_gridded == False:
        obs = obs[abs(obs['depth']) <= buffer] 
    
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
            
            statistics_list = []
            if var_obs == 'Chlfa':
                obs_season_list = []
                obs_season = series_crop.where(series_crop.index.month.isin([3,4,5,6,7,8,9]))   # only CHL
                obs_season_list.append(obs_season)
                statistics='count_growing_season'
                statistics_list.append(statistics)
            elif var_obs == 'pH':
                obs_season_list = []
                for s in [[4,5], [8,9], [12,1]]: 
                    obs_season = series_crop.where(series_crop.index.month.isin(s))
                    obs_season_list.append(obs_season)
                    if s==[4,5]:
                        statistics='count_spring'
                    elif s==[8,9]:
                        statistics = 'count_summer'
                    elif s==[12,1]:
                        statistics = 'count_winter'
                    statistics_list.append(statistics)
            elif var_obs == 'pCO2':
                obs_season_list = []
                for s in [[4,5], [8,9], [12,1]]: 
                    obs_season = series_crop.where(series_crop.index.month.isin(s))
                    obs_season = obs_season*0.101325
                    obs_season_list.append(obs_season)
                    if s==[4,5]:
                        statistics='count_spring'
                    elif s==[8,9]:
                        statistics = 'count_summer'
                    elif s==[12,1]:
                        statistics = 'count_winter'
                    statistics_list.append(statistics)
            elif var_obs == 'NO3':
                obs_season_list = []
                obs_season = series_crop.where(series_crop.index.month.isin([1,2,12]))   # only NO3
                obs_season = obs_season*1000/14
                obs_season_list.append(obs_season)
                statistics='count_winter'
                statistics_list.append(statistics)
            elif var_obs == 'PO4':
                obs_season_list = []
                obs_season = series_crop.where(series_crop.index.month.isin([1,2,12]))   # only NO3
                obs_season = obs_season*1000/30.97
                obs_season_list.append(obs_season)
                statistics='count_winter'
                statistics_list.append(statistics)
            elif var_obs == 'OXY':
                obs_season_list = []
                obs_season = series_crop.where(series_crop.index.month.isin([6,7,8]))  
                obs_season = obs_season*1000/32 
                obs_season_list.append(obs_season)
                statistics='count_summer'
                statistics_list.append(statistics)
            else:
                print('Observation variable not recognized.')
            
            latitude = []
            longitude = []
            colour = []
            for obs_season in obs_season_list:

                obs_season = obs_season.dropna()
                # Group the Series by year
                grouped = obs_season.groupby(obs_season.index.year)
                # Filter out years that have fewer than a defined number of values
                obs_season = grouped.filter(lambda x: len(x) >= min_obs_count)
                if len(obs_season) == 0:
                    colour.append([])
                    latitude.append([])
                    longitude.append([])                       
                else:
                    obs_seasonal_mean = obs_season.count()
                        
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
    print(f'Plotting for {var}') 
    i=0
    for statistics in statistics_list:
        #read the correct list items
        statistics=statistics_list[i]
        longitude=[]
        latitude=[]
        colour=[]
        for r in range(0, len(colours)):
            longitude.append(longitudes[r][i])
            latitude.append(latitudes[r][i])
            colour.append(colours[r][i])
        
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
        vmin,vmax = 0,10
        cmap = cmocean.cm.matter
        
        boundaries = np.array([b for b in np.arange(vmin,vmax+1E-6,(vmax-vmin)/10)])
              
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
        cbar = plt.colorbar(ob, fraction=0.025, pad=0.02, extend='neither')                                    # add the colorbar
        cbar.set_label(f'{var} {text}', rotation=90,fontsize=10, labelpad=15)   # add the title of the colorbar

        title = plt.title(f'{start_year}-{end_year}') # to overwrite the standard title
        
        # ## Save figure
        plt.xlim((LON_MIN, LON_MAX))
        plt.ylim((LAT_MIN, LAT_MAX))
        plt.tight_layout()   # so labels don't get cut off
        if NWDM_gridded == True:
            plt.savefig(os.path.join(outdir, f'{zooming}_map_obs_{start_year}_{end_year}_{var}_{slice_2d}_{statistics}_gridded.png'), dpi=400, bbox_inches='tight')
        elif fixed_stations == False and NWDM_gridded == False:
            plt.savefig(os.path.join(outdir, f'{zooming}_map_obs_{start_year}_{end_year}_{var}_{slice_2d}_{statistics}_non-fixed.png'), dpi=400, bbox_inches='tight')
        else:
            plt.savefig(os.path.join(outdir, f'{zooming}_map_obs_{start_year}_{end_year}_{var}_{slice_2d}_{statistics}_fixed.png'), dpi=400, bbox_inches='tight')
        # plt.close()
        print(f'Plot saved for {var}, {statistics}')
        print(' ')
        i=i+1

# %%
