#!/usr/bin/env python
# coding: utf-8

# In[1]:


## Packages

import os
import numpy as np
import xarray as xr
import xugrid as xu
import pandas as pd
import datetime as dt

import dfm_tools as dfmt
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
        var_DFM = 'mesh2d_Chlfa'
    return var_sat, var_IBI, var_NWS, var_DFM

# In[2]:


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


# In[3]:

# Selection:
start_year = 2021
end_year = 2022
offices = ['IBI', 'NWS'] #['IBI', 'NWS', 'DFM']  #IBI, #DFM
model = 'nrt'
slice_2d = 'surface' 
var = 'CHL'

#%%

selected_years = years_in_order(start_year, end_year)

var_sat, var_IBI, var_NWS, var_DFM = var_mapping(var) #get variable names for each model
 
rootdir = r'P:\11209810-cmems-nws\model_output\regridded_onto_NWS' if os.name == 'nt' else r'/p/11209810-cmems-nws/model_output/regridded_onto_NWS'

# path to output folder for plots
outdir = fr'P:\11209810-cmems-nws\figures\maps_differences\{start_year}_{end_year}_growing_season_CHL' if os.name == 'nt' else fr'/p/11209810-cmems-nws/figures/maps_differences/{start_year}_{end_year}_growing_season_CHL'
if not os.path.exists(outdir):
    os.makedirs(outdir)


# In[4]:


# Read the regridded satellite file:
satellite_xr_ds = []
print(f'Opening regridded_satellite_{start_year}_{end_year}')
for year in selected_years:    
    satellite_xr = xr.open_dataset(os.path.join(rootdir, f'regridded_satellite_{year}.nc'))
    satellite_xr = satellite_xr[var_sat]
    # satellite_xr = satellite_xr.sel(longitude=slice(LON_MIN,LON_MAX),latitude=slice(LAT_MIN,LAT_MAX))
    satellite_xr_ds.append(satellite_xr)
    
# Merge:
print(f'Merging regridded_satellite_{start_year}_{end_year}')
satellite_xr_ds_merge = xr.concat(satellite_xr_ds, dim='time')
satellite_xr_ds_merge = satellite_xr_ds_merge.where(satellite_xr_ds_merge.time.dt.month.isin([3,4,5,6,7,8,9]), drop=True)  # get seasonal (Mar-Sep) output
satellite_xr_ds_merge = satellite_xr_ds_merge.mean(dim='time')
satellite_xr_ds_merge = satellite_xr_ds_merge.rename({'latitude':'y', 'longitude':'x'})

# In[12]:


## Read the model map files:

for office in offices:
    waq_xr_ds = []
    print(f'Opening regridded_{slice_2d}_{office}_{model}_{start_year}_{end_year}')
    for year in selected_years: 
        if office == 'NWS':
            basedir = fr'P:\11209810-cmems-nws\model_output\combined_yearly\{slice_2d}_{office}_{model}_{year}.nc' if os.name == 'nt' else fr'/p/11209810-cmems-nws/model_output/combined_yearly/{slice_2d}_{office}_{model}_{year}.nc'  # not regridded, so different rootdir!
        else:
            basedir = fr'P:\11209810-cmems-nws\model_output\regridded_onto_NWS\regridded_{slice_2d}_{office}_{year}.nc' if os.name == 'nt' else fr'/p/11209810-cmems-nws/model_output/regridded_onto_NWS/regridded_{slice_2d}_{office}_{year}.nc'  
    
        waq_xr = xr.open_dataset(basedir)
        if office == 'NWS':
            waq_xr = waq_xr[var_NWS]#[:,0,:,:]
            waq_xr = waq_xr.rename({'latitude':'y', 'longitude':'x'})                            # select variable and depth
        elif office == 'IBI':
            waq_xr = waq_xr[var_IBI]#[0,:,:,:]
        elif office == 'DFM':
            # waq_xr = waq_xr.rename({'y':'latitude', 'x':'longitude'})
            waq_xr = waq_xr[var_DFM]#[:,:,:,-1]
        # waq_xr = waq_xr.sel(longitude=slice(LON_MIN,LON_MAX),latitude=slice(LAT_MIN,LAT_MAX))    
        waq_xr_ds.append(waq_xr)
        
    # Merge:
    print(f'Merging regridded_{slice_2d}_{office}_{model}_{start_year}_{end_year}')    
    waq_xr_ds_merge = xr.concat(waq_xr_ds, dim='time')        
    waq_xr_ds_merge = waq_xr_ds_merge.where(waq_xr_ds_merge.time.dt.month.isin([3,4,5,6,7,8,9]), drop=True)  # get seasonal (Mar-Sep) output
    waq_xr_ds_merge = waq_xr_ds_merge.mean(dim='time')
    
    # In[22]:
    
    fig = plt.figure(figsize=(7,6))
    waq_xr_ds_merge.plot(vmin=0,vmax=15)
    
    
    # In[23]:
    
    fig = plt.figure(figsize=(7,6))
    satellite_xr_ds_merge.plot(vmin=0,vmax=15)
    
    
    # In[13]:
    
    
    ## Plotting -- can't figure out the discrete to make the centre white, when the values are outside the bounds...
    
    # Define the characteristics of the plot
    fig = plt.figure(figsize=(7,6))
    ax = plt.axes(projection=ccrs.PlateCarree())                     # create ax + select map projection
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True)      # add the longitude / latitude lines
    gl.right_labels = False                                          # remove latitude labels on the right
    gl.top_labels = False                                            # remove longitude labels on the top
    gl.xlines = False
    gl.ylines = False
    ax.add_feature(cfeature.LAND, zorder=1, edgecolor='grey', linewidth=0.3, facecolor='linen')      # add land mask
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor='grey')
    
    mod = (waq_xr_ds_merge-satellite_xr_ds_merge).plot(ax=ax, linewidth=0.5, edgecolors='face', cmap='cmo.balance', add_colorbar=False)
    
    vmin,vmax = -8,8
    mod.set_clim(vmin,vmax)
    
    ## Colourbar
    cbar = plt.colorbar(mod, fraction=0.025, pad=0.02, extend='both')                # add the colorbar
    cbar.set_label(f'Difference in Chlorophyll concentration \n ({office} - satellite)', rotation=90,fontsize=10, labelpad=15)  # colorbar title 
    
    title = plt.title('') # to overwrite the standard title
    
    # ## Save figure
    plt.xlim((LON_MIN, LON_MAX))
    plt.ylim((LAT_MIN, LAT_MAX))
    plt.tight_layout()   # so labels don't get cut off
    plt.savefig(os.path.join(outdir, f'{zooming}_map_{office}_{model}_satellite_{start_year}_{end_year}_CHL.png'), dpi=400, bbox_inches='tight')
    # plt.close() 
    

# In[ ]:





# In[45]:


# ## Plotting -- can't figure out the discrete to make the centre white, when the values are outside the bounds...

# # Define the characteristics of the plot
# fig = plt.figure(figsize=(7,6))
# ax = plt.axes(projection=ccrs.PlateCarree())                     # create ax + select map projection
# ax.coastlines(linewidth=0.8)                                     # add the coastlines
# gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True)      # add the longitude / latitude lines
# gl.right_labels = False                                          # remove latitude labels on the right
# gl.top_labels = False                                            # remove longitude labels on the top
# gl.xlines = False
# gl.ylines = False
# ax.add_feature(cfeature.LAND, zorder=1, edgecolor='k', facecolor='w')      # add land mask
# ax.add_feature(cfeature.BORDERS, linewidth=0.8)

# # colorbar settings:
# vmin,vmax = -8,8
# cmap = plt.cm.seismic
# # boundaries = np.array([b for b in np.arange(vmin,vmax+1E-6,(vmax-vmin)/10)])
# boundaries = np.array([-8,-6,-4,-2,-0.5,0.5,2,4,6,8])

# ## Plotting model difference
# mod = (waq_xr-satellite_xr).plot(ax=ax, linewidth=0.5, edgecolors='face', cmap=cmap, add_colorbar=False, levels=boundaries)#, norm=colors.TwoSlopeNorm(0))   
# # mod.set_clim(vmin,vmax)
# # mod.cmap.set_over('k')
# # mod.cmap.set_extremes(bad='w', under='k', over='k')

# ## Colourbar
# cbar = plt.colorbar(mod, fraction=0.025, pad=0.02, extend='both', ticks=[-8,-6,-4,-2,-0.5,0.5,2,4,6,8])                # add the colorbar
# cbar.set_label('Difference in Chlorophyll concentration \n (model - satellite)', rotation=90,fontsize=10, labelpad=15)  # colorbar title 

# title = plt.title('') # to overwrite the standard title

# # ## Save figure
# plt.tight_layout()   # so labels don't get cut off
# plt.savefig(os.path.join(outdir, f'{zooming}_map_{office}_{model}_satellite_{year}_CHL.png'), dpi=400)
# # plt.close() 


# In[ ]:




