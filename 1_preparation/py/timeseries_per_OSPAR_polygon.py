#!/usr/bin/env python
# coding: utf-8

# In[1]:

## Packages

import os
import rasterio

import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd

from shapely.geometry import Point

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

## Import assessment areas:
fp = r'P:\11209810-cmems-nws\OSPAR_areas'
gdf = gpd.read_file(fp)

gdf.plot()  # includes regions outside NWS domain


# In[3]:
rootdir = r'P:\11209810-cmems-nws\model_output\regridded_onto_NWS' if os.name == 'nt' else r'/p/11209810-cmems-nws/model_output/regridded_onto_NWS'
outdir = r'P:\11209810-cmems-nws\model_output\timeseries_per_polygon' if os.name == 'nt' else r'/p/11209810-cmems-nws/model_output/timeseries_per_polygon'

# Selection:
offices = ['satellite']#['IBI', 'NWS']#['satellite', 'IBI', 'NWS']#, 'DFM']
model = 'nrt'
slice_2d = 'surface'
start_year = 2021
end_year = 2022
selected_years = years_in_order(start_year, end_year)
var = 'CHL'

var_sat, var_IBI, var_NWS, var_DFM = var_mapping(var) #get variable names for each model

# In[4]:
# Use regridded surface data
# Read Models:

for office in offices:
    waq_xr_ds = []
    print(f'Opening regridded_{slice_2d}_{office}_{start_year}_{end_year}')
    for year in selected_years: 
        if office == 'NWS':
            basedir = fr'P:\11209810-cmems-nws\model_output\combined_yearly\{slice_2d}_{office}_{model}_{year}.nc' if os.name == 'nt' else fr'/p/11209810-cmems-nws/model_output/combined_yearly/{slice_2d}_{office}_{model}_{year}.nc'  # not regridded, so different rootdir!
        elif office == 'satellite':
            basedir = os.path.join(rootdir, f'regridded_{office}_{year}.nc')
        else:
            basedir = os.path.join(rootdir,fr'regridded_{slice_2d}_{office}_{year}.nc')
    
        waq_xr = xr.open_dataset(basedir)
        if office == 'NWS':
            waq_xr = waq_xr[var_NWS]#[:,0,:,:]
            waq_xr = waq_xr.rename({'latitude':'y', 'longitude':'x'})                            # select variable and depth
        elif office == 'satellite':
            waq_xr = waq_xr[var_sat] 
            waq_xr = waq_xr.rename({'latitude':'y', 'longitude':'x'}) 
        elif office == 'IBI':
            waq_xr = waq_xr[var_IBI]#[0,:,:,:]
        elif office == 'DFM':
            # waq_xr = waq_xr.rename({'y':'latitude', 'x':'longitude'})
            waq_xr = waq_xr[var_DFM]#[:,:,:,-1]
        # waq_xr = waq_xr.sel(longitude=slice(LON_MIN,LON_MAX),latitude=slice(LAT_MIN,LAT_MAX))    
        waq_xr_ds.append(waq_xr)
             
    # Merge:
    print(f'Merging regridded_{slice_2d}_{office}_{start_year}_{end_year}')    
    waq_xr_ds_merge = xr.concat(waq_xr_ds, dim='time')        
    waq_xr_ds_merge = waq_xr_ds_merge.where(waq_xr_ds_merge.time.dt.month.isin([3,4,5,6,7,8,9]), drop=True)  # get seasonal (Mar-Sep) output
     
    waq_xr_ds_merge


# In[5]:
    # Create and select timeseries
    
    ds = waq_xr_ds_merge   
    df_all_polys = []
    
    print(f'Extracting time series per polygon for regridded_{slice_2d}_{office}_{start_year}_{end_year}')  
    for poly in range(0,len(gdf)):  # loop over polygons
        print(fr'polygon {poly}/{len(gdf)}')  
        # Create an assessment area mask for satellites:
        data_mask = ds[0,:,:]                     # remove time (maybe not necessary?)
        geom = [gdf.geometry.values[poly]]        # select the polygon (later make loop over each)
        mask = rasterio.features.geometry_mask(   # create polygon mask
                    geometries = geom, out_shape=data_mask.shape, transform=data_mask.rio.transform(), 
                    all_touched=False, invert=False)
        mask = xr.DataArray(mask, dims=("y", "x"))
    
        # Loop over every timesetep and apply mask to get a timeseries
        mean_timeseries = []
        for t in range(0,len(ds.time)):
            xr_crop = ds[t,:,:]                              # select time
            xr_crop = xr_crop.where(mask == False)  # mask ds with rasterized gdf
            xr_crop_mean = xr_crop.mean()
            mean_timeseries.append(xr_crop_mean)
        
        ts = xr.concat(mean_timeseries, dim='time')  # concat timeseries
        df_per_poly = ts.to_dataframe()              # convert to dataframe
        df_per_poly['polygon'] = np.repeat(gdf.iloc[poly].ID, len(df_per_poly))  # add polygon name
        # df_per_poly = df_per_poly.reset_index()
    
        df_all_polys.append(df_per_poly)
        # break
    
    # Append and save:
    print(f'Appending and saving time series for regridded_{slice_2d}_{office}_{start_year}_{end_year}') 
    print(' ')
    data_concat = pd.concat(df_all_polys, ignore_index=False) 
    data_concat.to_csv(os.path.join(outdir, fr'{start_year}_{end_year}_{office}_ts.csv'))



