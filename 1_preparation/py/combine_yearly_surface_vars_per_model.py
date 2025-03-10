#!/usr/bin/env python
# coding: utf-8


## Import packages

import os
import xarray as xr
import xugrid as xu
import pandas as pd
import dask


# ## Choose the model domain  - keep full for now.

# min_lon = -19.89 
# max_lon = 13
# min_lat = 40.07
# max_lat = 65


outdir = r'p:\11209810-cmems-nws\model_output\combined_yearly' if os.name == 'nt' else r'/p/11209810-cmems-nws/model_output/combined_yearly'


## Functions:

def combine_year_per_IBI_variable(rootdir, variable, year):
    directory = rootdir
    fps = []
    if variable in ['pp', 'spco2']:
        variable = 'bgc'
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            if ('_'+year) in file and variable in file:
                fps.append(os.path.join(subdir, file))
    ds = xr.open_mfdataset(fps)
    return ds

def combine_year_per_NWS_variable(rootdir, variable, year, model):
    fps = []
    
    # Construct directory based on the model type
    if model == 'nrt':
        directory = os.path.join(rootdir, f'cmems_mod_nws_bgc-{variable}_anfc_7km-3D_P1D-m', year)  # for nrt = anfc

        # Check if the 3D directory exists
        if not os.path.isdir(directory):
            print(r"No such 3D directory, trying 2D version")
            directory = os.path.join(rootdir, f'cmems_mod_nws_bgc-{variable}_anfc_7km-2D_P1D-m', year)  # for 2D version
            
            # Check if the 2D directory exists
            if not os.path.isdir(directory):
                raise FileNotFoundError(r"No such 2D directory or 3D directory.")

    elif model == 'rea':
        directory = os.path.join(rootdir, f'cmems_mod_nws_bgc-{variable}_my_7km-3D_P1D-m', year)    # for rea = my
        
        # Check if the 3D directory exists
        if not os.path.isdir(directory):
            print(r"No such 3D directory, trying 2D version")
            directory = os.path.join(rootdir, f'cmems_mod_nws_bgc-{variable}_my_7km-2D_P1D-m', year)  # for 2D version
            
            # Check if the 2D directory exists
            if not os.path.isdir(directory):
                raise FileNotFoundError(r"No such 2D directory or 3D directory.")
    else:
        raise ValueError("Invalid model specified. Use 'nrt' or 'rea'.")
    
    print(f"Searching in directory: {directory}")

    # Ensure the directory exists
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"The directory {directory} does not exist.")
    
    # Traverse directory and collect file paths
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            fps.append(os.path.join(subdir, file))
        
    # Open and combine datasets
    if not fps:
        raise FileNotFoundError("No files found to combine.")
    
    ds = xr.open_mfdataset(fps)
    return ds

# def combine_years_for_satellite_chl(rootdir, year):  # only necessary for the L3 products
#     fps = []
#     for subdir, dirs, files in os.walk(rootdir):  
#         for file in files:
#             fps.append(os.path.join(subdir, file))
#     ds = xr.open_mfdataset(fps)
#     return ds

def convert_3d_IBI_into_3d_NWS(ds, variable):    
    # Correcting the variable name check
    if variable == 'pp':
        variable = 'npp' # Assign 'npp' if variable is 'pp'
        
    # Check if 'deptht' exists in the dataset and use it, otherwise set default depth
    if 'deptht' in ds:
        depth_data = ds.deptht.data
        depth_attrs = ds.deptht.attrs
    else:
        depth_data = [0.5]  # Manually set depth to 0 if it doesn't exist
        depth_attrs = {}  # Empty attributes for missing depth
    
    if len(ds[variable].shape) == 3: # If it has dimensions (time_counter, y, x)
        ds[variable] = ds[variable].expand_dims(depth=1)  # Expands to (depth, time_counter, y, x)
        ds[variable] = ds[variable].transpose("time_counter", "depth", "y", "x") # reshuffles to (time_counter, depth, y, x)
        
    if len(ds.nav_lat.data.shape) > 2:
        lat_2d = ds.nav_lat.isel(time_counter=0).data
        lon_2d = ds.nav_lon.isel(time_counter=0).data
        
        # Create the new dataset with manually handled depth
        ds_new = xr.Dataset(
            data_vars=dict(
                variable=(["time", "depth", "y", "x"], ds[variable].data, ds[variable].attrs)),
            coords=dict(
                time      = (["time"], ds.time_counter.data, ds.time_counter.attrs),
                latitude  = (["y", "x"], lat_2d, ds.nav_lat.attrs),
                longitude = (["y", "x"], lon_2d, ds.nav_lon.attrs),
                depth     = (["depth"], depth_data, depth_attrs)),
            attrs=ds.attrs
        )
        
    else:    
        # Create the new dataset with manually handled depth
        ds_new = xr.Dataset(
            data_vars=dict(
                variable=(["time", "depth", "y", "x"], ds[variable].data, ds[variable].attrs)),
            coords=dict(
                time      = (["time"], ds.time_counter.data, ds.time_counter.attrs),
                latitude  = (["y", "x"], ds.nav_lat.data, ds.nav_lat.attrs),
                longitude = (["y", "x"], ds.nav_lon.data, ds.nav_lon.attrs),
                depth     = (["depth"], depth_data, depth_attrs)),
            attrs=ds.attrs
        )

    # Rename 'variable' to the actual variable name
    ds_new = ds_new.rename({'variable': variable})
    return ds_new

# For NWS
def convert_struct_to_ugrid(ds):
    ds = ds.rename_dims({"longitude":"x","latitude":"y"})
    ds = ds.rename_vars({"longitude":"x","latitude":"y"})
    list_vars = []
    for var in ds.data_vars:
        uda = xu.UgridDataArray.from_structured(ds[var])
        list_vars.append(uda)
    uds = xu.UgridDataset(grids=[uda.grid])
    for var in list_vars:
        uds[var.name] = var
    return uds

# For IBI
def derive_verts(ds):
    ds['vertices_longitude'] = xr.concat([ds.longitude, 
                                   ds.longitude.shift(x=1),
                                   ds.longitude.shift(x=1,y=1), 
                                   ds.longitude.shift(y=1)],
                                  dim="vertices").T #TODO: requires transpose since we use hardcoded vertices dim-index in open_dataset_curvilinear
    ds['vertices_latitude'] = xr.concat([ds.latitude, 
                                   ds.latitude.shift(x=1),
                                   ds.latitude.shift(x=1,y=1), 
                                   ds.latitude.shift(y=1)],
                                  dim="vertices").T
    return ds

def save_dataset_with_dask(ds_merge, output_file):
    # Define chunks based on available dimensions
    chunks = {}
    
    if 'time' in ds_merge.dims:
        chunks['time'] = 'auto'
    
    if 'depth' in ds_merge.dims:
        chunks['depth'] = 'auto'
        
    if 'latitude' in ds_merge.dims:
        chunks['latitude'] = 'auto'
    
    if 'longitude' in ds_merge.dims:
        chunks['longitude'] = 'auto'
    
    if 'y' in ds_merge.dims:
        chunks['y'] = 'auto'
        
    if 'x' in ds_merge.dims:
        chunks['x'] = 'auto'
        
    if 'mesh2d_nFaces' in ds_merge.dims:
        chunks['mesh2d_nFaces'] = 'auto'
        
    if 'mesh2d_nNodes' in ds_merge.dims:
        chunks['mesh2d_nNodes'] = 'auto'

    # Chunk the dataset with the defined chunks
    ds_merge = ds_merge.chunk(chunks)
            
    # Save the dataset to a NetCDF file with Dask
    ds_merge.to_netcdf(output_file, engine='netcdf4', compute=True)
    
def years_in_order(start_year, end_year):
    return list(range(start_year, end_year + 1))

## Combine the .nc files into yearly files

# Selection:
offices = ['NWS', 'IBI'] #'IBI' or 'NWS'
model = 'nrt' # 'rea' or 'nrt'   
start_year = 2020
end_year = 2023
selected_years = years_in_order(start_year, end_year)
variables = ['spco2', 'chl', 'no3', 'o2', 'ph', 'po4'] 
slice2d = 'surface' # 'surface' or 'bottom' or layer number

if slice2d == 'surface':
    slice2d_id = 0
elif slice2d == 'bottom':
    slice2d_id = -1
else:
    slice2d_id = int(slice2d)

# IBI Available: bgc (cflux, spco2, zeu, npp), alk, chl, dic, nh4, no3, o2, ph, po4, si. Not use bgc (except npp) and alk.
# NWS: pp gets name: nppv. Available: chl, kd, no3, o2, ph, phyc, po4, pp, spco2. kd and spco2 not in validation proposal (leave for now). 
    
for office in offices:
    for year in selected_years:    
        year=str(year)
        # Define the output file path
        output_file = os.path.join(outdir, f'{slice2d}_{office}_{model}_{year}.nc')
        if os.path.exists(output_file):
            print(f"File {output_file} already exists. Skipping.")
        else:
            print(f'Running {office}_{model}_{year}')
            rootdir =  fr'p:\11209810-cmems-nws-data-{model}\{office}' if os.name == 'nt' else fr'/p/11209810-cmems-nws-data-{model}/{office}'
           
            # Combine into yearly ds with specified variables:
            datasets = []
            for variable in variables:
                print(f'Combining {variable}')
                if office == 'IBI':
                    # read the 1-year data and convert IBI to NWS format:
                    ds = combine_year_per_IBI_variable(rootdir, variable, year=year)
                    ds = convert_3d_IBI_into_3d_NWS(ds,variable)
                elif office == 'NWS' and model == 'rea':
                    ds = combine_year_per_NWS_variable(os.path.join(rootdir, r'NWSHELF_MULTIYEAR_BGC_004_011'), variable, year, model)
                elif office == 'NWS' and model == 'nrt':
                    ds = combine_year_per_NWS_variable(os.path.join(rootdir, r'NWSHELF_ANALYSISFORECAST_BGC_004_002'), variable, year, model) 
                # Crop time 
                ds = ds.sel(time=slice(f'{year}-01',f'{year}-12'))  # redundant, but okay
                # # Crop domain
                # ds = ds.sel(latitude=slice(min_lat,max_lat), longitude=slice(min_lon,max_lon))
                
                # Add a depth dimension with a value of 0 if dataset doesn't have depth - applies only to NWS
                if 'depth' not in ds.dims:
                    ds = ds.expand_dims('depth')
                    ds['depth'] = [0.0] # Manually set depth to 0.0 if it doesn't exist
                    ds['depth'].attrs = {}  # Empty attributes for missing depth
                
                # Extract slice2d
                try:
                    ds = ds.isel(depth=slice2d_id) 
                except Exception:
                    print(fr"Data doesn't have {slice2d} coordinate.")
                    
                # Append to datasets to later merge:
                datasets.append(ds)
            
            # Make sure each dataset has the correct date format (based on NWS - midday):
            if office == 'IBI':
                updated_datasets = []
                for nr in range(0,len(datasets)):
                    datasets[nr]['time'] = datasets[nr].indexes['time'].normalize() + pd.DateOffset(hours=12)
                    updated_datasets.append(datasets[nr])
                print('Correcting date format')
        
            # Merge and save:
            print('Merging')
            if office == 'IBI':
                ds_merge = xr.merge(updated_datasets, compat='override')
            else:
                ds_merge = xr.merge(datasets, compat='override') # For NWS, not necessary to save the in-between set!
            print('Saving')    
            save_dataset_with_dask(ds_merge, output_file) # Save, to get path to use below

            print("=================================")            
            print(f'Done for {office}_{model}_{year}')
            print("=================================")
    
