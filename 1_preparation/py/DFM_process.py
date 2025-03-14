# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 16:46:44 2024

@author: lorinc
"""


## Import packages
import os
import re
import xarray as xr
import xugrid as xu
import pandas as pd
import dfm_tools as dfmt
import glob
import random


#Functions
   
def years_in_order(start_year, end_year):
    return list(range(start_year, end_year + 1))

def find_files_and_extract_full_pattern(directory, pattern):
    # Use glob to match the file pattern
    matching_files = glob.glob(os.path.join(directory, pattern))
    extracted_patterns = []
    
    # Define a regular expression to capture the full pattern
    full_pattern_regex = re.compile(r"(DCSM-FM_0_5nm_waq_\d+_map)")

    # Extract the full pattern from each matching file
    for file_path in matching_files:
        match = full_pattern_regex.search(file_path)
        if match:
            extracted_patterns.append(match.group(1))  # The full pattern (e.g., 'DCSM-FM_0_5nm_waq_0127_map')

    return extracted_patterns

def rechunk(ds):
    # Define chunks based on available dimensions
    chunks = {}
    
    if 'time' in ds.dims:
        chunks['time'] = 'auto' #int(1) #'auto'
    
    if 'depth' in ds.dims:
        chunks['depth'] = 'auto'
        
    if 'latitude' in ds.dims:
        chunks['latitude'] = 'auto' #int(ds.dims['latitude']/2) 
    
    if 'longitude' in ds.dims:
        chunks['longitude'] = 'auto' #int(ds.dims['longitude']/2)
    
    if 'y' in ds.dims:
        chunks['y'] = 'auto' #int(ds.dims['y']/2) 
        
    if 'x' in ds.dims:
        chunks['x'] = 'auto' #int(ds.dims['y']/2)
        
    if 'mesh2d_nFaces' in ds.dims:
        chunks['mesh2d_nFaces'] = 'auto'
        
    if 'mesh2d_nNodes' in ds.dims:
        chunks['mesh2d_nNodes'] = 'auto'

    # Chunk the dataset with the defined chunks
    ds = ds.chunk(chunks)
    
    return ds

# Filepaths
basedir = r'P:\11209810-cmems-nws\model_output' if os.name == 'nt' else r'/p/11209810-cmems-nws/model_output'
outdir = os.path.join(basedir, 'DFM_processed')

#Choose year
start_year = 2014  # for the new run, only until 2014 available for now!
end_year = 2014
selected_years = years_in_order(start_year, end_year)

search_pattern = r"DCSM-FM_0_5nm_waq_*_map.nc"  # The pattern with wildcard for any characters


for year in selected_years:
    DFM_model = fr'P:\11210284-011-nose-c-cycling\runs_fine_grid\B05_waq_{year}_PCO2_DenWat_stats_2023.01\DFM_OUTPUT_DCSM-FM_0_5nm_waq' if os.name == 'nt' else fr'/p/11210284-011-nose-c-cycling/runs_fine_grid/B05_waq_{year}_PCO2_ChlC_NPCratios_DenWat_stats_2023.01/DFM_OUTPUT_DCSM-FM_0_5nm_waq'
    full_patterns = find_files_and_extract_full_pattern(DFM_model, search_pattern)
    for partition in full_patterns:
        output_file = os.path.join(outdir,fr'{year}',fr'{partition}_processed.nc')
        if os.path.exists(output_file):
            print(f"File {output_file} already exists. Skipping.")
        else:
            print(f"Running {year} and {partition}.")
   
            f = os.path.join(DFM_model, fr"{partition}.nc")          # adds a .map file to the base directory
            # DFM_xr = dfmt.open_partitioned_dataset(f.replace('_0000_','_0*_'))    # opens every .map file in this folder
            # DFM_xr = dfmt.open_partitioned_dataset(f.replace('_0000_','_0*_'), remove_edges=False, remove_ghost=False, chunks = "auto")    # opens every .map file in this folder
            DFM_xr = dfmt.open_partitioned_dataset(f, remove_edges=False, remove_ghost=False, chunks = "auto")    # opens one partition
            DFM_xr = dfmt.rename_waqvars(DFM_xr)  
            
            # Select variables -- new:
            pattern = r'^mesh2d_MEAN_.*_(s1|Chlfa|NO3|OXY|pH|PO4|pCO2water)$'
            selected_vars = [var for var in DFM_xr.data_vars if re.match(pattern, var)]
            
            #Variabes to keep
            DFM_vars_basic = ['mesh2d_taus', 'mesh2d_tausx', 'mesh2d_tausy', 'mesh2d_ucx', 'mesh2d_ucy', 
                              "mesh2d_s1", "mesh2d_bldepth", 'mesh2d_flowelem_zw', 'mesh2d_flowelem_bl'] 
                        
            #Subset variables
            DFM_xr = DFM_xr[selected_vars + DFM_vars_basic]
            
            # Convert to float32
            DFM_xr = DFM_xr.astype('float32')
            # DFM_rasterized['mesh2d_layer_sigma_z'] = DFM_rasterized['mesh2d_layer_sigma_z'].astype('float32')
                                       
            print(f"Combining variables for {year} and {partition}.")
            # For new DFM, combine variables and add time dim:
            # Initialize a dictionary to store combined data arrays by parameter (e.g., 'OXY', 'PO4')
            combined_data = {}
    
            # Loop through each variable in the dataset
            for var_name in DFM_xr[selected_vars]:
                # Extract month and parameter from variable name
                parts = var_name.split('_')
                month_str, parameter = parts[2], fr'mesh2d_{parts[3]}'
    
                # Map month strings to datetime objects for January 2014, February 2014, etc.
                month_map = {
                    'jan': f'{year}-01', 'feb': f'{year}-02', 'mar': f'{year}-03',
                    'apr': f'{year}-04', 'may': f'{year}-05', 'jun': f'{year}-06',
                    'jul': f'{year}-07', 'aug': f'{year}-08', 'sep': f'{year}-09',
                    'oct': f'{year}-10', 'nov': f'{year}-11', 'dec': f'{year}-12'
                }
                time_val = pd.to_datetime(month_map[month_str])
    
                # Expand variable with new time dimension
                expanded_var = DFM_xr[var_name].expand_dims(time=[time_val])
    
                # Add this variable to the combined dictionary under its parameter key
                if parameter in combined_data:
                    combined_data[parameter].append(expanded_var)
                else:
                    combined_data[parameter] = [expanded_var]
    
            # Concatenate each parameter's variables along the new time dimension
            new_vars = {param: xu.concat(variables, dim="time") for param, variables in combined_data.items()}

            new_uds = xu.UgridDataset(grids=DFM_xr.ugrid.grids)        
            
            new_uds.attrs = DFM_xr.attrs  # Keep the original dataset's attributes

            for name, da in new_vars.items():
                new_uds[name] = da
                
            for name, da in DFM_xr[DFM_vars_basic].items():
                new_uds[name] = da
                
            new_uds['mesh2d_flowelem_zw'] = DFM_xr['mesh2d_flowelem_zw']
                        
            # Save
            # Save the dataset to a NetCDF file with Dask
            #ds_to_save = rechunk(new_uds)                  
            print(f"Saving {year} and partition {partition}.")        
            #ds_to_save.ugrid.to_netcdf(output_file, engine='netcdf4', compute=True)
            new_uds.ugrid.to_netcdf(output_file)
            print(" ")