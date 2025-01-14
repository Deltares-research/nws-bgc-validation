#!/usr/bin/env python
# coding: utf-8


## Import packages
import os
import re
import numpy as np
import xarray as xr
import pandas as pd
import scipy.interpolate
import dfm_tools as dfmt
import dask


## Functions:

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
    
def years_in_order(start_year, end_year):
    return list(range(start_year, end_year + 1))
#%%

# Choose the model domain
min_lon = -20 
max_lon = 13
min_lat = 48
max_lat = 62

# Filepaths
basedir = r'P:\11209810-cmems-nws\model_output' if os.name == 'nt' else r'/p/11209810-cmems-nws/model_output'
rootdir = os.path.join(basedir, 'combined_yearly')
outdir = os.path.join(basedir, 'regridded_onto_NWS')

# Selection:
model = 'nrt' # 'rea' or 'nrt'   

start_year = 2020
end_year = 2020
slice_2d = 'surface'  # surface / bottom
office = 'IBI'

selected_years = years_in_order(start_year, end_year)

#%%# Interpolation grid
if model == 'rea':
    filei = os.path.join(basedir, 'combined_yearly', fr'surface_NWS_{model}_2003.nc')
elif model == 'nrt':
    filei = os.path.join(basedir, 'combined_yearly', fr'surface_NWS_{model}_2021.nc')
nci = xr.open_dataset(filei)

try:
    # Select the subset based on longitude and latitude slices
    nci = nci.sel(longitude=slice(min_lon, max_lon), latitude=slice(min_lat, max_lat))
except NameError:
    # Handle the case where min_lon is not defined
    print("Bounding box slicing for IBI is not applied.")

# In[5]:
if office == 'satellite':
    ### Satellite ### -- Keep in structure as NWS also kept in structure... Just check nearest interpolation method...
    
    # Read file
    rootdir_sat = r'P:\11209810-cmems-nws\Data\OCEANCOLOUR_ATL_BGC_L4_MY_009_118' if os.name == 'nt' else r'/p/11209810-cmems-nws/Data/OCEANCOLOUR_ATL_BGC_L4_MY_009_118'
    
    # Selection:
    for year in selected_years:
        output_file = os.path.join(outdir, fr'regridded_satellite_{year}.nc')
        if os.path.exists(output_file):
            print(f"File {output_file} already exists. Skipping.")
        else:
            print(f"Running {output_file}.")
            satellite_xr = xr.open_dataset(os.path.join(rootdir_sat, f'cmems_obs-oc_atl_bgc-plankton_my_l4-gapfree-multi-1km_P1D_CHL_19.98W-12.98E_48.02N-61.98N_{year}-01-01-{year}-12-31.nc'))
            try:
                satellite_xr = satellite_xr.sel(longitude=slice(min_lon,max_lon),latitude=slice(min_lat,max_lat)) # crop domain - not necessary as downloaded correctly                                                                 # select only CHL as variable
            except NameError:
                # Handle the case where min_lon is not defined
                print("Bounding box slicing for satelite is not applied.")
            
            satellite_xr = satellite_xr['CHL'] 
            # Regrid (structured to structured, so can just interpolate)
            regridded_sat = satellite_xr.interp(latitude=nci.latitude, longitude=nci.longitude, method="nearest")  # bi-linear possible?
            
            # Save
            # Save the dataset to a NetCDF file with Dask
            regridded_sat = rechunk(regridded_sat)
            regridded_sat.to_netcdf(output_file, engine='netcdf4', compute=True)
            print(f'Done for regridded_{office}_{year}') 


### IBI: rasterize = ugrid to structured, regrid = ugrid to urgid. Both should work.

# Selection:
if office == 'IBI' or office == 'NWS':

    for year in selected_years:
        output_file = os.path.join(outdir,fr'regridded_{slice_2d}_{office}_{year}.nc')
        if os.path.exists(output_file):
            print(f"File {output_file} already exists. Skipping.")
        else:
            print(' ')
            print(f"Running regridded_{slice_2d}_{office}_{year}.")
            #file_nc = dfmt.open_partitioned_dataset(fr'P:\11209810-cmems-nws\model_output\combined_yearly\{slice_2d}_{office}_{model}_{year}.nc')
            
            # Convert to ugrid and save again: Note: ugrid makes the files a bit larger, but not by much!
            uds = dfmt.open_dataset_curvilinear(file_nc=os.path.join(rootdir, fr'{slice_2d}_{office}_{model}_{year}.nc'), 
                                                varn_vert_lon='vertices_longitude', #'grid_x'
                                                varn_vert_lat='vertices_latitude', #'grid_y'
                                                ij_dims=['x','y'], #['N','M']
                                                preprocess=derive_verts)
        
            # rename interp grid
            if 'latitude' in nci.dims:
                nci = nci.rename({'latitude':'y', 'longitude':'x'})  # rename nci to y,x, IF NWS is kept as structured grid...
            
            #Rasterize
            print('Rasterizing')
            ds = uds.ugrid.rasterize_like(other=nci)  # expects nci to have x,y coords... 
            
            # Dropping unnecessary variables
            try:
                ds_dropped = ds.drop_vars(['vertices_longitude', 'vertices_latitude'])
            except Exception:
                ds_dropped = ds
            
            #ds.ugrid.to_netcdf(os.path.join(outdir,fr'regridded_IBI_{year}.nc'))  # gives an error...
            print('Saving')  
            # Save the dataset to a NetCDF file with Dask
            ds_dropped = rechunk(ds_dropped)
            ds_dropped.to_netcdf(output_file, engine='netcdf4', compute=True)
            
            ## Skip bi-linear stuff for now and keep nearest for simplicity...
            print(f'Done for regridded_{slice_2d}_{office}_{year}') 


if office == 'DFM':
    ### DFM ###
    if slice_2d == 'surface':
        layer=-1   # Have to rewrite!
    elif slice_2d == 'bottom':
        layer = 'bedlevel'          # bedlevel / waterlevel
    
    vars_list = ['mesh2d_Chlfa', 'mesh2d_NO3', 'mesh2d_OXY', 'mesh2d_pH', 'mesh2d_PO4', 'mesh2d_pCO2water']
    
    # Interpolation grid - read above
    if 'latitude' in nci.dims:
        nci = nci.rename({'latitude':'y', 'longitude':'x'})  # rename nci to y,x
    
    for year in selected_years:
        output_file = os.path.join(outdir,fr'regridded_{slice_2d}_{office}_{year}.nc')
        if os.path.exists(output_file):
            print(f"File {output_file} already exists. Skipping.")
        else:
            print(f"Running regridded_{slice_2d}_{office}_{year}.")
            #DFM_model folder
            base_path = r'p:\11210370-011-nutrientwad\04_simulations\waq_runs' if os.name == 'nt' else r'/p/11210370-011-nutrientwad/04_simulations/waq_runs'
            for folder in os.listdir(base_path):
                if folder.startswith(f'waq_{year}'):  # Matches folders starting with waq_{year}
                    DFM_model = os.path.join(base_path, folder, 'DFM_OUTPUT_DCSM-FM_0_5nm_waq')
            # old format:
            # DFM_model = fr'P:\archivedprojects\11209731-002-nutrient-reduction-ta\runs_OSPAR\B05_waq_withDOM_{year}\DFM_OUTPUT_DCSM-FM_0_5nm_waq' if os.name == 'nt' else fr'/p/archivedprojects/11209731-002-nutrient-reduction-ta/runs_OSPAR/B05_waq_withDOM_{year}/DFM_OUTPUT_DCSM-FM_0_5nm_waq'
            # new format:
            # DFM_model = fr'P:\11210284-011-nose-c-cycling\runs_fine_grid\B05_waq_{year}_PCO2_ChlC_NPCratios_DenWat_stats_2023.01\DFM_OUTPUT_DCSM-FM_0_5nm_waq' if os.name == 'nt' else fr'/p/11210284-011-nose-c-cycling/runs_fine_grid/B05_waq_{year}_PCO2_ChlC_NPCratios_DenWat_stats_2023.01/DFM_OUTPUT_DCSM-FM_0_5nm_waq'       
            # processed:
            # DFM_model = os.path.join(basedir, 'DFM_processed', str(year))
    
    
            f = os.path.join(DFM_model, r"DCSM-FM_0_5nm_waq_0000_map.nc")          # adds a .map file to the base directory
            print(r"Opening and merging with dfm_tools.")
            DFM_xr = dfmt.open_partitioned_dataset(f.replace('_0000_','_0*_'), remove_edges=False, remove_ghost=False, chunks = "auto")    # opens every .map file in this folder
            # DFM_xr = dfmt.open_partitioned_dataset(f, remove_edges=False, remove_ghost=False, chunks = "auto")    # opens one partition
            DFM_xr = dfmt.rename_waqvars(DFM_xr)  
            
            print(r"Cropping bounding box, layer, variables.")   
            # Select top/bottom layer
            if slice_2d == 'surface':
                DFM_crop = DFM_xr.isel(mesh2d_nLayers=layer, nmesh2d_layer=layer, missing_dims='ignore')  # old
            elif slice_2d == 'bottom':
                DFM_crop = dfmt.get_Dataset_atdepths(data_xr=DFM_xr, depths=0, reference=layer)
                    
            # Crop spatial domain
            try:
                DFM_crop = DFM_crop.ugrid.sel(x=slice(min_lon,max_lon),y=slice(min_lat,max_lat))
            except NameError:
                # Handle the case where min_lon is not defined
                print("Bounding box slicing for DFM is not applied.")
             
            # Select variables:
            DFM_crop = DFM_crop[vars_list]
    
            print(r"Cropped bounding box, layer, variables.") 
                    
            print(f"Rasterizing regridded_{slice_2d}_{office}_{year}.")
            # Rasterize / regrid onto NWS
            DFM_rasterized = dfmt.rasterize_ugrid(DFM_crop, ds_like=nci)   
            
            print(f"Rasterized regridded_{slice_2d}_{office}_{year}.")
                   
            #Drop unnecessary coordinates
            DFM_rasterized = DFM_rasterized.drop_vars(['mesh2d_face_x', 'mesh2d_face_y', 'mesh2d_nFaces'])
                        
            # Save
            # Save the dataset to a NetCDF file with Dask
            ds_to_save = rechunk(DFM_rasterized)
    
            print(f"Saving regridded_{slice_2d}_{office}_{year}.")        
            ds_to_save.to_netcdf(output_file, engine='netcdf4', compute=True)
            #DFM_rasterized.to_netcdf(output_file)
            print(f"Saved regridded_{slice_2d}_{office}_{year}.")
            print(" ")
