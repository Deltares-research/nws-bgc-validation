#!/usr/bin/env python
# coding: utf-8

# ## Preparation

# In[1]:

## Import packages

import os
import numpy as np
import xarray as xr
import xugrid as xu
import pandas as pd
import dfm_tools as dfmt
import dask
from metpy.interpolate import cross_section  # for NWS transect extraction
from geopy.distance import geodesic

# In[2]:


# ## Choose the model domain  - keep full for now.

# min_lon = -19.89 
# max_lon = 13
# min_lat = 40.07
# max_lat = 65


# In[3]:


outdir = r'p:\11209810-cmems-nws\model_output\combined_yearly' if os.name == 'nt' else r'/p/11209810-cmems-nws/model_output/combined_yearly'


# In[4]:


## Functions:

import datetime as dt

def open_dataset_curvilinear(ds,
                             varn_lon='longitude',
                             varn_lat='latitude',
                             varn_vert_lon='vertices_longitude', #'grid_x'
                             varn_vert_lat='vertices_latitude', #'grid_y'
                             ij_dims=['i','j'], #['N','M']
                             convert_360to180=False,
                             **kwargs):
    """
    This is a first version of a function that creates a xugrid UgridDataset from a curvilinear dataset like CMCC. Curvilinear means in this case 2D lat/lon variables and i/j indexing. The CMCC dataset does contain vertices, which is essential for conversion to ugrid.
    It also works for WAQUA files that are converted with getdata
    """
    # TODO: maybe get varn_lon/varn_lat automatically with cf-xarray (https://github.com/xarray-contrib/cf-xarray)
    
    if 'chunks' not in kwargs:
        kwargs['chunks'] = {'time':1}
    
    # data_vars='minimal' to avoid time dimension on vertices_latitude and others when opening multiple files at once
    # ds = xr.open_mfdataset(file_nc, data_vars="minimal", **kwargs)
    
    print('>> getting vertices from ds: ',end='')
    dtstart = dt.datetime.now()
    vertices_longitude = ds.variables[varn_vert_lon].to_numpy()
    vertices_longitude = vertices_longitude.reshape(-1,vertices_longitude.shape[-1])
    vertices_latitude = ds.variables[varn_vert_lat].to_numpy()
    vertices_latitude = vertices_latitude.reshape(-1,vertices_latitude.shape[-1])
    print(f'{(dt.datetime.now()-dtstart).total_seconds():.2f} sec')
    
    # convert from 0to360 to -180 to 180
    if convert_360to180:
        vertices_longitude = (vertices_longitude+180) % 360 - 180
    
    # face_xy = np.stack([longitude,latitude],axis=-1)
    # face_coords_x, face_coords_y = face_xy.T
    #a,b = np.unique(face_xy,axis=0,return_index=True) #TODO: there are non_unique face_xy values, inconvenient
    face_xy_vertices = np.stack([vertices_longitude,vertices_latitude],axis=-1)
    face_xy_vertices_flat = face_xy_vertices.reshape(-1,2)
    uniq,inv = np.unique(face_xy_vertices_flat, axis=0, return_inverse=True)
    #len(uniq) = 104926 >> amount of unique node coords
    #uniq.max() = 359.9654541015625 >> node_coords_xy
    #len(inv) = 422816 >> is length of face_xy_vertices.reshape(-1,2)
    #inv.max() = 104925 >> node numbers
    node_coords_x, node_coords_y = uniq.T
    
    face_node_connectivity = inv.reshape(face_xy_vertices.shape[:2]) #fnc.max() = 104925
    
    #remove all faces that have only 1 unique node (does not result in a valid grid) #TODO: not used yet except for print
    fnc_all_duplicates = (face_node_connectivity.T==face_node_connectivity[:,0]).all(axis=0)
    
    #create bool of cells with duplicate nodes (some have 1 unique node, some 3, all these are dropped) #TODO: support also triangles?
    fnc_closed = np.c_[face_node_connectivity,face_node_connectivity[:,0]]
    fnc_has_duplicates = (np.diff(fnc_closed,axis=1)==0).any(axis=1)
    
    #only keep cells that have 4 unique nodes
    bool_combined = ~fnc_has_duplicates
    print(f'WARNING: dropping {fnc_has_duplicates.sum()} faces with duplicate nodes ({fnc_all_duplicates.sum()} with one unique node)')
    face_node_connectivity = face_node_connectivity[bool_combined]
    
    grid = xu.Ugrid2d(node_x=node_coords_x,
                      node_y=node_coords_y,
                      face_node_connectivity=face_node_connectivity,
                      fill_value=-1,
                      )
    
    print('>> stacking ds i/j coordinates: ',end='') #fast
    dtstart = dt.datetime.now()
    face_dim = grid.face_dimension
    # TODO: lev/time bnds are dropped, avoid this. maybe stack initial dataset since it would also simplify the rest of the function a bit
    ds_stacked = ds.stack({face_dim:ij_dims}).sel({face_dim:bool_combined})
    #latlon_vars = [varn_lon, varn_lat, varn_vert_lon, varn_vert_lat]
    latlon_vars = [varn_vert_lon, varn_vert_lat]
    ds_stacked = ds_stacked.drop_vars(ij_dims + latlon_vars)
    print(f'{(dt.datetime.now()-dtstart).total_seconds():.2f} sec')
    
    print('>> init uds: ',end='') #long
    dtstart = dt.datetime.now()
    uds = xu.UgridDataset(ds_stacked,grids=[grid])
    print(f'{(dt.datetime.now()-dtstart).total_seconds():.2f} sec')
    
    # drop 0-area cells (relevant for CMCC global datasets)
    bool_zero_cell_size = uds.grid.area==0
    if bool_zero_cell_size.any():
        print(f"WARNING: dropping {bool_zero_cell_size.sum()} 0-sized cells from dataset")
        uds = uds.isel({uds.grid.face_dimension: ~bool_zero_cell_size})
    
    #remove faces that link to node coordinates that are nan (occurs in waqua models)
    bool_faces_wnannodes = np.isnan(uds.grid.face_node_coordinates[:,:,0]).any(axis=1)
    if bool_faces_wnannodes.any():
        print(f'>> drop {bool_faces_wnannodes.sum()} faces with nan nodecoordinates from uds: ',end='') #long
        dtstart = dt.datetime.now()
        uds = uds.sel({face_dim:~bool_faces_wnannodes})
        print(f'{(dt.datetime.now()-dtstart).total_seconds():.2f} sec')
    
    return uds

def combine_year_per_IBI_variable(rootdir, variable, year):
    directory = rootdir
    fps = []
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            if ('_'+year) in file and variable in file:
                fps.append(os.path.join(subdir, file))
    ds = xr.open_mfdataset(fps, data_vars="minimal")
    #ds = xr.open_mfdataset(fps)
    return ds

def combine_year_per_NWS_variable(rootdir, variable, year, model):
    fps = []
    
    if model == 'nrt':
        directory = os.path.join(rootdir, f'cmems_mod_nws_bgc-{variable}_anfc_7km-3D_P1D-m', year)  # for nrt = anfc
    elif model == 'rea':
        directory = os.path.join(rootdir, f'cmems_mod_nws_bgc-{variable}_my_7km-3D_P1D-m', year)    # for rea = my
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
                depth     = (["depth"], ds.deptht.data, ds.deptht.attrs)),
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
                depth     = (["depth"], ds.deptht.data, ds.deptht.attrs)),
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

#Norway1: 10849 (-0.64, 60.8) -  13786 (4.64,60.8) 
#Norway2: 9950 (-2.24, 59.36) -  13955 (4.96,59.36)   
#Denmark: 10024 (-2.08, 56.96) -  15809 (8.32,56.96)
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
         'y2': 56.96}
}

# In[5]:
## Combine the .nc files into yearly files

# Selection:
offices = ['NWS', 'IBI'] #'IBI' 'NWS' 'DFM'
model = 'nrt' # 'rea' or 'nrt'   
slice_2d = 'transect'
start_year = 2020
end_year = 2023
selected_years = years_in_order(start_year, end_year)
variables = ['o2', 'chl', 'no3', 'ph', 'po4'] 
variables_DFM = ['mesh2d_Chlfa', 'mesh2d_NO3', 'mesh2d_OXY', 'mesh2d_pH', 'mesh2d_PO4']
# IBI Available: bgc (cflux, spco2, zeu, npp), alk, chl, dic, nh4, no3, o2, ph, po4, si. Not use bgc (except npp) and alk.
# NWS: pp gets name: nppv. Available: chl, kd, no3, o2, ph, phyc, po4, pp, spco2. kd and spco2 not in validation proposal (leave for now). 
# DFM: 'mesh2d_Chlfa', 'mesh2d_NO3', 'mesh2d_OXY', 'mesh2d_pH', 'mesh2d_PO4'

# Choose transects
transects = ['NORWAY1','NORWAY2','DENMARK', 'NOORDWK', 'TERSLG', 'ROTTMPT', 'WALCRN'] 


# read a database file
obs_dir = r'P:\11209810-cmems-nws\Data\NWDM_observations\combined_years\2003_2017_Chlfa_obs_surface.csv' if os.name == 'nt' else r'/p/11209810-cmems-nws/Data/NWDM_observations/combined_years/2003_2017_Chlfa_obs_surface.csv' 
obs = pd.read_csv(obs_dir)
  
obs['datetime'] = pd.to_datetime(obs['datetime'], format='mixed', dayfirst = False)

np.unique(obs['station'].astype(str).values)  # use to identify max and min locs

# In[]:  
for trans in transects:
    if trans in ['NORWAY1','NORWAY2','DENMARK']:
        x1, y1 = trans_dict[trans]['x1'], trans_dict[trans]['y1']
        x2, y2 = trans_dict[trans]['x2'], trans_dict[trans]['y2']
    else:
        trans_st = pd.DataFrame(obs[obs['station'].str.startswith(trans, na=False)].station)
        
        # Extract numeric values at the end of each station name
        trans_st['number'] = trans_st['station'].str.extract(r'(\d+)$').astype(int)
        
        # Find the rows with the minimum and maximum values
        min_loc = trans_st.loc[trans_st['number'].idxmin(), 'station']
        max_loc = trans_st.loc[trans_st['number'].idxmax(), 'station']
    
        loc_min = obs[obs['station'].str.endswith(min_loc, na=False)]#.geom.iloc[0]
        loc_max = obs[obs['station'].str.endswith(max_loc, na=False)]#.geom.iloc[0]
        
        x1, y1 = float(loc_min.geom.iloc[0].split(' ')[1:][0][1:]), float(loc_min.geom.iloc[0].split(' ')[1:][1][:-1])
        x2, y2 = float(loc_max.geom.iloc[0].split(' ')[1:][0][1:]), float(loc_max.geom.iloc[0].split(' ')[1:][1][:-1])
    
    line_array = np.array([[x1,y1], [x2,y2]])
    # line_array = np.array([[-1.8,49.0], [-1.9,49.4]])
    
    for office in offices:
        for year in selected_years:    
            year=str(year)
            # Define the output file path
            output_file = os.path.join(outdir, fr'{slice_2d}_{trans}_{office}_{model}_{year}.nc')
            if os.path.exists(output_file):
                print(f"File {output_file} already exists. Skipping.")
            else:
                print(f'Running {office}_{model}_{year}')
                rootdir =  fr'p:\11209810-cmems-nws-data-{model}\{office}' if os.name == 'nt' else fr'/p/11209810-cmems-nws-data-{model}/{office}'
              
                if office == 'NWS' or office == 'IBI':
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
                        
                        if office == 'IBI':
                            ds = derive_verts(ds)
                            # Convert to ugrid
                            uds = open_dataset_curvilinear(ds=ds, 
                                                                varn_vert_lon='vertices_longitude', #'grid_x'
                                                                varn_vert_lat='vertices_latitude', #'grid_y'
                                                                ij_dims=['x','y'], #['N','M']
                                                                preprocess=derive_verts)
                        
                            # Extract transect
                            ## Add the necessary dimensions for the vertical slicing:
                            zeros = np.array([0] * len(uds.mesh2d_nFaces))
                            uds['mesh2d_s1'] = (('mesh2d_nFaces'), zeros)
                            uds['mesh2d_flowelem_bl'] = (('mesh2d_nFaces'), zeros)
                        
                            #loop over days
                            ds_trans_day_ds = []
                            for ts in range(0,len(ds.time)):
                                ds_trans_day = dfmt.polyline_mapslice(uds.isel(time=ts), line_array)  # specific time stamp
                                # invert the depth values
                                ds_trans_day['depth'] = -ds_trans_day.depth  
                                
                                # Append to datasets to later merge:
                                ds_trans_day_ds.append(ds_trans_day)
            
                            ds_trans=xu.concat(ds_trans_day_ds, dim ='time')
                            
                            
                        elif office == 'NWS':
                            # add crs coordinate and extract transect
                            with_crs = ds.metpy.parse_cf()  # keeping all the vars
        
                            # extract cross-section
                            #steps = (trans_st['number'].max() - trans_st['number'].min())/1 # steps [in km] = (last station number [in km] - first station number [in km]) / 1 [in km]. 
                            # Calculate the distance
                            steps = round(geodesic((y1,x1), (y2,x2)).kilometers/1)  # steps [in km]
                            ds_trans = cross_section(with_crs, [y1,x1],[y2, x2], steps=steps, interp_type='linear').set_coords(('longitude', 'latitude'))
                            ds_trans['depth'] = -ds_trans.depth              # make depths negative
                            ds_trans = ds_trans.drop_vars('metpy_crs')
                                                
                        # Append to datasets to later merge:
                        datasets.append(ds_trans)
                
                # Make sure each dataset has the correct date format (based on NWS - midday):
                if office == 'IBI':
                    updated_datasets = []
                    for nr in range(0,len(datasets)):
                        datasets[nr]['time'] = datasets[nr].indexes['time'].normalize() - pd.DateOffset(hours=12)
                        updated_datasets.append(datasets[nr])
                    print('Correcting date format')
    
                if office == 'DFM' and 2014 <= int(year) <= 2017:  # add DFM only for reanalysis!]:
                    base_path = fr'p:\11210370-011-nutrientwad\04_simulations\waq_runs'
                    #base_path = fr'P:\archivedprojects\11209731-002-nutrient-reduction-ta\runs_OSPAR'
                    for folder in os.listdir(base_path):
                        if folder.startswith(f'waq_{year}'):  # Matches folders starting with waq_{year}
                        #if folder.startswith(f'B05_waq_withDOM_{year}'):  # Matches folders starting with waq_{year}
                            DFM_model = os.path.join(base_path, folder, 'DFM_OUTPUT_DCSM-FM_0_5nm_waq')
                    #DFM_model = fr'P:\archivedprojects\11209731-002-nutrient-reduction-ta\runs_OSPAR\B05_waq_withDOM_{year}\DFM_OUTPUT_DCSM-FM_0_5nm_waq'
                    #DFM_model = fr"p:\11209810-cmems-nws\model_output\DFM_processed\{year}" if os.name == 'nt' else fr'/p/11209810-cmems-nws/model_output/DFM_processed/{year}'
                    #f = os.path.join(DFM_model, r"DCSM-FM_0_5nm_waq_0000_map_processed.nc")        # adds a .map file to the base directory
                    f = os.path.join(DFM_model, r"DCSM-FM_0_5nm_waq_0000_map.nc")        # adds a .map file to the base directory
                    ds = dfmt.open_partitioned_dataset(f.replace('_0000_','_0*_'), remove_edges=False, remove_ghost=False, chunks = "auto")    # opens every .map file in this folder
                    #ds = dfmt.open_partitioned_dataset(os.path.join(DFM_model, r"DCSM-FM_0_5nm_waq_0255_map_processed.nc"), chunks = "auto")    # one partition
                    ds = dfmt.rename_waqvars(ds)  
                                       
                    # Extract transect               
                    #loop over days
                    print('Extracting transect')
                    ds_trans_day_ds = []
                    for ts in range(0,len(ds.time)):
                        ds_trans_day = dfmt.polyline_mapslice(ds.isel(time=ts), line_array)  # specific time stamp
    
                        # Append to datasets to later merge:
                        ds_trans_day_ds.append(ds_trans_day)
    
                    ds_trans=xu.concat(ds_trans_day_ds, dim ='time')
                      
                else:
                    print('Selected year is not valid for DFM')
            
                # Merge and save:
                try: 
                    print('Merging and Saving')
                    if office == 'IBI':
                        ds_merge = xu.merge(updated_datasets, compat='override')
                        #save_dataset_with_dask(ds_merge, output_file) # Save, to get path to use below
                        ds_merge.ugrid.to_netcdf(output_file)
                    elif office == 'DFM':
                        DFM_vars_basic = ['mesh2d_ucx', 'mesh2d_ucy', 
                      "mesh2d_s1", "mesh2d_bldepth","mesh2d_flowelem_zw", "mesh2d_flowelem_zcc", 'mesh2d_taus'] #'mesh2d_taus', 'mesh2d_tausx', 'mesh2d_tausy', 
                        
                        #ds_merge = ds_trans
                        ds_merge = ds_trans[variables_DFM + DFM_vars_basic]
                        #save_dataset_with_dask(ds_merge, output_file) # Save, to get path to use below
                        ds_merge.ugrid.to_netcdf(output_file)
                    else:            
                        ds_merge = xr.merge(datasets) # For NWS, not necessary to save the in-between set!
                        save_dataset_with_dask(ds_merge, output_file) # Save, to get path to use below
                        #ds_merge.to_netcdf(output_file)
          
                    print("=================================")            
                    print(fr'Done for {trans}_{office}_{model}_{year}')
                    print("=================================")
                except NameError:
                    print('Skipping this year')
                
    

