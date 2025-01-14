#!/usr/bin/env python
# coding: utf-8

# ## Import Packages

# In[1]:


import io
import os

import numpy as np
import pandas as pd
import urllib
import requests
from requests.auth import HTTPBasicAuth

import datetime as dt
import geopandas as gpd
from shapely.geometry import Point, box

#import matplotlib.dates as mdates
#import cartopy
#import cartopy.crs as ccrs
#import matplotlib.pyplot as plt
#from shapely.geometry import Point


# ## Functions

# In[2]:


def wfsbuild(typename, cql_dict = {}, geom_dict= {},  
             outputFormat = "csv", maxFeatures = None, columns = None):
    '''
    Build url to download data from geoserver.

    Parameters
    ----------
    typename : str
        WFS layer e.g. "NWDM:location".
    cql_dict : dict, optional
        dictionary with cql conditions provide column(s) and value. 
        The default is {}.
    geom_dict : dict, optional
        dictionary with geometry filter information. 
        The default is {}.
        three options are possible:
            filter on geometry: {intersects: {geom: geomtery}}
            filter on geometry + buffer area: {dwithin: {geom: geometry, 
                                                    distance: float, 
                                                    unit: str}}
            filter on bounding box: {bbox: {xmin:, ymin:, xmax:, ymax:}}
    outputFormat : str, optional
        desired format for download. The default is "csv".
    maxFeatures : int, optional
        maximum number of features to download. The default is None.
    columns : list, optional
        desired columns to download. The default is None.

    Returns
    -------
    url: str
        url to download data from geoserver.
    '''
    
    # URL of the geoserver
    baseurl = "https://nwdm.openearth.eu/geoserver/NWDM/ows?service=WFS&"
    
    # filters
    if columns == None:
        propertyName = None 
    else:
        propertyName = ",".join([f"MWDM:{c}" for c in columns])
  
    if len(cql_dict) == 0. and len(geom_dict) == 0.:
        cql_filter = None

    else:
        # list geometry filters
        geom_list = [f"{k}(geom,{v['geom']})" if k == 'intersects'\
                    else f"dwithin(geom, {v['geom']}, {v['distance']}, {v['unit']})" if k == 'within' \
                    else f"bbox(geom, {v['xmin']}, {v['ymin']}, {v['xmax']}, {v['ymax']})" if k == 'bbox' \
                    else " " for (k,v) in geom_dict.items()] 
        
        # list cql filters
        cql_list = [ "(" +' OR '.join([f"{column}='{v}'" for v in value]) + ")" if type(value) == list else f"{column}='{value}'" for column, value in cql_dict.items()]
               
        # combine lists
        geom_cql_list = geom_list + cql_list
        
        # create cql filter string 
        cql_filter = "(" + " and ".join(geom_cql_list)+")"
        print(cql_filter)
        
    # build query  
    query = { "version":"1.0.0", 
              "request":"GetFeature", 
              "typeName":typename,
              "cql_filter":cql_filter,
              "outputFormat":outputFormat, 
              "maxFeatures":maxFeatures,
              "propertyName":propertyName}        
    
    query = {k:v for k, v in query.items() if v is not None}

    return baseurl + urllib.parse.urlencode(query, quote_via=urllib.parse.quote) 


# To read NWDM data from url 
def readUrl(url, user, password):
    """
    Extract data as pandas dataframe from geoserver using url.

    Parameters
    ----------
    url : str
        url to access geoserver.
    user : str
        name of the user.
    password : str
        password for that user.

    Returns
    -------
    nwdmData_raw : pandas dataframe
        dataframe from geoserved extracted based on url.

    """
    s = requests.get(url, auth=HTTPBasicAuth(user, password))
    nwdmData_raw = pd.read_csv(io.StringIO(s.content.decode('utf-8')))
    
    return nwdmData_raw

def years_in_order(start_year, end_year):
    return list(range(start_year, end_year + 1))


#%%
#paraeter dictionary
parameter_to_p35code = {
    'Chlfa': 'EPC00105',
    'PO4': 'EPC00007',
    'salinity': 'EPC00001',
    'temperature': 'WATERTEMP',
    'NO3': 'EPC00004',
    'NH4': 'EPC00009',
    'Si': 'EPC00008',
    'OXY': 'EPC00002',
    'SS': 'EXTRA004',
    'POC': 'EPC00157',
    'PON': 'EPC00212',
    'POP': 'EPC00201',
    'TotN': 'EPC00134',
    'TotP': 'EPC00135',
    'DOC': 'EPC00190',
    'fPPtot': 'INPPPIC1',
    'pH': 'EPC00168',
    'pCO2': 'EPC00133'
}

## Settings

user = 'nwdm'
password = 'tkofschip'

# WFS layer
typename_sts = "NWDM:location"
typename_data = "NWDM:measurement_p35_all"

# list columns
columns = ["location_code", "date", "depth", "vertical_reference_preflabel",
            "vertical_reference_code", "p35code", "p35preflabel", 
            "value", "unit_preflabel", "quality_code", "station", "geom", "data_owner"]

#%%

##INPUT
#output directory
outdir = r'P:\11209810-cmems-nws\Data\NWDM_observations' if os.name == 'nt' else r'/p/11209810-cmems-nws/Data/NWDM_observations'

#NWDM statistics. It prints a summary of the NWDM dataset.
NWDM_stats=False

# set time frame 
start_year = 2003
end_year = 2022

#gridded or not, if yes, define bounding box
gridded = True #True - yes, False - no
geom_dict={'bbox': {'xmin': -20.0, 'ymin': 48, 'xmax': 13, 'ymax': 62}} #bounding box coordinates
# geom_dict={'bbox': {'xmin': 3.0, 'ymin': 48, 'xmax': 6, 'ymax': 62}} #bounding box coordinates
resolution = 0.16 #choose resolution in degrees
#Buffer for surface observations
buffer = 10.5 # (meters) take all measuerements from the top x meters

#choose vertical reference system
depths = ['surface','bottom'] #Choose from: surface, bottom
#choose variale
#parameters = ['NO3', 'PO4', 'pH', 'OXY', 'Chlfa']#, 'pCO2'] #
parameters = ['PO4']#['pCO2']

##################
#Available parameters
#[Chlfa, PO4, NO3, pH, OXY, pCO2, NH4, Si, SS, POC, PON, POP, TotN, TotP, DOC, fPPtot, , salinity, temperature]
##################

# Choose locations:  - if leave blank, extracts all!!

# single loc
# stationid = 'NOORDWK2'  # if want 1 station. Find from e.g.: https://wstolte.github.io/nwdm/workflow-description.html

# list of locs
# plot_locs = ['NOORDWK2','NOORDWK10','NOORDWK20','NOORDWK30','NOORDWK50','NOORDWK70',
#               'ROTTMPT3','ROTTMPT10','ROTTMPT100','ROTTMPT15','ROTTMPT20','ROTTMPT30','ROTTMPT50',
#               'ROTTMPT70', 'TERSLG10','TERSLG100', 'TERSLG135','TERSLG175','TERSLG20','TERSLG235',
#               'TERSLG30', 'TERSLG50','TERSLG70','WALCRN2','WALCRN10','WALCRN20','WALCRN30',
#               'WALCRN50','WALCRN70']

# area
# geometry (EPSG:4326)
# station_geom = Point(5.099480234957902, 53.46036612855593)  # if want to choose specific location

# bbox (EPSG:4326)
# bbox = {'xmin':-10., 'xmax':12.,
#         'ymin':45., 'ymax':48.}  # if want to choose specific area

selected_years = years_in_order(start_year, end_year)

#%%

if NWDM_stats:
    ## Build urls to import overview of stations
    url_sts = wfsbuild(typename = "NWDM:location", outputFormat = "csv")
    obs_sts_info = readUrl(url_sts, user, password)
    obs_sts_info.to_csv(os.path.join(outdir, r'NWDM_station_stats.csv') )
    #Derive statistics
    station_count = len(obs_sts_info)
    obs_date_min, obs_date_max = int(obs_sts_info['first_year'].min()), int(obs_sts_info['first_year'].max())
    data_owners_list=obs_sts_info['data_owner'].unique()
    locations_list=obs_sts_info['location_code'].unique().tolist()
    obs_nr_min, obs_nr_max = int(obs_sts_info['number_of_observations'].min()), int(obs_sts_info['number_of_observations'].max())
    # Calculate the difference between 'first_year' and 'last_year'
    obs_sts_info['year_difference'] = obs_sts_info['last_year'] - obs_sts_info['first_year']
    # Count how many stations have a difference greater than 1 year
    longer_than_one_year = obs_sts_info[obs_sts_info['year_difference'] > 1]
    count_longer_than_one_year = (obs_sts_info['year_difference'] > 1).sum()
    # Calculate the percentage
    percentage_longer_than_one_year = round((count_longer_than_one_year / len(obs_sts_info)) * 100,2)
    # Calculate the mean length for stations longer than one year
    mean_length = round(longer_than_one_year['year_difference'].mean(),1)
    
    #Write stats report
    print('')
    print(f'Data between {obs_date_min} - {obs_date_max}')
    print(f'Number of stations: {station_count}')
    print(f'Data owners: {data_owners_list}')
    print(f'Number of observations per station ranging between {obs_nr_min} - {obs_nr_max}')
    print(f'Percentage of stations with observation length longer than 1 year is {percentage_longer_than_one_year}%')
    print(f'Mean observation length of stations with more than 1 year observation is {mean_length} years')
    print('')
    
#EXTRACT
for depth in depths:
    for parameter in parameters:
        
        #build query
        if depth == 'surface':
            para2extract = {'parameter':parameter, 'p35code': parameter_to_p35code[parameter],   
                            'depth': ['sea level', 'unknown', '']}
        elif depth == 'bottom':
            para2extract = {'parameter':parameter, 'p35code': parameter_to_p35code[parameter],   
                            'depth': ['sea floor']}
          
        for year in selected_years:       
            startyr = year
            endyr = year
            
            tstart = dt.datetime(startyr,1,1)
            tend = dt.datetime(endyr,12,31)
            if startyr == endyr:
                tdate = startyr
            else:
                tdate = f'{startyr}_{endyr}'
                
            if gridded == True:
                outfile = os.path.join(outdir, f'{str(startyr)}_{para2extract["parameter"]}_obs_gridded_{depth}.csv')           
            else:
                outfile = os.path.join(outdir, f'{str(startyr)}_{para2extract["parameter"]}_obs_{depth}.csv')      
    
            if os.path.exists(outfile):
                print(f"File {outfile} already exists. Skipping.")
            else:               
                print(f'Running {year} for {para2extract["parameter"]}, gridded = {gridded}, depth = {depth}')               
                      
                ## Build urls to download measurement data
                
                if gridded == True:
                    
                    url = wfsbuild(
                        geom_dict = geom_dict,
                        typename=typename_data,
                        cql_dict={
                            'p35code': para2extract['p35code'],
                           'vertical_reference_preflabel': para2extract['depth']
                        },
                        outputFormat="csv",
                        columns=columns
                        )               
                    
                else: #point extraction     
                    
                    url = wfsbuild(
                        typename_data,
                        cql_dict={
                            'p35code': para2extract['p35code'],
                            'vertical_reference_preflabel': para2extract['depth']
                        },
                        outputFormat="csv",
                        columns=columns
                        )
                
                # # filter on geometry (points must intersect)
                # url = wfsbuild(typename_data, geom_dict={'intersects':{'geom':station_geom}}, 
                #                 outputFormat = "csv", columns = columns)
                
                # # filter on geometry (points must intersect), parametercode and reference label
                # url = wfsbuild(typename_data, cql_dict={'p35code':para2extract['p35code'],
                #                                         "vertical_reference_preflabel": para2extract['depth']},
                #                 geom_dict={'intersects':{'geom':station_geom}},               
                #                 outputFormat = "csv", columns = columns)
                
                # # filter on buffer around geometry, indicate the distance and unit
                # url = wfsbuild(typename_data, geom_dict={'within':{'geom':station_geom,
                #                                                    'distance':100.,
                #                                                    'unit':'meters'}}, 
                #                 outputFormat = "csv", columns = columns)
                
                # # filter on bbox, and parameter code
                # url = wfsbuild(typename_data, cql_dict={'p35code':para2extract['p35code']},
                #                 geom_dict = {'bbox':bbox},
                #                 outputFormat = "csv", columns = columns)
                    
                # In[21]:
                
                ## Import measurement data as csv
                print(' ')
                print('Import measurement data as csv')
                obsdata = readUrl(url, user, password)  
                obsdata['datetime'] = pd.to_datetime(obsdata['date'], dayfirst = False)
                print(obsdata.data_owner.unique())
                                
                # Cropping time
                obsdata = obsdata.loc[(obsdata['datetime'] >= tstart) & 
                                          (obsdata['datetime'] <= tend)].reset_index(drop=True)
                
                #Dropping ferry box data
                if parameter == 'Chlfa':
                    obsdata = obsdata[obsdata['data_owner'] != 'HZG']
                    print('Ferry box data dropped')
                               
                # Convert units to mg/l
                if parameter == 'NO3':
                    obsdata.loc[obsdata['unit_preflabel'].isin(['Micromoles per litre', 'Micromoles per kilogram']), 'value'] *= 14.006720 / 1000
                if parameter == 'PO4':
                    obsdata.loc[obsdata['unit_preflabel'].isin(['Micromoles per litre', 'Micromoles per kilogram']), 'value'] *= 30.973762 / 1000

                # Sort times
                obsdata = obsdata.sort_values('datetime')                

                if gridded == True:
                    print(f'Aggregating {para2extract["parameter"]} in a grid')
                    # Convert geom field to geometry type and add to geodataframe
                    obsdata["geom"] = gpd.GeoSeries.from_wkt(obsdata["geom"])
                    obsdata_gdf = gpd.GeoDataFrame(obsdata, geometry="geom", crs='epsg:4326')
            
                    # Convert date to datetime and extract month and year
                    obsdata_gdf['month'] = obsdata_gdf['datetime'].dt.month
                    obsdata_gdf['year'] = obsdata_gdf['datetime'].dt.year
                    #obsdata_gdf['cruise_date'] = obsdata_gdf['datetime'].dt.date
            
            
                    '''
                    Create a grid of a given resolution (in degreees) which will be used to aggregate the point data to create monthly averages
            
                    '''
                    # Specify the bounds of the North Sea and create extent
            
                    # North sea extent
                    min_lon, max_lon = geom_dict['bbox']['xmin'], geom_dict['bbox']['xmax']
                    min_lat, max_lat = geom_dict['bbox']['ymin'], geom_dict['bbox']['ymax']
                          
                    ns_bounds = box(min_lon, min_lat,
                                    max_lon, max_lat)
            
                    # Add to a geodataframe
                    ns_extent = gpd.GeoDataFrame(geometry=[ns_bounds], crs='epsg:4326')
        
            
                    longitudes = np.arange(min_lon, max_lon + resolution, resolution)
                    latitudes = np.arange(min_lat, max_lat + resolution, resolution)
            
                    # Create the grids
                    lon_grid, lat_grid = np.meshgrid(longitudes, latitudes)
            
                    grid_cells = []
                    for long in longitudes:
                        for lat in latitudes:
                            grid_cells.append({
                            'geometry': Point(long, lat).buffer(resolution/2, cap_style='square'),
                            'latitude': lat, 
                            'longitude' : long
                            })
            
                    grid_gdf= gpd.GeoDataFrame(grid_cells, crs='epsg:4326')
            
                    # Create grid index column
                    grid_gdf['grid_id'] = grid_gdf.index
            
            
                    # Join the grid to the point data and aggregate
                    gridded_data = gpd.sjoin(obsdata_gdf, grid_gdf, how='inner', predicate='intersects').drop(columns='index_right')
            
                    #Select observations in the top layer using buffer
                    if depth == 'surface':
                        gridded_data = gridded_data[(pd.isna(gridded_data['depth'])) | (abs(gridded_data['depth']) <= buffer)]
           
                    # Calculate monthly mean values (separate per year) for all grid boxes
                    monthly_aggregate_data = gridded_data.groupby(['grid_id', 'latitude', 'longitude', 'year', 'month'])['value'].agg({
                        'mean'}).reset_index()
                    monthly_aggregate_data.rename(columns={'mean': 'value'}, inplace=True)
                    monthly_aggregate_data.rename(columns={'grid_id': 'station'}, inplace=True)
            
                    # Round the values to 3 d.p
                    monthly_aggregate_data['value'] = monthly_aggregate_data['value'].round(3)
                    
                    # Create datatime from year and month
                    monthly_aggregate_data['datetime'] = pd.to_datetime(monthly_aggregate_data[['year', 'month']].assign(day=1))
            
                    # Create geom from latitude and longitude
                    monthly_aggregate_data['geom'] = monthly_aggregate_data.apply(lambda row: f"POINT ({row['longitude']} {row['latitude']})", axis=1)
            
                    # Harmonize with NWDM structure
                    clean = monthly_aggregate_data[['station', 'geom', 'datetime', 'value']]
                    
                else:
                    clean = obsdata[['station', 'geom', 'datetime', 'depth','value']]
                
                clean = clean.set_index('station')
            
                # In[22]:
                
                ## save the above as csv:
                    
                print(f'Saving {year} for {para2extract["parameter"]}')
                print('')
                
                clean.to_csv(outfile) # one year        
                clean
        
