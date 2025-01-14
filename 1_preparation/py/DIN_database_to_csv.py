#!/usr/bin/env python
# coding: utf-8

# In[1]:


### Packages

import io
import re

import numpy as np
import pandas as pd
import urllib
import requests
from requests.auth import HTTPBasicAuth

import datetime as dt
import matplotlib.dates as mdates

import cartopy
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from shapely.geometry import Point


# ### Functions

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


# In[3]:


## Settings

user = 'nwdm'
password = 'tkofschip'

# WFS layer
typename_sts = "NWDM:location"
typename_data = "NWDM:measurement_p35_all"

# list columns
columns = ["location_code", "date", "depth", "vertical_reference_preflabel",
            "vertical_reference_code", "p35code", "p35preflabel", 
            "value", "unit_preflabel", "quality_code", "station", "geom"]


# In[4]:


## Time frame 

startyr = 2005
endyr = 2015
tstart = dt.datetime(startyr,9,1)
tend = dt.datetime(endyr,9,1)
if startyr == endyr:
    tdate = startyr
else:
    tdate = f'{startyr}_{endyr}'
    
tstart, tend  


# ### Calculate DIN from separate NO2, NO3 and NH4 csv files

# #### Download vars separately

# In[11]:


## NWDM variable settings

# para2extract = {'parameter':'nh4', 
#                 'p35code': 'EPC00009',
#                 'depth': ['sea level', 'unknown']}

# para2extract = {'parameter':'no3', 
#                 'p35code': 'EPC00004',
#                 'depth': ['sea level', 'unknown']}

para2extract = {'parameter':'no2', 
                'p35code': 'EPC00006',
                'depth': ['sea level', 'unknown']}

# para2extract = {'parameter':'DIN_raw', 
#                 'p35code': 'EPC00198',
#                 'depth': ['sea level', 'unknown']}


# In[12]:


## Extraction

url_sts = wfsbuild(typename = "NWDM:location", outputFormat = "csv")
obs_sts_info = readUrl(url_sts, user, password)

url = wfsbuild(typename_data, cql_dict={ # or :plot_locs[0] or stationid, if only 1 location
                                        'p35code':para2extract['p35code'],
                                        'vertical_reference_preflabel' : para2extract['depth']}, 
                outputFormat = "csv", columns = columns)

obsdata = readUrl(url, user, password)  
obsdata['datetime'] = pd.to_datetime(obsdata['date'], dayfirst = False)

# Cropping time
obsdata = obsdata.loc[(obsdata['datetime'] >= tstart) & 
                          (obsdata['datetime'] <= tend)].reset_index(drop=True)

# Sort times
obsdata = obsdata.sort_values('datetime')

obsdata['station'] = obsdata['station'].astype(str) # convert labels to strings


# In[13]:


## Save
clean = obsdata[['station', 'geom', 'datetime', 'value']]
clean = clean.set_index('station')
clean.to_csv(fr'P:\11206304-futuremares\data\NWDM_observations\2005_to_2015_{para2extract["parameter"]}_obs.csv')
clean


# #### Read the above in again

# In[3]:


# list of locs
#plot_locs = ['NOORDWK2','NOORDWK10','NOORDWK20','NOORDWK30','NOORDWK50','NOORDWK70',
#              'ROTTMPT3','ROTTMPT10','ROTTMPT100','ROTTMPT15','ROTTMPT20','ROTTMPT30','ROTTMPT50',
#              'ROTTMPT70', 'TERSLG10','TERSLG100', 'TERSLG135','TERSLG175','TERSLG20','TERSLG235',
#              'TERSLG30', 'TERSLG50','TERSLG70','WALCRN2','WALCRN10','WALCRN20','WALCRN30',
#              'WALCRN50','WALCRN70']

## read these files:
nh4 = pd.read_csv(r'P:\11209810-cmems-nws\Data\NWDM_observations\2005_to_2015_nh4_obs.csv')
no3 = pd.read_csv(r'P:\11209810-cmems-nws\Data\NWDM_observations\2005_to_2015_no3_obs.csv')
no2 = pd.read_csv(r'P:\11209810-cmems-nws\Data\NWDM_observations\2005_to_2015_no2_obs.csv')
din_raw = pd.read_csv(r'P:\11209810-cmems-nws\Data\NWDM_observations\2005_to_2015_DIN_raw_obs.csv')

# Extract only the stations that we are interested in
#nh4 = nh4[nh4['station'].isin(plot_locs)]
#no3 = no3[no3['station'].isin(plot_locs)]
#no2 = no2[no2['station'].isin(plot_locs)]
#din_raw = din_raw[din_raw['station'].isin(plot_locs)]

# rename the value columns (so not all 'value')
nh4 = nh4.rename(columns={'value':'nh4'})
no3 = no3.rename(columns={'value':'no3'})
no2 = no2.rename(columns={'value':'no2'})
din_raw = din_raw.rename(columns={'value':'din_raw'})


# In[4]:


din_raw  # empty, so no need to append ...


# In[5]:


# Merge dataframes on the first three columns. Note: using 'on' means that first three columns must match in df's.
merged_df = pd.merge(nh4, no3, on=['station', 'geom', 'datetime'])
merged_df = pd.merge(merged_df, no2, on=['station', 'geom', 'datetime'])

# Calculate the sum of the fourth column from each original dataframe
merged_df['DIN'] = merged_df['nh4'] + merged_df['no3'] + merged_df['no2']

# Select only the first three columns and the new sum column
DIN_df = merged_df[['station', 'geom', 'datetime', 'DIN']]

# Save as new DIN variable:
DIN_df = DIN_df.set_index('station')
DIN_df.to_csv(fr'P:\11209810-cmems-nws\Data\NWDM_observations\combined_years\{startyr}_{endyr}_DIN_obs.csv')
DIN_df


# In[ ]:




