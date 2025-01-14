# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 09:46:54 2022

@author: linde_ae
"""

# load packages
# -----------------------------------------------------------------------------

import io
import numpy as np
import pandas as pd
import urllib
import requests
from requests.auth import HTTPBasicAuth
from shapely.geometry import Point

# =============================================================================
# Functions
# =============================================================================

# modify wfsbuild to take e.g. two depths!
def wfsbuild(typename, cql_dict = {}, geom_dict= {},  
             outputFormat = "csv", maxFeatures = None, columns = None):
    """
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

    """
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
        geom_list = [f"{k}(geom,{v['geom']})" if k == 'intersect'\
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


# function to read NWDM data from url and add datetime column 
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

def calculateDIN(nwdmLayer, depth, columnsNWDM, user, password):
    """
    Calculate DIN from NH4, NO3 and NO2.

    Parameters
    ----------
    nwdmLayer : str
        nwdm layer to extract data from.
    depth : str
        depth which should be extracted.
    columnsNWDM : list
        column names to be extracted.
    user : str
        user of NWDM.
    password : str
        password for user.

    Returns
    -------
    obsdata_rawDIN : dataframe
        concatenated dataframe of p35code for DIN and Din calculated from
        NH4, NO3 and NO2.

    """
    # extract NH4
    url = wfsbuild(typename = nwdmLayer, 
                   cql_dict={'p35code' : 'EPC00009', 
                             'vertical_reference_preflabel' : depth}, 
                   outputFormat = "csv", columns = columnsNWDM)
    obsdata_rawNH4 = readUrl(url, user, password)
    obsdata_rawNH4['depth'] = np.where(~np.isnan(obsdata_rawNH4['depth']), 0., obsdata_rawNH4['depth'])

    
    # extract NO3
    url = wfsbuild(typename = nwdmLayer, 
                   cql_dict={'p35code' : 'EPC00004', 
                             'vertical_reference_preflabel' : depth}, 
                   outputFormat = "csv", columns = columnsNWDM)
    obsdata_rawNO3 = readUrl(url, user, password)
    obsdata_rawNO3['depth'] = np.where(np.isnan(obsdata_rawNO3['depth']), 0., obsdata_rawNO3['depth'])

    
    # extract NO2
    url = wfsbuild(typename = nwdmLayer, 
                   cql_dict={'p35code' : 'EPC00006', 
                             'vertical_reference_preflabel' :depth}, 
                   outputFormat = "csv", columns = columnsNWDM)
    # read url and return raw NWDM data 
    obsdata_rawNO2 = readUrl(url, user, password)
    obsdata_rawNO2['depth'] = np.where(~np.isnan(obsdata_rawNO2['depth']), 0., obsdata_rawNO2['depth'])
    
    # merge NH4, NO3 and NO2 datasets together
    obsdata_var = obsdata_rawNH4.merge(obsdata_rawNO3, how = 'inner', on=['location_code', 'depth', 'date'])
    obsdata_var = obsdata_var.merge(obsdata_rawNO2, how = 'inner', on=['location_code', 'depth', 'date'])
    obsdata_var['DIN'] = obsdata_var['value_x'] + obsdata_var['value_y'] + obsdata_var['value']
    
    # select wanted columns and remove x in names of columns 
    obsdata_var = obsdata_var[['FID','location_code', 'geom', 'date', 'depth', 'vertical_reference_code', 'vertical_reference_preflabel',
                             'unit_preflabel', 'DIN', 'quality_code', 'p35code', 'p35preflabel', 'station']] 
    obsdata_var = obsdata_var.rename(columns = {'DIN':'value'})
    # rename the column p35code and p35preflabel
    obsdata_var["p35code"] = "EPC00198"
    obsdata_var["p35preflabel"] = "Water body DIN"
    
    # extract DIN p35code
    url = wfsbuild(typename = nwdmLayer, 
                    cql_dict={'p35code' : 'EPC00198', 
                              'vertical_reference_preflabel' : depth}, 
                    outputFormat = "csv", columns = columnsNWDM)
    obsdata_rawDIN = readUrl(url, user, password)
    
    # append calculated DIN to p35 DIN
    obsdata_rawDIN = obsdata_rawDIN.append(obsdata_var)
    
    return obsdata_rawDIN
    
def calculateStationDIN(nwdmLayer, station, depth, columnsNWDM, user, password):
    """
    Calculate DIN from NH4, NO3 and NO2 for specific stations.

    Parameters
    ----------
    nwdmLayer : str
        nwdm layer to extract data from.
    station :
        station(s) to be extracted
    depth : str
        depth which should be extracted.
    columnsNWDM : list
        column names to be extracted.
    user : str
        user of NWDM.
    password : str
        password for user.

    Returns
    -------
    obsdata_rawDIN : dataframe
        concatenated dataframe of p35code for DIN and Din calculated from
        NH4, NO3 and NO2.

    """
    # extract NH4, NO3 and NO2
    url = wfsbuild(typename = nwdmLayer, 
                   cql_dict={'station' : station,
                             'p35code' : ['EPC00009', 'EPC00004', 'EPC00006'],                             
                             'vertical_reference_preflabel' : depth}, 
                   outputFormat = "csv", columns = columnsNWDM)
    # read url and return raw NWDM data 
    obsdata_raw = readUrl(url, user, password)
    
    if not(len(obsdata_raw) > 0):
        pass
    else:
        # assumed unknown depths are surface layer
        obsdata_raw['depth'] = np.where(np.isnan(obsdata_raw['depth']), 0., obsdata_raw['depth'])
        
        # pivot to get NH4, NO3 and NO2 for same date, depth and location_code
        obsdata_var = pd.pivot_table(obsdata_raw, values = 'value', 
                                     index=['location_code', 'geom', 'date', 
                                            'depth',  'vertical_reference_code', 
                                            'vertical_reference_preflabel',
                                            'unit_preflabel', 'quality_code',
                                            'station'], 
                                     columns = 'p35code').reset_index()
        subs = list(obsdata_raw['p35code'].unique())
        
        obsdata_var['value'] = obsdata_var[subs].sum(axis=1)
    
        # insert columns FID p35code and p35preflabel
        obsdata_var["FID"] = np.nan
        obsdata_var["p35code"] = "EPC00198"
        obsdata_var["p35preflabel"] = "Water body DIN"
        
        # select wanted columns and remove x in names of columns 
        obsdata_var = obsdata_var[['location_code', 'geom', 'date', 'depth', 
                                   'vertical_reference_code', 'vertical_reference_preflabel',
                                   'unit_preflabel', 'value', 'quality_code', 
                                   'p35code', 'p35preflabel', 'station']]    
        
    # extract DIN p35code
    url = wfsbuild(typename = nwdmLayer, 
                    cql_dict={'station' : station, 'p35code' : 'EPC00198', 
                              'vertical_reference_preflabel' : depth}, 
                    outputFormat = "csv", columns = columnsNWDM)
    obsdata_rawDIN = readUrl(url, user, password)
    
    # append calculated DIN to p35 DIN
    try:
        obsdata_rawDIN = obsdata_rawDIN.append(obsdata_var)
    except:
        pass
    
    return obsdata_rawDIN
    