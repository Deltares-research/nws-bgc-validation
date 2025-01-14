# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 10:20:49 2022

@author: schne_la
"""

# =============================================================================
# load packages
# =============================================================================

from netCDF4 import Dataset, num2date #, num2date #, date2num, date2index

# =============================================================================
# Functions
# =============================================================================

def get_modkey(df, var_name):
    """
    Select model variable name from his file based on variable name selected in file (lng_name).

    Parameters
    ----------
    df : dataframe
        model hisfile.
    var_name : str
        variable name that should be extracted.

    Returns
    -------
    key : str
        key used in model output.

    """
    # salinity and temperature dot not have a long name
    # most other DWAQ relevant parameters do
    # otherwise add here in the list
    if (var_name in ['salinity', 'temperature']): 
        key = var_name
    else:
        key = df[df['long_name'] == var_name].index[0]
        # key = [x for x, y in zip(df['nc_varkeys'], df['long_name']) if y == var_name][0] 
    return key


def get_modTime(hisfile):
    """
    Extract date and time from model hisfile.

    Parameters
    ----------
    hisfile : dataframe
        hisfile of model output.

    Returns
    -------
    mdtimes : list
        data and time extracted from the hisfile.

    """
    # get data from hisfile
    mod = Dataset(hisfile)
    # get time from origin
    mod_times = mod.variables['time'][:]
    # get units
    mod_tunit = mod.variables['time'].units
    # calculate time into date
    mod_dtimes = num2date(mod_times, mod_tunit, only_use_cftime_datetimes=False)
    it_end = len(mod.variables['time'])    # wy do I need this??? 
    # get date and time based on length of variables from hisfile       
    mdtimes = list(mod_dtimes[:it_end])
    
    return mdtimes