#!/usr/bin/env python
# coding: utf-8

#%%
## Packages

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
import cmocean
from sklearn.metrics import mean_squared_error
from sklearn.metrics import root_mean_squared_error


#%%
# path to output folder for plots
rootdir = r'P:\11209810-cmems-nws\model_output\timeseries_per_polygon' if os.name == 'nt' else r'/p/11209810-cmems-nws/model_output/timeseries_per_polygon'

start_year = 2021
end_year = 2022
offices = ['NWS', 'IBI', 'satellite'] #DFM can be added only if the years are between 2015-2017

outdir_tables = fr'P:\11209810-cmems-nws\figures\tables\{start_year}_{end_year}' if os.name == 'nt' else r'/p/11209810-cmems-nws/figures/tables/{start_year}_{end_year}'
if not os.path.exists(outdir_tables):
    os.makedirs(outdir_tables)

outdir_maps = fr'P:\11209810-cmems-nws\figures\maps_stats\{start_year}_{end_year}' if os.name == 'nt' else r'/p/11209810-cmems-nws/figures/tables/{start_year}_{end_year}'
if not os.path.exists(outdir_maps):
    os.makedirs(outdir_maps)

#%%
# ## Read the models

## Select growing season CHL and remove empty / not matching assessment areas

office = 'satellite'

try:
    obs_path = os.path.join(rootdir, fr'{start_year}_{end_year}_{office}_ts.csv') 
    satellite_df = pd.read_csv(obs_path)
except:
    obs_path = os.path.join(rootdir, fr'2003_2017_{office}_ts.csv')
    satellite_df = pd.read_csv(obs_path)

satellite_df = satellite_df.set_index('time')
satellite_df.index = pd.to_datetime(satellite_df.index)
satellite_df = satellite_df[satellite_df['CHL'].notna()]                 # drop rows with NaNs (outside domain)
satellite_df = satellite_df.drop(columns='depth')
satellite_df.rename(columns={'CHL': 'chl'}, inplace=True)
#satellite_df = satellite_df[satellite_df.polygon != 'Kattegat Coastal']  # remove as not in NWS!  - But keep for IBI!
#satellite_df = satellite_df[satellite_df.polygon != 'Kattegat Deep']     # remove as not in NWS!
#satellite_df = satellite_df.loc[(satellite_df.index >= f'{year}-03-01') & (satellite_df.index <= f'{year}-09-30')]  # GROWING SEASON -- already done in ts!


#%%
## Select growing season CHL and remove empty / not matching assessment areas

office = 'NWS'

try:
    obs_path = os.path.join(rootdir, fr'{start_year}_{end_year}_{office}_ts.csv') 
    NWS_df = pd.read_csv(obs_path)
except:
    obs_path = os.path.join(rootdir, fr'2003_2017_{office}_ts.csv')
    NWS_df = pd.read_csv(obs_path)

NWS_df = NWS_df.set_index('time')
NWS_df.index = pd.to_datetime(NWS_df.index)  # drop the hours and minutes to match satellites
NWS_df.index = NWS_df.index.date
NWS_df.index.names = ['time']   # add index name and format back - removed by the .date operation
NWS_df.index = pd.to_datetime(NWS_df.index)
NWS_df = NWS_df.drop(columns='depth')
NWS_df = NWS_df[NWS_df['chl'].notna()]
#NWS_df = NWS_df.loc[(NWS_df.index >= f'{year}-03-01') & (NWS_df.index <= f'{year}-09-30')]

# ## Comparing the Date Columns of Two Dataframes and Keeping the Rows with the same Dates:
# common_index = list(set(satellite_df.index).intersection(NWS_df.index))      # Need for DFM + IBI (not NWS)
# satellite_df = satellite_df.loc[common_index].copy()
# NWS_df = NWS_df.loc[common_index].copy()
# print(len(satellite_df), len(NWS_df))


#%%
## Select growing season CHL and remove empty / not matching assessment areas

office = 'IBI'  # remember to keep Kattegat and other station in satellite_df here!

try:
    obs_path = os.path.join(rootdir, fr'{start_year}_{end_year}_{office}_ts.csv') 
    IBI_df = pd.read_csv(obs_path)
except:
    obs_path = os.path.join(rootdir, fr'2003_2017_{office}_ts.csv')
    IBI_df = pd.read_csv(obs_path)

IBI_df = IBI_df.set_index('time')
IBI_df.index = pd.to_datetime(IBI_df.index)  # drop the hours and minutes to match satellites
IBI_df.index = IBI_df.index.date
IBI_df.index.names = ['time']   # add index name and format back - removed by the .date operation
IBI_df.index = pd.to_datetime(IBI_df.index)
IBI_df = IBI_df.drop(columns='depth')
IBI_df = IBI_df[IBI_df['chl'].notna()]
#IBI_df = IBI_df.loc[(IBI_df.index >= f'{year}-03-01') & (IBI_df.index <= f'{year}-09-30')]

# ## Comparing the Date Columns of Two Dataframes and Keeping the Rows with the same Dates:
# common_index = list(set(satellite_df.index).intersection(IBI_df.index))      # Need for DFM + IBI (not NWS)
# satellite_df = satellite_df.loc[common_index].copy()
# IBI_df = IBI_df.loc[common_index].copy()
# print(len(satellite_df), len(IBI_df))

#%%

if 'DFM' in offices:
    ## Select growing season CHL and remove empty / not matching assessment areas
    
    office = 'DFM'  # Note, here have to select same time steps for satellites, as not daily resolution!
    
    try:
        obs_path = os.path.join(rootdir, fr'{start_year}_{end_year}_{office}_ts.csv') 
        DFM_df = pd.read_csv(obs_path)
    except:
        obs_path = os.path.join(rootdir, fr'2015_2017_{office}_ts.csv')
        DFM_df = pd.read_csv(obs_path)

    DFM_df = DFM_df.set_index('time')
    DFM_df.index = pd.to_datetime(DFM_df.index)
    try:
        DFM_df = DFM_df.drop(columns='mesh2d_layer_sigma_z')
    except:
        pass
    DFM_df = DFM_df[DFM_df['mesh2d_Chlfa'].notna()]
    DFM_df.rename(columns={'mesh2d_Chlfa': 'chl'}, inplace=True)
    # DFM_df = DFM_df[DFM_df.polygon != 'Adour plume']  # remove as not in NWS!
    # DFM_df = DFM_df[DFM_df.polygon != 'Gironde plume']  
    # DFM_df = DFM_df[DFM_df.polygon != 'Gulf of Biscay coastal waters'] 
    # DFM_df = DFM_df[DFM_df.polygon != 'Gulf of Biscay shelf waters'] 
    # DFM_df = DFM_df[DFM_df.polygon != 'Kattegat Coastal'] 
    # DFM_df = DFM_df[DFM_df.polygon != 'Kattegat Deep'] 
    # DFM_df = DFM_df[DFM_df.polygon != 'Loire plume'] 
    #DFM_df = DFM_df.loc[(DFM_df.index >= f'{year}-03-01') & (DFM_df.index <= f'{year}-09-30')]
    
    # ## Comparing the Date Columns of Two Dataframes and Keeping the Rows with the same Dates:
    # common_index = list(set(satellite_df.index).intersection(DFM_df.index))      # Need for DFM + IBI (not NWS)
    # satellite_df = satellite_df.loc[common_index].copy()
    # DFM_df = DFM_df.loc[common_index].copy()
    # print(len(satellite_df), len(DFM_df))

# Merge dataframes
# Reset the index to include 'time' as a column for merging
satellite_df_reset = satellite_df.reset_index()
NWS_df_reset = NWS_df.reset_index()
IBI_df_reset = IBI_df.reset_index()
if 'DFM' in offices:
    DFM_df_reset = DFM_df.reset_index()

# Merge satellite_df and NWS_df on 'time' and 'polygon'
merged_df = pd.merge(satellite_df_reset, NWS_df_reset, on=['time', 'polygon'], suffixes=('_satellite', '_NWS'))

# Merge the result with IBI_df
merged_df = pd.merge(merged_df, IBI_df_reset, on=['time', 'polygon'], suffixes=('', '_IBI'))

if 'DFM' in offices:
    # Merge the result with DFM_df
    merged_df = pd.merge(merged_df, DFM_df_reset, on=['time', 'polygon'], suffixes=('', '_DFM'))

# Rename columns to make it clear which 'chl' comes from which dataset
merged_df.rename(columns={'chl': 'chl_IBI'}, inplace=True)

# Set the index back to 'time' if desired
merged_df.set_index('time', inplace=True)


#%%

## Calc n (points per polygon! --  get this from timeseries creation code!), bias, MAD, corr coef etc: (using taylor diagram metrics to simplify...)

mean_sat, mean_NWS, mean_IBI, mean_DFM = [],[],[],[]
nstd_NWS, nstd_IBI, nstd_DFM = [],[],[]
ccoef_NWS, ccoef_IBI, ccoef_DFM = [],[],[]
mse_NWS, mse_IBI, mse_DFM = [],[],[]
rmse_NWS, rmse_IBI, rmse_DFM = [],[],[]

for poly in np.unique(merged_df.polygon):

    ref = merged_df.loc[merged_df['polygon'] == poly].chl_satellite
    # ref = np.log(ref) 
    for office in offices:
        if office == 'NWS':
            pred = merged_df.loc[merged_df['polygon'] == poly].chl_NWS      #NWS
        elif office == 'IBI':
            pred = merged_df.loc[merged_df['polygon'] == poly].chl_IBI      #IBI
        elif office == 'DFM':
            pred = merged_df.loc[merged_df['polygon'] == poly].chl_DFM      #DFM
    
        pred.index = pd.to_datetime(pred.index)
        pred.index = pred.index.date
        # pred = np.log(pred)

        if office == 'satellite':
            mean_sat.append(np.mean(ref))     
        if office == 'NWS':
            mean_NWS.append(np.mean(pred))
            nstd_NWS.append(np.std(pred) / np.std(ref))
            ccoef_NWS.append(abs(np.corrcoef(pred.values,ref.values)[0,1]))  # use absolute ccoef, to match Taylor Diagr.
            mse_NWS.append(mean_squared_error(ref.values, pred.values))
            rmse_NWS.append(root_mean_squared_error(ref.values, pred.values))
        elif office == 'IBI':
            mean_IBI.append(np.mean(pred))
            nstd_IBI.append(np.std(pred) / np.std(ref))
            ccoef_IBI.append(abs(np.corrcoef(pred.values,ref.values)[0,1]))
            mse_IBI.append(mean_squared_error(ref.values, pred.values))
            rmse_IBI.append(root_mean_squared_error(ref.values, pred.values))
        elif office == 'DFM':
            mean_DFM.append(np.mean(pred))
            nstd_DFM.append(np.std(pred) / np.std(ref))
            ccoef_DFM.append(abs(np.corrcoef(pred.values,ref.values)[0,1]))
            mse_DFM.append(mean_squared_error(ref.values, pred.values))
            rmse_DFM.append(root_mean_squared_error(ref.values, pred.values))

    # break


#%%
# Create the excel sheet with the output:

if 'DFM' in offices:
    excel_df = pd.DataFrame({'polygon':np.unique(merged_df.polygon),
                             'mean_sat': mean_sat,
                             'mean_NWS': mean_NWS,
                             'mean_IBI': mean_IBI,
                             'mean_DFM': mean_DFM,
                            'nstd_NWS':nstd_NWS,
                            'nstd_IBI':nstd_IBI,
                            'nstd_DFM':nstd_DFM,
                            'ccoef_NWS':ccoef_NWS,
                            'ccoef_IBI':ccoef_IBI,
                            'ccoef_DFM':ccoef_DFM,
                            # 'mse_NWS':mse_NWS,
                            # 'mse_IBI':mse_IBI,
                            # 'mse_DFM':mse_DFM,
                            'rmse_NWS':rmse_NWS,
                            'rmse_IBI':rmse_IBI,
                            'rmse_DFM':rmse_DFM               
                               })      
                            
else:
    excel_df = pd.DataFrame({'polygon':np.unique(merged_df.polygon),
                             'mean_sat': mean_sat,
                             'mean_NWS': mean_NWS,
                             'mean_IBI': mean_IBI,
                            'nstd_NWS':nstd_NWS,
                            'nstd_IBI':nstd_IBI,
                            'ccoef_NWS':ccoef_NWS,
                            'ccoef_IBI':ccoef_IBI,
                            # 'mse_NWS':mse_NWS,
                            # 'mse_IBI':mse_IBI,
                            'rmse_NWS':rmse_NWS,
                            'rmse_IBI':rmse_IBI,
                            })

excel_df = excel_df.set_index('polygon')

#Save to df
if 'DFM' in offices:
    excel_df.to_excel(os.path.join(outdir_tables, fr'{start_year}_{end_year}_satellite_vs_NWS_IBI_DFM.xlsx'), sheet_name='map_stats', index=True)
else:
    excel_df.to_excel(os.path.join(outdir_tables, fr'{start_year}_{end_year}_satellite_vs_NWS_IBI.xlsx'), sheet_name='map_stats', index=True)


#%%
# Make a map of the values per assessment area:

# read assessment areas:
area_shp = gpd.read_file(r'P:\11209810-cmems-nws\OSPAR_areas\COMP4_assessment_areas_v8a.shp')

for vari in excel_df.columns:
    print(vari)

    # Set colorbar limits and colormap based on the column type
    if vari.startswith('mean_'):
        cmap = cmocean.cm.algae  # Set colormap for nstd_* columns
        vmin, vmax = 0, 10  # Limit for standard deviation  --- can be zero to infinity. The larger the value, the larger the spread
        extend = 'max'
    if vari.startswith('nstd_'):
        cmap = cmocean.cm.amp  # Set colormap for nstd_* columns
        vmin, vmax = 0, 1  # Limit for standard deviation  --- can be zero to infinity. The larger the value, the larger the spread
        extend = 'max'
    elif vari.startswith('ccoef_'):
        cmap = cmocean.cm.amp  # Set colormap for ccoef_* columns
        vmin, vmax = 0,1  # Limit for correlation coefficient  --- can be -1 to 1., where 0 is no correlation and 1 is perfect positive correlation.
        extend = None
    elif vari.startswith('mse_'):
        cmap = cmocean.cm.balance  # Set colormap for ccoef_* columns
        vmin, vmax = -1,1  # Limit for correlation coefficient  --- can be -1 to 1., where 0 is no correlation and 1 is perfect positive correlation.
        extend = 'both'
    elif vari.startswith('rmse_'):
        cmap = cmocean.cm.balance  # Set colormap for ccoef_* columns
        vmin, vmax = -1,1  # Limit for correlation coefficient  --- can be -1 to 1., where 0 is no correlation and 1 is perfect positive correlation.
        extend = 'both'

    fig = plt.figure(figsize=(8,6))
    ax = plt.axes(projection=ccrs.PlateCarree())                     # create ax + select map projection
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True)      # add the longitude / latitude lines
    gl.right_labels, gl.top_labels  = False, False                   # remove latitude labels on the right
    gl.xlines, gl.ylines = False, False
    ax.add_feature(cfeature.LAND, zorder=1, edgecolor='grey', linewidth=0.3, facecolor='linen')      # add land mask
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor='grey')
    #ax.set_extent([-16,15, 42,64])

    # Plot all OSPAR assessment areas:
    # area_shp.plot(ax=ax, linewidth = 1, facecolor = (1, 1, 1, 0), edgecolor = (0.5, 0.5, 0.5, 1))

    # Reset index to make 'index' a column, so it can be merged with 'ID' in the GeoDataFrame
    excel_df_reset = excel_df.reset_index()
    # Merge the DataFrame with the GeoDataFrame on polygon names
    merged_df = excel_df_reset.merge(area_shp[['ID', 'geometry']], left_on='polygon', right_on='ID')
    # Convert the merged DataFrame to a GeoDataFrame
    geo_df = gpd.GeoDataFrame(merged_df, geometry='geometry')
    # Drop the 'ID' column if it's not needed
    geo_df = geo_df.drop(columns='ID')

    # Plot the polygons with the colours as the nstd values 
    plot = geo_df.plot(column=vari, cmap=cmap, legend=False, ax=ax)

    # colorbar:
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cbar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar.set_array([])                             # Needed for matplotlib to generate the colorbar
    cb = fig.colorbar(cbar, ax=ax, extend=extend)   # Add the colorbar to the figure
    cb.set_label(vari, rotation=90, labelpad=15)   # Add a title to the colorbar

    # save:
    plt.savefig(os.path.join(outdir_maps, fr'map_{start_year}_{end_year}_{vari}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
# %%
