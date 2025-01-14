#!/usr/bin/env python
# coding: utf-8

# In[1]:


## Packages

import os

import numpy as np
import pandas as pd
import geopandas as gpd
import skill_metrics as sm

import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# Function to check the region based on arrays
def assign_region(polygon):
    if polygon in plume:
        return 'plume'
    elif polygon in coastal:
        return 'coastal'
    elif polygon in oceanic:
        return 'oceanic'
    elif polygon in shelf:
        return 'shelf'
    else:
        return 'other'  # Empty string if not in any region
# In[2]:


# Set paths and year

rootdir = r'P:\11209810-cmems-nws\model_output\timeseries_per_polygon' if os.name == 'nt' else r'/p/11209810-cmems-nws/model_output/timeseries_per_polygon'

start_year = 2015
end_year = 2017
year = fr'{start_year}_{end_year}'
offices = ['NWS', 'IBI'] #DFM can be added only if the years are between 2015-2017
taylor = 1 # yes - 1 | no - 0
target = 0 # yes - 1 | no - 0

outdir_tables = fr'P:\11209810-cmems-nws\figures\tables\{start_year}_{end_year}' if os.name == 'nt' else r'/p/11209810-cmems-nws/figures/tables/{start_year}_{end_year}'
if not os.path.exists(outdir_tables):
    os.makedirs(outdir_tables)

# In[5]:


## Import assessment areas and categories:

fp = r'P:\11209810-cmems-nws\OSPAR_areas' if os.name == 'nt' else r'/p/11209810-cmems-nws/OSPAR_areas'
gdf = gpd.read_file(fp)

Assessment_area_categories_file = r'P:\11209810-cmems-nws\OSPAR_areas\Assessment_area_categories.xlsx' if os.name == 'nt' else r'/p/11209810-cmems-nws/OSPAR_areas/Assessment_area_categories.xlsx'
categories = pd.read_excel(Assessment_area_categories_file,
                           sheet_name='Assessment_area_categories')

# Sort the ID columns alphabetically so both dataset have the same order
gdf = gdf.sort_values('ID')
categories = categories.sort_values('New_UnitCode')

# Add category to the df:
gdf['category'] = categories.Category.values
gdf


# In[6]:


# Categories for the colours:

# gdf.groupby('category')

coastal = gdf.loc[gdf['category'] == 'Coastal']['ID'].values
plume = gdf.loc[gdf['category'] == 'Plume']['ID'].values
shelf = gdf.loc[gdf['category'] == 'Shelf']['ID'].values
oceanic = gdf.loc[gdf['category'] == 'Oceanic']['ID'].values


# ## Read the models

# In[37]:


## Select growing season CHL and remove empty / not matching assessment areas

model = 'satellite'

satellite_df = pd.read_csv(os.path.join(rootdir, fr'{year}_{model}_ts.csv'))
satellite_df = satellite_df.set_index('time')
satellite_df.index = pd.to_datetime(satellite_df.index)
satellite_df = satellite_df.drop(columns='depth')
satellite_df = satellite_df[satellite_df['CHL'].notna()]                 # drop rows with NaNs (outside domain)
satellite_df.rename(columns={'CHL': 'chl'}, inplace=True)
satellite_df = satellite_df.loc[(satellite_df.index >= f'{start_year}-03-01') & (satellite_df.index <= f'{end_year}-09-30')]  # GROWING SEASON -- already done in ts!
satellite_df  # 39 regions


# In[28]:


## Select growing season CHL and remove empty / not matching assessment areas

model = 'NWS'

NWS_df =  pd.read_csv(os.path.join(rootdir, fr'{year}_{model}_ts.csv'))
NWS_df = NWS_df.set_index('time')
NWS_df.index = pd.to_datetime(NWS_df.index)  # drop the hours and minutes to match satellites
NWS_df.index = NWS_df.index.date
NWS_df.index.names = ['time']   # add index name and format back - removed by the .date operation
NWS_df.index = pd.to_datetime(NWS_df.index)
NWS_df = NWS_df.drop(columns='depth')
NWS_df = NWS_df[NWS_df['chl'].notna()]
NWS_df = NWS_df.loc[(NWS_df.index >= f'{start_year}-03-01') & (NWS_df.index <= f'{end_year}-09-30')]
NWS_df

# # remove these satellites (only here)
# satellite_df = satellite_df[satellite_df.polygon != 'KC']  # remove as not in NWS!  - But keep for IBI!
# satellite_df = satellite_df[satellite_df.polygon != 'KD']  

# print(len(satellite_df), len(NWS_df))
# np.unique(NWS_df.polygon.values)   # 37 regions


# In[34]:


## Select growing season CHL and remove empty / not matching assessment areas

model = 'IBI'  # remember to keep Kattegat and other station in satellite_df here!

IBI_df =  pd.read_csv(os.path.join(rootdir, fr'{year}_{model}_ts.csv'))
IBI_df = IBI_df.set_index('time')
IBI_df.index = pd.to_datetime(IBI_df.index)  # drop the hours and minutes to match satellites
IBI_df.index = IBI_df.index.date
IBI_df.index.names = ['time']   # add index name and format back - removed by the .date operation
IBI_df.index = pd.to_datetime(IBI_df.index)
IBI_df = IBI_df.drop(columns='depth')
IBI_df = IBI_df[IBI_df['chl'].notna()]
IBI_df = IBI_df.loc[(IBI_df.index >= f'{start_year}-03-01') & (IBI_df.index <= f'{end_year}-09-30')]

# ## Comparing the Date Columns of Two Dataframes and Keeping the Rows with the same Dates:
# common_index = list(set(satellite_df.index).intersection(IBI_df.index))      # Need for DFM + IBI (not NWS)
# satellite_df = satellite_df.loc[common_index].copy()
# IBI_df = IBI_df.loc[common_index].copy()
# print(len(satellite_df), len(IBI_df))
# # len(np.unique(IBI_df.polygon.values))

IBI_df  # 39 regions


# In[18]:

if 'DFM' in offices:
    ## Select growing season CHL and remove empty / not matching assessment areas
    
    model = 'DFM'  # Note, here have to select same time steps for satellites, as not daily resolution!
    
    DFM_df =  pd.read_csv(os.path.join(rootdir, fr'{year}_{model}_ts.csv'))
    DFM_df = DFM_df.set_index('time')
    DFM_df.index = pd.to_datetime(DFM_df.index)
    try:
        DFM_df = DFM_df.drop(columns='mesh2d_layer_sigma_z')
    except KeyError:
        print('')
    DFM_df = DFM_df[DFM_df['mesh2d_Chlfa'].notna()]
    DFM_df.rename(columns={'mesh2d_Chlfa': 'chl'}, inplace=True)
    
    # # Remove the polygons in DFM, but not in comparable satellites: 
    # # DFM_df = DFM_df[DFM_df.polygon != 'Adour plume']  # remove as not in satellite!
    # # DFM_df = DFM_df[DFM_df.polygon != 'Gironde plume']  
    # # DFM_df = DFM_df[DFM_df.polygon != 'Gulf of Biscay coastal waters'] 
    # # DFM_df = DFM_df[DFM_df.polygon != 'Gulf of Biscay shelf waters'] 
    # # DFM_df = DFM_df[DFM_df.polygon != 'Kattegat Coastal'] 
    # # DFM_df = DFM_df[DFM_df.polygon != 'Kattegat Deep'] 
    # # DFM_df = DFM_df[DFM_df.polygon != 'Loire plume'] 
    
    # DFM_df = DFM_df[DFM_df.polygon != 'ADPM']
    # DFM_df = DFM_df[DFM_df.polygon != 'GBCW']
    # DFM_df = DFM_df[DFM_df.polygon != 'GBSW']
    # DFM_df = DFM_df[DFM_df.polygon != 'GDPM']
    # DFM_df = DFM_df[DFM_df.polygon != 'LPM']
    # DFM_df = DFM_df[DFM_df.polygon != 'NAAC1B']
    # DFM_df = DFM_df[DFM_df.polygon != 'NAAC1C']
    # DFM_df = DFM_df[DFM_df.polygon != 'NAAC1D']
    # DFM_df = DFM_df[DFM_df.polygon != 'NAAC2']
    # DFM_df = DFM_df[DFM_df.polygon != 'NAAC3']
    # DFM_df = DFM_df[DFM_df.polygon != 'NAAO1']
    # DFM_df = DFM_df[DFM_df.polygon != 'NAAP2']
    # DFM_df = DFM_df[DFM_df.polygon != 'NAAPF']
    
    DFM_df = DFM_df.loc[(DFM_df.index >= f'{start_year}-03-01') & (DFM_df.index <= f'{end_year}-09-30')]
    
    # ## Comparing the Date Columns of Two Dataframes and Keeping the Rows with the same Dates:
    # common_index = list(set(satellite_df.index).intersection(DFM_df.index))      # Need for DFM + IBI (not NWS)
    # satellite_df = satellite_df.loc[common_index].copy()
    # DFM_df = DFM_df.loc[common_index].copy()
    # print(len(satellite_df), len(DFM_df))
    
    DFM_df  # 52 polygons!


# In[66]:


# # Check that same stations with NaNs removed in both dataframes:
# [x for x in np.unique(DFM_df.polygon) if x not in np.unique(satellite_df.polygon)]  # For NWS need opposite code arrangement!

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

# Display the merged DataFrame
merged_df



# ## Taylor Diagrams

# In[9]:


# path to output folder for plots

outdir = fr'P:\11209810-cmems-nws\figures\taylor_diagrams\{year}' if os.name == 'nt' else r'/p/11209810-cmems-nws/figures/taylor_diagrams/{year}'
if not os.path.exists(outdir):
    os.makedirs(outdir)

# #### Multiple points

# In[19]:

for office in offices:
    # Loop over assessment areas and extract the stats for the Taylor Diagram:
    # Note: LOG_TRANSFORMED!
    
    mean = []
    mean_sat = []
    crmsd = []
    std = []
    ccoef = []
    
    colors = []
    polys = []
    
    for poly in np.unique(merged_df.polygon):
        # print(poly)
    
        ref = merged_df.loc[merged_df['polygon'] == poly].chl_satellite
        # ref = np.log(ref) 
        if office == 'NWS':
            pred = merged_df.loc[merged_df['polygon'] == poly].chl_NWS              # NWS
        elif office == 'IBI':
            pred = merged_df.loc[merged_df['polygon'] == poly].chl_IBI              # IBI
        elif office == 'DFM':
            pred = merged_df.loc[merged_df['polygon'] == poly].chl_DFM              # DFM
        pred.index = pd.to_datetime(pred.index)
        pred.index = pred.index.date
        # pred = np.log(pred)
        
        mean.append(np.mean(pred))
        mean_sat.append(np.mean(ref))
        crmsd.append(sm.centered_rms_dev(pred.values,ref.values) / np.std(ref)) 
        std.append(np.std(pred) / np.std(ref))
        ccoef.append(np.corrcoef(pred.values,ref.values)[0,1])
    
        if poly in plume:
            colors.append('k')
        elif poly in coastal:
            colors.append('y')
        elif poly in shelf:
            colors.append('c')
        elif poly in oceanic:
            colors.append('b')
        else:
            colors.append('m')
    
        polys.append(poly)
        # break
    
    std = np.insert(np.array(std),0,1)
    crmsd = np.insert(np.array(crmsd),0, 0)
    ccoef = np.insert(np.array(ccoef),0, 1)

#%%
    # Create the excel sheet with the output:
    excel_df = pd.DataFrame({'polygon':np.unique(merged_df.polygon),
                                 'mean_sat': mean_sat,
                                 f'mean_{office}': mean,
                                f'nstd_{office}':std[1:],
                                f'ccoef_{office}':ccoef[1:],
                                # f'mse_{office}':mse_NWS,
                                f'rmsd_{office}':crmsd[1:]
                })
        
    # Add 'region' column based on the regions
    excel_df['region'] = excel_df['polygon'].apply(assign_region)
    
    # Drop a column named 'column_name'
    excel_df = excel_df.drop('polygon', axis=1)
    
    # Group by the 'region' column and calculate the mean for the other columns
    excel_df = excel_df.groupby('region').mean()

    # Round all numeric values to 2 decimal places
    excel_df = excel_df.round(2)
    
    #Save to df

    excel_df.to_excel(os.path.join(outdir_tables, fr'{start_year}_{end_year}_satellite_vs_{office}_clustered_regions.xlsx'), sheet_name='map_stats', index=True)

# In[22]:
    # # Plotting
    if taylor == 1:
        
        ## Taylor Diagrams
    
        fig = plt.figure()
        ax = plt.axes()
        
        sm.taylor_diagram(np.array([std[0], std[1]]),np.array([crmsd[0], crmsd[1]]),np.array([ccoef[0], abs(ccoef[1])]),
                          styleOBS = '-',colOBS = 'r', markerobs = 'o', titleOBS = 'Observation', 
                          markerLegend = 'on', markerlabel=['REAN',polys[0]], markerColor=colors[0],
                          widthcor=0.5, widthrms=0.5, widthstd=0.5, labelrms='', titlecor='off',
                          axismax=2) 
        
        #Add the other points
        for d in range(2,len(std)):
            sm.taylor_diagram(np.array([std[0], std[d]]),np.array([crmsd[0], crmsd[d]]),np.array([ccoef[0], abs(ccoef[d])]),
                              styleOBS = '-',colOBS = 'r', markerobs = 'o', titleOBS = 'Observation',
                              markerLegend = 'on', markerlabel=['REAN',polys[d-1]], markerColor=colors[d-1],
                              overlay='on')    
        
        ylabel = plt.ylabel('Normalized Standard Deviation', labelpad=8)
        yticks = plt.yticks([])  # because not accurate anymore if use them (not bent)
        xlim = plt.xlim(left=0)
        title = plt.title(fr'Chlorophyll-a {office} vs satellite in OSPAR assessment areas', y=1.15)
        
        point00 = mlines.Line2D([], [], color='b', linestyle ='--', linewidth=0.5, label='Absolute \n Correlation Coefficient')  # CORR label
        point0 = mlines.Line2D([], [], color='g', linestyle ='--', linewidth=0.5, label='RMSD')  # RMSD label
        
        point1 = mlines.Line2D([], [], color='k', marker='+', markersize=10, label='Plume',linestyle='None')
        point2 = mlines.Line2D([], [], color='y', marker='+', markersize=10, label='Coastal',linestyle='None')
        point3 = mlines.Line2D([], [], color='c', marker='+', markersize=10, label='Shelf',linestyle='None')
        point4 = mlines.Line2D([], [], color='b', marker='+', markersize=10, label='Oceanic',linestyle='None')
        # point5 = mlines.Line2D([], [], color='m', marker='+', markersize=10, label='Not assigned',linestyle='None')
        plt.legend(handles=[point00,point0,point1,point2,point3,point4],bbox_to_anchor=(1, 1))
        
        # plt.tight_layout()
        plt.savefig(os.path.join(outdir, fr'{office}-satellite_{year}_taylor_diagram_CHL_growing_season.png'), dpi=400, bbox_inches='tight')
        print(f'Done with Taylor diagram for {office}.')
        print(' ')
        
        ## Can't figure out how to access the label: Correlation Coefficient... (to rename to: absolute correlation coefficient)

    if target == 1:
        ## Target Diagrams
        
        # In[21]:
        
        
        # path to output folder for plots
        
        outdir = fr'P:\11209810-cmems-nws\figures\target_diagrams\{year}' if os.name == 'nt' else r'/p/11209810-cmems-nws/figures/target_diagrams/{year}'
        if not os.path.exists(outdir):
            os.makedirs(outdir)
              
        #### Multiple Targets
        
        # In[35]:
        
        for office in offices:
            bias = []
            crmsd = []
            rmsd = []
            
            colors = []
            polys = []
            
            for poly in np.unique(merged_df.polygon):
                # print(poly)
            
                ref = merged_df.loc[merged_df['polygon'] == poly].chl_satellite
                if office == 'NWS':
                    pred = merged_df.loc[merged_df['polygon'] == poly].chl_NWS    # NWS
                elif office == 'IBI':
                    pred = merged_df.loc[merged_df['polygon'] == poly].chl_IBI    # IBI
                elif office == 'DFM':
                    pred = merged_df.loc[merged_df['polygon'] == poly].chl_DFM    # DFM
                pred.index = pd.to_datetime(pred.index)
                pred.index = pred.index.date
            
                ## Normalized!
                ts = sm.target_statistics(pred.values,ref.values, norm=True)
                bias.append(ts['bias'])
                crmsd.append(ts['crmsd'])
                rmsd.append(ts['rmsd'])
            
                # Not normalised!  -- Could normalize later, but causes crash due to typo in code... Note: DFM looks much better without norm...?
                # bias.append(sm.bias(pred.values,ref.values)) 
                # crmsd.append(sm.centered_rms_dev(pred.values,ref.values)) 
                # rmsd.append(sm.rmsd(pred.values,ref.values))
            
                if poly in plume:
                    colors.append('k')
                elif poly in coastal:
                    colors.append('y')
                elif poly in shelf:
                    colors.append('c')
                elif poly in oceanic:
                    colors.append('b')
                else:
                    colors.append('m')
            
                polys.append(poly)
                # break
            
            bias = np.array(bias)
            crmsd = np.array(crmsd)
            rmsd = np.array(rmsd)
                   
        
            ## Correct colour-coded targets:
            
            sm.target_diagram(bias[0],  crmsd[0],  rmsd[0],
                              markerLegend = 'on', markerlabel=[polys[0]], markerColor=colors[0],
                              axismax=2, circles=[0,1,2,3,4])      # NWS / IBI
                              # axismax=16, circles=[0,4,8,12,16])   # DFM
            
            #Add the other points
            for d in range(1,len(bias)):
                sm.target_diagram(bias[d],  crmsd[d],  rmsd[d],
                                  markerLegend = 'on', markerlabel=[polys[d]], markerColor=colors[d],
                                  overlay='on', circles=[0,1,2,3,4])     # NWS / IBI
                                  # overlay='on', circles=[0,4,8,12,16])   # DFM
            
            # title = plt.title(f'Chlorophyll-a {model} vs satellite in OSPAR assessment areas', y=1.15)
            plt.title('')
            
            point1 = mlines.Line2D([], [], color='k', marker='+', markersize=10, label='Plume',linestyle='None')
            point2 = mlines.Line2D([], [], color='y', marker='+', markersize=10, label='Coastal',linestyle='None')
            point3 = mlines.Line2D([], [], color='c', marker='+', markersize=10, label='Shelf',linestyle='None')
            point4 = mlines.Line2D([], [], color='b', marker='+', markersize=10, label='Oceanic',linestyle='None')
            # point5 = mlines.Line2D([], [], color='m', marker='+', markersize=10, label='Not assigned',linestyle='None')
            plt.legend(handles=[point1,point2,point3,point4],bbox_to_anchor=(1, 1))
            
            # plt.tight_layout()
            plt.savefig(os.path.join(outdir, f'{office}-satellite_{year}_target_diagram_CHL_growing_season.png'), dpi=400, bbox_inches='tight')
            print(f'Done with Target diagram for {office}.')
            print(' ')
            
            # why not include a line of the observations - should be a ring, no? According to the word doc. But not in other examples on GitHub...
        

        


