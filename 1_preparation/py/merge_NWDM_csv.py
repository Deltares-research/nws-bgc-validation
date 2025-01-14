# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 15:18:52 2024

@author: lorinc
"""

import os
import pandas as pd

#%%

#Functions
def list_and_filter_files(rootdir, variable, depth, start_year, end_year):
    # List all files in the directory
    all_files = os.listdir(rootdir)
    
    # Filter files that match the pattern
    filtered_files = [
        f for f in all_files 
        if f.endswith('.csv') 
        and variable in f 
        and depth in f
        and (gridded is False or 'gridded' in f)
        and start_year <= int(f.split('_')[0]) <= end_year
    ]
    
    return filtered_files

def combine_files(rootdir, files):
    # Read and concatenate the CSV files
    dfs = [pd.read_csv(os.path.join(rootdir, file)) for file in files]
    
    # Concatenate along the time axis (assuming a time column exists and needs to be parsed)
    combined_df = pd.concat(dfs, ignore_index=True)
    
    return combined_df

#%%
#Input
rootdir =  r'p:\11209810-cmems-nws\Data\NWDM_observations' if os.name == 'nt' else r'/p/11209810-cmems-nws/Data/NWDM_observations'
outdir = os.path.join(rootdir, 'combined_years')
start_year = 2021
end_year = 2022

#gridded or not
gridded = False #True - yes, False - no

variables = ['pH', 'Chlfa', 'OXY', 'NO3', 'PO4', 'pCO2'] #
#variables = ['NO3', 'PO4']
depth = 'surface' #surface, bottom

#%%
#Loop
for variable in variables:
    
    print(f'Running {variable} for {start_year}_{end_year}')
    
    # Get the list of filtered files
    filtered_files = list_and_filter_files(rootdir, variable, depth, start_year, end_year)
    
    # Combine the filtered files
    combined_df = combine_files(rootdir, filtered_files)
    
    print(f'Saving combined .csv for {variable} for {start_year}_{end_year}')
    
    # Save the combined DataFrame to a new CSV file   
    # Generate the output filename
    if gridded == True:
        output_filename = f"{start_year}_{end_year}_{variable}_obs_gridded_{depth}.csv"
    else:
        output_filename = f"{start_year}_{end_year}_{variable}_obs_{depth}.csv"

    # Save the combined DataFrame to a new CSV file
    combined_df.to_csv(os.path.join(outdir, output_filename), index=False)
    print(f"Combined DataFrame saved to {output_filename}")
    print(' ')

