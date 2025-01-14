# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 10:40:31 2025

@author: lorinc
"""
import pandas as pd
import skill_metrics as sm
import numpy as np
import matplotlib.pyplot as plt
import os
import re

# Define the time interval
start_time = pd.to_datetime('2009-01-01')
end_time = pd.to_datetime('2014-12-31')

#Open OSPAR data
obs_path = r"p:\11209810-cmems-nws\Data\OSPAR_paper\OSPAR_station_data.csv" 
obs = pd.read_csv(obs_path)
#Process OSPAR data
obs['Time'] = pd.to_datetime(obs['Time'])
obs['Station'] = obs['Station'].astype('string')

obs = obs[['Station', 'Time', 'Variable', 'Value']]
obs = obs.rename(columns={
    'Value': 'obs',
})
# Subset the DataFrame based on the time interval
obs = obs[(obs['Time'] >= start_time) & (obs['Time'] <= end_time)]
# Filter out PO4 values greater than 3
obs = obs[~((obs['Variable'] == 'PO4') & (obs['obs'] > 3.0))]

# Extract the month from the 'datetime' column
obs['month'] = obs['Time'].dt.month

# Calculate multi-annual monthly means
obs = obs.groupby(['Station','Variable','month'])['obs'].mean().reset_index()

#Open model data
#DFM
DFM_path = r'p:\11209810-cmems-nws\Data\OSPAR_paper\timeseries_CSV_20210927'  # Replace with your folder path
# List all files in the folder
files = [f for f in os.listdir(DFM_path) if f.endswith('.csv')]

# Initialize an empty list to collect data from all files
all_data = []

# Loop through each CSV file
for file in files:
    # Create the full file path
    file_path = os.path.join(DFM_path, file)
    
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    
    # Add a new column for the station name (the file name without the '.csv' extension)
    df['Station'] = re.sub(r'_\d{4}$', '', file.split('.')[0])  # Remove the year (e.g., _2009)
    
    # Append the DataFrame to the list
    all_data.append(df)

# Combine all DataFrames into one
DFM = pd.concat(all_data, ignore_index=True)
DFM = DFM[['date', 'DIN', 'PO4', 'Chlfa', 'Station']]
DFM = DFM.rename(columns={
    'DIN': 'NO3',
    'Chlfa': 'CHL',
    'date': 'Time'
    })

# Melting the dataframe to collapse the three columns
DFM = DFM.melt(id_vars=[col for col in DFM.columns if col not in ['NO3', 'PO4', 'CHL']],
                                   value_vars=['NO3', 'PO4', 'CHL'],
                                   var_name='Variable',
                                   value_name='DFM')

DFM['Time'] = pd.to_datetime(DFM['Time'])
DFM = DFM[(DFM['Time'] >= start_time) & (DFM['Time'] <= end_time)]
DFM.replace(-999.0, np.nan, inplace=True)

# Extract the month from the 'datetime' column
DFM['month'] = DFM['Time'].dt.month

# Calculate multi-annual monthly means
DFM = DFM.groupby(['Station','Variable','month'])['DFM'].mean().reset_index()

#NWS
NWS_path = r"p:\11209810-cmems-nws\Data\OSPAR_paper\NWS_2009_2014.csv" 
NWS = pd.read_csv(NWS_path)
NWS = NWS.rename(columns={
    'Value': 'NWS',
    'time': 'Time'
})
NWS['Time'] = pd.to_datetime(NWS['Time'])
NWS = NWS[['Station', 'Time', 'Variable', 'NWS']]
NWS = NWS[(NWS['Time'] >= start_time) & (NWS['Time'] <= end_time)]

# Extract the month from the 'datetime' column
NWS['month'] = NWS['Time'].dt.month

# Calculate multi-annual monthly means
NWS = NWS.groupby(['Station','Variable','month'])['NWS'].mean().reset_index()

#IBI
IBI_path = r"p:\11209810-cmems-nws\Data\OSPAR_paper\IBI_2009_2014.csv" 
IBI = pd.read_csv(IBI_path)
IBI = IBI.rename(columns={
    'Value': 'IBI',
    'time': 'Time'
})
IBI['Time'] = pd.to_datetime(IBI['Time'])
IBI = IBI[['Station', 'Time', 'Variable', 'IBI']]
IBI = IBI[(IBI['Time'] >= start_time) & (IBI['Time'] <= end_time)]

# Extract the month from the 'datetime' column
IBI['month'] = IBI['Time'].dt.month

# Calculate multi-annual monthly means
IBI = IBI.groupby(['Station','Variable','month'])['IBI'].mean().reset_index()

#Merge models to observatin table
# Merge the two DataFrames on 'date', 'station', and 'variable'
merged_df = pd.merge(obs, NWS, on=['Station','Variable', 'month'], how='inner')
merged_df = pd.merge(merged_df, IBI, on=['Station','Variable', 'month'], how='inner')
merged_df = pd.merge(merged_df, DFM, on=['Station','Variable', 'month'], how='inner')
# merged_df = pd.merge(obs, DFM, on=['Station','Variable', 'month'], how='inner')
 
# Assign marker shapes for variables
variable_markers = {
    "PO4": "o",        # Circle
    "NO3": "s",      # Square
    "CHL": "D",           # Diamond
}

# Assign colors for models
model_colors = {
    "NWS": "blue",
    "IBI": "green",
    "DFM": "red",
}

# Initialize the plot
fig, ax = plt.subplots(figsize=(8, 8))

# Calculate the aspect ratio (1/3 for x-axis to y-axis)
aspect_ratio = 1 / 3

# Add concentric circles
circle_intervals = [0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1]
for r in circle_intervals:
    adjusted_radius_x = r * aspect_ratio
    circle = plt.Circle((0, 0), r, color="gray", linestyle="--", fill=False, alpha=0.7)
    ax.add_artist(circle)

# Create handles for the legends
variable_handles = []
model_handles = []

#model=merged_df.copy()
model=merged_df.copy()

# Loop through each variable and model in the DataFrame
for variable in model['Variable'].unique():
    print(f'{variable}:')
    variable_data = model[model['Variable'] == variable]
    ref = variable_data['obs'].values
    ref_std = np.std(ref)
    ref_mean = np.mean(ref)
    # print(f'obs mean: {round(ref_mean,2)} for {variable}')

    # Add a marker shape to the variable legend only once
    variable_handles.append(plt.Line2D([], [], color="black", marker=variable_markers[variable], linestyle="None", markersize=10, label=variable))

    for model_name in ['IBI', 'NWS', 'DFM']:
        model_values = variable_data[model_name].values  # Fix the error here: .values (no parentheses)

        # Calculate statistics
        # rmsd = np.sqrt(np.mean((model_values - obs) ** 2))  # RMSD
        # nrmsd = rmsd / obs_std  # Normalized RMSD
        mean = np.mean(model_values)
        nrmsd = sm.centered_rms_dev(model_values, ref) / ref_std
        corr_coeff = np.corrcoef(model_values, ref)[0, 1]  # Correlation Coefficient
        one_minus_corr = 1 - corr_coeff  # 1 - Correlation
        # print(f'mean: {round(mean,2)} for {model_name} and {variable}')
        print(f'nrmsd: {round(nrmsd,2)} for {model_name}')
        print(f'corr_coeff: {round(corr_coeff,2)} for {model_name} and {variable}')
        
        # Plot the point with the corresponding marker and color
        ax.scatter(
            one_minus_corr,  # Switch X-axis to 1 - Correlation
            nrmsd,           # Switch Y-axis to Normalized RMSD
            color=model_colors[model_name],
            marker=variable_markers[variable],
            s=100,
        )

        # Add a color to the model legend only once
        if variable == "CHL":  # Ensure models are added only once
            model_handles.append(plt.Line2D([], [], color=model_colors[model_name], marker="o", linestyle="None", markersize=10, label=model_name))

# Customize plot
ax.set_xlim(0, 1.0)  # For 1 - correlation coefficient
ax.set_ylim(0, 2.1)  # For normalized RMSD
ax.set_xlabel("(1 - Correlation Coefficient)", fontsize=16)
ax.set_ylabel("Normalized RMSD", fontsize=16)
# Set tick labels font size
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# Remove the rectangular grid background
ax.grid(False)

# Customize plot aspect to avoid stretching
ax.set_aspect("auto")  # Allow axes to scale independently

# Add title
plt.title("Taylor Diagram", fontsize=14)

# Add legends
legend1 = ax.legend(handles=variable_handles, title="Variables", loc="upper left", fontsize=10)
legend2 = ax.legend(handles=model_handles, title="Models", loc="upper right", fontsize=10)
ax.add_artist(legend1)  # Ensure both legends appear

#Savefig
plt.savefig(r'p:\11209810-cmems-nws\Data\OSPAR_paper\OSPAR_Taylor.png', dpi=400, bbox_inches='tight')

# Show the plot
plt.show()
