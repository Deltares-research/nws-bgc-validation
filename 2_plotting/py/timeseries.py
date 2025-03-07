#!/usr/bin/env python
# coding: utf-8

# Imports

import os
import numpy as np
import xarray as xr
import pandas as pd
import cartopy as cp

import datetime as dt
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.lines as mlines

import skill_metrics as sm

# dfm_tools 
import dfm_tools as dfmt
from dfm_tools.get_nc_helpers import get_ncvarproperties #, get_timesfromnc
from dfm_tools.xarray_helpers import preprocess_hisnc



# Funtions:

def find_nearest_location_from_2D_output(ds, lat, lon):
    '''
    ds needs to be a datarray: 1 variable, 1 depth and 1 timestamp.
    Also, 2D lat/lon needs to be named: latitude and longitude.
    '''
    # Find the index of the grid point nearest a specific lat/lon
    abslat = np.abs(ds.latitude-lat)
    abslon = np.abs(ds.longitude-lon)
    c = np.maximum(abslon, abslat)
    ([xloc], [yloc]) = np.where(c == np.min(c))
    # Then use that index location to get the values at the x/y diminsion
    point_ds = ds.sel(x=yloc, y=xloc)     ### Note: x and y inverted as in netCDF listed as y,x (in that order!)
    return point_ds

def years_in_order(start_year, end_year):
    return list(range(start_year, end_year + 1))

def var_mapping(var):
    if var == 'CHL':
        var_sat = 'CHL'
        var_IBI = 'chl'
        var_NWS = 'chl'
        var_DFM = 'Chlfa'
        var_obs = 'Chlfa'
    elif var == 'OXY':
        var_sat = ''
        var_IBI = 'o2'
        var_NWS = 'o2'
        var_DFM = 'OXY'
        var_obs = 'OXY'
    elif var == 'NO3':
        var_sat = ''
        var_IBI = 'no3'  
        var_NWS = 'no3'  
        var_DFM = 'NO3'
        var_obs = 'NO3'
    elif var == 'PH':
        var_sat = ''
        var_IBI = 'ph'  
        var_NWS = 'ph'  
        var_DFM = 'pH'
        var_obs = 'pH'
    elif var == 'PO4':
        var_sat = ''
        var_IBI = 'po4'  
        var_NWS = 'po4'  
        var_DFM = 'PO4'
        var_obs = 'PO4'
    elif var == 'PCO2':
        var_sat = ''
        var_IBI = 'spco2'  
        var_NWS = 'spco2'  
        var_DFM = 'pCO2water'
        var_obs = 'pCO2'
              
    return var_sat, var_IBI, var_NWS, var_DFM, var_obs

# Function to extract lat and lon from the 'geom' column
def extract_lat_lon(geom):
    # Split the string, assuming it's formatted as 'POINT (lon lat)'
    lon, lat = geom.replace('POINT ', '').strip('()').split()
    return float(lat), float(lon)

#parameter dictionary
station_dict = {
    'WCO_L4_PML': 'L4_PML',
    '74E9_0040 (Liverpool_Bay)': 'LIVBAY',
    'channel_area_france': '10511',
    'north_northsea': 'J12',
    'M15898': 'M444',
    'M5514': 'GE_Westl_Sderpiep',
    'Aa13': 'A13',
    'M10539': 'ANHOLTE',
    'West_Gabbard': 'EN_WESTGAB',}


# Choose the year, model and output directory
start_year = 2015
end_year = 2017
model = 'rea'
selected_years = years_in_order(start_year, end_year)
slice_2d = 'surface' 
taylor = 1 # yes - 1 | no - 0
unit = 'CMEMS'  # CMEMS / DFM
# TODO: Include option to select which models we want to plot... 

# Buffer for surface observations
buffer = 5.5 # (meters) take all measuerements from the top x meters

# Minimum amount of observations per year
min_obs_count = 10 

rootdir = r'P:\11209810-cmems-nws\model_output\combined_yearly' if os.name == 'nt' else r'/p/11209810-cmems-nws/model_output/combined_yearly'

outdir = fr'P:\11209810-cmems-nws\figures\timeseries\{start_year}_{end_year}'
#outdir = fr'P:\11209810-cmems-nws\figures\timeseries\{start_year}_{end_year}\nutrientwad' if os.name == 'nt' else fr'/p/11209810-cmems-nws/figures/timeseries/{start_year}_{end_year}/nutrientwad'
#outdir = fr'P:\11209810-cmems-nws\figures\timeseries\{start_year}_{end_year}\B05_waq_2014_PCO2_ChlC_NPCratios_DenWat_stats_2023.01' if os.name == 'nt' else fr'/p/11209810-cmems-nws/figures/timeseries/{start_year}_{end_year}/B05_waq_2014_PCO2_ChlC_NPCratios_DenWat_stats_2023.01'

if not os.path.exists(outdir):
    os.makedirs(outdir)

## Set start and end time for the timeseries plot
tstart = dt.datetime(start_year,1,1)
tend = dt.datetime(end_year,12,31)


## Plotting 

variables = ['PH', 'NO3', 'PO4', 'OXY', 'CHL']

for var in variables:
    print(fr'Reading NWDM points for {var}')

    var_sat, var_IBI, var_NWS, var_DFM, var_obs = var_mapping(var) #get variable names for each model
    # Read NWDM obs points:
    try:
        obs_path = fr'P:\11209810-cmems-nws\Data\NWDM_observations\combined_years\{start_year}_{end_year}_{var_obs}_obs_{slice_2d}.csv' if os.name == 'nt' else fr'/p/11209810-cmems-nws/Data/NWDM_observations/combined_years/{start_year}_{end_year}_{var_obs}_obs_{slice_2d}.csv'
        obs = pd.read_csv(obs_path)
    except:
        obs_path = fr'P:\11209810-cmems-nws\Data\NWDM_observations\combined_years\2003_2017_{var_obs}_obs_{slice_2d}.csv' if os.name == 'nt' else fr'/p/11209810-cmems-nws/Data/NWDM_observations/combined_years/2003_2017_{var_obs}_obs_{slice_2d}.csv'
        obs = pd.read_csv(obs_path)
    
    obs['datetime'] = pd.to_datetime(obs['datetime'], format='mixed')
    obs['datetime'] = obs['datetime'].astype('datetime64[ns]')
    obs['depth'] = obs['depth'].fillna(0)
    obs['station'] = obs['station'].fillna(obs['geom'])
    obs['station'] = obs['station'].astype('string')
    
    #Select observations in the top layer using buffer
    if slice_2d == 'surface':
        obs = obs[abs(obs['depth']) <= buffer] 
    
    obs = obs.loc[(obs.datetime>=f'{start_year}-01-01') & (obs.datetime<=f'{end_year}-12-31')]
    # Get a unique list of the locations:
    plot_locs = np.unique(np.unique(obs.station.values.astype(str)))

    # obs_locs = obs[obs['station'].isin(plot_locs)]
    # # Drop duplicate stations, keeping only the first occurrence (first time)
    # unique_stations_obs_locs = obs_locs.drop_duplicates(subset='station', keep='first')

    # ## Lats/lons:
    # unique_stations_obs_locs['lat'], unique_stations_obs_locs['lon'] = zip(*unique_stations_obs_locs['geom'].apply(extract_lat_lon))
    # unique_stations_obs_locs = unique_stations_obs_locs[['lat', 'lon']]
     
    
    # Loop over stations
    # Dutch stations / OSPAR paper
    # plot_locs = ['NOORDWK2','NOORDWK10','NOORDWK20','NOORDWK30','NOORDWK50','NOORDWK70',
    #               'ROTTMPT3','ROTTMPT10','ROTTMPT100','ROTTMPT15','ROTTMPT20','ROTTMPT30','ROTTMPT50',
    #               'ROTTMPT70', 'TERSLG10', 'TERSLG50','TERSLG100', 'TERSLG135','TERSLG175','TERSLG20','TERSLG235',
    #               'TERSLG30', 'TERSLG50','TERSLG70','WALCRN2','WALCRN10','WALCRN20','WALCRN30',
    #               'WALCRN50','WALCRN70'] 
        
    # "Rockall_Stn_18", "WES_Stn_347", "WES_Stn_209", "WES_Stn_34", "se_faroe",
    # "WES_Stn_669", "WCO_L4_PML", "74E9_0040 (Liverpool_Bay)", "Bay_of_Biscay_north",
    # "Stonehaven", "Bay_of_Biscay_south", "channel_area_france", "nw_shetland",
    # "north_northsea", "TH1", "TERSLG235", "W01", "NOORDWK70", "TERSLG135",
    # "NOORDWK10", "M15898", "TERSLG50", "EMPM_DK", "M5514", "Aa13", "M10539",
    # "WES_Stn_329", "WES_Stn_104", "WES_Stn_642", "Granville", "Seine_bloom",
    # "CCTI_FR", "West_Gabbard", "GOERE6", "TERSLG10", "ENS_central",
    # "EMPM_DE_1", "CO_central", "GBC_DE_2", "ELPM_DE_1", "Gniben"
    
    # # OSPAR Taylor stations
    # plot_locs = ['M5514', 'NOORDWK70', 'TERSLG50', 'TERSLG235', 'M10539', 'M15898',
    #         'Aa13', 'W03', 'TH1', 'Stonehaven', '45CV',
    #         '74E9_0040 (Liverpool_Bay)', 'NOORDWK10', 'WCO_L4_PML',
    #         'channel_area_france', 'north_northsea', 'nw_shetland',
    #         'WES_Stn_104']
 
    # OSPAR paper station in NWDM
    # plot_locs = [ 'TERSLG50','TERSLG135', 'TERSLG235', 'NOORDWK10','NOORDWK70', 'ROTTMPT70', 'West_Gabbard', 'GOERE6', 'TH1', 'W01', 'Stonehaven', 
    #               'north_northsea', 'M15898',  'WCO_L4_PML',  '74E9_0040 (Liverpool_Bay)', 'channel_area_france', 'M5514', 'Aa13', 'M10539', '15525' ]
    
    excel_df = pd.DataFrame(columns=['station', 'mean_insitu', 'mean_NWS', 'mean_IBI', 'mean_DFM', 'mean_satellite', 
                                     'std_NWS', 'std_IBI', 'std_DFM', 'std_satellite', 'crmsd_NWS', 'crmsd_IBI', 
                                     'crmsd_DFM', 'crmsd_satellite', 'ccoef_NWS', 'ccoef_IBI', 'ccoef_DFM', 'ccoef_satellite'])
    
    l=1
    for loc in plot_locs:  # loop over all locations
        print(' ')    
        print(loc)
        print(' ')

        outfile_ts = os.path.join(outdir, f'ts_{start_year}_{end_year}_{var}_{loc}.png')
        outfile_taylor = os.path.join(outdir, f'taylor_{start_year}_{end_year}_{var}_{loc}.png')
        outfile_table = os.path.join(outdir, f'taylor_{start_year}_{end_year}_{var}.xlsx')
        if os.path.exists(outfile_ts):
            print(f"File {outfile_ts} already exists. Skipping.")

        else:
            ### NWDM OBSERVATIONS:
            if loc in station_dict:
                loc_NWDM = station_dict[loc]
            else:
                loc_NWDM = loc
            station = obs[obs['station'].str.endswith(loc_NWDM, na=False)]
            # remove empty df's
            if len(station) == 0:
                print('No data at this location')
                continue
            if len(station) < (end_year-start_year+1)*min_obs_count:
                print('Not enough data at this location')
                continue            
            try:
                series = station[station['station'].str.endswith(loc_NWDM)] # select 1 station
                series_crop = station.set_index('datetime')
                series_crop = series_crop.value
                series_crop.index = pd.to_datetime(series_crop.index)
                series_crop = series_crop.dropna()
                series_crop = series_crop[(series_crop < series_crop.quantile(0.98))] #Filter to robust percentiles. Cut the top 2 percentile
                
                if var_obs == 'NO3' and unit == 'CMEMS':
                    series_crop = series_crop*1000/14.006720    
                elif var_obs == 'OXY' and unit == 'CMEMS':
                    series_crop = series_crop*1000/31.998
                elif var_obs == 'PO4' and unit == 'CMEMS':
                    series_crop = series_crop*1000/30.973762
                elif var_obs == 'pCO2' and unit == 'CMEMS':
                    series_crop = series_crop*0.101325
                ## Lat/lon:
                latitude = float(series.geom.iloc[0].split(' ')[1:][1][:-1])
                longitude = float(series.geom.iloc[0].split(' ')[1:][0][1:])
            except:
                continue
                
            
            ### Satellites (only Chl!) -- only for growing season for some stations north (where in winter cropped off!)
            if var == 'CHL':
                # multi-year
                office = 'satellite'  
    
                satellite_xr_ds = []
                print(f'Opening {office}_{start_year}_{end_year}')
                for year in selected_years: 
                    basedir = fr'P:\11209810-cmems-nws\Data\OCEANCOLOUR_ATL_BGC_L4_MY_009_118\cmems_obs-oc_atl_bgc-plankton_my_l4-gapfree-multi-1km_P1D_CHL_19.98W-12.98E_48.02N-61.98N_{year}-01-01-{year}-12-31.nc' if os.name == 'nt' else fr'/p/11209810-cmems-nws/Data/OCEANCOLOUR_ATL_BGC_L4_MY_009_118/cmems_obs-oc_atl_bgc-plankton_my_l4-gapfree-multi-1km_P1D_CHL_19.98W-12.98E_48.02N-61.98N_{year}-01-01-{year}-12-31.nc'
                    satellite_xr_year = xr.open_dataset(basedir)
                    satellite_crop = satellite_xr_year[var_sat]
                    satellite_crop = satellite_xr_year.sel(latitude=latitude, longitude=longitude, method='nearest')
                    
                    satellite_xr_ds.append(satellite_crop)
                    
                # Merge:
                print(f'Merging {office}_{start_year}_{end_year}')    
                satellite_xr = xr.concat(satellite_xr_ds, dim='time') 
            
            ### Models
            office = 'NWS'  
            
            NWS_xr_ds = []
            print(f'Opening {slice_2d}_{office}_{model}_{start_year}_{end_year}')
            for year in selected_years: 
                basedir = os.path.join(rootdir,fr'{slice_2d}_{office}_{model}_{year}.nc')
                NWS_xr_year = xr.open_dataset(basedir)
                NWS_crop = NWS_xr_year[var_NWS]#[:,0,:,:]                                             # extract surface
                NWS_crop = NWS_crop.sel(longitude=longitude, latitude=latitude, method='nearest')     # Select point
                #waq_xr = waq_xr.rename({'latitude':'y', 'longitude':'x'})                            # select variable and depth
                NWS_xr_ds.append(NWS_crop)
                
            # Merge:
            print(f'Merging {slice_2d}_{office}_{model}_{start_year}_{end_year}')    
            NWS_xr = xr.concat(NWS_xr_ds, dim='time')     
        
            office = 'IBI'
                
            IBI_xr_ds = []
            print(f'Opening {slice_2d}_{office}_{model}_{start_year}_{end_year}')
            for year in selected_years: 
                basedir = os.path.join(rootdir,fr'{slice_2d}_{office}_{model}_{year}.nc')
                IBI_xr_year = xr.open_dataset(basedir)
                IBI_crop = IBI_xr_year[var_IBI]#[:,0,:,:]                                                    # extract surface
                IBI_crop = find_nearest_location_from_2D_output(ds=IBI_crop, lat=latitude, lon=longitude)   # Select point
                #waq_xr = waq_xr.rename({'latitude':'y', 'longitude':'x'})                            # select variable and depth
                IBI_xr_ds.append(IBI_crop)
                
            # Merge:
            print(f'Merging {slice_2d}_{office}_{model}_{start_year}_{end_year}')    
            IBI_xr = xr.concat(IBI_xr_ds, dim='time') 
    
            if 2015 <= start_year <= 2017 and 2015 <= end_year <= 2017:  # add DFM only for reanalysis!
                try:
                    office = 'DFM'
                    
                    print(f'Opening {office}_{start_year}_{end_year}')
                    
                    DFM_xr_ds = []
                    for year in selected_years: 
                        base_path = r'p:\11210370-011-nutrientwad\04_simulations\waq_runs'
                        for folder in os.listdir(base_path):
                            if folder.startswith(f'waq_{year}'):  # Matches folders starting with waq_{year}
                                DFM_model = os.path.join(base_path, folder, 'DFM_OUTPUT_DCSM-FM_0_5nm_waq')
                                f = os.path.join(DFM_model, r"DCSM-FM_0_5nm_waq_0000_his.nc")  
                        #DFM_model = fr'P:\archivedprojects\11209731-002-nutrient-reduction-ta\runs_OSPAR\B05_waq_withDOM_{year}_spinup\DFM_OUTPUT_DCSM-FM_0_5nm_waq\DCSM-FM_0_5nm_waq_0000_his.nc' if os.name == 'nt' else fr'/p/archivedprojects/11209731-002-nutrient-reduction-ta/runs_OSPAR/B05_waq_withDOM_{year}_spinup/DFM_OUTPUT_DCSM-FM_0_5nm_waq/DCSM-FM_0_5nm_waq_0000_his.nc'
                        #DFM_model = fr'p:\11210284-011-nose-c-cycling\runs_fine_grid\B05_waq_{year}_PCO2_ChlC_NPCratios_DenWat_stats_2023.01\DFM_OUTPUT_DCSM-FM_0_5nm_waq\DCSM-FM_0_5nm_waq_0000_his.nc' if os.name == 'nt' else fr'/p/11210284-011-nose-c-cycling/runs_fine_grid/B05_waq_{year}_PCO2_ChlC_NPCratios_DenWat_stats_2023.01/DFM_OUTPUT_DCSM-FM_0_5nm_waq/DCSM-FM_0_5nm_waq_0000_his.nc'
                        #DFM_model = fr'p:\11210284-011-nose-c-cycling\runs_fine_grid\B05_waq_{year}_PCO2_DenWat_stats_2023.01\DFM_OUTPUT_DCSM-FM_0_5nm_waq\DCSM-FM_0_5nm_waq_0000_his.nc'
                        DFM_xr = xr.open_mfdataset(f, preprocess=preprocess_hisnc)
                        
                        DFM_xr = dfmt.rename_waqvars(DFM_xr)
                        DFM_crop = DFM_xr[var_DFM].sel(stations=loc, laydim=-1)                               # select point and surface
                        
                        if var == 'NO3' and unit == 'CMEMS':               
                            DFM_crop = DFM_crop*1000/14.006720 
                        elif var == 'PO4' and unit == 'CMEMS':               
                            DFM_crop = DFM_crop*1000/30.973762 
                        elif var == 'OXY' and unit == 'CMEMS':
                            DFM_crop = DFM_crop*1000/31.998 
                        elif var == 'PCO2' and unit == 'CMEMS':
                            DFM_crop = DFM_crop*1000/(12+2*32)  
                    
                        DFM_xr_ds.append(DFM_crop)        
                    
                    print(f'Merging {office}_{start_year}_{end_year}')
                    DFM_xr = xr.concat(DFM_xr_ds, dim='time')
                        
                except Exception as e:
                    print(f"An error occurred: {e}")
                    pass
            
            # Plotting:
            print('Plotting')
            fig = plt.figure(figsize=(9,5), dpi=300)
            plt.rcParams.update({'font.size': 13}) 
        
            left_margin  = 1.25 / 9.
            right_margin = 0.2 / 9.
            bottom_margin = 0.5 / 5.
            top_margin = 0.6 / 5.
        
            # dimensions are calculated relative to the figure size
            x = left_margin    # horiz. position of bottom-left corner
            y = bottom_margin  # vert. position of bottom-left corner
            w = 1 - (left_margin + right_margin) # width of axes
            h = 1 - (bottom_margin + top_margin) # height of axes
            ax = fig.add_axes([x, y, w, h])
            
            ## Plotting NWDM obs:
            ax.plot(series_crop.index, series_crop, color='k', zorder=100, linestyle='None', marker='.', markersize=8, label = 'NWDM observations')
    
            # Plotting satellite chl:
            if var == 'CHL':
                ax.scatter(satellite_xr.time, satellite_xr['CHL'], color='grey', linestyle='None', marker='.', s=12, label = 'Satellites')
            
            ## Plotting models:
            ax.plot(NWS_xr.time, NWS_xr.values, linestyle='-', color='indianred', linewidth=1, label = 'NWS model')   
            ax.plot(IBI_xr.time, IBI_xr.values, linestyle='-', color='slateblue', linewidth=1, label = 'IBI model') 
            if 2015 <= start_year <= 2017 and 2015 <= end_year <= 2017:
                try:
                    ax.plot(DFM_xr.time, DFM_xr.values, linestyle='-', color='green', linewidth=2, label = 'DFM model')  
                except Exception as e:
                    print(f"An error occurred: {e}")
                    pass        
            # y-axis settings
            ax.set_ylabel(NWS_xr.attrs['long_name'])          
            #ax.set_ylim(ymin=0)
            # x-axis settings    
            ax.set_xlim([tstart, tend])    
            myFmt = mdates.DateFormatter('%b%y')
        #     myFmt = mdates.DateFormatter('%b')
            ax.xaxis.set_major_formatter(myFmt)
            ax.tick_params(axis="x")
            plt.xticks(rotation=45)
    
            # Add location map
            ax2 = fig.add_axes([0.97,0.1,0.3,0.3], projection=ccrs.PlateCarree())
            ax2.set_extent([-10,13,48,62])
            ax2.coastlines(resolution='10m')
            ax2.add_feature(cp.feature.LAND, edgecolor='k')
            ax2.add_feature(cp.feature.BORDERS, edgecolor='k', linewidth=0.8)
            
            ax2.plot(longitude,latitude, 'rx', ms=7, mew=2)
    
            # Add a legend and title
            leg = ax.legend(bbox_to_anchor=(1.03,0.73,1,0.2), loc="lower left", borderaxespad=0, fontsize = 10)  # with satellites legend
            # leg = ax.legend(bbox_to_anchor=(1.03,0.78,1,0.2), loc="lower left", borderaxespad=0, fontsize = 10)  # no satellites legend
            ax.set_title(f'{loc}\n') # add title
            plt.gcf().subplots_adjust(left=0.5)
            plt.tight_layout()
            # save plot
            plt.savefig(outfile_ts, dpi=400, bbox_inches='tight')
            print('Time series figure saved')
            plt.close()
            
            if taylor == 1:
                
                if os.path.exists(outfile_taylor):
                    print(f"File {outfile_taylor} already exists. Skipping.")
                else:
                
                    ## Taylor Diagrams
                    
                    #Merge the the different time series
                    merged_df=series_crop.to_frame(name='NWDM')
                    merged_df.index = merged_df.index.date  # Convert index to date
                    # Convert the index to a DatetimeIndex
                    merged_df.index = pd.to_datetime(merged_df.index)
                    
                    # Extract the month from the 'datetime' column
                    merged_df['month'] = merged_df.index.month

                    # Calculate multi-annual monthly means
                    merged_df =  merged_df.groupby(['month'])['NWDM'].mean().reset_index()
        
                    # Convert NWS to DataFrame
                    NWS_df = NWS_xr.to_dataframe(name='NWS')
                    NWS_df = NWS_df[['NWS']]          
                    NWS_df.index = NWS_df.index.date  # Convert index to date
                    # Convert the index to a DatetimeIndex
                    NWS_df.index = pd.to_datetime(NWS_df.index)
                    
                    # Extract the month from the 'datetime' column
                    NWS_df['month'] = NWS_df.index.month
                    
                    # Calculate multi-annual monthly means
                    NWS_df =  NWS_df.groupby(['month'])['NWS'].mean().reset_index()
                    # Merge DataFrames on the time index
                    merged_df = merged_df.merge(NWS_df, on='month')
                    
                    # Convert IBI to DataFrame
                    IBI_df = IBI_xr.to_dataframe(name='IBI')
                    IBI_df = IBI_df[['IBI']]          
                    IBI_df.index = IBI_df.index.date  # Convert index to date
                    # Convert the index to a DatetimeIndex
                    IBI_df.index = pd.to_datetime(IBI_df.index)
                    
                    # Extract the month from the 'datetime' column
                    IBI_df['month'] = IBI_df.index.month
                    
                    # Calculate multi-annual monthly means
                    IBI_df =  IBI_df.groupby(['month'])['IBI'].mean().reset_index()
                    # Merge DataFrames on the time index
                    merged_df = merged_df.merge(IBI_df, on='month')
                    
                    if 2015 <= start_year <= 2017 and 2015 <= end_year <= 2017:
                        try:
                            # Convert DFM to DataFrame
                            DFM_df = DFM_xr.to_dataframe(name='DFM')
                            DFM_df = DFM_df[['DFM']]          
                            DFM_df.index = DFM_df.index.date  # Convert index to date
                            # Convert the index to a DatetimeIndex
                            DFM_df.index = pd.to_datetime(DFM_df.index)
                            
                            # Extract the month from the 'datetime' column
                            DFM_df['month'] = DFM_df.index.month
                            
                            # Calculate multi-annual monthly means
                            DFM_df =  DFM_df.groupby(['month'])['DFM'].mean().reset_index()
                            # Merge DataFrames on the time index
                            merged_df = merged_df.merge(DFM_df, on='month')
                            
                        except Exception:
                            print('DFM results not available')
                            pass 
                    
                    if var == 'CHL':
                        # Convert satellite to DataFrame
                        sat_df = satellite_xr['CHL'].to_dataframe(name='satellite')
                        sat_df = sat_df[['satellite']]          
                        sat_df.index = sat_df.index.date  # Convert index to date
                        # Convert the index to a DatetimeIndex
                        sat_df.index = pd.to_datetime(sat_df.index)
                        
                        # Extract the month from the 'datetime' column
                        sat_df['month'] = sat_df.index.month
                        
                        # Calculate multi-annual monthly means
                        sat_df =  sat_df.groupby(['month'])['satellite'].mean().reset_index()
                        # Merge DataFrames on the time index
                        merged_df = merged_df.merge(sat_df, on='month')
                    
                    #Convert all columns to float (or desired dtype)
                    merged_df = merged_df.astype(float)
                    
                    #Check if any of the columns have only nans. if yes, then skip the loop
                    if 'NWS' in merged_df.columns[merged_df.isna().all()] or 'IBI' in merged_df.columns[merged_df.isna().all()]:
                        print('Not all models are available at this station. No Taylor diagramms are produced.')
                        continue
        
                    #Derive statistics         
                    mean_insitu = []
                    mean = []
                    crmsd = []
                    std = []
                    ccoef = []
                    
                    colors = []
                    offices_names = []
                    
                    ref = merged_df['NWDM'].copy()
                    
                    for office in ['NWS', 'IBI', 'DFM', 'satellite']:
                    
                        if office == 'NWS':
                            pred = merged_df['NWS']             # NWS
                        elif office == 'IBI':
                            pred = merged_df['IBI']              # IBI
                        elif office == 'DFM':
                            try:
                                pred = merged_df['DFM']              # DFM
                            except Exception:
                                print('No DFM data')
                                pred = ref.copy()
                                pred.name = 'DFM'
                                pred[:] = np.nan
                        elif office == 'satellite':
                            try:
                                pred = merged_df['satellite']        # satellite    
                            except Exception:
                                print('No satellite data')
                                pred = ref.copy()
                                pred.name = 'satellite'
                                pred[:] = np.nan
                        
                        # Drop NaN values
                        ref_pred_clean = pd.concat([ref, pred], axis=1)
                        ref_pred_clean.columns = ['ref', 'pred']
                        ref_pred_clean = ref_pred_clean.dropna()
                        
                        if len(ref_pred_clean) < 2:
                            mean_insitu.append(np.mean(ref_pred_clean['ref']))
                            mean.append(np.mean(ref_pred_clean['pred']))
                            crmsd.append(np.nan) 
                            std.append(np.nan)
                            ccoef.append(np.nan)                 
                        else:
                            mean_insitu.append(np.mean(ref_pred_clean['ref']))
                            mean.append(np.mean(ref_pred_clean['pred']))
                            crmsd.append(sm.centered_rms_dev(ref_pred_clean['pred'],ref_pred_clean['ref']) / np.std(ref_pred_clean['ref'])) 
                            std.append(np.std(ref_pred_clean['pred']) / np.std(ref_pred_clean['ref']))
                            ccoef.append(np.corrcoef(ref_pred_clean['pred'],ref_pred_clean['ref'])[0,1])
                    
                        if office == 'NWS':
                            colors.append('k')
                        elif office == 'IBI':
                            colors.append('y')
                        elif office == 'DFM':
                            colors.append('c')
                        elif office == 'satellite':
                            colors.append('b')
                        else:
                            colors.append('m')
                    
                        offices_names.append(office)
                        # break
                    
                    mean = np.insert(np.array(mean),0,np.nanmean(mean_insitu))
                    std = np.insert(np.array(std),0,1)
                    crmsd = np.insert(np.array(crmsd),0, 0)
                    ccoef = np.insert(np.array(ccoef),0, 1)
                    ccoef[ccoef < 0] = 0 #replace negative correlation coefficients with zero
        
                    #store metrics
                    
                    # Create the dataframe
                    station_df = pd.DataFrame({'station':[loc],
                                                 'mean_insitu': mean[0],
                                                 'mean_NWS': mean[1],
                                                 'mean_IBI': mean[2],
                                                 'mean_DFM': mean[3],
                                                 'mean_satellite': mean[4],
                                                 'std_NWS': std[1],
                                                 'std_IBI': std[2],
                                                 'std_DFM': std[3],
                                                 'std_satellite': std[4],
                                                 'crmsd_NWS': crmsd[1],
                                                 'crmsd_IBI': crmsd[2],
                                                 'crmsd_DFM': crmsd[3],
                                                 'crmsd_satellite': crmsd[4],
                                                 'ccoef_NWS': ccoef[1],
                                                 'ccoef_IBI': ccoef[2],
                                                 'ccoef_DFM': ccoef[3],
                                                 'ccoef_satellite': ccoef[4],
                                })
                    
                    #station_df = station_df.set_index('station')
                    
                    # Round all numeric values to 2 decimal places
                    station_df = station_df.round(2)
                    
                    # Concatenate the new rows with the original DataFrame
                    excel_df = pd.concat([excel_df, station_df], ignore_index=True)
                    
        
                    #Plotting
                
                    fig = plt.figure()
                    ax = plt.axes()
                    
                    sm.taylor_diagram(np.array([std[0], std[1]]),np.array([crmsd[0], crmsd[1]]),np.array([ccoef[0], abs(ccoef[1])]),
                                      styleOBS = '-',colOBS = 'r', markerobs = 'o', titleOBS = 'Observation', 
                                      markerLegend = 'on', markerlabel=['REAN',offices_names[0]], markerColor=colors[0],
                                      widthcor=0.5, widthrms=0.5, widthstd=0.5, labelrms='', titlecor='off',
                                      axismax=3) 
                    
                    #Add the other points
                    for d in range(2,len(std)):
                        sm.taylor_diagram(np.array([std[0], std[d]]),np.array([crmsd[0], crmsd[d]]),np.array([ccoef[0], abs(ccoef[d])]),
                                          styleOBS = '-',colOBS = 'r', markerobs = 'o', titleOBS = 'Observation',
                                          markerLegend = 'on', markerlabel=['REAN',offices_names[d-1]], markerColor=colors[d-1],
                                          overlay='on')    
                    
                    ylabel = plt.ylabel('Normalized Standard Deviation', labelpad=8)
                    yticks = plt.yticks([])  # because not accurate anymore if use them (not bent)
                    xlim = plt.xlim(left=0)
                    title = plt.title(fr'{var} - predictions vs observation at {loc}', y=1.15)
                    
                    point00 = mlines.Line2D([], [], color='b', linestyle ='--', linewidth=0.5, label='Absolute \n Correlation Coefficient')  # CORR label
                    point0 = mlines.Line2D([], [], color='g', linestyle ='--', linewidth=0.5, label='NRMSD')  # RMSD label
                    
                    point1 = mlines.Line2D([], [], color='k', marker='+', markersize=10, label='NWS',linestyle='None')
                    point2 = mlines.Line2D([], [], color='y', marker='+', markersize=10, label='IBI',linestyle='None')
                    point3 = mlines.Line2D([], [], color='c', marker='+', markersize=10, label='DFM',linestyle='None')
                    point4 = mlines.Line2D([], [], color='b', marker='+', markersize=10, label='Satellite',linestyle='None')
                    # point5 = mlines.Line2D([], [], color='m', marker='+', markersize=10, label='Not assigned',linestyle='None')
                    plt.legend(handles=[point00,point0,point1,point2,point3,point4],bbox_to_anchor=(1, 1))
                    
                    #save figure
                    plt.savefig(outfile_taylor, dpi=400, bbox_inches='tight')
                    
                    # Drop columns where all values are NaN
                    excel_df = excel_df.dropna(axis=1, how='all')
                    #save excel file
                    excel_df.to_excel(outfile_table, sheet_name='timeseries_stats', index=True)
                    
    # Calculate column averages for numeric columns
    averages = excel_df.select_dtypes(include='number').mean()
    
    # Append the row with averages
    averages['station'] = 'Average'  # Add a label for the 'station' column
    excel_df = pd.concat([excel_df, pd.DataFrame([averages])], ignore_index=True)
    #save excel file
    excel_df.to_excel(outfile_table, sheet_name='timeseries_stats', index=True)