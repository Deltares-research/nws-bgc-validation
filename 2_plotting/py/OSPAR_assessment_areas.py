#!/usr/bin/env python
# coding: utf-8

#%%

import pandas as pd
import geopandas as gpd
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import cartopy.feature as cfeature

# read assessment areas:
area_shp = gpd.read_file(r'P:\11209810-cmems-nws\OSPAR_areas\COMP4_assessment_areas_v8a.shp')

#%%

fig = plt.figure(figsize=(8,6))
ax = plt.axes(projection=ccrs.PlateCarree())                     # create ax + select map projection
ax.coastlines(linewidth=0.8)                                     # add the coastlines
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True)      # add the longitude / latitude lines
gl.right_labels = False                                          # remove latitude labels on the right
gl.top_labels = False                                            # remove longitude labels on the top
gl.xlines = False
gl.ylines = False
ax.add_feature(cfeature.LAND, zorder=1, edgecolor='k', facecolor='linen')      # add land mask
ax.add_feature(cfeature.BORDERS, linewidth=0.8)
ax.set_extent([-16,15, 42,64])

# OSPAR assessment areas:
area_shp.plot(ax=ax, linewidth = 1, facecolor = (1, 1, 1, 0), edgecolor = (0.5, 0.5, 0.5, 1))


#%%

area_shp = gpd.read_file(r'P:\11209810-cmems-nws\OSPAR_areas\COMP4_assessment_areas_v8a.shp')

# import excel file with categories:
categories = pd.read_excel(r'P:/11209810-cmems-nws/OSPAR_areas/Assessment_area_categories.xlsx',
                           sheet_name='Assessment_area_categories')

# Sort the ID columns alphabetically so both dataset habe the same order
gdf = area_shp.sort_values('ID')
categories = categories.sort_values('UnitCode')

# Add category to the df:
gdf['category'] = categories.Category.values


# In[26]:
# subdivide gdf's per category:

gdf_plume = gdf[gdf['category'].str.contains("Plume")]
gdf_shelf = gdf[gdf['category'].str.contains("Shelf")]
gdf_coastal = gdf[gdf['category'].str.contains("Coastal")]
gdf_oceanic = gdf[gdf['category'].str.contains("Oceanic")]


# In[33]:
# plotting different categories

fig = plt.figure(figsize=(8,6))
ax = plt.axes(projection=ccrs.PlateCarree())                     # create ax + select map projection
ax.coastlines(linewidth=0.8)                                     # add the coastlines
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True)      # add the longitude / latitude lines
gl.right_labels = False                                          # remove latitude labels on the right
gl.top_labels = False                                            # remove longitude labels on the top
gl.xlines = False
gl.ylines = False
ax.add_feature(cfeature.LAND, zorder=1, edgecolor='k', facecolor='linen')      # add land mask
ax.add_feature(cfeature.BORDERS, linewidth=0.8)
ax.set_extent([-17,15, 34,64])

# OSPAR assessment areas:
gdf_plume.plot(ax=ax, linewidth = 1, facecolor = 'k')
gdf_shelf.plot(ax=ax, linewidth = 1, facecolor = 'y')
gdf_coastal.plot(ax=ax, linewidth = 1, facecolor = 'c')
gdf_oceanic.plot(ax=ax, linewidth = 1, facecolor = 'b')

plt.savefig(r'P:\11209810-cmems-nws\figures\OSPAR_regions_and_categories\regions_and_categories.png', dpi=400, bbox_inches='tight')


# In[ ]:




