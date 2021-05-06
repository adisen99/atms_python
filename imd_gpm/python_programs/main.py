"""
Python program for plotting GPM data by masking import
and to plot the total error, RMSE and monthly time series for IMD and GPM data
"""

## Importing packages
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import netCDF4
import pandas as pd
from functions import *

import scipy.interpolate as interp

## Colormap selection
xr.set_options(cmap_divergent='bwr', cmap_sequential='turbo')

## Extracting GPM data for 2009 and 2019
mfdataDIR1 = '../data/GPM/2009/3B-MO.MS.MRG.3IMERG.*.V06B.HDF5.SUB.nc4'
mfdataDIR2 = '../data/GPM/2019/3B-MO.MS.MRG.3IMERG.*.V06B.HDF5.SUB.nc4'

ds1 = xr.open_mfdataset(mfdataDIR1, parallel=True)
ds2 = xr.open_mfdataset(mfdataDIR2, parallel=True)

## Masking and Plotting GPM data for 2009 and 2019

# Convert precipitation rate to precipitation
ds1 = convert_to_precipitaion(ds1)
ds2 = convert_to_precipitaion(ds2)

# Transpose the data to get lat first and lon after - 
ds1 = ds1.transpose("time", "lat", "lon")
ds2 = ds2.transpose("time", "lat", "lon")

# Get India data
ds1_ind = ds1.sel(lat=slice(7,36), lon=slice(67,98)).dropna("time")
ds2_ind = ds2.sel(lat=slice(7,36), lon=slice(67,98)).dropna("time")

# Get seasonal mean

ds1_ind_sm = season_mean(ds1_ind)
ds2_ind_sm = season_mean(ds2_ind)

# Convert to dataarray

da1 = ds1_ind_sm.precipitation
da2 = ds2_ind_sm.precipitation

# Masking the data using function from masking
shp_dir = '../shapefiles'
da1_ind = add_shape_coord_from_data_array(da1, shp_dir, "awash")
awash_da1 = da1_ind.where(da1_ind.awash==0, other=np.nan)
da2_ind = add_shape_coord_from_data_array(da2, shp_dir, "awash")
awash_da2 = da2_ind.where(da2_ind.awash==0, other=np.nan)

# Plotting for 2009 GPM

fig = plt.figure(figsize=(20, 15))
fig.tight_layout()

titles = ["MAM", "JJA", "SON", "DJF"]

for i,season in enumerate(titles):

    ax = fig.add_subplot(2, 2, i+1, projection=ccrs.PlateCarree())
    ax.set_extent([67, 98, 7, 36], crs=ccrs.PlateCarree())
    awash_da1.sel(season=titles[i]).plot()
    gridliner = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True)
    gridliner.top_labels = False
    gridliner.bottom_labels = True
    gridliner.left_labels = True
    gridliner.right_labels = False
    gridliner.ylines = False  # you need False
    gridliner.xlines = False  # you need False
    ax.set_title("Months"+ " " + "-" + " " + "("+titles[i]+")", pad=10, fontsize=14)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS)

fig.suptitle('Precipitation over India (in mm) year 2009', fontsize=20, y=0.95)

plt.savefig('./images/GPM2009.png')

# Plotting for FPM 2019

fig = plt.figure(figsize=(20, 15))
fig.tight_layout()
titles = ["MAM", "JJA", "SON", "DJF"]

for i,season in enumerate(titles):
 
    ax = fig.add_subplot(2, 2, i+1, projection=ccrs.PlateCarree())
    ax.set_extent([67, 98, 7, 36], crs=ccrs.PlateCarree())
    awash_da2.sel(season=titles[i]).plot()
    gridliner = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True)
    gridliner.top_labels = False
    gridliner.bottom_labels = True
    gridliner.left_labels = True
    gridliner.right_labels = False
    gridliner.ylines = False  # you need False
    gridliner.xlines = False  # you need False
    ax.set_title("Months"+ " " + "-" + " " + "("+titles[i]+")", pad=10, fontsize=14)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS)

fig.suptitle('Precipitation over India (in mm) year 2019', fontsize=20, y=0.95)
plt.savefig('./images/GPM2019.png')

## Importing IMD data

data3 = '../data/IMD/_Clim_Pred_LRF_New_RF25_IMD0p252009.nc'
data4 = '../data/IMD/_Clim_Pred_LRF_New_RF25_IMD0p252019.nc'

ds3 = xr.open_dataset(data3)
ds4 = xr.open_dataset(data4)

# rename dimension names
ds3_ind = ds3.rename({"LONGITUDE":"lon", "LATITUDE":"lat","TIME":"time"})
ds4_ind = ds4.rename({"LONGITUDE":"lon", "LATITUDE":"lat","TIME":"time"})


# Getting seasonal mean for IMD data
ds3_ind_sm = season_mean(ds3_ind)
ds4_ind_sm = season_mean(ds4_ind)

gpm2009 = awash_da1
gpm2019 = awash_da2

imd2009 = ds3_ind_sm.RAINFALL
imd2019 = ds4_ind_sm.RAINFALL

# Interpolating IMD like GPM data
# using interp_like
imd2009_interp = imd2009.interp_like(gpm2009)
imd2019_interp = imd2019.interp_like(gpm2019)

## Calculation of Performance metrics

# Error

# 2009 and 2019 GPM variation comparison to IMD

err2009 = gpm2009 - imd2009_interp
err2019 = gpm2019 - imd2019_interp

# Plotting overall error for 2009 and 2019

# Plotting 2009 and 2019 RMSE

fig = plt.figure(figsize=(20, 15))
fig.tight_layout()

ax = fig.add_subplot(1, 2, 1, projection=ccrs.PlateCarree())
ax.set_extent([67, 98, 7, 36], crs=ccrs.PlateCarree())
err2009.mean(dim='season').plot(cbar_kwargs = {"orientation":"horizontal", "pad":0.05})
gridliner = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True)
gridliner.top_labels = False
gridliner.bottom_labels = True
gridliner.left_labels = True
gridliner.right_labels = False
gridliner.ylines = False  # you need False
gridliner.xlines = False  # you need False
ax.set_title("2009", pad=10, fontsize=14)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS)

ax = fig.add_subplot(1, 2, 2, projection=ccrs.PlateCarree())
ax.set_extent([67, 98, 7, 36], crs=ccrs.PlateCarree())
err2019.mean(dim='season').plot(cbar_kwargs = {"orientation":"horizontal", "pad":0.05})
gridliner = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True)
gridliner.top_labels = False
gridliner.bottom_labels = True
gridliner.left_labels = True
gridliner.right_labels = False
gridliner.ylines = False  # you need False
gridliner.xlines = False  # you need False
ax.set_title("2019", pad=10, fontsize=14)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS)

fig.suptitle('GPM data error compared to IMD gridded data (in mm)', fontsize=20, y=0.78)

plt.savefig('./images/err.png')

# Plotting 2009 seasonal error

fig = plt.figure(figsize=(20, 15))
fig.tight_layout()

titles = ["MAM", "JJA", "SON", "DJF"]

for i,season in enumerate(titles):
 
    ax = fig.add_subplot(2, 2, i+1, projection=ccrs.PlateCarree())
    ax.set_extent([67, 98, 7, 36], crs=ccrs.PlateCarree())
    err2009.sel(season=titles[i]).plot()
    gridliner = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True)
    gridliner.top_labels = False
    gridliner.bottom_labels = True
    gridliner.left_labels = True
    gridliner.right_labels = False
    gridliner.ylines = False  # you need False
    gridliner.xlines = False  # you need False
    ax.set_title("Months"+ " " + "-" + " " + "("+titles[i]+")", pad=10, fontsize=14)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS)

fig.suptitle('GPM data error compared to IMD gridded data (in mm)-2009', fontsize=20, y=0.95)

plt.savefig('./images/err2009.png')

# Plotting 2019 seasonal error

fig = plt.figure(figsize=(20, 15))
fig.tight_layout()

titles = ["MAM", "JJA", "SON", "DJF"]

for i,season in enumerate(titles):
 
    ax = fig.add_subplot(2, 2, i+1, projection=ccrs.PlateCarree())
    ax.set_extent([67, 98, 7, 36], crs=ccrs.PlateCarree())
    err2019.sel(season=titles[i]).plot()
    gridliner = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True)
    gridliner.top_labels = False
    gridliner.bottom_labels = True
    gridliner.left_labels = True
    gridliner.right_labels = False
    gridliner.ylines = False  # you need False
    gridliner.xlines = False  # you need False
    ax.set_title("Months"+ " " + "-" + " " + "("+titles[i]+")", pad=10, fontsize=14)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS)

fig.suptitle('GPM data error compared to IMD gridded data (in mm)-2019', fontsize=20, y=0.95)

plt.savefig('./images/err2019.png')

# RMSE

# Resetting Colormap selection for RMSE
xr.set_options(cmap_divergent='bwr', cmap_sequential='CMRmap') # divergent doesn't matter here

rmse2009 = np.sqrt((err2009 * err2009).mean(dim = 'season'))
rmse2019 = np.sqrt((err2019 * err2019).mean(dim = 'season'))

# Plotting 2009 and 2019 RMSE

fig = plt.figure(figsize=(20, 15))
fig.tight_layout()

ax = fig.add_subplot(1, 2, 1, projection=ccrs.PlateCarree())
ax.set_extent([67, 98, 7, 36], crs=ccrs.PlateCarree())
rmse2009.plot(cbar_kwargs = {"orientation":"horizontal", "pad":0.05})
gridliner = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True)
gridliner.top_labels = False
gridliner.bottom_labels = True
gridliner.left_labels = True
gridliner.right_labels = False
gridliner.ylines = False  # you need False
gridliner.xlines = False  # you need False
ax.set_title("2009", pad=10, fontsize=14)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS)

ax = fig.add_subplot(1, 2, 2, projection=ccrs.PlateCarree())
ax.set_extent([67, 98, 7, 36], crs=ccrs.PlateCarree())
rmse2019.plot(cbar_kwargs = {"orientation":"horizontal", "pad":0.05})
gridliner = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True)
gridliner.top_labels = False
gridliner.bottom_labels = True
gridliner.left_labels = True
gridliner.right_labels = False
gridliner.ylines = False  # you need False
gridliner.xlines = False  # you need False
ax.set_title("2019", pad=10, fontsize=14)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS)

fig.suptitle('GPM RMSE compared to IMD gridded data (in mm)', fontsize=20, y=0.78)

plt.savefig('./images/rmse.png')

# Plotting rmse error for seasonal means for 2009

fig = plt.figure(figsize=(20, 15))
fig.tight_layout()
titles = ["MAM", "JJA", "SON", "DJF"]

for i,season in enumerate(titles):
 
    ax = fig.add_subplot(2, 2, i+1, projection=ccrs.PlateCarree())
    ax.set_extent([67, 98, 7, 36], crs=ccrs.PlateCarree())
    rmse_calc(err2009, titles[i]).plot()
    gridliner = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True)
    gridliner.top_labels = False
    gridliner.bottom_labels = True
    gridliner.left_labels = True
    gridliner.right_labels = False
    gridliner.ylines = False  # you need False
    gridliner.xlines = False  # you need False
    ax.set_title("Months"+ " " + "-" + " " + "("+titles[i]+")", pad=10, fontsize=14)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS)

fig.suptitle('2009 GPM RMSE compared to IMD gridded data (in mm)', fontsize=20, y=0.95)
plt.savefig('./images/rmse2009.png')

# Plotting rmse error for seasonal means for 2019

fig = plt.figure(figsize=(20, 15))
fig.tight_layout()
titles = ["MAM", "JJA", "SON", "DJF"]

for i,season in enumerate(titles):
 
    ax = fig.add_subplot(2, 2, i+1, projection=ccrs.PlateCarree())
    ax.set_extent([67, 98, 7, 36], crs=ccrs.PlateCarree())
    rmse_calc(err2019, titles[i]).plot()
    gridliner = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True)
    gridliner.top_labels = False
    gridliner.bottom_labels = True
    gridliner.left_labels = True
    gridliner.right_labels = False
    gridliner.ylines = False  # you need False
    gridliner.xlines = False  # you need False
    ax.set_title("Months"+ " " + "-" + " " + "("+titles[i]+")", pad=10, fontsize=14)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS)

fig.suptitle('2019 GPM RMSE compared to IMD gridded data (in mm)', fontsize=20, y=0.95)
plt.savefig('./images/rmse2019.png')

## Time series plots

# get the time series dataarrays using the function. Here "ts" is timeseries

gpm2009_ts = daily_to_monthly(ds1_ind.precipitation)
imd2009_ts = daily_to_monthly(ds3_ind.RAINFALL)

gpm2019_ts = daily_to_monthly(ds2_ind.precipitation)
imd2019_ts = daily_to_monthly(ds4_ind.RAINFALL)

# Time series 2009

fig= plt.figure(figsize=(9,6))
imd2009_ts.mean(dim='lat').mean(dim='lon').plot.line("-*",color='orange', label = 'IMD')
gpm2009_ts.mean(dim='lat').mean(dim='lon').plot.line("b-^", label = 'GPM')

plt.legend()
plt.title('2009 precipitaion time-series for India', fontdict={"size":18})
plt.ylabel('Precipitation (in mm)', fontdict={"size":14})
plt.xlabel('time',fontdict={"size":14})

plt.savefig('./images/time_series2009.png')

# Time series 2019

fig = plt.figure(figsize=(9,6))
imd2019_ts.mean(dim='lat').mean(dim='lon').plot.line("-*",color='orange', label = 'IMD')
gpm2019_ts.mean(dim='lat').mean(dim='lon').plot.line("b-^", label = 'GPM')

plt.legend()
plt.title('2019 precipitaion time-series for India', fontdict={"size":18})
plt.ylabel('Precipitation (in mm)', fontdict={"size":14})
plt.xlabel('time',fontdict={"size":14})

plt.savefig('./images/time_series2019.png')
