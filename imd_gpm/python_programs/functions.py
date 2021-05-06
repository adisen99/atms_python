## Function required in the main program

import numpy as np
import xarray as xr
import netCDF4
import pandas as pd

# function to make preciptation rate to preciptation
def convert_to_precipitaion(ds):
    temp = ds * 24
#     temp = temp.to_dataset()
    return temp

# Function to obtain seasonal mean
def season_mean(ds, calendar='standard'):
    # Make a DataArray with the number of days in each month, size = len(time)
    month_length = ds.time.dt.days_in_month

    # Calculate the weights by grouping by 'time.season'
    weights = month_length.groupby('time.season') / month_length.groupby('time.season').sum()

    # Test that the sum of the weights for each season is 1.0
    np.testing.assert_allclose(weights.groupby('time.season').sum().values, np.ones(4))

    # Calculate the weighted average
    return (ds * weights).groupby('time.season').sum(dim='time')

# function for masking the data Masking the data

import geopandas as gpd
from rasterio import features
from affine import Affine

def transform_from_latlon(lat, lon):
    """ input 1D array of lat / lon and output an Affine transformation
    """
    lat = np.asarray(lat)
    lon = np.asarray(lon)
    trans = Affine.translation(lon[0], lat[0])
    scale = Affine.scale(lon[1] - lon[0], lat[1] - lat[0])
    return trans * scale

def rasterize(shapes, coords, latitude='lat', longitude='lon',
              fill=np.nan, **kwargs):
    """Rasterize a list of (geometry, fill_value) tuples onto the given
    xray coordinates. This only works for 1d latitude and longitude
    arrays.

    usage:
    -----
    1. read shapefile to geopandas.GeoDataFrame
          `states = gpd.read_file(shp_dir+shp_file)`
    2. encode the different shapefiles that capture those lat-lons as different
        numbers i.e. 0.0, 1.0 ... and otherwise np.nan
          `shapes = (zip(states.geometry, range(len(states))))`
    3. Assign this to a new coord in your original xarray.DataArray
          `ds['states'] = rasterize(shapes, ds.coords, longitude='X', latitude='Y')`

    arguments:
    ---------
    : **kwargs (dict): passed to `rasterio.rasterize` function

    attrs:
    -----
    :transform (affine.Affine): how to translate from latlon to ...?
    :raster (numpy.ndarray): use rasterio.features.rasterize fill the values
      outside the .shp file with np.nan
    :spatial_coords (dict): dictionary of {"X":xr.DataArray, "Y":xr.DataArray()}
      with "X", "Y" as keys, and xr.DataArray as values

    returns:
    -------
    :(xr.DataArray): DataArray with `values` of nan for points outside shapefile
      and coords `Y` = latitude, 'X' = longitude.


    """
    transform = transform_from_latlon(coords['lat'], coords['lon'])
    out_shape = (len(coords['lat']), len(coords['lon']))
    raster = features.rasterize(shapes, out_shape=out_shape,
                                fill=fill, transform=transform,
                                dtype=float, **kwargs)
    spatial_coords = {latitude: coords['lat'], longitude: coords['lon']}
    return xr.DataArray(raster, coords=spatial_coords, dims=('lat', 'lon'))

def add_shape_coord_from_data_array(xr_da, shp_path, coord_name):
    """ Create a new coord for the xr_da indicating whether or not it 
         is inside the shapefile

        Creates a new coord - "coord_name" which will have integer values
         used to subset xr_da for plotting / analysis/

        Usage:
        -----
        precip_da = add_shape_coord_from_data_array(precip_da, "awash.shp", "awash")
        awash_da = precip_da.where(precip_da.awash==0, other=np.nan) 
    """
    # 1. read in shapefile
    shp_gpd = gpd.read_file(shp_path)

    # 2. create a list of tuples (shapely.geometry, id)
    #    this allows for many different polygons within a .shp file (e.g. States of US)
    shapes = [(shape, n) for n, shape in enumerate(shp_gpd.geometry)]

    # 3. create a new coord in the xr_da which will be set to the id in `shapes`
    xr_da[coord_name] = rasterize(shapes, xr_da.coords, 
                               longitude='longitude', latitude='latitude')

    return xr_da

# function to calculate rmse error for given dataarray error value

def rmse_calc(da_err, season):
    """
    The RMSE calc function calculates the rmse from given input error value
    and also takes a season string as input for selecting the seasonal
    mean whose rmse needs to be calculated
    """
    months = ['MAM','JJA','SON','DJF']
    if season == 'MAM':
        rmse = np.sqrt((da_err * da_err).sel(season = months[0]))
    elif season == 'JJA':
        rmse = np.sqrt((da_err * da_err).sel(season = months[1]))
    elif season == 'SON':
        rmse = np.sqrt((da_err * da_err).sel(season = months[2]))
    elif season == 'DJF':
        rmse = np.sqrt((da_err * da_err).sel(season = months[3]))
    else:
        print("ERROR : Please enter a correct season value")
 
    return rmse

# Function to convert the IMD daily data to one month data
# we do the same with GPM for consistency

def daily_to_monthly(da, year):

    temp = da.groupby('time.month').mean(dim='time')
    if year == '2009':
        times = pd.date_range(start = "2009-01-15", end = "2010-01-15",freq='M')
    elif year == '2019':
        times = pd.date_range(start = "2019-01-15", end = "2020-01-15",freq='M')
    else:
        print("Enter either 2009 or 2019 as a year")

    output = xr.DataArray(
        temp,
        coords={
            "time": times,
            "lon": temp.lon,
            "lat": temp.lat
        },
        dims=["time", "lat", "lon"],
    )
 
    return output
