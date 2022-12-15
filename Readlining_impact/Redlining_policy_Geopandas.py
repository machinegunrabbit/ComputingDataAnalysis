
# **Goal and workflow.** Your goal is to see if there is a relationship between the rating a neighborhood received in the 1930s and two attributes we can observe today: the average temperature of a neighborhood and the average home price.
#
# - Temperature tells you something about the local environment. Areas with more parks, trees, and green space tend to experience more moderate temperatures.
# - The average home price tells you something about the wealth or economic well-being of the neighborhood's residents.
#
# Your workflow will consist of the following steps:
#
# 1. You'll start with neighborhood rating data, which was collected from public records
# 2. You'll then combine these data with satellite images, which give information about climate.
# 3. Lastly, you'll merge these data with home prices from the real estate website, [Zillow](https://zillow.com).
# ## Part 0: Setup ##

import sys
print(f"* Python version: {sys.version}")
# Standard packages you know and love:
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import geopandas
print("* geopandas version:", geopandas.__version__)


# ## Part 1: Neighborhood ratings

# The neighborhood rating data is stored in a special extension of a pandas `DataFrame` called a `GeoDataFrame`. Let's load the data into a variable named `neighborhood_ratings` and have a peek at the first few rows:

neighborhood_ratings = load_geopandas('fullDownload.geojson')
print(type(neighborhood_ratings))
neighborhood_ratings.head()

def filter_ratings(ratings, city_st, targets=None):
    assert isinstance(ratings, geopandas.GeoDataFrame)
    assert isinstance(targets, set) or (targets is None)
    assert {'city', 'state', 'holc_grade', 'geometry'} <= set(ratings.columns)
    city_list = city_st.split(',')
    print(city_list[1])
    df = ratings[(ratings['city']==city_list[0])] #filter city
    display(df)
    if targets != None:
        targets = list(targets)
        df = df[df['holc_grade'].isin(targets)]
    return df


# ### Bounding boxes ###
#
# Recall that a geopandas dataframe includes a `'geometry'` column, which defines the geographic shape of each neighborhood using special multipolygon objects. To simplify some geometric calculations, a useful operation is to determine a multipolygon's _bounding box_, which is the smallest rectangle that encloses it.
#
# Getting a bounding box is easy! For example, recall the neighborhood in row 4 of the `neighborhood_ratings` geopandas dataframe:
# The bounding box is given to you by the multipolygon's `.bounds` attribute. This attribute is a Python 4-tuple (tuple with four components) that encodes both the lower-left corner and the upper-right corner of the shape. Here is what that tuple looks like for the previous example:
# The first two elements of the tuple are the smallest possible x-value and the smallest possible y-value among all points of the multipolygon. The last two elements are the largest x-value and y-value.
# If it's helpful, here is a plot that superimposes the bounding box on `g4_example`:

plot_multipolygon(gdf_ex1_demo.loc[3, 'geometry'], color='blue')
plot_bounding_box(gdf_ex1_demo.loc[3, 'geometry'].bounds, color='blue', linestyle=':')
plot_multipolygon(gdf_ex1_demo.loc[4, 'geometry'], color='gray')
plot_bounding_box(gdf_ex1_demo.loc[4, 'geometry'].bounds, color='gray', linestyle=':')

gdf_ex1_demo_bounding_box = (-86.815458, 33.464794, -86.749156, 33.533325)
plot_bounding_box(gdf_ex1_demo_bounding_box, color='black', linestyle='--')

def get_bounds(gdf):
    assert isinstance(gdf, geopandas.GeoDataFrame)
    if len(gdf) == 0:
        return None
    assert len(gdf) >= 1
    a = gdf['geometry'].apply(lambda x: x.bounds[0])
    b = gdf['geometry'].apply(lambda x: x.bounds[1])
    c = gdf['geometry'].apply(lambda x: x.bounds[2])
    d = gdf['geometry'].apply(lambda x: x.bounds[3])
    print(min(a),min(b),max(c),max(d))
    return (min(a),min(b),max(c),max(d))


# ## Part 2: Temperature analysis ##
#
# We have downloaded satellite images that cover some of the cities in the `neighborhood_ratings` dataset. Each pixel of an image is the estimated temperature at the earth's surface. The images we downloaded were taken by the satellite on a summer day.
# Here is an example of a satellite image that includes the Atlanta, Georgia neighborhoods used in earlier examples. The code cell below loads this image, draws it, and superimposes the Atlanta bounding box. The image is stored in the variable `sat_demo`. The geopandas dataframe for Atlanta is stored in `gdf_sat_demo`, and its bounding box in `bounds_sat_demo`.


# ### Exercise 2: Cleaning masked images (2 points) ###

def masked_to_degC(masked_array):
    assert isinstance(masked_array, np.ndarray)
    assert masked_array.ndim >= 1
    assert np.issubdtype(masked_array.dtype, np.integer)
#     print((len(masked_array),len(masked_array[0])))
    new_array = masked_array.astype('float')
    new_array[new_array == -9999] = np.nan
#     display(new_array)
    return new_array/10 -273.15

#
# ```python
# [[   nan  21.85    nan]
#  [   nan  43.55 -71.75]
#  [   nan  34.35  49.05]
#  [  6.95    nan -31.55]]
# ```
def mean_temperature(masked_array):
    assert isinstance(masked_array, np.ndarray)
    assert np.issubdtype(masked_array.dtype, np.floating)
    return np.nanmean(masked_array)

home_prices = load_df("Zip_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_mon.csv") # From Zillow
print("\nColumns:\n", home_prices.columns, "\n")
home_prices.head(3)

def clean_zip_prices(df):
    assert isinstance(df, pd.DataFrame)
    result = pd.DataFrame()
    result['ZipCode'] = df['RegionName'].astype('int')
    result['City'] = df['City']
    result['State'] = df['State']
    latest = sorted(df.filter(regex ='\d{4}-\d{2}-\d{2}').columns)[-1]
#     print(latest)
    result['Price'] = df[latest].astype('float')
    return result


plot_multipolygon(zip_geo.loc[2, 'geometry'], color='blue')
plot_bounding_box(zip_geo.loc[2, 'geometry'].bounds, color='black', linestyle='dashed')


# ### Exercise 5 (last one!): Merging price and geographic boundaries (2 points) ###
#
# Complete the function, `merge_prices_with_geo(prices_clean, zip_gdf)`, so that it merges price information stored in `prices_clean` with geographic boundaries stored in `zip_gdf`.
#
# - The `prices_clean` object is a **pandas** dataframe that will have four columns, `'ZipCode'`, `'City'`, `'State'`, and `'Price'`, as would be produced by `clean_home_prices` (Exercise 4).
# - The `zip_gdf` input is a **geopandas** dataframe with two columns, `'GEOID10'` and `'geometry'`.
# - Your function should return a new **geopandas** dataframe with five columns: `'ZipCode'`, `'City'`, `'State'`, `'Price'`, and `'geometry'`.
#
# > **Note 0:** Recall that the `'ZipCode'` column of `prices_clean` stores values as integers, whereas the `'GEOID10'` column of `zip_gdf` stores values as strings. In your final result, store the `'ZipCode'` column using integer values.
# >
# > **Note 1:** We are only interested in zip codes with _both_ price information _and_ known geographic boundaries. That is, if a zip code is missing in either `prices_clean` or `zip_gdf`, you should ignore and omit it from your output.
# >
# > **Note 2:** If `df` is a pandas dataframe, you can convert it to a geopandas one simply by calling `geopandas.GeoDataFrame(df)`.

def merge_prices_with_geo(prices_clean, zip_gdf):
    assert isinstance(prices_clean, pd.DataFrame)
    assert isinstance(zip_gdf, geopandas.GeoDataFrame)
    price_gdf = geopandas.GeoDataFrame(prices_clean)
#     name_change = {'GEOID10':'ZipCode'}
    zip_gdf['ZipCode'] = zip_gdf['GEOID10'].astype('int')
    result = price_gdf.merge(zip_gdf[['ZipCode','geometry']], on='ZipCode')
#     display(result)
    return result
