
# ## Goal: Implement some basic analyses of NYC Taxi Cab data ##
# In this problem, we'll use real New York City Yellow Taxi fare and travel data to some simple analyses, including an analysis of routes or "paths" in the data.
# Once you've loaded the data, the overall workflow consists of the following steps:

import pandas as pd
import numpy as np
import scipy as sp

from matplotlib.pyplot import figure, subplot, plot
from matplotlib.pyplot import text, title, xlabel
from seaborn import histplot

from pprint import pprint # For pretty-printing native Python data structures
from testing_tools import load_df, load_geopandas


# ## Part A: Taxi Zones and Paths (Exercises 0 and 1) ##

# The NYC Taxi Dataset that you will analyze contains records for taxi rides or _trips._ Each trip starts in one "zone" and ends in another. The NYC Metropolitan area is divided into 266 "zones."
#
# Run the cell below, which loads a pandas dataframe holding metadata about these zones, which are stored in the dataframe named `zones`.

# In[2]:


zones = load_df('nyc-taxi-data/taxi+_zone_lookup.csv').drop('service_zone', axis=1).fillna('Unknown')
zones.head()


# Each zone has a unique integer ID (the `LocationID` column), a name (`Zone`), and an administrative district (`Borough`).
#
# Note that all location IDs from 1 to `len(zones)` are represented in this dataframe. However, you should not assume that in the exercises below.

# In[3]:


print("# of unique location IDs:", len(zones['LocationID'].unique()))
print("Some stats on location IDs:")
zones['LocationID'].describe()


#`zones_to_dict` (1 point) ###

def zones_to_dict(zones):
    df = zones.copy()
    df[['Borough', 'Zone']] = df[['Borough', 'Zone']].astype(str)
    df['Borough'] = df['Borough'].str.strip()
    df['Zone'] = df['Zone'].str.strip()
    df['combined'] = df['Zone'].str.cat(df['Borough'], sep = ', ')
    return {i:v for i,v in zip(df['LocationID'],df['combined'])}

zones_to_dict(zones.iloc[:3])

zones_to_dict(zones.iloc[:3]) # Sample output on the first three examples of `zones`

# `path_to_zones` (1 point) ###

def path_to_zones(p, zones_dict):
    location_list = list()
    string_p = [str(x) for x in p]
    for i in string_p:
        address = zones_dict[int(i)]
        address_list = [i, address]
        address_str = '. '.join(address_list)
#         formated_add = '"'+address_str+'"'
        location_list.append(address_str)
    return location_list

# Demo:
path_to_zones([3, 2, 1], zones_dict)

from testing_tools import mt2_ex1__check
print("Testing...")
for trial in range(250):
    mt2_ex1__check(path_to_zones)

zones_to_dict__passed = True
print("\n(Passed!)")

# Taxi trip data (Exercise 2)

get_ipython().system('date')
taxi_trips_raw_dfs = []
for month in ['06']: #, '07', '08']:
    taxi_trips_raw_dfs.append(load_df(f"nyc-taxi-data/yellow_tripdata_2019-{month}.csv",
                                      parse_dates=['tpep_pickup_datetime', 'tpep_dropoff_datetime']))
taxi_trips_raw = pd.concat(taxi_trips_raw_dfs)
del taxi_trips_raw_dfs # Save some memory
get_ipython().system('date')


# In[14]:


print(f"The raw taxi trips data has {len(taxi_trips_raw):,} records (rows). Here's a sample:")
taxi_trips_raw.head()


# Let's start by "focusing" our attention on just the columns we'll need in this problem.

# `focus` (1 point) ###

def focus(trips_raw):
    df = pd.DataFrame(columns=['I','J','D','C','T_start','T_end'])
    df['I'] = trips_raw['PULocationID']
    df['J'] = trips_raw['DOLocationID']
    df['D'] = trips_raw['trip_distance']
    df['C'] = trips_raw['fare_amount']
    df['T_start'] = trips_raw['tpep_pickup_datetime']
    df['T_end'] = trips_raw['tpep_dropoff_datetime']
    return df
focus(taxi_trips_raw.iloc[:3])
# Demo:
focus(taxi_trips_raw.iloc[:3])
# In[19]:

from testing_tools import mt2_trips as trips
display(trips.head())


# Date/Time objects (Exercise 3) ##
# Our "focused" dataframe includes two columns with trip start and stop times. These are stored as native Python `datetime` objects, which you would have encountered if you did the recommended Practice Midterm 2, Problem 18 (pmt2.18). By way of review, here is how they work.# Recall the first few rows of the `trips` dataframe:

trips.head(3)


print(type(trips['T_start'].iloc[0]))

t_start_demo = trips['T_start'].iloc[0]
t_end_demo = trips['T_end'].iloc[0]
dt_demo = t_end_demo - t_start_demo
print(dt_demo, "<==", type(dt_demo))

# This ride was evidently a short one, lasting just over 1 minute (1 minute and 4 seconds).
# These timedelta objects have special accessor fields, too. For example, if you want to convert this value to seconds, you can use the `.total_seconds()` function [[docs](https://pandas.pydata.org/docs/reference/api/pandas.Timedelta.total_seconds.html)]:

dt_demo.total_seconds()


# **Vectorized datetime accessors via `.dt`.** Beyond one-at-a-time access, there is another, faster way to do operations on any datetime or timedelta `Series` object using the `.dt` accessor. For example, here we calculate the time differences and extract the seconds for the first 3 rows:

print("Trip times for the first three rows:\n")
dt_series_demo = (trips['T_end'] - trips['T_start']).iloc[:3]
display(dt_series_demo)

print("\nConverting to total number of seconds:")
display(dt_series_demo.dt.total_seconds())

# `get_minutes` (1 point) ###
#
# Complete the function, `get_minutes(trips)`, below. It should take a "focused" trips dataframe, like `trips` above with `'T_start'` and `'T_end'` columns, as input. It should then return a **pandas `Series` object** of floating-point values corresponding to the total number of **minutes** that elapsed between the trip start and end.
#
# For example, suppose `trips` is as follows:
#
# |    |   I |   J |   D |    C | T_start             | T_end               |
# |---:|----:|----:|----:|-----:|:--------------------|:--------------------|
# |  0 | 145 | 145 | 0   |  3   | 2019-06-01 00:55:13 | 2019-06-01 00:56:17 |
# |  1 | 262 | 263 | 0   |  2.5 | 2019-06-01 00:06:31 | 2019-06-01 00:06:52 |
# |  2 |  74 |   7 | 4.4 | 17.5 | 2019-06-01 00:17:05 | 2019-06-01 00:36:38 |
#
# Then your function would return the Series,
#
# ```
# 0     1.066667
# 1     0.350000
# 2    19.550000
# dtype: float64
# ```
#
# > _Note 0:_ Timedelta objects have both `.total_seconds()` and `.seconds` accessors. However, it is **not** correct to use `.seconds` for this problem.
# >
# > _Note 1:_ The index of your `Series` should match the index of the input `trips`.


def get_minutes(trips):
    minutes = pd.Series()
    minutes = trips["T_end"] - trips["T_start"]
    minutes = minutes.dt.total_seconds()
    minutes = minutes/60
    return minutes


# Demo:
get_minutes(trips.head(3))

# `get_minutes` $\implies$ updated `trips` ###
# If you had a working solution to Exercise 3, then in principle you could use it to generate trip times and add them as a new column to the `trips` dataframe. We have precomputed that for you; run the cell below to load this dataframe, which will add a column named `'T'` to the `trips` dataframe.
# > **Read and run this cell even if you skipped or otherwise did not complete Exercise 3.**
from testing_tools import mt2_trip_times as trip_times
trips['T'] = trip_times
display(trips.head())


#  Filtering (Exercises 4 and 5) ##
# Our data has several issues, which basic descriptive statistics reveals:

assert 'trip_times' in globals(), "*** Be sure you ran the 'sample results' cell for Exercise 3 ***"

trips[['D', 'C', 'T']].describe()

def filter_bounds(s, lower=None, upper=None, include_lower=False, include_upper=False):
    if lower is None:
        lower = -np.inf
    if upper is None:
        upper = np.inf
    check_lower = (s >= lower) if include_lower else (s > lower)
    check_upper = (s <= upper) if include_upper else (s < upper)
    inbound = check_lower & check_upper
    return inbound

# Demo
ex4_s_demo = pd.Series([10, 2, -10, -2, -9, 9, 5, 1, 2, 8])
print(f"Input:\n{ex4_s_demo.values}")

ex4_demo1 = filter_bounds(ex4_s_demo, lower=-2, upper=4, include_lower=True, include_upper=True)
print(f"\n* [-2, 4], i.e., -2 <= x <= 4:\n{ex4_demo1.values}")

ex4_demo2 = filter_bounds(ex4_s_demo, lower=2)
print(f"\n* (2, infinity), i.e., 2 < x:\n{ex4_demo2.values}")


# > _Note:_ There are three test cells below for Exercise 4, meaning it is possible to get partial credit if only a subset pass.



from testing_tools import mt2_trips_clean as trips_clean

print(f"{len(trips_clean):,} of {len(trips):,} trips remain "
      f"after bounds filtering (~ {len(trips_clean)/len(trips)*100:.1f}%).")

trips_clean[['D', 'C', 'T']].describe()


#  `count_trips` (2 points) ###
def count_trips(trip_coords, min_trips=0):
    trips = trip_coords.copy()
    N = pd.DataFrame()
    N = trips.groupby(['I','J'], as_index=True)['I'].count()
    N = N.to_frame().rename(columns={'I': 'N'}).reset_index()
    result = N[N['N']>=min_trips]
    return result

# Demo:
ex5_df = trips_clean[((trips_clean['I'] == 85) & (trips_clean['J'] == 217))
                     | ((trips_clean['I'] == 139) & (trips_clean['J'] == 28))
                     | ((trips_clean['I'] == 231) & (trips_clean['J'] == 128))
                     | ((trips_clean['I'] == 169) & (trips_clean['J'] == 51))] \
                    [['I', 'J']] \
                    .reset_index(drop=True)
display(ex5_df)
count_trips(ex5_df, min_trips=3)


# `part_of_day` (2 points) ###

def part_of_day(tss):
    hour = tss.dt.hour
    mapping = {
        0:'wee hours',
        1:'morning',
        2:'afternoon',
        3:'evening'
    }
    segment = hour//6
    output = segment.map(mapping)
    return output

# Demo:
print("* Sample input `Series`:")
ex6_demo = trips_clean['T_start'].iloc[[20, 37752, 155816, 382741]]
display(ex6_demo)

print("\n* Your output:")
part_of_day(ex6_demo)

pair_costs = trips_clean[['I', 'J', 'C']].groupby(['I', 'J']).median().reset_index()
pair_costs.head()
pair_costs.sort_values(by='C', ascending=False).head()

#  `make_csr` (2 points) ###
# Complete the function, `make_csr(pair_costs, n)`, below. It should take as input a pair-costs dataframe, like the one shown above, as well as the matrix dimension, `n`.

from scipy.sparse import csr_matrix
def make_csr(pair_costs, n):
    rows = pair_costs['I']
    cols = pair_costs['J']
    cost = pair_costs['C']
    matrix = csr_matrix((cost,(rows,cols)),shape=(n,n))

    return matrix

# Call your code to convert:
ex7_csr = make_csr(ex7_demo, 5)
assert isinstance(ex7_csr, sp.sparse.csr.csr_matrix), "Not a Scipy CSR sparse matrix?"

# Try to visualize:
from matplotlib.pyplot import spy
spy(ex7_csr);

from networkx import from_scipy_sparse_matrix
Cost_graph = from_scipy_sparse_matrix(Cost_matrix)
print("Matrix entry (83, 1):", Cost_matrix[83, 1])
print("Graph edge (83, 1):", Cost_graph[83][1]['weight'])


# **Shortest paths.** One cool aspect of the NetworkX graph representation is that we can perform graph queries. For example, here is a function that will look for the shortest path---that is, the sequence of vertices such that traversing their edges yields a path whose total weight is the smallest among all possible paths. Indeed, that path can be _smaller_ than the direct path, as you'll see momentarily!
# The function `get_shortest_path(G, i, j)` finds the shortest path in the graph `G` going between `i` and `j`, and returns the path as a list of vertices along with the length of that path:
# In[53]:

def get_shortest_path(G, i, j):
    from networkx import shortest_path, shortest_path_length
    p = shortest_path(G, source=i, target=j, weight='weight')
    l = shortest_path_length(G, source=i, target=j, weight='weight')
    return p, l
# Demo: Check out the shortest path between 83 and 1
path_83_1, length_83_1 = get_shortest_path(Cost_graph, 83, 1)
print("Path:", path_83_1)
print("Length", length_83_1, "via the above path vs.", Cost_matrix[83, 1], '("direct")')
shapes = load_geopandas('nyc-taxi-data/zone-shapes/geo_export_28967859-3b38-43de-a1a2-26aee980d05c.shp')
shapes['location_i'] = shapes['location_i'].astype(int)
