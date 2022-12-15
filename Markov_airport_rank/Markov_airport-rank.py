# Use graph to simulate US air traffic by airports. Using MarkovChain to assess the importance of each airport and find critical point

# Step 0: Preparation
import sys
print(f"=== Python version ===\n{sys.version}\n")
import numpy as np
import scipy as sp
import scipy.sparse
import pandas as pd
print(f"- Numpy version: {np.__version__}")
print(f"- Scipy version: {sp.__version__}")
print(f"- Pandas version: {pd.__version__}")
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

def spy(A, figsize=(6, 6), markersize=0.5):
    """Visualizes a sparse matrix."""
    fig = plt.figure(figsize=figsize)
    plt.spy(A, markersize=markersize)
    plt.show()

from IPython.display import display, Markdown # For pretty-printing tibbles
from nb11utils import download_airport_dataset, get_data_path
from nb11utils import canonicalize_tibble, tibbles_are_equivalent
download_airport_dataset()
print("\n(All data appears to be ready.)")
airport_codes = pd.read_csv(get_data_path('L_AIRPORT_ID.csv'))
# airport_codes.head()
airport_codes[airport_codes.Description.str.contains("Los.*")]
airport_codes.iloc[373]['Description']
flights = pd.read_csv(get_data_path('us-flights--2017-08.csv'))
print("Number of flight segments: {} [{:.1f} million]".format (len(flights), len(flights)*1e-6))
del flights['Unnamed: 7'] # Cleanup extraneous column
flights.head(n=5)


# Step 1: Use the `airport_codes` data frame to figure out the integer airport codes for Atlanta's Hartsfield-Jackson International (ATL) and Los Angeles International (LAX).
# PART A) Define `ATL_ID` and `LAX_ID` to correspond to the codes in `airport_codes` for ATL and LAX, respectively.
pd.set_option('display.max_rows',airport_codes.shape[0]+1)
airport_codes['ATL'] = pd.Series(airport_codes['Description'].str.contains("Hartsfield-Jackson"))
airport_codes['LAX'] = pd.Series(airport_codes['Description'].str.contains("Los Angeles International"))

ATL_df = airport_codes[airport_codes['ATL'] == True]
LAX_df = airport_codes[airport_codes["LAX"] == True]

ATL_ID = ATL_df['Code'].iloc[0]
LAX_ID = LAX_df['Code'].iloc[0]

# Print the descriptions of the airports with your IDs:
ATL_DESC = airport_codes[airport_codes['Code'] == ATL_ID]['Description'].iloc[0]
LAX_DESC = airport_codes[airport_codes['Code'] == LAX_ID]['Description'].iloc[0]
print("{}: ATL -- {}".format(ATL_ID, ATL_DESC))
print("{}: LAX -- {}".format(LAX_ID, LAX_DESC))
airport_codes

# PART B) Construct `flights_atl_to_lax`
flights_atl_to_lax = flights.loc[(flights['ORIGIN_AIRPORT_ID'] == ATL_ID) &(flights['DEST_AIRPORT_ID'] == LAX_ID)]
# Displays a few of your results
print("Your code found {} flight segments.".format(len(flights_atl_to_lax)))
display(flights_atl_to_lax.head())

#Step 1: Aggregate (Origin, destination) pair that occured multiple times

# 1. It considers just the flight date, origin, and destination columns.
# 2. It _logically_ groups the rows having the same origin and destination, using `groupby()`.
# 3. It then aggregates the rows, counting the number of rows in each (origin, destination) group.

flights_cols_subset = flights[['FL_DATE', 'ORIGIN_AIRPORT_ID', 'DEST_AIRPORT_ID']]
segment_groups = flights_cols_subset.groupby(['ORIGIN_AIRPORT_ID', 'DEST_AIRPORT_ID'], as_index=False)
segments = segment_groups.count()
segments.rename(columns={'FL_DATE': 'FL_COUNT'}, inplace=True)
segments.head()


# Step 2: Determine actual flights

origins = segments[['ORIGIN_AIRPORT_ID', 'FL_COUNT']].groupby('ORIGIN_AIRPORT_ID', as_index=False).sum()
origins.rename(columns={'FL_COUNT': 'ORIGIN_COUNT'}, inplace=True)
print("Number of actual origins:", len(origins))
origins.head()


# Step 3: Construct a dataframe, `origins_top10`, containing the top 10 airports in descending order of outgoing segments dataframe with three columns:
# * `ID`: The ID of the airport
# * `Count`: Number of outgoing segments.
# * `Description`: The plaintext descriptor for the airport that comes from the `airport_codes` dataframe.

origins_extract = origins[['ORIGIN_AIRPORT_ID','ORIGIN_COUNT']]
origins_extract.rename(columns={'ORIGIN_AIRPORT_ID':'Code'}, inplace = True)

origins_merge = pd.merge(origins_extract, airport_codes, on='Code')
origins_drop = origins_merge.drop(columns = ['ATL', 'LAX'])

origins_rank = origins_drop.sort_values('ORIGIN_COUNT', ascending = False)
origins_rank.rename(columns={'ORIGIN_COUNT':'Count','Code':'ID'}, inplace = True)

origins_ranked = origins_rank.reset_index(drop = True)

# Prints the top 10, according to your calculation:
origins_top10 = origins_ranked[:10]
origins_top10


# Step 4: The preceding code computed a tibble, `origins`, containing all the unique origins and their number of outgoing flights. Write some code to compute a new tibble, `dests`, which contains all unique destinations and their number of _incoming_ flights. Its columns should be named `DEST_AIRPORT_ID` (airport code) and `DEST_COUNT` (number of direct inbound segments).

dests = segments[['DEST_AIRPORT_ID','FL_COUNT']].groupby('DEST_AIRPORT_ID',as_index = False).sum()
dests.rename(columns = {'FL_COUNT':"DEST_COUNT"}, inplace = True)
print("Number of unique destinations:", len(dests))
dests.head()

# Step 5: Compute a tibble, `dests_top10`, containing the top 10 destinations (i.e., rows of `dests`) by inbound flight count. The column names should be the same as `origins_top10` and the rows should be sorted in decreasing order by count.

# In[19]:


dests_extract = dests[['DEST_AIRPORT_ID','DEST_COUNT']]
dests_extract.rename(columns = {'DEST_AIRPORT_ID':'Code'}, inplace = True)

dests_merge = pd.merge(dests_extract,airport_codes, on = 'Code')
dests_drop = dests_merge.drop(columns = ['ATL','LAX'])

dests_rank = dests_drop.sort_values('DEST_COUNT', ascending = False)
dests_rank.rename(columns = {'Code':'ID','DEST_COUNT':'Count'}, inplace = True)

dests_ranked = dests_rank.reset_index(drop = True)
# print("Your computed top 10 destinations:")
dests_top10 = dests_ranked[:10]
dests_top10


# Step 6: Constructing the state-transition matrix

n_airports = airport_codes.index.max() + 1
print("Note: There are", n_airports, "airports.")

# Extract the `Code` column and index from `airport_codes`, storing them in
# a temporary tibble with new names, `ORIGIN_AIRPORT_ID` and `ORIGIN_INDEX`.
origin_indices = airport_codes[['Code']].rename(columns={'Code': 'ORIGIN_AIRPORT_ID'})
origin_indices['ORIGIN_INDEX'] = airport_codes.index
# Since you might run this code cell multiple times, the following
# check prevents `ORIGIN_ID` from appearing more than once.
if 'ORIGIN_INDEX' in segments.columns:
    del segments['ORIGIN_INDEX']
# Perform the merge as a left-join of `segments` and `origin_ids`.
segments = segments.merge(origin_indices, on='ORIGIN_AIRPORT_ID', how='left')
segments.head()


# Step7: Analogous to the preceding procedure, create a new column called `segments['DEST_INDEX']` to hold the integer index of each segment's _destination_.
dest_indices = airport_codes[['Code']].rename(columns = {'Code': "DEST_AIRPORT_ID"})
dest_indices['DEST_INDEX'] = airport_codes.index

segments = segments.merge(dest_indices, on = 'DEST_AIRPORT_ID', how = 'left')
# Visually inspect your result:
segments.head()


# Step 8: Computing edge weights.** Armed with the preceding mapping, let's next determine each segment's transition probability, or "weight," pij
#
# For each origin i, let d_i be the number of outgoing edges, or _outdegree_. Note that this value is *not* the same as the total number of (historical) outbound _segments_; rather, let's take $d_i$ to be just the number of airports reachable directly from $i$. For instance, consider all flights departing the airport whose airport code is 10135:

display(airport_codes[airport_codes['Code'] == 10135])
abe_segments = segments[segments['ORIGIN_AIRPORT_ID'] == 10135]
display(abe_segments)
print("Total outgoing segments:", abe_segments['FL_COUNT'].sum())
k_ABE = abe_segments['FL_COUNT'].sum()
d_ABE = len(abe_segments)
i_ABE = abe_segments['ORIGIN_AIRPORT_ID'].values[0]
display(Markdown('''
Though `ABE` has {} outgoing segments,
its outdegree or number of outgoing edges is just {}.
Thus, `ABE`, whose airport id is $i={}$, has $d_{{{}}} = {}$.
'''.format(k_ABE, d_ABE, i_ABE, i_ABE, d_ABE)))
segments.head(10)


# Step 9: Add a new column named `OUTDEGREE` to the `segments` tibble that holds the outdegrees, d_i. That is, for each row whose airport _index_ (as opposed to code) is i,
#its entry of `OUTDEGREE` should be d_i.
#
# For instance, the rows of segments corresponding to airport ABE (code 10135 and matrix index 119) would look like this:
#
# ORIGIN_AIRPORT_ID | DEST_AIRPORT_ID | FL_COUNT | ORIGIN_INDEX | DEST_INDEX | OUTDEGREE
# ------------------|-----------------|----------|--------------|------------|----------
# 10135             | 10397           | 77       | 119          | 373        | 3
# 10135             | 11433           | 85       | 119          | 1375       | 3
# 10135             | 13930           | 18       | 119          | 3770       | 3

# This `if` removes an existing `OUTDEGREE` column
# in case you run this cell more than once.
if 'OUTDEGREE' in segments.columns:
    del segments['OUTDEGREE']

outdegree = pd.DataFrame(segments.groupby(['ORIGIN_INDEX'])['ORIGIN_INDEX'].count())
outdegree = outdegree.rename(columns = {'ORIGIN_INDEX':'OUTDEGREE'})
outdegree["OUTDEGREE"] = outdegree["OUTDEGREE"].astype(object)
segments = segments.merge(outdegree,left_on='ORIGIN_INDEX',right_on = outdegree.index)
# Visually inspect the first ten rows of your result:
segments.head(10)

###parking lot for wrong commands
# outdegree["ORIGIN_INDEX_COL"] = outdegree.index
# outdegree.reset_index()
# segments.drop(columns = 'ORIGIN_INDEX_COL')
# outdegree
# outdegree.dtypes

#Step 10: from outdegree to probabilitiess
if 'WEIGHT' in segments:
    del segments['WEIGHT']
segments['WEIGHT'] = 1.0 / segments['OUTDEGREE']
display(segments.head(10))
# These should sum to 1.0!
origin_groups = segments[['ORIGIN_INDEX', 'WEIGHT']].groupby('ORIGIN_INDEX')
assert np.allclose(origin_groups.sum(), 1.0, atol=10*n_actual*np.finfo(float).eps), "Rows of $P$ do not sum to 1.0"
segments.shape


# Step 11: With your updated `segments` tibble, construct a sparse matrix, `P`, corresponding to the state-transition matrix P. Use SciPy's [scipy.sparse.coo_matrix()]

airport_codes = airport_codes.drop(columns = ['ATL', "LAX"])
airport_codes
from scipy.sparse import coo_matrix
row = np.array(segments['ORIGIN_INDEX'])
col = np.array(segments['DEST_INDEX'])
data = np.array(segments['WEIGHT'])
P = coo_matrix((data,(row,col)), dtype='float', shape=(n_airports,n_airports))

# Visually inspect your result:
spy(P)
P


# Step 12: Armed with the state-transition matrix $P$, you can now compute the steady-state distribution.

#  At time t=0, suppose the random flyer is equally likely to be at any airport with an outbound segment. Create a NumPy vector `x0[:]` such that `x0[i]` equals this initial probability of being at airport `i`.

# Your task: Create `x0` as directed above.
segments = segments.rename(columns = {'ORIGIN_AIRPORT_ID':'Code'})
combined = airport_codes.merge(segments, how = 'left', on = 'Code')

extracted = combined[['Code','WEIGHT']]

extracted1 = extracted.groupby(['Code']).sum()
print(extracted1.size)
extracted1.reset_index(drop = False, inplace = True)

actual_airports = extracted1['WEIGHT'].sum()
actual = extracted1[['Code','WEIGHT']]
actual['actual_airports'] = actual['WEIGHT']/actual['WEIGHT'].sum()

x0 = actual['actual_airports'].values

# Visually inspect your result:
def display_vec_sparsely(x, name='x'):
    i_nz = np.argwhere(x).flatten()
    df_x_nz = pd.DataFrame({'i': i_nz, '{}[i] (non-zero only)'.format(name): x[i_nz]})
    display(df_x_nz.head())
    print("...")
    display(df_x_nz.tail())

display_vec_sparsely(x0, name='x0')


# Step 13: Given the state-transition matrix `P`, an initial vector `x0`, and the number of time steps `t_max`, complete the function `eval_markov_chain(P, x0, t_max)` so that it computes and returns.

from copy import deepcopy
def eval_markov_chain(P, x0, t_max):
    num_row = max(x0.shape)+1
    y = x0.copy()
    z = np.zeros(num_row)
    for t in range(t_max):
        z = P.T * y
        y = z
    return z


T_MAX = 50
x0_copy = x0.copy()
x = eval_markov_chain(P, x0, T_MAX)
# x
display_vec_sparsely(x)
assert all(x0 == x0_copy), "Did your implementation modify `x0`? It shouldn't do that!"

print("\n=== Top 10 airports ===\n")
ranks = np.argsort(-x)

# print(ranks)
top10 = pd.DataFrame({'Rank': np.arange(1, 11),
                      'Code': airport_codes.iloc[ranks[:10]]['Code'],
                      'Description': airport_codes.iloc[ranks[:10]]['Description'],
                      'x(t)': x[ranks[:10]]})
top10[['x(t)', 'Rank', 'Code', 'Description']]
top10_with_ranks = top10[['Code', 'Rank', 'Description']].copy()

origins_top10_with_ranks = origins_top10[['ID', 'Description']].copy()
origins_top10_with_ranks.rename(columns={'ID': 'Code'}, inplace=True)
origins_top10_with_ranks['Rank'] = origins_top10.index + 1
origins_top10_with_ranks = origins_top10_with_ranks[['Code', 'Rank', 'Description']]

top10_compare = top10_with_ranks.merge(origins_top10_with_ranks, how='outer', on='Code',
                                       suffixes=['_MC', '_Seg'])
top10_compare
