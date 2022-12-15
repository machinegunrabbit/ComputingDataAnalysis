#Human migration — where is everyone going? #
# This analysis will allow you to compare the most populous places in the US today with those predicted for the year 2070.

# ## Part 0: Setup ##

# Run the following code cell to preload some standard modules you may find useful for this notebook.


import sys
print(f"* Python version: {sys.version}")

import pandas as pd
import numpy as np
import scipy as sp
from testing_tools import load_db, get_db_schema, data_fn

# ## Dataset: IRS Tax Migration Data ##

conn = load_db('irs-migration/irs-migration.db')
conn

# The database has three tables, which were created using the following SQL commands:
for k, table in enumerate(get_db_schema(conn)):
    print(f'* [Table #{k}]', table[0])

# Let's inspect the contents of each of these tables.

# ### `States` table ###
#
# The first table, `States`, is a list of the US's fifty states (provinces). Each has a unique integer ID (`States.id`) and an abbreviated two-letter name (`States.name`). Here are the first few rows of this table:

pd.read_sql_query('SELECT * FROM States LIMIT 5', conn)
pd.read_sql_query('SELECT * FROM Counties LIMIT 5', conn)

# Each county has a unique integer ID and a name. The names are _not_ unique. For instance, there are 8 counties named `"Fulton County"`:

pd.read_sql_query('SELECT * FROM Counties WHERE name="Fulton County"', conn)
pd.read_sql_query('SELECT * FROM Counties WHERE id >= 13000 AND id < 14000', conn)

def count_counties(conn):
    df = pd.read_sql_query('''
    SELECT States.name AS name, COUNT(*) AS count
    FROM States JOIN Counties ON (Counties.id/1000)=States.id
    Group By States.id
    ''', conn)
    return dict(zip(df['name'],df['count']))
demo_counts = count_counties(conn)
print("Found", len(demo_counts), "states.")
print("The state of 'GA' has", demo_counts['GA'], "counties.")

pd.read_sql_query('SELECT * FROM Flows LIMIT 10', conn)

# where the totals are taken over all destinations and years for each of the two sources.

def sum_outflows(conn):
    df = pd.read_sql_query('''
    SELECT source, SUM(num_returns) AS total_returns
    FROM Flows
    Group By source
    ''', conn)
    df.astype(int)
    return df

demo_sum_outflows = sum_outflows(conn)
demo_sum_outflows[demo_sum_outflows['source'].isin({13125, 27077})]

from testing_tools import mt2_ex1__check
print("Testing...")
for trial in range(250):
    mt2_ex1__check(sum_outflows)

print("\n(Passed!)")

def estimate_probs(conn):
    df1 = pd.read_sql_query('''
    SELECT source, SUM(num_returns) as denom
    FROM Flows
    GROUP BY source
    ''', conn)
    df2 = pd.read_sql_query('''
    SELECT source, dest, SUM(num_returns) as numer
    FROM Flows
    GROUP BY source, dest
    ''', conn)
    merged = df1.merge(df2, on='source')
    merged['prob'] = merged['numer']/merged['denom']
    return merged[['source','dest','prob']]

demo_probs = estimate_probs(conn)
demo_probs[demo_probs['source'] == 13125]

from testing_tools import mt2_ex2__check
print("Testing...")
for trial in range(250):
    mt2_ex2__check(estimate_probs)

print("\n(Passed!)")
# `map_counties` ###
#
def map_counties(conn):
    df = pd.read_sql_query('''
    SELECT id
    FROM Counties
    ORDER BY id
    ''', conn)

    df_dict = df.to_dict()
    result = {v:i for i,v in df_dict['id'].items()}
    return result
demo_map_counties = map_counties(conn)

for i in [1001, 1003, 1005, None, 56041, 56043, 56045]:
    if i is None:
        print("...")
        continue
    print(i, "==>", demo_map_counties[i])

def make_matrix(probs, county_map):
    from scipy.sparse import coo_matrix
    assert isinstance(probs, pd.DataFrame)
    assert isinstance(county_map, dict)
    rows = probs['source'].map(county_map)
    cols = probs['dest'].map(county_map)
    values = probs['prob']
    matrix = coo_matrix((values,(rows,cols)),shape=(len(county_map),len(county_map)))
    return matrix

demo_P = make_matrix(demo_probs, demo_map_counties)
demo_n = max(demo_map_counties.values())+1
print("* Shape:", demo_P.shape, "should equal", (demo_n, demo_n))
print("* Number of nonzeros:", demo_P.nnz, "should equal", len(demo_probs))


# ## Part 2: Calculating the initial distribution ##

def load_pop_data(fn='census/co-est2019-alldata.csv'):
    pop = pd.read_csv(data_fn(fn), encoding='latin_1')
    pop = pop[['STATE', 'COUNTY', 'POPESTIMATE2019', 'BIRTHS2019', 'DEATHS2019']]
    pop = pop[pop['COUNTY'] > 0]
    pop = pop[(pop['STATE'] != 15) & (pop['COUNTY'] != 5)]
    return pop

population = load_pop_data()
population.sample(5) # Show 5 randomly selected rows
def normalize_pop(population, county_map):
    assert isinstance(population, pd.DataFrame)
    assert set(population.columns) == {'BIRTHS2019', 'COUNTY', 'DEATHS2019', 'POPESTIMATE2019', 'STATE'}
    pop = population.copy()
    total = pop['POPESTIMATE2019'].sum()
    norm = pop['POPESTIMATE2019']/total
    index = (1000*pop['STATE']+pop['COUNTY']).map(county_map)
    result_arr = np.zeros(len(county_map))
    result_arr[index] = norm
    return result_arr

demo_pop = population[population.apply(lambda row: (row['STATE'], row['COUNTY']) in [(47, 69), (50, 1), (26, 117), (55, 23), (22, 99)], axis=1)]
demo_map = {47069: 2, 50001: 3, 26117: 1, 55023: 4, 22099: 0}
normalize_pop(demo_pop, demo_map)


def estimate_pop(population, t):
    assert isinstance(population, pd.DataFrame)
    assert set(population.columns) == {'BIRTHS2019', 'COUNTY', 'DEATHS2019', 'POPESTIMATE2019', 'STATE'}
    assert isinstance(t, int) and t >= 0
    pop = population.copy()
    n0 = pop['POPESTIMATE2019'].sum()
    b0 = pop['BIRTHS2019'].sum()
    d0 = pop['DEATHS2019'].sum()
    beta = b0/n0
    sigma = d0/n0
    nt = n0*(1+beta-sigma)**t
    return nt


# Demo cell
demo_pop = population[population.apply(lambda row: (row['STATE'], row['COUNTY']) in [(47, 69), (50, 1), (26, 117), (55, 23), (22, 99)], axis=1)]
estimate_pop(demo_pop, 50)


def calc_ipr(conn):
    self = pd.read_sql_query('''
    SELECT source AS county_id, SUM(income_thousands)*1000 AS s_inc, SUM(num_returns) AS s_ret
    FROM Flows
    WHERE source = dest
    Group BY source
    ''',conn)
    outflows = pd.read_sql_query('''
    SELECT source AS county_id, SUM(income_thousands)*1000 AS o_inc, SUM(num_returns) AS o_ret
    FROM Flows
    WHERE source <> dest
    Group BY source
    ''',conn)
    inflows = pd.read_sql_query('''
    SELECT dest AS county_id, SUM(income_thousands)*1000 AS i_inc, SUM(num_returns) AS i_ret
    FROM Flows
    WHERE source <> dest
    Group BY dest
    ''',conn)
    merged = self.merge(inflows, how = 'left', on = 'county_id').merge(outflows, how = 'left', on='county_id').fillna(0)
    merged['ipr'] = (merged['s_inc']+0.5*(merged['i_inc']+merged['o_inc']))/(merged['s_ret']+0.5*(merged['i_ret']+merged['o_ret']))

    return merged[['county_id', 'ipr']]


# Demo cell 0: Should get approximately 73791.05 for county 2068
income = calc_ipr(conn)
income[income['county_id'] == 2068]


income.sort_values(by='ipr', ascending=False).head(5)

# Run PageRank
P = load_pickle('matrix.pickle')
x_0 = load_pickle('dist0.pickle') # initial distribution
x = x_0.copy() # holds final distribution
for _ in range(50):
    x = P.T.dot(x)
x_final = x.copy()

# Build DataFrame
def get_ranking(x):
    k = np.argsort(x)
    r = np.empty(len(x), dtype=int)
    r[k] = np.arange(len(x))
    return r

# Get population ranking
county_map = load_json('map_counties.json') # county IDs -> physical indices
inv_county_map = {v: k for k, v in county_map.items()} # physical indices -> county IDs

rankings = pd.DataFrame({'county_id': [inv_county_map[k] for k in range(len(county_map))],
                         'rank_2019': get_ranking(-x_0), 'x_2019': x_0,
                         'rank_2070': get_ranking(-x_final), 'x_2070': x_final})
rankings['county_id'] = rankings['county_id'].astype(int)

# Add income data
top_incomes = pd.read_csv(data_fn('incomes.csv'))
top_incomes['rank_ipr'] = top_incomes.index

# Construct location metadata
locations = pd.read_sql_query("""SELECT Counties.id AS county_id,
                                        Counties.name||', '||States.name AS name
                                 FROM Counties, States
                                 WHERE Counties.id/1000 == States.id""", conn)

# Merge
rankings = rankings.merge(locations, how='left', on='county_id').merge(top_incomes, how='left', on='county_id')[['county_id', 'name', 'rank_2019', 'rank_2070', 'x_2019', 'x_2070', 'ipr', 'rank_ipr']]
rankings.head()


# View Top 10 according to their year 2070 rankings:
rankings.sort_values(by='rank_2070').head(10)
