
# Actor Network Analysis

def load_data(path):
    import pandas as pd
    return pd.read_csv(path, names=['film_id', 'film_name', 'actor', 'year'], skiprows=1)

# ## Exercise 1 (1 Point):

def explore_data(df):
    shape = df.shape
    first_five = df.head(5)
    num = df.groupby('year').apply(lambda group: group['film_id'].unique().shape[0])
    num_year = num.to_dict()
    tup = (shape, first_five, num_year)
    return tup

### define demo inputs
import pickle
with open('resource/asnlib/publicdata/movie_data.pkl', 'rb') as f:
    movie_data = pickle.load(f)
demo_df_ex1 = movie_data.sample(15, random_state=6040)


### call demo funtion
explore_data(demo_df_ex1)

# ## Exercise 2 (2 Points):

import pandas as pd
def top_10_actors(df):
    actor = pd.DataFrame(df.groupby('actor',as_index=False)['film_id'].count())
    actor = actor.rename(columns={'film_id':'count'})
    actor.sort_values(by=['count','actor'], ascending=[False,True], inplace=True,ignore_index=True)

    return actor.nlargest(10,'count',keep = 'all')

import pickle
with open('resource/asnlib/publicdata/movie_data.pkl', 'rb') as f:
    movie_data = pickle.load(f)
demo_df_ex2 = movie_data.sample(3000, random_state=6040)

### call demo funtion
print(top_10_actors(demo_df_ex2))


# ## Exercise 3 (1 Point):

def actor_years(df, actor):
    actor_years = df.loc[df['actor'] == actor, 'year'].sort_values().unique().tolist()
    result = {actor:actor_years}
    return result

### define demo inputs
import pickle
with open('resource/asnlib/publicdata/movie_data.pkl', 'rb') as f:
    movie_data = pickle.load(f)
demo_df_ex3 = movie_data.sample(3000, random_state=6040)

### call demo funtion
actor_years(demo_df_ex3, 'James Franco')

# ## Exercise 4 (2 Points):

def movie_size_by_year(df):
    ### BEGIN SOLUTION
    import pandas as pd
    def summarize_year(group):
#         print("type:", type(group))
        counts = group.groupby('film_id').apply(lambda x: len(x))
        return pd.Series({
            'min': counts.min(),
            'max': counts.max(),
            'mean': round(counts.mean())
        })
    return df.groupby(['year'])        .apply(summarize_year)        .to_dict('index')
    ### END SOLUTION

import pandas as pd
df = pd.DataFrame({'A': 'a a b'.split(),
                   'B': [1,2,3],
                   'C': [4,6,5]})

g1 = df.groupby('A')

g1.groups

for group_name, indices in g1.groups.items():
    group_df = df.iloc[indices]
    print(group_df, '\n')

import numpy as np
df.apply(np.sum, axis=0)

### Define movie_size_by_year
import pandas as pd
def movie_size_by_year(df):
    def summarize_year(group):
        counts = group[['film_id']].groupby('film_id').apply(lambda x: len(x))
        return pd.Series({
            'min':counts.min(),
            'max':counts.max(),
            'mean':round(counts.mean())
        })

    result = df.groupby('year').apply(summarize_year).to_dict('index')
    return result

# In[22]:


### define demo inputs
import pickle
with open('resource/asnlib/publicdata/movie_data.pkl', 'rb') as f:
    movie_data = pickle.load(f)
demo_df_ex4 = movie_data.sample(3000, random_state=6040)

# ## Exercise 5:

from collections import defaultdict
def make_network_dict(df):
    d = defaultdict(set)
    actor_pairs = df[['film_id','actor']].merge(df[['film_id','actor']], how='inner', on='film_id')    .query('actor_x < actor_y')    .drop(columns = 'film_id')
    for row in actor_pairs.itertuples():
        d[row[1]].add(row[2])
    return {k:v for k,v in d.items()}
### define demo inputs
import pickle
with open('resource/asnlib/publicdata/movie_data.pkl', 'rb') as f:
    movie_data = pickle.load(f)
demo_df_ex5 = movie_data.sample(300, random_state=6040)

### call demo funtion
make_network_dict(demo_df_ex5)

# ## Exercise 6:

import networkx as nx
def to_nx(dos):
    g = nx.Graph()
    for node, values in dos.items():
        g.add_node(node)
        for v in values:
            g.add_edge(node,v)
    return g

### define demo inputs
import pickle
import numpy as np
rng = np.random.default_rng(6040)
with open('resource/asnlib/publicdata/network_dict.pkl', 'rb') as f:
    network_dict = pickle.load(f)
demo_dos_ex6 = {k: {v for v in rng.choice(network_dict[k], 5)} for k in rng.choice(list(network_dict.keys()), 15)}


### call demo funtion
set(to_nx(demo_dos_ex6).edges)

# ## Exercise 7 :

def high_degree_actors(g, n=None):
    actor_degree = dict(g.degree)
    pair_list = list()
    for actor, degree in actor_degree.items():
        pair = [actor,degree]
        pair_list.append(pair)
    df = pd.DataFrame(pair_list, columns = ['actor','degree'])
    df.sort_values(by=['degree','actor'], ascending=[False,True], inplace=True,ignore_index=True)
    if n != None:
        df = df.nlargest(n,'degree','all')
    return df

### call demo funtion
print(high_degree_actors(demo_g_ex7, demo_n_ex7))

# In[56]:
# ## Exercise 8:

def notable_actors_in_comm(communities, degrees, actor):
    assert actor in {a for c in communities for a in c}, 'The given actor was not found in any of the communities!'
    degrees = degrees
    for community in communities:
        if actor in community:
            break
    degrees = degrees[degrees['actor'].isin(community)].nlargest(10,'degree','all').reset_index(drop=True)
    return degrees

### call demo funtion
print(notable_actors_in_comm(communities, degrees, demo_actor_ex8))
