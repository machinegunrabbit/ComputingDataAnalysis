# ## Problem goal: Category mining ##
# ## Using Scikit-learn to do a co-clustering analysis of reader/book preference and tendency to review same type of book


get_ipython().system('date')
global_overwrite = False
from testing_tools import get_mem_usage_GiB
print(get_mem_usage_GiB())
get_ipython().system('date')
# Some things *we* need...
from pprint import pprint
from testing_tools import load_json, plot_spy
# Some things *you* might need...
import re
import pandas as pd
import numpy as np
import scipy as sp


# ## Part A: Data cleaning (contains Exercises 0 and 1) ##
# We need to clean two parts of the dataset
# ### Part A-0: Reviews ###
# The first part are the reviews. The code cell below loads them into a list of dictionaries named `raw_reviews`:
get_ipython().system('date')
raw_reviews = load_json('kindle_reviews.json', line_by_line=True)
print(f'\n==> Found {len(raw_reviews):,} e-book reviews. Here are the first two:\n')
pprint(raw_reviews[:2])
print(get_mem_usage_GiB())
get_ipython().system('date')

# For example, suppose `reviews` has exactly these two elements:
#
# ```python
# [{'asin': 'B000F83SZQ', 'helpful': [0, 0], 'overall': 5.0, 'reviewerID': 'A1F6404F1VG29J',
#   'reviewText': 'I enjoy vintage books ... good read for me.',
#   ... },
#  {'asin': 'B000F83SZQ', 'helpful': [2, 2], 'overall': 4.0, 'reviewerID': 'AN0N05A9LIJEQ',
#   'reviewText': 'This book is ... well worth a look-see.',
#   ...}
# ]
# ```
# Then the output would be a `DataFrame` of the form:
#
# |    | reviewer       | ebook      |   rating | text                                        |   num_helpful |   num_evals |
# |---:|:---------------|:-----------|---------:|:--------------------------------------------|--------------:|------------:|
# |  0 | A1F6404F1VG29J | B000F83SZQ |        5 | I enjoy vintage books ... good read for me. |             0 |           0 |
# |  1 | AN0N05A9LIJEQ  | B000F83SZQ |        4 | This book is ... well worth a look-see.     |             2 |           2 |

def reviews_to_df(reviews):
    list_to_table = pd.DataFrame.from_records(reviews)
    list_to_table['num_helpful'] = list_to_table['helpful'].str[0]
    list_to_table['num_evals'] = list_to_table['helpful'].str[1]
    df = list_to_table.rename(columns={'asin':'ebook','overall':'rating','reviewerID':'reviewer','reviewText':'text'})
    df = df.drop(['helpful'], axis=1)
    df['rating'] = df['rating'].astype(int)
    new_cols = ['reviewer','ebook','rating','text','num_helpful','num_evals']
    df = df.reindex(columns = new_cols)
    return df.head()

# Demo:
ex1_demo_reviews =     [{'asin': 'B000F83SZQ', 'helpful': [0, 0], 'overall': 5.0, 'reviewerID': 'A1F6404F1VG29J',
      'reviewText': 'I enjoy vintage books ... good read for me.'},
     {'asin': 'B000F83SZQ', 'helpful': [2, 2], 'overall': 4.0, 'reviewerID': 'AN0N05A9LIJEQ',
      'reviewText': 'This book is ... well worth a look-see.'}]

reviews_to_df(ex1_demo_reviews)
# ### Part A-1: E-book metadata ###
# For many of the e-books, there is also some additional metadata. Run the code cell below, which will load these metadata separately into a list of nested dictionaries called `raw_metadata`:
get_ipython().system('date')
raw_metadata = load_json('kindle_metadata-2018.json', line_by_line=True)
print(f'\n==> Found {len(raw_metadata):,} e-books.')
get_mem_usage_GiB()
get_ipython().system('date')
pprint(raw_metadata[0])
print('   4:', raw_metadata[4]['category']) # Looks fine ...
print('   1:', raw_metadata[1]['category']) # 4 elements; but element 2 is still valid
print(' 244:', raw_metadata[244]['category']) # 2 elements
print('2890:', raw_metadata[2890]['category']) # 3 elements, but element 2 has "junk" in it
print('   9:', raw_metadata[9]['category']) # 3 elements, but element 2 has '&amp;' instead of '&'

# ```python
#     [{'asin': 'B000FA5KKA',
#       'category': ['Kindle Store', 'Kindle eBooks', 'Science Fiction & Fantasy']},
#      {'asin': 'B000FA5M3K',
#       'category': ['Kindle Store', 'Kindle eBooks', 'Engineering & Transportation', '</span>']},
#      {'asin': 'B000FA5KX2',
#       'category': ['Kindle Store', 'Kindle eBooks', 'Business & Money']},
#      {'asin': 'B000FA5L2C',
#       'category': ['Kindle Store', 'Kindle eBooks', 'Business &amp; Money']},
#      {'asin': 'B000FC2LCS',
#       'category': ['Kindle Store', 'Kindle eBooks']},
#      {'asin': 'B001CBBL7M',
#       'category': ['Kindle Store', 'Kindle eBooks', '</span>']}]
# ```
#
# Then your function would return a dataframe equivalent to this one:
#
# |    | ebook      | genre                        |
# |---:|:-----------|:-----------------------------|
# |  0 | B000FA5KKA | Science Fiction & Fantasy    |
# |  1 | B000FA5M3K | Engineering & Transportation |
# |  2 | B000FA5KX2 | Business & Money             |
# |  3 | B000FA5L2C | Business & Money             |

def metadata_to_df(metadata):
    display(metadata)
    def is_ok(x):
        y = x['category']
        return len(y)>=3                and (y[0] == 'Kindle Store') and (y[1] == 'Kindle eBooks')                and ('<' not in y[2]) and ('>' not in y[2])
    def clean_genre(x):
        return x['category'][2].replace('&amp;','&')
    metadata_paired = [{'ebook':item['asin'],
                       'genre':clean_genre(item)} for item in metadata if is_ok(item)]
#     df = pd.DataFrame.from_records(metadata_paired)?\
    from pandas import DataFrame
    return DataFrame(metadata_paired)

ex1_demo_metadata =     [{'asin': 'B000FA5KKA',
      'category': ['Kindle Store', 'Kindle eBooks', 'Science Fiction & Fantasy']},
     {'asin': 'B000FA5M3K',
      'category': ['Kindle Store', 'Kindle eBooks', 'Engineering & Transportation', '</span>']},
     {'asin': 'B000FA5KX2',
      'category': ['Kindle Store', 'Kindle eBooks', 'Business & Money']},
     {'asin': 'B000FA5L2C',
      'category': ['Kindle Store', 'Kindle eBooks', 'Business &amp; Money']},
     {'asin': 'B000FC2LCS',
      'category': ['Kindle Store', 'Kindle eBooks']},
     {'asin': 'B001CBBL7M',
      'category': ['Kindle Store', 'Kindle eBooks', '</span>']}]

metadata_to_df(ex1_demo_metadata)

# ### Exercise 2 (1 point): Combining the data ###
#
# To simplify your work later, let's combine these separate dataframes into a single one. Complete the function, `combine_dfs(reviews_df, metadata_df)`, so that it returns a new dataframe with the following properties:
#
# 1. There is one review per row
# 2. The dataframe has seven (7) columns, which essentially combine all columns from the two dataframes: `'reviewer'`, `'ebook'`, `'rating'`, `'text'`, `'num_helpful'`, `'num_evals'`, and `'genre'`
# 3. Any review whose `'ebook'` is **missing** from `metadata_df` should be omitted
#
# > _Note 0:_ Your function must _not_ modify the input dataframes. The test cell will check for that and may fail with strange errors if you do so.
# >
# > _Note 1:_ The order of rows does not matter, as the test cell will use tibble comparison functions.

# In[14]:


def combine_dfs(reviews_df, metadata_df):
    df = reviews_df.merge(metadata_df, on = 'ebook', how = 'inner')
    return df

# Demo: Should reduce the number of reviews to `484,708`
print(f"`reviews_df` has {len(reviews_df):,} reviews (rows).")

ex2_demo_result = len(combine_dfs(reviews_df, metadata_df))
print(f"Combining with `metadata_df` leaves {ex2_demo_result:,} reviews.")
# ## Part B: Discovering genres? (Exercises 3-5) ##
#
# Each review connects a user to a book. Suppose we want to cluster users into groups based on what books they reviewed; or similarly, suppose we want to cluster books together based on who reviewed them. This kind of clustering between **two** times of objects is sometimes referred to as a **biclustering** problem.
#
# To do a biclustering, we first need to build a (sparse) matrix that encodes these connections. Here is one way. Let $A$ an $m \times n$ matrix of $m$ users (rows) and $n$ books (columns). Let each entry $a_{i, j}$ be the rating that user $i$ gave to book $j$. With such a matrix, we can then use a biclustering algorithm called _spectral co-clustering_ to construct both user-user and book-book clusters.

# ### Exercise 3 (2 points): Map string IDs to logical indices ###
#
# Suppose you are given a pandas `Series` object, `x`, whose values are strings. For example, suppose `x` is the following `Series`:
#
# |    x     |
# |:--------:|
# |   cat    |
# |   dog    |
# |   cat    |
# |   cat    |
# | aardvark |
# |   dog    |
# | anemone  |
# | aardvark |
#
# Write a function, `create_map(x)`, so that it does the following:
#
# 1. It determines the unique values in `x`.
# 2. It sorts these unique values in ascending order.
# 3. It assigns each unique value to a unique integer, starting from 0 and corresponding to the value's position in sorted order.
# 4. It constructs and returns a Python dictionary that maps `x`-values (keys) to integers (values).
#
# For instance, for the preceding `x`, the unique sorted values would be _aardvark_, _anemone_, _cat_, and _dog_. Therefore, we would assign _aardvark_ to 0, _anemone_ to 1, _cat_ to 2, and _dog_ to 3. Therefore, the function would return the dictonary,
#
# ```python
#    {'aardvark': 0, 'anemone': 1, 'cat': 2, 'dog', 3}
# ```

# In[18]:
def create_map(x):
    assert isinstance(x, pd.Series)
    series = x.sort_values().drop_duplicates(keep='first')
    index = list(range(0,series.size))
    value = list(series.values)
    result = dict(zip(value, index))
    return result


# In[19]:


# Demo
ex3_demo_input = pd.Series(['cat', 'dog', 'cat', 'cat', 'aardvark', 'dog', 'anemone', 'aardvark'])
create_map(ex3_demo_input)


# Suppose you are given a dataframe, `ratings`, with three columns:
#
# * `'reviewer'`: The string ID of a user
# * `'ebook'`: The string ID of a book that the user rated
# * `'rating'`: The value of the user's rating, which is an integer from 1 to 5.
#
# In addition, suppose you are given string-to-integer maps for reviewers and e-books, as `r_map` and `e_map`, respectively.
#
# Complete the function, `ratings_to_coo(ratings, r_map, e_map)`, so that it returns a Scipy sparse matrix in COO (coordinate) format, constructed according to these rules:
#
# 1. Let `(r, e, v)` denote the values of `'reviewer'`, `'ebook'`, and `'rating'` in a row of `ratings`. Each row will be a nonzero of the sparse matrix.
# 2. The reviewer with string ID `r` corresponds to row `i = r_map[r]` of the sparse matrix.
# 3. An e-book with string ID `e` corresponds to column `j = e_map[e]` of the sparse matrix.
# 4. The `(i, j)` entry of the sparse matrix is `v`.
#
# To get you started, the function below imports [Scipy's `coo_matrix` function](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.html), which you should use to construct your matrix. The starter code also determines the shape (number of rows and columns) for you. Use the `dtype=int` argument to ensure the nonzero values are stored as integers.
#
# **Example.** Suppose `ratings` is as follows:
#
# | reviewer   | ebook   |   rating |
# |:-----------|:--------|---------:|
# | alice      | happy   |        5 |
# | bob        | sad     |        2 |
# | carol      | happy   |        5 |
# | dave       | happy   |        4 |
# | edith      | sad     |        1 |
#
# Next, suppose `r_map` and `e_map` are as follows:
#
# ```python
# r_map = {'alice': 0, 'bob': 1, 'carol': 2, 'dave': 3, 'edith': 4}
# e_map = {'happy': 0, 'sad': 1}
# ```
#
# Then the sparse matrix would have a structure that looks like the following (rendered here as a table for ease of visualization):
#
# |         | _col 0_ | _col 1_ |
# |:-------:|:-------:|:-------:|
# | _row 0_ |    5    |         |
# | _row 1_ |         |    2    |
# | _row 2_ |    5    |         |
# | _row 3_ |    4    |         |
# | _row 4_ |         |    1    |
#


def ratings_to_coo(ratings, r_map, e_map):
    from scipy.sparse import coo_matrix

    # Precompute the matrix shape, `m` rows by `n` columns:
    m, n = max(r_map.values())+1, max(e_map.values())+1
    data = ratings['rating']
    display(data)
    rows = ratings['reviewer'].map(r_map)
    display(rows)
    cols = ratings['ebook'].map(e_map)
    display(cols)
    result_coo = coo_matrix((data,(rows,cols)),shape=(m,n))
    return result_coo


# ### Spectral co-clustering ###
# The `scikit-learn` package provides a spectral co-clustering implementation. It takes your sparse matrix as input, as well as a target number of clusters; it returns an assignment of rows and columns to clusters, as we'll explain below.
# Let's start by computing this clustering. As it happens, there are roughly 30 different usable genres, so let's ask for 30 clusters. The output will consist of two sets of labels, one for users (in a Numpy array named `row_labels`, below) and one for e-books (`col_labels`).
# > There are more principled ways to select the number of clusters, but let's save that for your next (O)MSA/VMM class!

def bicluster(matrix, num_clusters=10, verbose=True):
    from sklearn.cluster import SpectralCoclustering
    if verbose:
        print("Performing a biclustering...")
        print(f"* Input matrix shape: {matrix.shape[0]:,} x {matrix.shape[1]:,}")
        print(f"* Number of nonzeros: {matrix.nnz:,}")
        print(f"* Desired number of clusters: {num_clusters}")
    clustering = SpectralCoclustering(n_clusters=num_clusters, random_state=0).fit(matrix)
    if verbose:
        print("==> Done!")
    return clustering.row_labels_, clustering.column_labels_

get_ipython().system('date')
row_labels, col_labels = bicluster(ratings_matrix, num_clusters=30)
get_ipython().system('date')


# Let's focus on the e-book (column) labels. Here are the computed labels for the first eight e-books:
print(f"\nHere are the first few column (e-book) labels: {col_labels[:8]}")


# The labels are arbitrary integers. E-books belonging to the same cluster will have the same label.

# ### Exercise 5: Postprocessing the results (2 points) ###
#
# Suppose you are given two inputs, `x_map` and `labels`, formatted as follows:
#
# - `x_map` is a dictionary just like `reviewer_map` and `ebook_map` from Exercise 3: each key `s` is a string ID, and the corresponding value `i = x_map[s]` is its integer index.
# - `labels` is a Numpy array holding cluster labels. For an integer index `i`, the value `labels[i]` is the label assigned to `i`.
#
# Complete the function, `map_labels(x_map, labels)` so that it returns a Python dictionary where each key is a string label `s` and the value is its assigned label.

def map_labels(x_map, labels):
    users = list(x_map.keys())
    label_lst = list(labels)
    result = dict(zip(users, label_lst))
    return result

# Demo:
ex5_demo_x_map = {'alice': 0, 'bob': 1, 'carol': 2, 'dave': 3, 'edith': 4}
ex5_demo_labels = np.array([1, 0, 1, 1, 0])
map_labels(ex5_demo_x_map, ex5_demo_labels)

# ## Part C: Cluster analysis (contains Exercise 6-8) ##
#
# Let's inspect the e-book clusters and see what they reveal.

# ### Exercise 6: k-largest clusters (3 points) ###
#
# Suppose we are given a Numpy array of cluster labels called `labels` and an integer `k >= 1`. Complete the function, `top_k_labels(labels, k)`, so that
#
# 1. it counts how many times each label occurs;
# 2. ranks the labels from most-to-least frequent; and
# 3. returns a **Python set** containing the `k` most frequent labels.
# In the case of ties, include all labels with the same count. Therefore, your function might return more than `k` labels. And if there are fewer than `k` distinct labels, then your function should return them all.

def top_k_labels(labels, k):
    assert isinstance(labels, np.ndarray)
    from pandas import DataFrame
    df = DataFrame({'label':labels})
    df = df.groupby('label', as_index=False).size()
    result_pd =df.nlargest(k,'size',keep='all')
    unique_set = set(result_pd['label'].values)
    return unique_set

# Demo:
ex6_demo_labels = np.array([5, 3, 5, 5, 0, 6, 6, 4, 1, 1, 4, 7, 5, 6])
print("* Input:", ex6_demo_labels)
print("* Your output:")
for k in range(1, 10):
    print(f"k={k} ==>", top_k_labels(ex6_demo_labels, k))

# ### Exercise 7: Top genres in each cluster (2 points) ###
# Recall that our analysis has assigned some of the e-books to clusters. For instance, here are a few such assignments, taken from `ebook_labels` as computed in Exercise 5:

ex7_demo_ebooks = {'B00BQR4MYG', 'B00BM672WA', 'B00BQK8YEC', 'B005V8XX1Y', 'B004LP2GXE'}
{e: ebook_labels[e] for e in ex7_demo_ebooks}

# Each of these e-books is associated with a genre, available in the metadata dataframe from Exercise 1:
metadata_df[metadata_df['ebook'].isin(ex7_demo_ebooks)]

# **Your task.** Suppose you are given an e-book-to-label mapping, `ebook_labels`, and an e-book metadata dataframe, `metadata`. Complete the function, `merge_labels(metadata, ebook_labels)`, below, so that it does the following:
#
# 1. It constructs a new dataframe with three columns: `'ebook'`, `'genre'`, and `'label'`.
# 2. In this new dataframe, the `'ebook'` and `'genre'` columns come from `metadata`. The `'label'` column comes from `ebook_labels`. In other words, you need to merge the `ebook_labels` labels with `metadata`.
# 3. Not all e-books in `metadata` will have labels. The new dataframe should **only** include e-books with labels; any other e-book should not appear in it.
#
# > _Note 0:_ Your function must _not_ modify the inputs, `metadata` and `ebook_labels`. The test cell will check for that and may fail with strange errors if you do so.
# >
# > _Note 1:_ Be sure the column of labels has an integer.
# >
# > _Note 2:_ Your function should work even if `ebook_labels` is empty (is a dictionary with no keys, or no keys that match any e-book in `metadata`).

def merge_labels(metadata, ebook_labels):
    ebook_df = pd.DataFrame(ebook_labels.items(),columns =['ebook','label'])
    ebook_df['label'] = ebook_df['label'].astype(int)
    merged = metadata.merge(ebook_df,on='ebook')
    return merged

# Demo:
merge_labels(metadata_df, ebook_labels).head()

# You now have the building blocks you need to inspect some simple features of each cluster, to see how "distinct" clusters are from one another. Let's do that by analyzing the top genres represented in each cluster.

# **Your task.** Complete the function, `calc_top_genres(labeled_metadata, top_labels)`, below. It takes as input two objects:
# - `labeled_metadata`: A pandas dataframe, formatted like `labeled_metadata_df` above.
# - `top_labels`: A Python set of labels, like `top_k_ebook_labels` above.
# It should then do the following:
#
# - For each label in `top_labels`, it should determine the **two** most frequently occurring genres among the e-books with that label.
# - It should then return a single dataframe with two columns, `'label'` and `'genre'`. Each row should correspond to one (label, genre) pair. And per the preceding bullet, you expect to see two rows per label.
#
# Regarding the number of rows per label, there are two exceptions. First, if a given label only has e-books from one genre, then there will only be one row. Second, if there are ties, then you should retain all pairs, in the same way you would have done in Exercise 6.
#
# > _Note 0:_ Your function must _not_ modify the input arguments. The test cell will check for that and may fail with strange errors if you do so.
# >
# > _Note 1:_ The order of rows does not matter, as the test cell will use tibble comparison functions.

# **Example.** A correct implementation will produce, for the call `calc_top_genres(labeled_metadata_df, top_k_ebook_labels)` on the input dataset, the following result:
#
# |   label | genre                     |
# |--------:|:--------------------------|
# |       0 | Literature & Fiction      |
# |       0 | Romance                   |
# |      15 | Children's eBooks         |
# |      15 | Literature & Fiction      |
# |      25 | Health, Fitness & Dieting |
# |      25 | Literature & Fiction      |
# |      26 | Science Fiction & Fantasy |
# |      26 | Literature & Fiction      |
# |      29 | Literature & Fiction      |
# |      29 | Romance                   |
#
# Though not definitive, this result does suggest that the clustering captures distinct groups of books, here, for instance, with a cluster having `'Romance'` novels (label 0) being distinct from a cluster with `'Children's eBooks'` (label 15) and from another with `'Health, Fitness & Dieting'` (label 25), for instance.


def calc_top_genres(labeled_metadata, top_labels):
#     display(top_labels)
#     display(labeled_metadata)
    is_top = labeled_metadata['label'].isin(top_labels)
#     display(is_top)
    df = labeled_metadata[is_top].groupby('label', as_index=False)
    result = df.apply(analyze_cluster)
#     display(result)
    return result.drop(columns = 'size').reset_index(drop=True)

def analyze_cluster(df):
    return df.groupby(['label','genre'], as_index=False).size().nlargest(2,'size',keep='all')

# Demo
calc_top_genres(labeled_metadata_df, top_k_ebook_labels)
