
import numpy as np
import scipy as sp
import pandas as pd
from pandas import DataFrame

# Support code for data loading, code testing, and visualization:
import sys
sys.path.insert(0, 'resource/asnlib/public')
from cse6040utils import tibbles_are_equivalent, pandas_df_to_markdown_table, hidden_cell_template_msg, cspy

from matplotlib.pyplot import figure, subplots
get_ipython().magic('matplotlib inline')
from networkx.convert_matrix import from_pandas_edgelist
from networkx.drawing.nx_pylab import draw
from networkx import DiGraph
from networkx.drawing import circular_layout

# Location of input data:
def dataset_path(base_filename):
    return f"resource/asnlib/publicdata/{base_filename}"


# ## Background: Relationship networks and partitioning
#
# Suppose we have data on the following five people, stored in the `nodes` data frame (run the next cell):

# In[2]:


nodes = DataFrame({'name': ['alice', 'bob', 'carol', 'dave', 'edith'],
                   'age': [35, 18, 27, 57, 41]})
nodes


edges = DataFrame({'source': ['alice', 'alice', 'dave',  'dave',  'dave',  'bob'],
                   'target': ['dave',  'edith', 'alice', 'edith', 'carol', 'carol']})
edges


G = from_pandas_edgelist(edges, source='source', target='target', create_using=DiGraph())
figure(figsize=(4, 4))
draw(G, arrows=True, with_labels=True, pos=circular_layout(G),
     node_size=1200, node_color=None, font_color='w',
     width=2, arrowsize=20)

def symmetrize_df(edges):
    assert 'source' in edges.columns
    assert 'target' in edges.columns
    edges_transpose = edges.rename(columns={'source':'target','target':'source'})
    edges_all = pd.concat([edges,edges_transpose], sort=False).drop_duplicates().reset_index(drop=True)

    return edges_all


# Demo of your function:
symmetrize_df(edges)

def df_to_coo(nodes, edges):
    """Returns a Scipy coordinate sparse matrix, given an input graph in a data frame representation."""
    assert 'name' in nodes.columns
    assert 'source' in edges.columns
    assert 'target' in edges.columns
    from scipy.sparse import coo_matrix  # Use this to construct your sparse matrix!

    n = nodes.index.max() + 1
    name_to_id = dict(zip(nodes['name'], nodes.index))
    rows = edges['source'].replace(name_to_id)
    cols = edges['target'].replace(name_to_id)
    values = [1.0]*len(edges)

    result = coo_matrix((values, (rows, cols)), shape = (n,n))
    return result

def check1(n=5, k=None):
    assert n < 26, "Currently limited to a max dimension of 26"
    from numpy.random import randint
    from numpy import array, vectorize, issubdtype, floating
    def gen_name(c):
        return chr(ord('A') + c)
    nodes = DataFrame({'name': array([gen_name(c) for c in range(n)])})
    name_to_id = dict(zip(nodes['name'], nodes.index))
    if k is None:
        k = randint(0, 5*n)
    if k > 0:
        sources = vectorize(gen_name)(randint(n, size=k))
        targets = vectorize(gen_name)(randint(n, size=k))
    else:
        sources = []
        targets = []
    edges = symmetrize_df(DataFrame({'source': sources, 'target': targets}))
    edges_by_ids = set([(name_to_id[a], name_to_id[b]) for a, b in zip(edges['source'], edges['target'])])
    try:
        A = df_to_coo(nodes, edges)
        assert A.nnz == len(edges_by_ids), f"Expected {len(edges_by_ids)} edges, but you produced {A.nnz}."
        assert issubdtype(A.dtype, floating), f"Value type is `{A.dtype}`, which is incorrect."
        A_edge_set = set()
        for i, j, v in zip(A.row, A.col, A.data):
            assert v == 1.0, f"All values should equal 1.0, but you have {v} at position ({i}, {j})."
            A_edge_set |= {(i, j)}
        assert edges_by_ids == A_edge_set,                f"Your matrix has incorrect edges in it:\n"                f"  * Expected: {edges_by_ids}\n"                f"  * You:      {A_edge_set}\n"
    except:
        print("=== Failing test case ===")
        print("* Nodes:")
        display(nodes)
        print("* Edges:")
        display(edges)
        raise

for _ in range(10):
    check1()

# Check empty test case
check1(k=0)
print("(Passed.)")
def partition_inds_by_sign(x):
    from numpy import where

    K_pos = where(x > 0)[0]
    K_neg = where(x<=0)[0]
    return K_pos,K_neg

# Demo:
x = np.array([1, -1, -2, 2, -3, 3, 4, -4, -5, 5])
k_gtz, k_lez = partition_inds_by_sign(x)
print(f"x[{k_gtz}] == {x[k_gtz]}") # Correct output: `x[[0 3 5 6 9]] == [1 2 3 4 5]`
print(f"x[{k_lez}] == {x[k_lez]}") # Correct output: `x[[1 2 4 7 8]] == [-1 -2 -3 -4 -5]`

a = np.array([1, -1, -2, 2, -3, 3, 4, -4, -5, 5])
np.where(a > 0)[0]

def check2(n=8):
    from random import randrange
    from numpy import array, issubdtype, integer, ndarray
    x = []
    kp = []
    kn = []
    for i in range(n):
        x.append(randrange(-10, 10))
        (kp if x[-1] > 0 else kn).append(i)
    x, kp, kn = array(x), array(kp, dtype=int), array(kn, dtype=int)
    try:
        k_gtz, k_lez = partition_inds_by_sign(x)
        assert isinstance(k_gtz, ndarray) and isinstance(k_lez, ndarray),                "You did not return Numpy arrays."
        assert issubdtype(k_gtz.dtype, integer) and issubdtype(k_lez.dtype, integer),                f"Both Numpy element types must be integers (yours are {k_gtz.dtype} and {k_lez.dtype})."
        assert (k_gtz == kp).all(), "Indices of positive values is incorrect."
        assert (k_lez == kn).all(), "Indices of non-positive values is incorrect."
    except:
        print("=== Error on test case! ===")
        print(f"Input: {x}")
        print(f"Your solution: {k_gtz}, {k_lez}")
        print(f"True solution: {kp}, {kn}")
        assert False, "Please double-check your approach on this test case."

print("Checking for correctness on small inputs...")
for _ in range(100):
    check2()

print("Checking for efficiency on a larger input...")
check2(2000000)

print("(Passed.)")

print(f"x == {x}")
k = np.concatenate((k_gtz, k_lez))
print(f"k == {k}")
y = x[k] # permute!
print(f"y = x[k] == {y}") # elements rearranged

def invert_perm(k):
    from numpy import ndarray, issubdtype, integer
    assert isinstance(k, ndarray), "Input permutation should be a Numpy array."
    assert issubdtype(k.dtype, integer), "Input permutation should have integer elements."
    assert (k >= 0).all(), "Input permutation should contain positive values."

    k_inv = np.empty(len(k), dtype=int)
    k_inv[k] = np.arange(len(k))
    return k_inv

# Demo
k_inv = invert_perm(k)
y = x[k]
print(f"x == {x}")
print(f"k == {k}")
print(f"y = x[k] == {y}")
print(f"k_inv == {k_inv}")
print(f"y[k_inv] == {y[k_inv]}")   # If all goes well, should equal `x`
assert (y[k_inv] == x).all(), "Demo failed!"

k = np.array([0, 3, 5, 6, 9, 1, 2, 4, 7, 8])
np.array([(i, k) for i,k in enumerate(k)])
inds = np.arange(len(k))
[inds[e] for e in k ]
k_inv = np.zeros_like(k)
np.arange(len(k))[k]


def check3(n=8):
    from numpy import arange, ndarray, issubdtype, integer
    from numpy.random import permutation
    try:
        x = arange(n)
        p = permutation(n)
        p_inv = invert_perm(p)
        assert isinstance(p_inv, ndarray) and (p_inv.ndim == 1), "You did not return a 1-D Numpy array."
        assert issubdtype(p_inv.dtype, integer), f"Numpy element type must be integer, not {p_inv.dtype}."
        y = x[p]
        z = y[p_inv]
        assert (x == z).all()
    except:
        print("=== Failed test case ===")
        print(f"Input: x == {x}")
        print(f"A random permutation: p == {p}")
        print(f"Permuted `x`: y == x[p] == {y}")
        print(f"Your inverse permutation: p_inv == {p_inv}")
        print(f"Result of y[p_inv] should be `x`, and yours is {z}")
        raise

for _ in range(100):
    check3()

print("(Passed.)")

def calc_degrees_matrix(A):
    from scipy.sparse import spdiags
    n = min(A.shape)
    return spdiags(A.sum(axis=1).reshape(n), diags=0, m=A.shape[0], n=A.shape[1])

def calc_fiedler_vector(A):
    from scipy.sparse.linalg import eigsh
    D = calc_degrees_matrix(A)
    L = D - A # Form Laplacian
    _, V = eigsh(L, k=2, which='SM') # Determine 2nd smallest eigenpair
    return V[:, 1]

# Demo:
v = calc_fiedler_vector(A)
print(v)

k_gtz, k_lez = partition_inds_by_sign(v)

print("=== Group 0 ===")
display(nodes.loc[k_gtz])

print("\n=== Group 1 ===")
display(nodes.loc[k_lez])

print("COO row indices:", A.row)
print("COO column indices:", A.col)
print("COO values:", A.data)

def reorder_by_fiedler(A, v):
    from scipy.sparse import coo_matrix
    assert isinstance(A, type(coo_matrix((1, 1))))
#     print(A)
#     print(v)
    k_gtz,k_lez = partition_inds_by_sign(v)
    k = np.concatenate((k_gtz,k_lez))
    inv_k = invert_perm(k)
    rows_perm = inv_k[A.row]
    cols_perm = inv_k[A.col]
    return coo_matrix((A.data,(rows_perm, cols_perm)), shape=A.shape)

A_perm = reorder_by_fiedler(A, v)
figure(figsize=(4, 4))
cspy(A_perm, cbar=False)

def check4(n=8):
    from random import randrange
    from numpy import array, arange, where
    import scipy.sparse as sparse
    v = []
    kp = []
    kn = []
    for i in range(n):
        v.append(randrange(-10, 10))
        (kp if v[-1] > 0 else kn).append(i)
    v, p = array(v), array(kp + kn)
    A = sparse.random(n, n, 0.25)
    A.data = arange(1, len(A.data)+1)
    try:
        A_perm = reorder_by_fiedler(A, v)
        assert isinstance(A_perm, type(A)),                f"You returned an object of type {type(A_perm)}, not {type(A)}."
        assert A.shape == A_perm.shape,                f"You returned a sparse matrix with dimensions {A_perm.shape} instead of {A.shape}."
        assert A.nnz == A_perm.nnz,                f"Your result has {A_perm.nnz} nonzeros instead of {A.nnz}."
        for i, j, a_perm_i_j in zip(A_perm.row, A_perm.col, A_perm.data):
            i0, j0 = p[i], p[j]
            k0 = where((A.row == i0) & (A.col == j0))[0]
            a_i0_j0 = A.data[k0]
            assert a_perm_i_j == a_i0_j0,                    f"Entry ({i}, {j}) of your solution does not appear to be correct."
    except:
        print("=== Error on test case! ===")
        print(f"Input matrix:")
        print(f"* rows = {A.row}")
        print(f"* cols = {A.col}")
        print(f"* vals = {A.data}")
        print(f"Fiedler vector: {v}")
        print(f"Permutation: {p}")
        assert False, "Please double-check your approach on this test case."

print("Checking for correctness...")
for _ in range(100):
    check4()

print("(Passed.)")

blog_nodes = pd.read_csv(dataset_path('polblogs3_nodes.csv'))
blog_edges = pd.read_csv(dataset_path('polblogs3_edges.csv'))

display(blog_nodes.head())
print(f"==> The `blog_nodes` data frame lists {len(blog_nodes)} blog sites.")
display(blog_edges.head())
print(f"==> The `blog_edges` data frame contains {len(blog_edges)} directed links"       f"among sites (~ {len(blog_edges)/len(blog_nodes):.1f} links/site).")

A_blogs = df_to_coo(blog_nodes, blog_edges)
v_blogs = calc_fiedler_vector(A_blogs)
A_blogs_perm = reorder_by_fiedler(A_blogs, v_blogs)

f, (ax1, ax2) = subplots(1, 2, figsize=(10, 5))
cspy(A_blogs, ax=ax1, cbar=False, s=1)
cspy(A_blogs_perm, ax=ax2, cbar=False, s=1)
kp_blogs, kn_blogs = partition_inds_by_sign(v_blogs)
len(kp_blogs), len(kn_blogs)

display(blog_nodes.loc[kp_blogs].sample(10))
display(blog_nodes.loc[kn_blogs].sample(10))


display(blog_nodes.loc[kp_blogs].describe())
display(blog_nodes.loc[kn_blogs].describe())

print(hidden_cell_template_msg())
