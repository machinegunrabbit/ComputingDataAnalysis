import sys
print("* Python version:", sys.version)
import pickle
from problem_utils import get_path
# The usual suspects
import numpy as np
import scipy as sp
import matplotlib
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

print("* Numpy version:", np.__version__)
print("* Scipy version:", sp.__version__)
print("* Matplotlib version:", matplotlib.__version__)

# > A sentence is _important_ if it contains many important words. And a word is important if it appears in many important sentences.
#
# At first glance, this idea seems a bit circular! But it turns out we can formulate it as a well-defined mathematical problem in the language of linear algebra.
# **An algorithm.** From the background above, we now have a "simple" algorithm to rank words and sentences.
# 1. Given a document, construct the word-sentence matrix $A$.
# 2. Compute the **largest** (or **principal**) singular value and corresponding singular vectors of $A$. Denote the singular value by $\sigma_0$ and the left- and right-singular vectors as $u_0$ and $v_0$, respectively.
# 3. Use $u_0$ and $v_0$ to rank words and sentences. (**You** need to figure out the relationship between $w$ and $s$ and $u_0$ and $v_0$!)


def read_doc(filepath):
    with open(filepath, 'rt', encoding='utf-8') as fp:
        doc = [line for line in fp.read().splitlines() if line.strip() != '']
    return doc

# Read sample file:
raw_doc = read_doc(get_path('nyt2.txt'))

print("=== Here are the first 7 lines of the sample document ===")
for k, line in enumerate(raw_doc[:7]):
    print(f"\n* raw_doc[{k}] == {repr(line)}")
print("\n... (and so on) ...")

import re # Maybe re.split() will help?

# Complete this function:
def get_sentences(doc):
    assert isinstance(doc, list)
    return(
    [s.strip() for line in doc for s in re.split('[.?!]', line) if s.strip() != '']
    )

# Demo:
get_sentences(ex0_demo_input)
# ### Stopwords ###
from problem_utils import STOPWORDS
print(f"There are {len(STOPWORDS)} stopwords. Examples:")
for w in ["ourselves", "you're", "needn't", "cat", "dog", "python"]:
    is_or_isnt = "is" if w in STOPWORDS else "is *not*"
    print(f"- {repr(w)} {is_or_isnt} a stopword.")
def clean(s):
    from re import finditer
    pattern = r"[a-z]+('[a-z])?[a-z]*"
    return [match.group(0) for match in finditer(pattern, s.lower())]

print("* Sentence #6 ==", repr(raw_sents[6]))
print("\n* clean(raw_sents[6]) ==", clean(raw_sents[6]))


# Your next task is to construct the _bag-of-words representation_ of each sentence. If `s` is a sentence, its bag-of-words representation is a set (or "bag") of its unique words. However, this set should never include stopwords.
#
# Complete the function `gen_bag_of_words(sents)` to perform this task. In particular, the input `sents` is a list of sentences, as might be produced by Exercise 0. Your function should do the following:
#
# - Split each sentence `sents[k]` into a list of words, using `clean()` defined above.
# - Convert this list into a set, removing any stopwords.
# - Return a list of these sets. If the returned list is named `bags`, then `bags[k]` should be the set for `sents[k]`.
#
# The next code cell shows an example of the expected input and output of your function.

ex1_demo_input = ["This is a phrase; this, too, is a phrase",
                  "But this is another sentence",
                  "Hark",
                  "Come what may    <-- save those spaces, but not these -->",
                  "What did you say",
                  "Split into 3 (even without a space)",
                  "Okie dokie"]
ex1_demo_output = [{'phrase'},
                   {'another', 'sentence'},
                   {'hark'},
                   {'come', 'may', 'save', 'spaces'},
                   {'say'},
                   {'even', 'space', 'split', 'without'},
                   {'dokie', 'okie'}]

def gen_bag_of_words(sents):
    assert isinstance(sents, list)
    bags = []
    for sent in sents:
        bag = clean(sent)
        bag = set(bag)-STOPWORDS
        bags.append(bag)
    return bags

# Demo:
gen_bag_of_words(ex1_demo_input)
with open(get_path('ex1_soln.pickle'), 'rb') as fp:
    bags = pickle.load(fp)

print("=== First few bags ===")
for k, b in enumerate(bags[:10]):
    print(f"bags[{k}] == {repr(b)}")


# ### Generating IDs
from random import choices

all_words = set()
for b in bags:
    all_words |= b
word_to_id = {w: k for k, w in enumerate(all_words)}
id_to_word = {k: w for k, w in enumerate(all_words)}

print(f"There are {len(all_words)} unique words.")
for w in choices(list(all_words), k=5):
    print("- ID for", repr(w), "==", word_to_id[w])
    assert id_to_word[word_to_id[w]] == w


# Constructing the coordinates for a sparse matrix (3 points) ###

ex2_demo_bags = [{'cat', 'dog', 'fish'},
                 {'red', 'blue'},
                 {'dog'},
                 {'one', 'two', 'dog', 'fish'}]
ex2_demo_w2id = {'cat': 0, 'dog': 1, 'fish': 2, 'red': 3, 'blue': 4, 'one': 5, 'two': 6}

from math import log   # log(x) == natural logarithm of x
ex2_rows = [       0,          1,          2,        3,        4,          1,        5,        6,          1,          2]
ex2_cols = [       0,          0,          0,        1,        1,          2,        3,        3,          3,          3]
ex2_vals = [1/log(5), 1/log(5/3), 1/log(2.5), 1/log(5), 1/log(5), 1/log(5/3), 1/log(5), 1/log(5), 1/log(5/3), 1/log(2.5)]


def gen_coords(bags, word_to_id):
    # Some code to help you get started:
    m, n = len(word_to_id), len(bags)
    rows, cols, vals = [], [], []

    #get word frequency
    words = list(word_to_id.keys())
    freq_list = []
#     print(words)
    for word in words:
        freq = 0
        for sent in range(n):
            if word in bags[sent]:
                freq += 1
        freq_list.append(freq)
#     print(freq_list)
    word_freq = dict(zip(words,freq_list))
#     print(word_freq)

    from numpy import log
    # Construct rows, cols, and vals:
    for sent in range(n):
        sentence = list(bags[sent])
        for word in sentence:
            cols.append(sent)
            row = word_to_id[word]
            rows.append(row)
            value = 1/np.log((n+1)/word_freq[word])
            vals.append(value)
#         print(rows,cols,vals)
    # Returns your arrays:
    return rows, cols, vals

# Runs your function on the demo:
gen_coords(ex2_demo_bags, ex2_demo_w2id)

with open(get_path('ex2_soln.pickle'), 'rb') as fp:
    rows = pickle.load(fp)
    cols = pickle.load(fp)
    vals = pickle.load(fp)

print("=== First few coordinates ===")
print(f"* rows[:5] == {rows[:5]}")
print(f"* cols[:5] == {cols[:5]}")
print(f"* vals[:5] == {vals[:5]}")


# ### Calculating the SVD ###

from scipy.sparse import csr_matrix
A = csr_matrix((vals, (rows, cols)), shape=(len(word_to_id), len(bags)))

plt.figure(figsize=(9, 9))
plt.spy(A, marker='.', markersize=1)
pass

def get_svds_largest(A):
    from scipy.sparse.linalg import svds
    from numpy import abs
    u, s, v = svds(A, k=1, which='LM', return_singular_vectors=True)
    return s, abs(u.reshape(A.shape[0])), abs(v.reshape(A.shape[1]))

sigma0, u0, v0 = get_svds_largest(A)
print("sigma_0 ==", sigma0)
print("u0.shape ==", u0.shape)
print("v0.shape ==", v0.shape)


# Ranking words
def rank_words(u0, v0):
    return np.argsort(u0)[::-1]

# Demo on the input document:
word_ranking = rank_words(u0, v0)
top_ten_words = [id_to_word[k] for k in word_ranking[:10]]
print("Top 10 words:", top_ten_words)
from problem_utils import ex3_check_one
for trial in range(10):
    print(f"=== Trial #{trial} / 9 ===")
    ex3_check_one(rank_words)

print("\n(Passed.)")


# Ranking sentences
def rank_sentences(u0, v0):
    return np.argsort(v0)[::-1]

sentence_ranking = rank_sentences(u0, v0)
top_five_sentences = [raw_sents[k] for k in sentence_ranking[:5]]
print("=== Top 5 sentences ===")
for k, s in enumerate(top_five_sentences):
    print(f"\n{k}.", repr(s))
