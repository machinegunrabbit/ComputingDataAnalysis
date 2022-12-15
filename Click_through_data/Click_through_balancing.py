
# coding: utf-8

# # Problem 19: Click-through Balancing Act (5 points) #
#
# _Version 1.1_
#
# This notebook is a (hopefully) simple one about two common data preprocessing issues in practical machine learning, namely, imputing missing values and "balancing" training data. Here, you'll see these ideas in the context of analyzing advertising click-through data, where you wish to predict when a user will click on an ad using logistic regression. This problem will make use of some basic Python, pandas, and Numpy.
#
# The problem is worth a total of 5 points, broken up into **3** exercises. They are independent, so you can complete them in any order. (However, they do build on one another so read them in sequence.) Their points values are as follows.
#
# * Exercise 0: 2 points
# * Exercise 1: 2 points
# * Exercise 2: 1 point
#
# **Pro-tips.**
# - If your program behavior seem strange, try resetting the kernel and rerunning everything.
# - If you mess up this notebook or just want to start from scratch, save copies of all your partial responses and use `Actions` $\rightarrow$ `Reset Assignment` to get a fresh, original copy of this notebook. (_Resetting will wipe out any answers you've written so far, so be sure to stash those somewhere safe if you intend to keep or reuse them!_)
# - If you generate excessive output (e.g., from a ill-placed `print` statement), causing the notebook to load slowly or not at all, use `Actions` $\rightarrow$ `Clear Notebook Output` to get a clean copy. The clean copy will retain your code but remove any generated output. **However**, it will also **rename** the notebook to `clean.xxx.ipynb`. Since the autograder expects a notebook file with the original name, you'll need to rename the clean notebook accordingly.
#
# **Revision history.**
# * Version 1.1 - Added more hints, fixed a missing link [Th Apr 23, 2020]
# * Version 1.0 - Initial release

# ## Setup ##
#
# Here are some of the basic modules you'll need for this problem.

# In[24]:


import sys
import pickle
import numpy as np
import pandas as pd

print("* Python version:", sys.version)
print("* Numpy version:", np.__version__)
print("* pandas version", pd.__version__)

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')


# ## Get to know the dataset: Click-through data ##
#
# The dataset consists of [click-through data](https://www.kaggle.com/c/avazu-ctr-prediction) collected from mobile users about whether they clicked on ads. Run this code cell to load the data into a pandas `DataFrame` named `df_train_0` and inspect a small sample.

# In[25]:


from problem_utils import get_path, load_click_through_data

df_train_0 = load_click_through_data(get_path('train-4M.csv'), adversary=42)

print("\nFirst five data points (rows):")
display(df_train_0.head(5))
print("Last five data points (rows):")
display(df_train_0.tail(5))
print("==> Training set:", len(df_train_0), "points")


# **Observations.** There are over 3.2 million data points. The response variable you are trying to predict is in the column named `'click'`, whose values are 0 (did not click on the ad) or 1 (did click on the ad).
#
# All remaining columns are (undisclosed) features or attributes that we will use to build a prediction model. Two of them contain "continuous" numerical values: `'x_pixels'` and `'y_pixels'`, which presumably relate to the pixel-dimensions of the ad. All remaining columns are categorical or discrete. In particular, they only contain 0 and 1 values.
#
# In some of the exercises, you may need to refer to the columns by name or type (numerical or categorical). Let's import some special variable names your code can use to make referencing these columns a little easier:

# In[26]:


from problem_utils import RESPONSE, PREDICTORS, PREDICTORS_NUMERICAL, PREDICTORS_CATEGORICAL

print("* The response variable is in the column named", repr(RESPONSE))
print("* Predictor variables are in these columns:", repr(PREDICTORS))
print("    => Numerical (continuous) variables:", repr(PREDICTORS_NUMERICAL))
print("    => Categorical (discrete) variables:", repr(PREDICTORS_CATEGORICAL))


# **How many clicks?** To get a better sense of what we are predicting, let's count how many times 0 and 1 occur in the `'click'` column.

# In[27]:


num_clicks = df_train_0.groupby(RESPONSE)[RESPONSE].count().to_frame(name='count')
num_clicks['%'] = (100 * num_clicks['count'] / num_clicks['count'].sum()).round(1)
num_clicks


# It seems that most of the time (~ 83%), unsurprisingly, users do not bother to click on the ad!

# As for the remaining categorical variables:

# In[28]:


df_train_0[PREDICTORS_CATEGORICAL].describe(include='all')


# These are all binary categories (2 unique values, which it turns out are 0 and 1), with the most frequent value and its frequency shown by the `'top'` and `'freq'` rows above.
#
# There's only one problem, which leads to the first exercise: some of the predictors' values are, mysteriously, missing! This phenomenon can occur if, for example, the user has enabled a privacy setting that prevents measurement of some attribute.
#
# Here is a summary of how many missing values ("not-a-number" values) exist in each column of our dataset:

# In[29]:


def count_missing_by_col(df):
    return df.isna().sum().to_frame(name="# of missing values ('NaNs')").T

count_missing_by_col(df_train_0)


# ## Exercise 0: Imputing missing values (2 points) ##
#
# Complete the function, `fill_missing(df)`, below, so that it finds any missing (NaN) values and returns a **copy** with all missing values filled in as described below.
#
# First, assume the input dataframe `df` has categorical and numerical predictors given by the `PREDICTORS_CATEGORICAL` and `PREDICTORS_NUMERICAL` variables defined earlier. The responses (`RESPONSE`) are never missing.
#
# Your function should then do the following:
#
# 1. Create a copy of `df`.
# 2. For each categorical column, replace any missing values by the _mode_ in that column, ignoring the missing values. Recall that the mode of a collection of values is the most commonly occurring one. In the event of ties, choose the smallest mode value.
# 3. For each numerical column, replace any missing values by the _mean_ value in that column, ignoring missing values.
# 4. Return the copy.
#
# > **Hint 0.** Let `s` be a `Series`. The method [`s.fillna()`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.fillna.html) may prove useful.
# >
# > **Hint 1.** Let `s` be a `Series`. You can calculate the mode, ignoring not-a-number values, using [`s.mode()`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.mode.html), which will return a `Series` object containing the mode or modes (in the event of ties). Similarly, you can calculate the mean using [`s.mean()`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.mean.html).
# >
# > **Note.** You may assume that a given `Series`, `s`, always has at least one present value, so that `s.mean()` and `s.mode()` always return usable values.

# In[30]:


def fill_missing(df):
    global PREDICTORS_NUMERICAL, PREDICTORS_CATEGORICAL
    mode_dict = df[PREDICTORS_CATEGORICAL].mode()
    mode_vals = list(mode_dict.loc[0].values)
    mode = dict(zip(PREDICTORS_CATEGORICAL,mode_vals))
    categorical = df[PREDICTORS_CATEGORICAL].fillna(value=mode)

    mean_dict = df[PREDICTORS_NUMERICAL].mean()
    mean_vals = list(mean_dict.values)
#     print(mean_vals)
    mean = dict(zip(PREDICTORS_NUMERICAL,mean_vals))
#     print(mean)
    numerical = df[PREDICTORS_NUMERICAL].fillna(value=mean)
    display(numerical)
    total = pd.concat([df['click'],numerical,categorical], axis=1, join='inner')
    display(total)
    return total


# In[31]:


# Demo:
df_train_1 = fill_missing(df_train_0)

assert df_train_0 is not df_train_1, "*** ERROR: Did you return a _copy_? ***"
count_missing_by_col(df_train_1)


# In[32]:


# Test cell: `ex0__fill_missing` (2 points)

###
### AUTOGRADER TEST - DO NOT REMOVE
###

def ex0_check(max_rows=10):
    from problem_utils import ex0_gen_soln
    from problem_utils import assert_tibbles_are_equivalent
    df_clean, df_dirty_orig = ex0_gen_soln(max_rows)
    df_dirty = df_dirty_orig.copy()
    try:
        df_yours = fill_missing(df_dirty)
        assert df_yours is not df_dirty, "*** ERROR: Did you return a copy? ***"
        assert_tibbles_are_equivalent(df_yours, df_clean)
    except:
        print("=== Input data frame ===")
        display(df_dirty_orig)
        display(df_dirty_orig.info())
        print("=== Expected output ===")
        display(df_clean)
        display(df_clean.info())
        print("=== Your output ===")
        display(df_yours)
        display(df_yours.info())
        raise

for trial in range(10):
    print(f"=== Trial #{trial} / 9 ===")
    ex0_check()

print("\n(Passed.)")


# ### Precomputed solution for Exercise 0 ###
#
# Here is some code to load a precomputed training dataset with filled-in missing values. Regardless of whether your Exercise 0 works or not, please run this cell now so subsequent exercises can continue. It will define a variable named **`df_train`**, a pandas `DataFrame` that holds the training dataset you'll need. Subsequent code uses it, so do not modify it!

# In[33]:


df_train = pd.read_csv(get_path('ex0_soln.csv'))
display(df_train.sample(5))
count_missing_by_col(df_train)


# ## Building a logistic regression model ##
#
# Armed with the preceding data, let's build a logistic regression model for it. Let's use [scikit-learn's version](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html), which is easy to use and implements a similar algorithm to what you built yourself from scratch in Notebook 13.

# **Constructing the training data matrix and response vector.** Just like Notebook 13, you need to construct a data matrix and array of labels. These are easy to get from the training dataframe, `df_train`, created above. The code cell below creates two Numpy arrays:
#
# - **`X_train`**, the matrix of training data (rows are data points, columns are predictors).
# - **`y_train`**, the vector (array) of responses.

# In[11]:


X_train = df_train[PREDICTORS].values
y_train = df_train[RESPONSE].values

print(X_train.shape)
print(y_train.shape)


# > **Note.** One subtle difference from Notebook 13 is that `y_train` is a 1-D array, rather than a column vector stored as a $m \times 1$ 2-D array. But don't worry too much about this detail as it won't matter for what follows.

# **Building the model, or _classifier_.** Next, we can call scikit-learn's `LogisticRegression` class, which returns a classifier. To simplify things for you, we've wrapped this step into a function called `fit` that returns the fitted model, which is a special object that holds the model parameters.
#
# The code cell below defines `fit()` and uses it to build a classifier from the training data, `X_train` and `y_train`. The classifier is stored in an object called **`baseline_classifier`**. It's not important to read the code; rather, pay attention to the various outputs.

# In[34]:


def fit(X, y, verbose=True):
    from sklearn.linear_model import LogisticRegression
    if verbose: print("Fitting to data of size:", X.shape)
    classifier = LogisticRegression(random_state=0).fit(X, y)
    if verbose: print("Done!")
    return classifier

baseline_classifier = fit(X_train, y_train)


# **Testing data.** We've prepared a separate _testing_ or _validation_ dataset. Let's load the test data into a dataframe named **`df_test`**. Again, the code here isn't critical, but do inspect the output.

# In[35]:


df_test = load_click_through_data(get_path('test-4M.csv'))
display(df_test.sample(5))
print("==> Testing set:", len(df_test), "points")

display(count_missing_by_col(df_test))

num_clicks_test = df_test.groupby(RESPONSE)[RESPONSE].count().to_frame(name='count')
num_clicks_test['%'] = (100 * num_clicks_test['count'] / num_clicks_test['count'].sum()).round(1)
num_clicks_test


# Observe that it does not have any missing values, and that the ratio of users who clicked on an ad resembles that of the training dataset (about 17%).

# As a final step for the testing data, let's extract the data matrix and response variables corresponding to it into two variables named **`X_test`** and **`y_test`**, respectively.

# In[14]:


X_test = df_test[PREDICTORS].values
y_test = df_test[RESPONSE].values
print(X_test.shape)
print(y_test.shape)


# ### Baseline accuracy ###
#
# For this baseline classifier, let's determine how accurate it is when predicting on the test dataset. We've provided a function called `test()` that does this evaluation for you. Run the next two code cells to define this function and try it out on the `baseline_classifier`. Again, the code is not critical; skip to the output.

# In[38]:


def confusion(y_row, row, y_col, col, verbose=True):
    from pandas import crosstab, Series
    C = crosstab(Series(y_row, name=row),
                 Series(y_col, name=col))
    if verbose: display(C)
    return C

def test(classifier, X_test, y_test, verbose=True):
    if verbose: print("Testing on data of size:", X_test.shape)
    y_pred = classifier.predict(X_test)

    C = confusion(y_test, "Truth", y_pred, "Predicted", verbose=verbose)
    if verbose:
        score = 1e2 * classifier.score(X_test, y_test)
        print(f"Accuracy: {score:.1f}%")

    return y_pred, C


# In[39]:


_, baseline_confusion = test(baseline_classifier, X_test, y_test)


# **Observations.** The overall prediction accuracy should be about 83%. The output includes a confusion matrix, which shows how many times the classifier predicted 0 or 1 (the columns of the table) given the true labels (rows). If the classifer made perfect predictions, you would only see values on the diagonal and the off-diagonal entries would be zero.
#
# The accuracy isn't perfect, but it seems pretty good, at 83%. Well, except for one thing: **you could have gotten the same result simply by always guessing 0 (not clicked)!** Recall from our analysis of the original data that only about 17% of the observations involved users who clicked on ads, and in this application, these are the predictions you care about!

# ## Rebalancing via _down-sampling_ ##
#
# There a variety of ways of making a training dataset more "balanced." One way is to _down-sample_: determine which group is smaller, and then _randomly_ select---_without_ replacement---an equal number of points from the larger group. Doing so focuses the learning algorithm on the more rare group while possibly sacrificing some loss of information from the larger group.
#
# For example, suppose you have a 1-D Numpy array named `y` whose entries are as follows:
#
# ```python
#     #     index:  0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15
#     y = np.array([0,  0,  1,  0,  1,  0,  1,  0,  0,  0,  1,  0,  0,  1,  0,  0])
# ```
#
# This array has 16 elements (numbered 0-15). There are five (5) occurrences of the value 1, at index positions 2, 4, 6, 10, and 13. The remaining eleven (11) elements have the value 0. Thus, the smaller group is the collection of "1" values.
#
# In down-sampling, keep everything from the smaller group, meaning all five (5) of the "1" elements. Of the remaining elements from the 0-valued group, we select five of them **uniformly at random _without_ replacement**, so that we can have an equal number of 0 and 1 elements. _(Recall that sampling without replacement means drawing a subset of unique elements, **without** repeats.)_ For example, if we were to keep elements from the following index positions,
#
# ```python
#     keep = [1, 2, 4, 5, 6, 7, 8, 10, 13, 15]
# ```
#
# Then
#
# ```python
#     y[keep] == [0, 1, 1, 0, 1, 0, 0, 1, 1, 0]
# ```
#
# has an equal number of 0 and 1 elements. In the next exercise, you need to compute `keep`, which we'll call the _"keep-set."_

# **`choice()` in Numpy.** For this kind of sampling, a handy function is [`numpy.random.choice()`](https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.random.choice.html#numpy.random.choice). Given a list or Numpy array `a` of values, `choice(a, size=k, replace=False)` will return a uniformly randomly chosen subset of `k` elements without replacement (`replace=False`). Here's a demo, which takes a list of the positions containing "0" values from the above example and returns randomly chosen subset of 4 of them. (Run it a few times to see different subsets.)

# In[17]:


np.random.choice([0, 1, 3, 5, 7, 8, 9, 11, 12, 14, 15], size=4, replace=False)


# ### Exercise 1: Down-sampling (2 points) ###
#
# Suppose you are given a 1-D Numpy vector, `y`, whose values are either 0 or 1. Implement a function, `downsample(y)`, that does the following.
#
# - Assume that there is at least one occurrence of 0 and at least one occurrence of 1 in `y`.
# - First determine which entries of `y` have a 0 value, and which have a 1 value.
# - Determine which of these two groups is **smaller** (i.e., the 0-group or the 1-group).
# - Create a new 1-D array, `keep`, which will hold a "keep-set" of down-sampled elements.
# - The keep-set should include the _index positions_ of all elements from the **smaller** group.
# - The keep-set should also include the index positions of a randomly selected subset of the **larger** group. It should choose these uniformly at random **without** replacement, so that the number of elements in the keep-set from each group is equal.
#
# The function should return `keep`. The order in which these values are returned does not matter.
#
# For example, suppose you run this code frament.
#
# ```python
#     #     index:  0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15
#     y = np.array([0,  0,  1,  0,  1,  0,  1,  0,  0,  0,  1,  0,  0,  1,  0,  0])
#     keep = downsample(y)
# ```
#
# The "1"-group is smaller, having just five elements. Therefore, the resulting output should have ten elements, such as the following:
#
# ```python
#     keep == np.array([2, 4, 6, 10, 13, 0, 1, 5, 15])
# ```
#
# > Verify that `y[keep]` has an equal number of 0 and 1 values.

# In[151]:


def downsample(y):
    # You can assume `y` is a 1-D Numpy array-like object
    assert hasattr(y, 'ndim') and hasattr(y, 'shape'), "*** `y` is not a Numpy array-like object? ***"
    assert y.ndim == 1, "*** `y` is not 1-D? ***"

    Zero_index = []
    One_index = []
    for index in range(y.shape[0]):
        if y[index] == 0:
            Zero_index.append(index)
        if y[index] == 1:
            One_index.append(index)
#     print(Zero_index)
#     print(One_index)
    if len(Zero_index) > len(One_index):
        long = Zero_index
        short = One_index
    else:
        long = One_index
        short = Zero_index
#     print(long)
#     print(short)
    left = np.random.choice(long, size=len(short), replace=False)
    left_list = list(left)
    print(left_list)
    keep_list = short+left_list
    print(keep_list)
    keep = np.array(keep_list)

    return keep


# In[152]:


y_demo = np.array([0,  0,  1,  0,  1,  0,  1,  0,  0,  0,  1,  0,  0,  1,  0,  0])
keep_demo = downsample(y_demo)

print("Input: y ==", y_demo)
print("Keeping: keep ==", keep_demo)
print("Equal zeros and ones? y[keep] ==", y_demo[keep_demo])


# In[153]:


# Test cell: `ex1__downsample` (2 points)

def ex1_check():
    from problem_utils import ex1_random_bin
    from numpy import isin, setdiff1d
    from random import randint
    n = randint(2, 20)
    y, k_smaller, k_larger = ex1_random_bin(n)
    n_out = 2 * len(k_smaller)
    try:
        keep = downsample(y)
        assert hasattr(keep, 'ndim') and hasattr(keep, 'shape'),             f"*** ERROR: Should return a Numpy array-like object, not a `{type(keep)}` object. ***"
        assert keep.ndim == 1,             f"*** ERROR: Should return a 1-D array, not a {keep.ndim}-D one. ***"
        assert len(keep) == n_out,             f"*** ERROR: Should have returned {n_out} elements, not {len(keep)}. ***"
        assert len(set(keep)) == n_out,             f"*** ERROR: Output is of the wrong length or contains duplicate elements. ***"
        assert ((0 <= keep) & (keep < n)).all(),             f"*** ERROR: Output contains invalid (out-of-bounds) values ***"
        assert isin(k_smaller, keep).all(),             f"*** ERROR: Elements {setdiff1d(k_smaller, keep)} are missing. ***"
    except:
        print("=== Inputs ===")
        print("- Input array, `y`:", y)
        print("- Smaller group positions (must be included):", k_smaller)
        print("- Larger group positions (choose an equal-sized subset):", k_larger)
        print("\n=== Your output ===")
        print("- Keep-set:", keep)
        raise

for trial in range(10):
    print(f"=== Trial #{trial} / 9 ===")
    ex1_check()

###
### AUTOGRADER TEST - DO NOT REMOVE
###

print("\n(Passed.)")


# ### Precomputed solution for Exercise 1 ###
#
# Here is some code to load a precomputed training dataset with filled-in missing values. Regardless of whether your Exercise 0 works or not, please run this cell now so subsequent exercises can continue. It will create two variables named **`keep_train_ds`** and **`keep_test_ds`**, two 1-D Numpy array-like objects that indicate which samples to keep from the training and testing datasets, respectively. You'll need these two _keep-sets_ later, so do not modify them!

# In[154]:


with open(get_path('ex1_soln.pickle'), 'rb') as fp:
    keep_train_ds = pickle.load(fp)
    keep_test_ds = pickle.load(fp)

print(keep_train_ds.shape)
print(keep_test_ds.shape)


# ### Reassessing the baseline classifier ###
#
# Suppose we downsample the _test set_, so that we test on equal numbers of "0" and "1" examples. How does the accuracy change?

# In[155]:


# Down-sample the testing data:
X_test_ds = X_test[keep_test_ds, :]
y_test_ds = y_test[keep_test_ds]

# Reevaluate the classifier:
test(baseline_classifier, X_test_ds, y_test_ds)


# **Observation.** You should see test accuracy drop to near 50%. Recall that this balanced, down-sampled test set has equal numbers of 0 and 1 examples. So this accuracy is no better than random guessing!

# ## Up-sampling ##
#
# Whereas down-sampling shrinks the larger groups so that they have an equal number of samples as the smaller group, _up-sampling_ does the opposite: it takes the smaller group and makes it bigger by randomly selecting elements _with replacement_. One needs to use replacement because the smaller group will necessarily require repeats to match the size of a larger group.
#
# > Conveniently, the `choice()` function mentioned before allows you to sample with replacement by the parameter, `replace=True`.
#
# In contrast to down-sampling, up-sampling avoids throwing out data. However, doing so comes at a price: the cost goes up for training (or testing) on a now-larger number of up-sampled inputs.

# ### Exercise 2: Up-sampling (1 point) ###
#
# Suppose you are given a 1-D Numpy vector, `y`, whose values are either 0 or 1. Implement a function, `upsample(y)`, that implements an _up-sampling strategy_, summarized as follows.
#
# - Assume that `y` has at least one occurrence of 0 and at least one occurrence of 1.
# - First determine which entries of `y` have a 0 value, and which have a 1 value.
# - Determine which of these two groups is **larger** (i.e., the 0-group or the 1-group).
# - Create a new 1-D array, `keep`, which will hold a "keep-set" of up-sampled elements.
# - The keep-set should include the _index positions_ of all elements from the **larger** group.
# - The keep-set should also include the index positions of a randomly selected subset of the **smaller** group. It should choose these so that the number of elements from each group in the keep set is equal, meaning elements from this smaller group must repeat.
#
# The function should return `keep`. The order in which these values are returned does not matter.
#
# For example, suppose you run this code frament.
#
# ```python
#     #     index:  0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15
#     y = np.array([0,  0,  1,  0,  1,  0,  1,  0,  0,  0,  1,  0,  0,  1,  0,  0])
#     keep = upsample(y)
# ```
#
# The "0"-group is larger, having 11 elements. Therefore, the resulting output should have 22 elements, which might be the following:
#
# ```python
#     keep == np.array([0, 1, 3, 5, 7, 8, 9, 11, 12, 14, 15, \
#                       2, 2, 4, 4, 4, 6, 6, 10, 13, 13, 13])
# ```
#
# (Verify that `y[keep]` has an equal number of 0 and 1 values.)

# In[156]:


def upsample(y):
    # You can assume `y` is a 1-D Numpy array-like object
    assert hasattr(y, 'ndim') and hasattr(y, 'shape'), "*** `y` is not a Numpy array-like object? ***"
    assert y.ndim == 1, "*** `y` is not 1-D? ***"

    Zero_index = []
    One_index = []
    for index in range(y.shape[0]):
        if y[index] == 0:
            Zero_index.append(index)
        if y[index] == 1:
            One_index.append(index)
#     print(Zero_index)
#     print(One_index)
    if len(Zero_index) > len(One_index):
        long = Zero_index
        short = One_index
    else:
        long = One_index
        short = Zero_index
#     print(long)
#     print(short)
    left = np.random.choice(short, size=len(long), replace=True)
    left_list = list(left)
    print(left_list)
    keep_list = long+left_list
    print(keep_list)
    keep = np.array(keep_list)

    return keep


# In[157]:


y_demo = np.array([0,  0,  1,  0,  1,  0,  1,  0,  0,  0,  1,  0,  0,  1,  0,  0])
keep_demo = upsample(y_demo)

print("Input: y ==", y_demo)
print("Keeping: keep ==", keep_demo)
print("Equal zeros and ones? y[keep] ==", y_demo[keep_demo])


# In[158]:


# Test cell: `ex2__upsample` (1 point)

def ex2_check():
    from problem_utils import ex1_random_bin
    from numpy import isin, setdiff1d
    from random import randint
    n = randint(2, 20)
    y, k_smaller, k_larger = ex1_random_bin(n)
    n_out = 2 * len(k_larger)
    try:
        keep = upsample(y)
        assert hasattr(keep, 'ndim') and hasattr(keep, 'shape'),             f"*** ERROR: Should return a Numpy array-like object, not a `{type(keep)}` object. ***"
        assert keep.ndim == 1,             f"*** ERROR: Should return a 1-D array, not a {keep.ndim}-D one. ***"
        assert len(keep) == n_out,             f"*** ERROR: Should have returned {n_out} elements, not {len(keep)}. ***"
        assert ((0 <= keep) & (keep < n)).all(),             f"*** ERROR: Output contains invalid (out-of-bounds) values ***"
        assert isin(k_larger, keep).all(),             f"*** ERROR: Elements {setdiff1d(k_larger, keep)} are missing. ***"
        assert isin(keep, k_larger).sum() == len(k_larger),             f"*** ERROR: Are elements from the larger set repeated? ***"
    except:
        print("=== Inputs ===")
        print("- Input array, `y`:", y)
        print("- Smaller group positions (any subset, which may repeat):", k_smaller)
        print("- Larger group positions (must be included and appear only once each):", k_larger)
        print("\n=== Your output ===")
        print("- Keep-set:", keep)
        raise

for trial in range(10):
    print(f"=== Trial #{trial} / 9 ===")
    ex2_check()

###
### AUTOGRADER TEST - DO NOT REMOVE
###

print("\n(Passed.)")


# ### Precomputed solution for Exercise 2 ###
#
# Here is some code to load a precomputed training dataset with filled-in missing values. Regardless of whether your Exercise 2 works or not, please run this cell now. It will create two variables named **`keep_train_us`** and **`keep_test_us`**, two 1-D Numpy array-like objects that indicate which samples to keep from the training and testing datasets, respectively. The remaining notebook code will need these two keep-sets, so do not modify them!

# In[159]:


with open(get_path('ex2_soln.pickle'), 'rb') as fp:
    keep_train_us = pickle.load(fp)
    keep_test_us = pickle.load(fp)

print(keep_train_us.shape)
print(keep_test_us.shape)


# ## Retraining with balanced data ##
#
# > There are no more exercises in this notebook. If you are pressed for time, be sure to submit your work and consider moving on!
#
# Given the ability to up-sample and down-sample the data, let's see how doing so affects training accuracy and generalization. You should see that accuracy goes down slightly on the original test data, but there is some small improvement when testing on the down-sampled test data. (Recall that the baseline classifier did no better than chance in this case.)

# In[160]:


methods = {'down-sampling': (keep_train_ds, keep_test_ds),
           'up-sampling': (keep_train_us, keep_test_us)}

X_test_ds, y_test_ds = X_test[keep_test_ds, :], y_test[keep_test_ds]
for method, (keep_train, _) in methods.items():
    print("\n=== Using", method, "to re-balance the training data ===\n")
    X_train_bal = X_train[keep_train, :]
    y_train_bal = y_train[keep_train]
    classifier = fit(X_train_bal, y_train_bal)

    print("\n==> Testing on the unmodified test data ...")
    test(classifier, X_test, y_test)

    print("\n==> Testing on the down-sampled test data ...")
    test(classifier, X_test_ds, y_test_ds)


# **Epilogue.** In fact, there are many interesting methods for resampling to create better datasets for training. You'll encounter these techniques in other courses of the analytics program.
#
# **Fin!** You’ve reached the end of this part. Don’t forget to restart and run all cells again to make sure it’s all working when run in sequence; and make sure your work passes the submission process. Good luck!
