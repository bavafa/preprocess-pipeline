#!/usr/bin/env python
# coding: utf-8

# # BAIT 509 Assignment 2: Preprocessing, Pipelines and Hyperparameter Tuning
# 

# ## Introduction and learning goals <a name="in"></a>
# <hr>
# 
# Welcome to the assignment! In this assignment, you will practice:
# 
# - Identify when to implement feature transformations such as imputation and scaling.
# - Apply `sklearn.pipeline.Pipeline` to build a machine learning pipeline.
# - Use `sklearn` for applying numerical feature transformations on the data.
# - Identify when it's appropriate to apply ordinal encoding vs one-hot encoding.
# - Explain strategies to deal with categorical variables with too many categories.
# - Use `ColumnTransformer` to build all our transformations together into one object and use it with `scikit-learn` pipelines.
# - Carry out hyperparameter optimization using `sklearn`'s `GridSearchCV` and `RandomizedSearchCV`.

# ## Introduction <a name="in"></a>
# <hr>
# 
# A crucial step when using machine learning algorithms on real-world datasets is preprocessing. This assignment will give you some practice to build a preliminary supervised machine learning pipeline on a real-world dataset. 

# ## Exercise 1: Introducing and Exploring the dataset <a name="1"></a>
# <hr>
# 
# In this assignment, you will be working on a sample of [the adult census dataset](https://www.kaggle.com/uciml/adult-census-income#) that we provide as `census.csv`. We have made some modifications to this data so that it's easier to work with. 
# 
# This is a classification dataset and the classification task is to predict whether income exceeds 50K per year or not based on the census data. You can find more information on the dataset and features [here](http://archive.ics.uci.edu/ml/datasets/Adult).
# 
# 
# *Note that many popular datasets have sex as a feature where the possible values are male and female. This representation reflects how the data were collected and is not meant to imply that, for example, gender is binary.*

# In[1]:


import pandas as pd

census_df = pd.read_csv("census.csv")
census_df


# ### 1.1 Data splitting 
# rubric={accuracy:2}
# 
# <div class="alert alert-info" style="color:black">
#     
# To avoid violation of the golden rule, the first step before we do anything is splitting the data. 
# 
# Split the data into `train_df` (80%) and `test_df` (20%). Keep the target column (`income`) in the splits so that we can use it in EDA. 
# Please use `random_state=893`, so that your results are consistent with what we expect.
#     
# </div>

# In[2]:


# Assign the splits to train_df and test_df
from sklearn.model_selection import train_test_split

# Split the dataset into 80% train and 20% test 
train_df, test_df = train_test_split(census_df, test_size = 0.2, train_size = 0.8, random_state = 893)


# Let's examine our `train_df`,
# you can just follow along for the next few cells.

# In[3]:


train_df


# In[4]:


train_df.info()


# It looks like things are in order,
# but there is a hidden gotcha with this dataframe.
# Let's look at the unique values of each column.

# In[5]:


from IPython.display import HTML  # This step is just to avoid the long columns being truncated as "..."

HTML(
    train_df
    .select_dtypes(object)
    .apply(lambda x: sorted(pd.unique(x)))
    .to_frame()
    .to_html()
)


# You can see that there are question marks in the columns "workclass", "occupation", and "native_country".
# Unfortunately it seems like the people collecting this data used a non-conventional way to indicate missing/unknown values
# instead of using the standard blank/NaN.
# Our first step would be to do this conversion manually,
# so that `?` is not interpreted as an actual value by our models.

# In[6]:


import numpy as np
train_df_nan = train_df.replace("?", np.nan)
test_df_nan = test_df.replace("?", np.nan)


# ### 1.2 Describing your data
# rubric={accuracy:2}
# 
# <div class="alert alert-info" style="color:black">
#     
# Use `.describe()` to show summary statistics of each feature in the `train_df_nan` dataframe.
# Figure out how to show numerical and categorical columns separately
# by reading the `describe` docstring to figure out which parameter to use.
# 
# </div>

# In[7]:


# Numerical
train_df_nan.describe(include = [np.number])


# In[8]:


# Categorical
train_df_nan.describe(include = 'object')


# ### 1.3 Identifying potentially important features
# rubric={reasoning:2}
# 
# <div class="alert alert-info" style="color:black">
#     
# Suggest which features you think seem relevant
# for the given prediction task of building a model to identify who makes over and under 50k.
# List these features and briefly explain your rationale in why you have selected them.
# 
# </div>

# #### Answer:
# 
# - "age"
# - "education_num"
# - "marital_status"
# 
# seem to be good predictors for the income as they show clear disticted peaks for the income values belwo and above 50K.

# In[9]:


import altair as alt

alt.data_transformers.disable_max_rows()  # Allows us to plot big datasets

alt.Chart(train_df.sort_values('income')).mark_bar(opacity=0.6).encode(
    alt.X(alt.repeat(), type='quantitative', bin=alt.Bin(maxbins=50)),
    alt.Y('count()', stack=None),
    alt.Color('income')
).properties(
    height=200
).repeat(
    train_df_nan.select_dtypes('number').columns.to_list(),
    columns=2
)


# In[10]:


alt.Chart(train_df.sort_values('income')).mark_bar(opacity=0.6).encode(
    alt.X(alt.repeat(), type='nominal'),
    alt.Y('count()', stack=None),
    alt.Color('income')
).properties(
    height=200
).repeat(
    train_df_nan.select_dtypes('object').columns.to_list(),
    columns=1
)


# ### 1.4 Separating feature vectors and targets  
# rubric={accuracy:2}
# 
# <div class="alert alert-info" style="color:black">
# 
# Create `X_train`, `y_train`, `X_test`, `y_test` from `train_df_nan` and `test_df_nan`. 
#     
# </div>

# In[11]:


X_train = train_df_nan.drop(columns = ['income'])
y_train = train_df_nan['income']
X_test = test_df_nan.drop(columns = ['income'])
y_test = test_df_nan['income']


# ### 1.5 Training?
# rubric={reasoning:2}
# 
# 
# <div class="alert alert-info" style="color:black">
# 
# If you train [`sklearn`'s `SVC`](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) model on `X_train` and `y_train` at this point, would it work? Why or why not?
#     
# </div>

# #### Answer:
# 
# It won't work as we have missing values in these data and the model cannot handle missing values.

# ## Exercise 2: Preprocessing <a name="3"></a>
# <hr>

# In this exercise, you'll be wrangling the dataset so that it's suitable to be used with `scikit-learn` classifiers. 

# ### 2.1 Identifying transformations that need to be applied
# rubric={reasoning:7}
# 
# <div class="alert alert-info" style="color:black">
# 
# Identify the columns on which transformations need to be applied and tell us what transformation you would apply in what order by filling in the table below. Example transformations are shown for the feature `age` in the table.  
# 
# Note that for this problem, no ordinal encoding will be executed on this dataset. 
# 
# Are there any columns that you think should be dropped from the features? If so, explain your answer.
# 
# </div>

# | Feature | Transformation |
# | --- | ----------- |
# | age | imputation, scaling |
# | workclass |  |
# | fnlwgt |  |
# | education |  |
# | education_num |  |
# | marital_status |  |
# | occupation |  |
# | relationship |  |
# | race |  |
# | sex |  |
# | capital_gain |  |
# | capital_loss |  |
# | hours_per_week |  |
# | native_country |  |

# #### Answers:
# 
# "workclass", "occupation", and "native_country" -> Should also be imputed then one hot encoded as they are categorical features that have missing values (now these missing values are marked as NaN).
# 
# Other numerical features like "age", "fnlwgt", "education_num", "capital_gain", "capital_loss", and "hours_per_week" should be scaled.
# 
# Other categorical features such as "education", "marital_status", "relationship", "race", and "sex" should be one hot encoded.
# 
# Moreover, "fnlwgt", "capital_gain", and "capital_loss" do not seem to impact the results based on the graphs. Hence, they can be omitted from the list of features.

# ### 2.2 Numeric vs. categorical features
# rubric={reasoning:2}
# 
# <div class="alert alert-info" style="color:black">
# 
# Since we will apply different preprocessing steps on the numerical and categorical columns,
# we first need to identify the numeric and categorical features and create lists for each of them
# (make sure not to include the target column).
# 
# *Save the column names as string elements in each of the corresponding list variables below*
#     
# </div>

# In[12]:


# Creating feauture list manually
numeric_features = ["age",
                    "fnlwgt", 
                    "education_num", 
                    "capital_gain", 
                    "capital_loss",
                    "hours_per_week"]

categorical_features = ["workclass",
                        "occupation",
                        "native_country",
                        "education", 
                        "marital_status", 
                        "relationship", 
                        "race",
                        "sex"]


# Alternatively we can create all the numeric and categorical features by one command

# numeric_features = X_train.select_dtypes('number').columns
# categorical_features = X_train.select_dtypes('object').columns


# ### 2.3 Numeric feature pipeline
# rubric={accuracy:2}
# 
# <div class="alert alert-info" style="color:black">
# 
# Let's start making our pipelines. Use `make_pipeline()` or `Pipeline()` to make a pipeline for the numeric features called `numeric_transformer`. 
#     
# </div>

# In[14]:


from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

numeric_transformer = Pipeline(
    steps=[("scaler", StandardScaler())]
)


# ### 2.4 Categorical feature pipeline
# rubric={accuracy:2}
# 
# <div class="alert alert-info" style="color:black">
# 
# Next, make a pipeline for the categorical features called `categorical_transformer`. 
# To keep things simple,
# we will impute on all columns,
# including those where we did not find missing values in the training data.
# Use `SimpleImputation()` with `strategy='most_frequent'`. 
# Add a OneHotEncoder as the second step and configure it to ignore unknown values in the test data.
#     
# </div>

# In[26]:


from sklearn.preprocessing import OneHotEncoder

categorical_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="most_frequent")),
           ("onehot", OneHotEncoder(handle_unknown="ignore"))]
)


# ### 2.5 ColumnTransformer
# rubric={accuracy:2}
# 
# <div class="alert alert-info" style="color:black">
# 
# Create a column transformer that applies our numeric pipeline transformations to the numeric feature columns
# and our categorical pipeline transformations to the categorical feature columns.
# Assign this columns transformer to the variable `preprocessor`.
# 
# </div>

# In[27]:


from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_transformer, numeric_features),
        ("categorical", categorical_transformer, categorical_features)
    ],
    remainder='passthrough'
)


# ## Exercise 3: Building a Model <a name="4"></a>
# <hr>

# ### 3.1 Dummy Classifier
# rubric={accuracy:3}
# 
# <div class="alert alert-info" style="color:black">
# 
# Now that we have our preprocessing pipeline setup,
# let's move on to the model building.
# First,
# it's important to build a dummy classifier to establish a baseline score to compare our model to.
# Make a `DummyClassifier` that predicts the most common label, train it, and then score it on the training and test sets
# (in two separate cells so that both scores are displayed).
#     
# </div>

# In[29]:


from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_validate

dummy = DummyClassifier(strategy="most_frequent")
scores = cross_validate(dummy, X_train, y_train, return_train_score=True)


# In[32]:


print('Mean training score', scores['train_score'].mean().round(2))
print('Mean validation score', scores['test_score'].mean().round(2))


# ### 3.2 Main pipeline
# rubric={accuracy:2}
# 
# <div class="alert alert-info" style="color:black">
# 
# Define a main pipeline that transforms all the different features and uses an `SVC` model with default hyperparameters. 
# If you are using `Pipeline` instead of `make_pipeline`, name each of your steps `columntransformer` and `svc` respectively. 
#     
# </div>

# In[40]:


from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

main_pipe = Pipeline(
    steps=[
        ("columntransformer", preprocessor),
        ("svc", SVC())])


# ### 3.3 Hyperparameter tuning/optimization
# 
# rubric={accuracy:3}
# 
# <div class="alert alert-info" style="color:black">
# 
# Now that we have our pipelines and a model, let's tune the hyperparameters `gamma` and `C`.
# For this tuning,
# construct a grid where each hyperparameter can take the values `0.1, 1, 10, 100`
# and randomly search for the best combination.
# 
# To save some running time on your laptops,
# use 3-fold crossvalidation to evaluate each result
# and only search for 7 iterations,
# and set `n_jobs=-1`.
# Return the train and testing score,
# set `random_state=289`,
# and optionally `verbose=2` if you want to see information as the search is occurring.
# Don't forget to fit the best model from the `RandomizedSearchCV` object
# on all the training data as the final step.
# 
# *This search is quite demanding computationally so be prepared for this to take 2 or 3 minutes and your fan may start to run!*
#     
# </div>

# In[41]:


from sklearn.model_selection import RandomizedSearchCV

param_grid = {
    "svc__gamma": [0.1, 1.0, 10, 100],
    "svc__C": [0.1, 1.0, 10, 100]
}


# In[44]:


random_search = RandomizedSearchCV(main_pipe, param_grid, cv=3, return_train_score=True, verbose=2, n_jobs=-1, n_iter=7, random_state=289)
random_search.fit(X_train, y_train);


# ### 3.4 Choosing your hyperparameters
# rubric={accuracy:2, reasoning:1}
# 
# <div class="alert alert-info" style="color:black">
# 
# We are displaying the results from the random hyperparameter search
# as a dataframe below.
# Looking at this table,
# which values for `gamma` and `C` would you choose for your final model and why? 
# You can answer this by either manually by using the table
# or by accessing the corresponding attributes from the random search object.
# 
# </div>

# In[45]:


pd.DataFrame(random_search.cv_results_)[["params", "mean_test_score", "mean_train_score", "rank_test_score"]]


# #### Answer:
# 
# Based on the ranking in the above table, gamma=0.1 and C=0.1 lead to the best results.
# 
# Moreover, as can be seen the mean_test_score of this combination has the highest value among all other tested alternatives.

# # 4. Evaluating on the test set <a name="5"></a>
# <hr>
# 
# Now that we have a best-performing model, it's time to assess our model on the test set. 

# ### 4.1 Scoring your final model
# rubric={accuracy:2}
# 
# <div class="alert alert-info" style="color:black">
# 
# What is the training and test score of the best scoring model? 
# Score the model in two separate cells so that both the training and test scores are displayed.
#     
#     
# </div>

# In[46]:


random_search.score(X_train, y_train)


# In[47]:


random_search.score(X_test, y_test)


# ### 4.2 Assessing your model
# rubric={reasoning:2}
# 
# <div class="alert alert-info" style="color:black">
# 
# Compare your final model accuracy with your baseline model from question 3.1,
# do you consider that our model is performing better than the baseline to such as extent that you would prefer it on deployment data?
# 
# Briefly describe one aspect of our model development in this notebook that either supports your confidence in the model we have,
# or one possible improvement to what we did here that you think could have increased our model score.
#     
# </div>

# #### Answer:
# 
# Our model outperforms the dummy classifier by far. The improvement of accuracy from 50% to 82% strongly suggest utilization of the final model over the base model.
# 
# As the test score in this case is very close to the validation score, we can be more confidnet that we have not run hyperparameter optimization too much, hence hyperparameter overfitting has not happened here.
# 
# To increase the model score, we could use the GridSearchCV() to fully explore the range of gamma and C and even test more granular values for gamma and C. However, repeating cross-validation over and over again, might result in optimization bias or overfitting the validation set. 
# 
# There are also python libraries (like scikit-optimize, hyperopt, and hyperband) for optimization of hyperparameter search that use machine learning to predict what hyperparameters will perfomr best. Using them also could improve the score to some extent.

# ### End
