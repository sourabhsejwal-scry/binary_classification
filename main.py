# Importing necessary libraries
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import itertools

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.svm import NuSVC, SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, StackingClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

import lightgbm as lgb
import sys

# Define a parser for getting arguments at the CLI
parser = argparse.ArgumentParser()
# Adding argument which is path of train data
parser.add_argument('-tr', '--train', type=str, help='It is the path to a train file')
# Adding argument which is path of test data
parser.add_argument('-te', '--test', type=str, help='It is the path to a test file')
# Parsing the arguments
args = parser.parse_args()

# Check if the path to the train file is specified or not
if not args.train:
    sys.exit(" Please specify the correct path to the train file ")

try:
    # Reading the train data
    data = pd.read_csv(args.train)
except FileNotFoundError:
    sys.exit("Could not find the file")

# Check if the path to the test file is specified or not
if not args.test:
    sys.exit("Please specify the correct path to the test file ")

try:
    # Reading the test data
    test_data = pd.read_csv(args.test)
except FileNotFoundError:
    sys.exit("Could not find the file")

# Reading the train and test feature datasets
train_f = pd.read_csv('train_feature.csv')
test_f = pd.read_csv('test_feature.csv')

# Assigning values to the train and test datasets
X_train = train_f  # train dataset
y_train = data[' Label']  # Label class from train dataset
X_test = test_f  # test dataset

# Initializing classifiers
classifier1 = SVC()
classifier2 = MLPClassifier()
classifier3 = DecisionTreeClassifier(max_depth=5)
classifier4 = RandomForestClassifier()
classifier5 = GaussianNB()
classifier6 = KNeighborsClassifier(3)

# Define estimators
estimator_list = [
    ('classifier1', classifier1),
    ('classifier2', classifier2),
    ('classifier3', classifier3),
    ('classifier4', classifier4),
    ('classifier5', classifier5),
    ('classifier6', classifier6),
]

# Build stack model
stack_model = StackingClassifier(
    estimators=estimator_list, final_estimator=LogisticRegression()
)

# Fit the stack model
stack_model.fit(X_train, y_train)

# Predict the probabilities of the test data
y_test_pred_prob = stack_model.predict_proba(X_test)
y_test_pred = y_test_pred_prob[:, 1]

# Create a dataframe for the predicted probabilities
df2 = pd.DataFrame(y_test_pred, columns=['Label'])

# Read the test data
test_data = pd.read_csv("test.csv")
# Get the IDs
id = test_data['ID']
# Concatenate the IDs and predicted labels into a single dataframe
result3 = pd.concat([id, df2], axis=1, join="inner")

# Write the output to a csv file
result3.to_csv('group10_output.csv', index=False)
