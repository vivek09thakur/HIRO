import csv
import re
import numpy as np
import pandas
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score,train_test_split
from sklearn.tree import DecisionTreeClassifier,_tree
from sklearn.svm import SVC

