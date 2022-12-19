from sklearn.tree import DecisionTreeRegressor
# from sklearn.preprocessing import LabelEncoder
import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, roc_auc_score, plot_roc_curve
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import pickle


def train_data():
    df = pd.read_csv("training-set.csv")
    df = df.drop_duplicates()
    df['inst_type'] = df['inst_type'].map({'bank':1, 'nbfc':0})
    df['program'] = df['program'].map({'gross profit':2, 'gross margin':1, 'gst':0})
    df['roi'] = df['roi'].astype("int")
    df['cost_of_goods'] = df['cost_of_goods'].astype("int")
    df['depriciation'] = df['depriciation'].astype("int")
    df['interest_on_loans'] = df['interest_on_loans'].astype("int")
    df['sales'] = df['sales'].astype("int")
    df['other_income_and_interest'] = df['other_income_and_interest'].astype("int")
    df['net_profit_before_tax'] = df['net_profit_before_tax'].astype("int")
    df['cash_profit'] = df['cash_profit'].astype("int")
    df['gross_profit'] = df['gross_profit'].astype("int")
    df['se_on_cash_profit'] = df['se_on_cash_profit'].astype("int")
    df['se_on_gross_profit'] = df['se_on_gross_profit'].astype("int")
    df['se_on_gross_margin'] = df['se_on_gross_margin'].astype("int")
    X = df.iloc[:, 0:-1].values
    y = df.iloc[:, -1].values
    from sklearn.preprocessing import StandardScaler  
    scaler = StandardScaler() 
    scaler.fit(X,y) 
    X_scaled = scaler.transform(X)
    dt = DecisionTreeRegressor(max_depth = 10)
    dt.fit(X, y)
    filename = "loan_match"
    pickle.dump(dt,open(filename,'wb'))
    