import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from patsy import dmatrices
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score

#Import .csv file to pandas
df = pd.read_csv('International_Yelp.csv')

#Create input and output data sets
X = df.drop('stars',1)
Y = df.stars

#Create a new binary output variable Success, Success = 1 if stars>=4, else 0 
df['success'] = [1 if x>=4 else 0 for x in df['stars'] ]

X = df.drop('stars',1)
X = X.drop('success',1)
Y = df.success
# A function to covert all the categorical predictors to numeric 
def dummy_df(df, list):
    for x in list:
        dummies = pd.get_dummies(df[x], prefix=x, dummy_na = False)
        df = df.drop(x,1)
        df = pd.concat([df, dummies], axis=1)
    return df 

#Create a list of categorical predictors 

list = ['Good_For_Latenight', 'Outdoor_Seating' , 'Alcohol', 'Ambience_classy' , 'Parking_Lot' , 'Ambience_Touristy' , 
         'Good_For_Brunch' , 'Waiter_Service' , 'Parking_Street' , 'Ambience_Hipster' , 'Good_For_Breakfast' , 
         'Parking_Garage' , 'Accepts_Credit_Cards' , 'Good_For_Lunch' , 'valet','Take_out','Good_For_dessert' ,
         'Takes_Reservations' , 'Ambience_Trendy' , 'Delivery' , 'WiFi', 'Wheelchair_Accessible' ,
         'Caters' , 'Good_For_Dinner','Good_For_Kids' , 'Parking_Validated', 'Has_TV' , 'Ambience_Casual',
         'Drive_Thru', 'Noise_Level' ,'Smoking' , 'Attire' , 'Good_For_Groups']

X = dummy_df(X,list)
# Removing NaN values 
from sklearn.preprocessing import Imputer
imp = Imputer(missing_values= 'NaN', strategy = 'median' , axis = 0)
imp.fit(X)
X = pd.DataFrame(imp.transform(X), columns=X.columns)

# normalization of values
from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler()
#X = min_max_scaler.fit_transform(X)
X = pd.DataFrame(min_max_scaler.fit_transform(X), columns=X.columns)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.7, random_state=1)
print (df.shape)
print (X.shape)
print(X_train.shape)
print(X_test.shape)

#Feature selection using Chi square 
import sklearn.feature_selection
from sklearn.feature_selection import chi2
select = sklearn.feature_selection.SelectKBest(score_func=chi2,k=60)
#select = sklearn.feature_selection.SelectKBest(k=60)
selected_features = select.fit(X_train, Y_train)
indices_selected = selected_features.get_support(indices=True)
colnames_selected = [X.columns[i] for i in indices_selected]

X_train_selected = X_train[colnames_selected]
X_test_selected = X_test[colnames_selected]

# check which columns were selected 
colnames_selected

# Function to calculate AUC using roc_auc_score
def model_score(x_train, y_train, x_test, y_test):
    model = LogisticRegression(penalty='l2', C=100)
    model.fit(x_train, Y_train)
    y_hat = [x[1] for x in model.predict_proba(x_test)]
    auc = roc_auc_score(y_test, y_hat)
    return auc

auc_processed = model_score(X_train_selected, Y_train, X_test_selected, Y_test)
print auc_processed

#Caluculate Model score by applying logistic regression
model = LogisticRegression(penalty='l2', C=10)
model.fit(X_train_selected, Y_train)
model.score(X_train_selected, Y_train, sample_weight=None)

#Caluculate Model score by applying logistic regression
model = LogisticRegression(penalty='l2', C=10)
model.fit(X_train_selected, Y_train)
model.score(X_train_selected, Y_train, sample_weight=None)

#Calculate 10 fold cross validation score
scores = cross_val_score(LogisticRegression(), X, Y, scoring='accuracy', cv=10)
print scores
print scores.mean()

import statsmodels.api as sm

logit = sm.Logit(Y, X)
result = logit.fit()
print result.summary()
                              
