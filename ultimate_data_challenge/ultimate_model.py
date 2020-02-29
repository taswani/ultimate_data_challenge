import pandas as pd
import numpy as np
from ultimate_data_wrangling import data_cleaning
import xgboost as xgb
from joblib import dump
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Random number seed to get more reproduceable results
np.random.seed(32)

# Calling data cleaning function to provide us with the dataframe
retention_df = data_cleaning()

# Scaling dates to prepare them for model
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(retention_df[['signup_date', 'last_trip_date']].values)
retention_df[['signup_date', 'last_trip_date']] = scaled

# Setting up X and y from retention_df for model consumption
X = retention_df[[
    'city', #1
    'trips_in_first_30_days', #4
    'signup_date', #8
    'avg_rating_of_driver', #6
    'avg_surge', #7
    'phone', #1
    'surge_pct', #2
    'ultimate_black_user', #1
    'weekday_pct', #3
    'avg_dist', #5
    'avg_rating_by_driver' #1
    ]]
y = retention_df['six_month_active']

# Setting up training and testing folds
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Model set up
clf = xgb.XGBClassifier()
clf.fit(X_train, y_train)
# Make predictions
y_pred = clf.predict(X_test)
predictions = [round(value) for value in y_pred]
# Evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

dump(clf, 'ultimate_data_challenge_model.joblib')

# Feature Selection for the classifier
estimator = clf
selector = RFE(estimator, 4, step=1)
selector = selector.fit(X, y)
print("Feature Ranking: ", selector.ranking_)

# Seems like the most important features are: city, phone, ultimate_black_user, and avg_rating_by_driver
# Using only those features only lowers the predictive accuracy by less than 2%
