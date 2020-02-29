import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import json

def data_cleaning():
    # Reading in json file to pandas dataframe
    retention_df = pd.read_json('ultimate_data_challenge.json', convert_dates=['signup_date', 'last_trip_date'])
    # Creating a column to see which users are active
    retention_df['active'] = np.where(retention_df['trips_in_first_30_days'] >= 1, 1, 0)
    # Creating a column to see which users are active 6 months out
    retention_df['six_month_active'] = np.where(retention_df['last_trip_date'] >= np.datetime64('2014-06-01'), 1, 0)
    # Checking to see if there are any NaN values that need to be filled
    # print(retention_df.isna().any())
    # Seems like avg_rating_of_driver, phone, avg_rating_by_driver have NaN values that need to be filled
    # We can choose to forward fill the phone, and take the average for the columns to fill for the ratings
    retention_df['phone'] = retention_df['phone'].fillna(method='ffill')
    retention_df['avg_rating_of_driver'] = retention_df['avg_rating_of_driver'].fillna(retention_df['avg_rating_of_driver'].mean())
    retention_df['avg_rating_by_driver'] = retention_df['avg_rating_by_driver'].fillna(retention_df['avg_rating_by_driver'].mean())
    # print(retention_df.isna().any())

    # Further typing categorical variables to num
    retention_df['city'] = retention_df['city'].astype('category')
    retention_df['phone'] = retention_df['phone'].astype('category')
    cat_columns = retention_df.select_dtypes(['category']).columns
    retention_df[cat_columns] = retention_df[cat_columns].apply(lambda x: x.cat.codes)
    return retention_df
