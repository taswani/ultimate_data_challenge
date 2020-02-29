import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

# Reading in json file to pandas dataframe
logins_df = pd.read_json('logins.json')

# Plotting df for visualization of dataframe
# Resampling data for the sake of 15 minute aggregations
logins_df.set_index('login_time', inplace=True)
logins_df['count'] = 1
logins_df_minutes = logins_df.resample('15T').sum()
X = logins_df_minutes.index.values
y = logins_df_minutes['count']

# Visualizing using a scatterplot
plt.scatter(X, y)
plt.title('Scatter plot of 15min')
plt.show()

# Visualizing using a bar chart
plt.bar(X, y)
plt.title('Bar chart of 15min')
plt.xticks(rotation='vertical')
plt.show()

# Resampling for day to see daily cycles
logins_df_day = logins_df.resample('D').sum()
X = logins_df_day.index.values
y = logins_df_day['count']

# Visualizing using a scatterplot
plt.scatter(X, y)
plt.title('Scatter plot of days')
plt.show()

# Visualizing using a bar chart
plt.bar(X, y)
plt.title('Bar chart of days')
plt.xticks(rotation='vertical')
plt.show()

# Conclusion when looking at the 15 min intervals is that there seems to be the highest number of logins during:
## Night time and after 5 has the largest amount of logins and the lowest amount of logins are during the day time
# Conclusion when looking at the daily intervals is that there seems to be the highest number of logins during:
## Saturdays have the largest logins, Mondays have the lowest logins, and logins increase as it gets closer to April
