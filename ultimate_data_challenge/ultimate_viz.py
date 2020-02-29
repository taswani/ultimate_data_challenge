import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ultimate_data_wrangling import data_cleaning
import json

retention_df = data_cleaning()

# Visualizing data:
plt.hist(retention_df['active'], bins=[0, .5, 1])
plt.title('Histogram of total retained riders')
plt.show()

plt.hist(retention_df['six_month_active'], bins=[0, .5, 1])
plt.title('Histogram of total retained riders after six months')
plt.show()

plt.scatter(retention_df['trips_in_first_30_days'], retention_df['six_month_active'])
plt.title('Scatterplot of retained riders vs active in six months')
plt.show()

# Conclusion: Roughly 3/4 of the riders were considered retained, but at six month mark only 1/3 of them were considered active/retained
# There also didn't seems to be any correlation between the amount of trips taken in the first 30 days versus how likely they were to remain active after 6 months
