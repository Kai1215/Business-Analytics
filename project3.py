import numpy as np
import pandas as pd

df = pd.read_pickle('./Project-3_NYC_311_Calls.pkl')
df = df.set_index(pd.DatetimeIndex(df['Created Date']))
del df['Created Date']

#Filter the DataFrame to include only the data from 2022
df_2022 = df[df.index.year == 2022]

# Use resample to count daily complaints
daily_complaints = df_2022['Unique Key'].resample('D').count()

# Calculate the average number of daily complaints
average_daily_complaints = daily_complaints.mean()
print(average_daily_complaints)
#Find the range of years in the dataset
years = df.index.year.unique()

# Dictionary to hold the date with maximum calls for each year
max_calls_per_year = {}

for year in years:
    # Filter the DataFrame for the current year
    df_year = df[df.index.year == year]
    
    # Resample to get daily counts
    daily_counts = df_year['Unique Key'].resample('D').count()
    
    # Find the date with the maximum number of calls/complaints
    max_calls_date = daily_counts.idxmax()
    max_calls = daily_counts.max()
    
    # Store the result in the dictionary
    max_calls_per_year[year] = (max_calls_date, max_calls)

max_calls_per_year
# Filter the DataFrame to include only the records from '2020-08-04'
max_calls_day_df = df[df.index.date == pd.to_datetime('2020-08-04').date()]

# Count the occurrences of each complaint type on that day
complaint_counts = max_calls_day_df['Complaint Type'].value_counts()

# Find the most common complaint type
most_common_complaint = complaint_counts.idxmax()
most_common_complaint_count = complaint_counts.iloc[0]

(most_common_complaint, most_common_complaint_count)

#Extract the month from the index
df['Month'] = df.index.month

# Group by the 'Month' column and count the number of entries for each month
monthly_counts_all_years = df.groupby('Month').size()

# Identify the month with the fewest number of calls across all years
quietest_month_all_years = monthly_counts_all_years.idxmin()
quietest_month_count_all_years = monthly_counts_all_years.min()

# Translate the numeric month to a month name
import calendar
quietest_month_name = calendar.month_name[quietest_month_all_years]

# (quietest_month_name, quietest_month_count_all_years)
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# First, resample the series to a daily frequency, summing the number of calls
daily_series = df['Unique Key'].resample('D').count()
# Perform ETS decomposition based on an additive model
# The model requires that there are no missing values, so we fill any with 0
daily_series_filled = daily_series.replace(0, np.nan).fillna(method='ffill')
decomposition = seasonal_decompose(daily_series_filled, model='additive', period=365)

# Get the seasonal component
seasonal = decomposition.seasonal

# What is the value of the seasonal component on 2020-12-25?
seasonal_value = seasonal.loc[pd.to_datetime('2020-12-25')]
seasonal_value_rounded = np.round(seasonal_value)

print(f"The value of the seasonal component on 2020-12-25 is {seasonal_value_rounded}.")


# Resample the series to a daily frequency, counting the number of calls
daily_series = df['Unique Key'].resample('D').count()

# Calculate the autocorrelation with a lag of 1
lag_1_autocorrelation = daily_series.autocorr(lag=1)

print(lag_1_autocorrelation)

from prophet import Prophet
from sklearn.metrics import mean_squared_error
# Assuming 'df' is your DataFrame and it's indexed by 'Created Date'
# Assuming the data has been resampled to a daily frequency
daily_series = df['Unique Key'].resample('D').count().fillna(0)

# Reset index to use in Prophet
df_prophet = daily_series.reset_index()
df_prophet.columns = ['ds', 'y']

# Split data into train and test sets
train = df_prophet.iloc[:-90]  # All data except last 90 days for training
test = df_prophet.iloc[-90:]   # Last 90 days for testing

# Initialize and fit the Prophet model
model = Prophet(daily_seasonality=True)  # turn on daily seasonality, since we deal with daily data
model.fit(train)

# Make a dataframe for predictions
future = model.make_future_dataframe(periods=90)

# Forecast the future
forecast = model.predict(future)

# Extract predicted values for the test set dates
y_pred = forecast['yhat'].iloc[-90:]

# Calculate the RMSE between the observed and predicted values in the test set
rmse = np.sqrt(mean_squared_error(test['y'], y_pred))

print(rmse)