import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Read the data from the CSV file
tsla_data = pd.read_csv("C:/Users/yarin/OneDrive/שולחן העבודה/TSLAanalysis/tsla data.csv")

# Convert the 'Date' column to datetime format with dayfirst=True
tsla_data['Date'] = pd.to_datetime(tsla_data['Date'], dayfirst=True)

# Create a new column for the year
tsla_data['Year'] = tsla_data['Date'].dt.year

# Group by year and calculate the average closing price and trading volume
yearly_avg_close = tsla_data.groupby('Year')['Close'].mean()
yearly_avg_volume = tsla_data.groupby('Year')['Volume'].mean()

# Calculate the percentage change from the previous year for both closing price and trading volume
yearly_close_pct_change = yearly_avg_close.pct_change() * 100
yearly_volume_pct_change = yearly_avg_volume.pct_change() * 100

# Plot the yearly average closing prices with percentage change annotations
plt.figure(figsize=(14, 7))
ax = yearly_avg_close.plot(kind='bar', color='orange', alpha=0.75)
plt.title('Tesla Yearly Average Closing Price with Percentage Change', fontsize=16)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Average Closing Price (in USD)', fontsize=14)
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)
for i, bar in enumerate(ax.patches):
    if not pd.isnull(yearly_close_pct_change.iloc[i]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{yearly_close_pct_change.iloc[i]:.2f}%',
                ha='center', va='bottom', fontsize=10, color='black', fontweight='bold')
plt.tight_layout()
plt.show()

# Convert the trading volume to millions
yearly_avg_volume = yearly_avg_volume / 1e6

# Plot the yearly average trading volume with percentage change annotations
plt.figure(figsize=(14, 7))
ax = yearly_avg_volume.plot(kind='bar', color='purple', alpha=0.75)
plt.title('Tesla Yearly Average Trading Volume with Percentage Change', fontsize=16)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Average Trading Volume (in millions)', fontsize=14)
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)
for i, bar in enumerate(ax.patches):
    if not pd.isnull(yearly_volume_pct_change.iloc[i]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{yearly_volume_pct_change.iloc[i]:.2f}%',
                ha='center', va='bottom', fontsize=10, color='black', fontweight='bold')
plt.tight_layout()
plt.show()

# Create a DataFrame with the yearly average closing price and trading volume
yearly_data = pd.DataFrame({
    'Average Closing Price': yearly_avg_close,
    'Average Trading Volume': yearly_avg_volume
})

# Create the scatter plot with regression line and year annotations
plt.figure(figsize=(14, 7))
sns.regplot(x='Average Closing Price', y='Average Trading Volume', data=yearly_data, scatter_kws={'s':100}, ci=None)
plt.title('Yearly Average Closing Price vs. Yearly Average Trading Volume', fontsize=16)
plt.xlabel('Average Closing Price (in USD)', fontsize=14)
plt.ylabel('Average Trading Volume (in millions)', fontsize=14)
plt.grid(True, alpha=0.3)
for year, (x, y) in zip(yearly_data.index, yearly_data.values):
    plt.text(x, y, str(year), fontsize=10)
plt.tight_layout()
plt.show()

# Calculate the daily returns as the percentage change in the closing price
tsla_data['Daily Return'] = tsla_data['Close'].pct_change() * 100

# Drop the first row with NaN daily return
tsla_data = tsla_data.dropna()

# Plot the distribution of daily returns
plt.figure(figsize=(14, 7))
sns.histplot(tsla_data['Daily Return'], kde=True, color='blue', bins=50)
plt.title('Distribution of Tesla Daily Returns', fontsize=16)
plt.xlabel('Daily Return (%)', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Calculate the total value of shares traded for each day
tsla_data['Value Traded'] = tsla_data['Close'] * tsla_data['Volume']

# Calculate the cumulative value traded and cumulative volume
tsla_data['Cumulative Value Traded'] = tsla_data['Value Traded'].cumsum()
tsla_data['Cumulative Volume'] = tsla_data['Volume'].cumsum()

# Calculate the VWAP
tsla_data['VWAP'] = tsla_data['Cumulative Value Traded'] / tsla_data['Cumulative Volume']

# Plot the closing price and VWAP
plt.figure(figsize=(14, 7))
plt.plot(tsla_data['Date'], tsla_data['Close'], label='Closing Price', color='blue')
plt.plot(tsla_data['Date'], tsla_data['VWAP'], label='VWAP', color='orange', linestyle='--')
plt.title('Tesla Closing Price and VWAP', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Price (in USD)', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Calculate the daily price change
tsla_data['Price Change'] = tsla_data['Close'].diff()

# Calculate the gain and loss for each day
tsla_data['Gain'] = tsla_data['Price Change'].where(tsla_data['Price Change'] > 0, 0)
tsla_data['Loss'] = -tsla_data['Price Change'].where(tsla_data['Price Change'] < 0, 0)

# Calculate the 14-day average gain and average loss
tsla_data['Avg Gain'] = tsla_data['Gain'].rolling(window=14).mean()
tsla_data['Avg Loss'] = tsla_data['Loss'].rolling(window=14).mean()

# Calculate the relative strength (RS)
tsla_data['RS'] = tsla_data['Avg Gain'] / tsla_data['Avg Loss']

# Calculate the RSI
tsla_data['RSI'] = 100 - (100 / (1 + tsla_data['RS']))

# Plot the closing price and RSI
fig, ax1 = plt.subplots(figsize=(14, 7))
ax2 = ax1.twinx()

ax1.plot(tsla_data['Date'], tsla_data['Close'], color='blue', label='Closing Price')
ax1.set_xlabel('Date', fontsize=14)
ax1.set_ylabel('Closing Price (in USD)', fontsize=14)
ax1.legend(loc='upper left')

ax2.plot(tsla_data['Date'], tsla_data['RSI'], color='orange', label='RSI')
ax2.axhline(70, color='red', linestyle='--', label='Overbought')
ax2.axhline(30, color='green', linestyle='--', label='Oversold')
ax2.set_ylabel('RSI', fontsize=14)
ax2.legend(loc='upper right')

plt.title('Tesla Closing Price and RSI', fontsize=16)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
