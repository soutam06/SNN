import pandas as pd
import yfinance as yf # type: ignore
import numpy as np
import os

# Step 1: Download NIFTY 500 table from Wikipedia
url = 'https://en.wikipedia.org/wiki/NIFTY_500'
nifty500 = pd.read_html(url)[4]

# Step 2: Ensure columns are named as expected
expected_columns = ['Sl.No', 'Company Name', 'Industry', 'Symbol', 'Series', 'ISIN Code']
if list(nifty500.columns) != expected_columns:
    nifty500.columns = expected_columns

# Step 3: Clean and format the Symbol column
nifty500['Symbol'] = nifty500['Symbol'].str.strip() + '.NS'

# Step 4: Get the top 250 symbols (skip header row if present)
symbols_list = nifty500['Symbol'].tolist()[1:]  # Adjust if first row is not a stock

print(f"Number of symbols (raw): {len(symbols_list)}")

# Step 5: Download historical price data using yfinance
start_date = '2017-01-01'
end_date = '2025-05-31'

print("Downloading price data. This may take a few minutes...")

data = yf.download(
    tickers=symbols_list,
    start=start_date,
    end=end_date,
    interval='1d',
    group_by='ticker',
    auto_adjust=False,
    threads=True
)

# Step 6: Extract Close prices and align with available data
close_df = pd.DataFrame({
    sym: data[sym]['Close']
    for sym in symbols_list if sym in data
}).ffill()

# Remove columns (stocks) with any missing data
close_df = close_df.dropna(axis=1, how='any')

print(f"Number of symbols with full price data: {close_df.shape[1]}")

# Step 7: Save processed prices as CSV for MATLAB
# os.makedirs('data', exist_ok=True)
close_df.to_csv('processed_prices.csv', index=True, date_format='%Y-%m-%d')
print("Saved processed_prices.csv for MATLAB.")

# Step 8: Create sector classification DataFrame for only the stocks in processed_prices.csv
unique_sectors = nifty500['Industry'].unique()
sector_id_map = {sector: i+1 for i, sector in enumerate(unique_sectors)}

sector_rows = []
for sym in close_df.columns:
    row = nifty500.loc[nifty500['Symbol'] == sym]
    if row.empty:
        sector_id = 0
        sector_name = 'Unknown'
    else:
        sector_name = row['Industry'].values[0]
        sector_id = sector_id_map[sector_name]
    sector_rows.append({'Symbol': sym, 'SectorID': sector_id, 'SectorName': sector_name})

sector_df = pd.DataFrame(sector_rows, columns=['Symbol', 'SectorID', 'SectorName'])

# Step 9: Save sector classification for MATLAB
sector_df.to_csv('sector_classification.csv', index=False)
print("Saved sector_classification.csv for MATLAB.")

# Step 10: Calculate daily returns (percentage change)
# returns_df = close_df.pct_change().iloc[1:]  # Skip first row (NaN)

# Step 11: Compute annualized mean returns and covariance matrix
# mu = returns_df.mean().values * 252  # Annualized mean returns
# sigma = returns_df.cov().values * 252  # Annualized covariance matrix

# Step 12: Save returns and statistics for reference (optional)
# returns_df.to_csv('daily_returns.csv')
# np.save('mu.npy', mu)
# np.save('sigma.npy', sigma)

print(f"Data shape: {close_df.shape} (Days × Stocks)")
# print(f"Mean returns vector shape: {mu.shape}")
# print(f"Covariance matrix shape: {sigma.shape}")
print("All files saved successfully.")
