% Step 1: Load prices
prices = readmatrix('data\processed_prices.csv'); % size: 2077 x 356

% Step 2: Compute daily returns
returns = diff(prices) ./ prices(1:end-1, :); % size: 2076 x 356

% Step 3: Compute mean returns (column vector)
mean_ret = mean(returns, 1)'; % size: 356 x 1

% Step 4: Compute covariance matrix
cov_mat = cov(returns); % size: 356 x 356

% Step 5: Save to MAT file
save('portfolio_data.mat', 'returns', 'mean_ret', 'cov_mat');
