%% SNN_PortfolioOptimization.m
% Spiking Neural Network Portfolio Optimizer with Data Cleaning

clearvars; clc;

%% Load Data
load('portfolio_data.mat'); % returns (days x stocks), mean_ret (stocks x 1), cov_mat (stocks x stocks)
[n_days, n_stocks] = size(returns);
fprintf('Data loaded: %d days, %d stocks\n', n_days, n_stocks);

%% Clean Returns Data (NaN/Inf handling)
nan_count = sum(isnan(returns), 'all');
inf_count = sum(isinf(returns), 'all');
if nan_count > 0 || inf_count > 0
    fprintf('Cleaning returns: %d NaN, %d Inf values detected...\n', nan_count, inf_count);
    % Option 1: Remove rows with any NaN/Inf
    bad_rows = any(isnan(returns),2) | any(isinf(returns),2);
    returns = returns(~bad_rows, :);
    fprintf('Removed %d rows with NaN/Inf.\n', sum(bad_rows));
    n_days = size(returns,1);
    % Option 2: (Alternative) Impute with column mean:
    % for j = 1:n_stocks
    %     col = returns(:,j);
    %     col(~isfinite(col)) = nanmean(col(isfinite(col)));
    %     returns(:,j) = col;
    % end
end

%% Recompute mean and covariance after cleaning
mean_ret = mean(returns, 1)';
cov_mat = cov(returns);

% Validate dimensions
if size(mean_ret,1) ~= n_stocks || size(cov_mat,1) ~= n_stocks || size(cov_mat,2) ~= n_stocks
    error('Input data dimensions do not match after cleaning.');
end

%% SNN Parameters
params = struct(...
    'n_epochs', 100, ...          % Training iterations
    'pop_size', 50, ...           % Number of spiking neurons
    'tau', 0.8, ...               % Decay rate for spike potentials
    'threshold', 1.0, ...         % Firing threshold
    'cardinality', [30,50], ...   % Target stock selection range
    'risk_aversion', 0.94 ...     % Risk-reward balance [0-1]
);

%% SNN Portfolio Optimization
[optimal_weights, selected_idx] = snn_portfolio_solver(mean_ret, cov_mat, params);

%% Display Results
% Stock symbols (replace with actual symbols if available)
stock_symbols = arrayfun(@(x) sprintf('STOCK%03d',x), 1:n_stocks, 'UniformOutput', false)';

fprintf('\n=== Optimal Portfolio ===\n');
fprintf('Selected Stocks: %d\n', sum(optimal_weights > 0));
fprintf('Expected Return: %.2f%%\n', mean_ret' * optimal_weights * 100);
fprintf('Portfolio Risk: %.2f%%\n', sqrt(optimal_weights' * cov_mat * optimal_weights) * 100);

% Display top 30 stocks by weight
[~, sort_idx] = sort(optimal_weights, 'descend');
top_stocks = stock_symbols(sort_idx(1:30));
top_weights = optimal_weights(sort_idx(1:30)) * 100;

fprintf('\nTop 30 Stocks:\n');
disp(table(top_stocks, top_weights, ...
    'VariableNames', {'Symbol','Weight_percent'}));

%% Visualization
figure;
subplot(1,2,1);
bar(optimal_weights(optimal_weights > 0));
title('Portfolio Weights Distribution');
xlabel('Selected Stocks'); ylabel('Weight');

subplot(1,2,2);
if all(isfinite(top_weights(1:10)))
    pie(top_weights(1:10), top_stocks(1:10));
    title('Top 10 Holdings');
else
    title('Top 10 Holdings: Data not finite');
end

%% Portfolio Concentration Index
herfindahl = sum(optimal_weights.^2);
fprintf('Portfolio Concentration Index (Herfindahl): %.3f\n', herfindahl);
