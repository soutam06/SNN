%% SNN_PortfolioOptimization.m - Updated Version 2.0 (June 2025)
% Spiking Neural Network Portfolio Optimizer with Enhanced Features

clearvars; clc; close all;

%% Load Data
fprintf('Loading and preparing data...\n');
load('portfolio_data.mat'); % Returns, mean_ret, cov_mat

[n_days, n_stocks] = size(returns);
fprintf('Data loaded: %d days, %d stocks\n', n_days, n_stocks);

%% Clean Returns Data
nan_count = sum(isnan(returns), 'all');
inf_count = sum(isinf(returns), 'all');

if nan_count > 0 || inf_count > 0
    fprintf('Cleaning returns: %d NaN, %d Inf values detected...\n', nan_count, inf_count);
    
    % Remove rows with any NaN/Inf
    bad_rows = any(isnan(returns),2) | any(isinf(returns),2);
    returns = returns(~bad_rows, :);
    fprintf('Removed %d rows with NaN/Inf.\n', sum(bad_rows));
    n_days = size(returns,1);
    
    % Recompute mean and covariance after cleaning
    mean_ret = mean(returns, 1)';
    cov_mat = cov(returns);
end

% Validate dimensions
if size(mean_ret,1) ~= n_stocks || size(cov_mat,1) ~= n_stocks || size(cov_mat,2) ~= n_stocks
    error('Input data dimensions do not match after cleaning.');
end

%% Optional: Load sector/industry information (if available)
try
    load('sector_data.mat'); % Should contain sector_info vector
    fprintf('Sector data loaded: %d sectors found\n', max(sector_info));
    use_sector_constraints = true;
catch
    fprintf('No sector data found. Proceeding without sector constraints.\n');
    use_sector_constraints = false;
    sector_info = [];
end

%% Optional: Pre-clustering for dimensionality reduction
use_clustering = false; % Set to true to enable clustering
if use_clustering
    fprintf('Applying hierarchical clustering for dimensionality reduction...\n');
    
    % Calculate correlation matrix from returns
    corr_matrix = corr(returns);
    
    % Convert to distance matrix (1 - correlation)
    dist_matrix = 1 - abs(corr_matrix);
    
    % Perform hierarchical clustering
    Z = linkage(squareform(dist_matrix), 'ward');
    
    % Determine optimal number of clusters using Calinski-Harabasz criterion
    max_clusters = min(30, floor(n_stocks/5));
    ch_values = zeros(max_clusters-1, 1);
    
    for k = 2:max_clusters
        clusters = cluster(Z, 'maxclust', k);
        ch_values(k-1) = calinski_harabasz(returns, clusters);
    end
    
    [~, idx] = max(ch_values);
    optimal_clusters = idx + 1;
    
    % Cluster the assets
    clusters = cluster(Z, 'maxclust', optimal_clusters);
    fprintf('Optimal number of clusters: %d\n', optimal_clusters);
    
    % Select representative assets from each cluster
    representative_assets = zeros(optimal_clusters, 1);
    for i = 1:optimal_clusters
        cluster_assets = find(clusters == i);
        
        % Select asset with highest Sharpe ratio in cluster
        asset_sharpes = mean_ret(cluster_assets) ./ sqrt(diag(cov_mat(cluster_assets, cluster_assets)));
        [~, max_idx] = max(asset_sharpes);
        representative_assets(i) = cluster_assets(max_idx);
    end
    
    % Subset data to only include representative assets
    returns = returns(:, representative_assets);
    mean_ret = mean_ret(representative_assets);
    cov_mat = cov_mat(representative_assets, representative_assets);
    
    if use_sector_constraints
        sector_info = sector_info(representative_assets);
    end
    
    fprintf('Data reduced to %d representative assets\n', length(representative_assets));
    n_stocks = length(representative_assets);
    asset_mapping = representative_assets; % Keep track of original indices
else
    asset_mapping = 1:n_stocks; % No mapping needed
end

%% Set SNN Parameters
fprintf('Configuring SNN optimization parameters...\n');

params = struct(...
    'n_epochs', 200, ... % Increased iterations for better convergence
    'pop_size', 100, ... % Increased population size
    'tau', 0.8, ... % Decay rate for spike potentials
    'threshold', 1.0, ... % Initial firing threshold
    'threshold_decay', 0.98, ... % Threshold decay rate
    'min_threshold', 0.2, ... % Minimum threshold
    'cardinality', [30, 50], ... % Target stock selection range [min, max]
    'risk_aversion', 0.85, ... % Risk-reward balance [0-1]
    'learning_rate', 0.15, ... % Learning rate for neuron updates
    'noise_factor', 0.05, ... % Noise factor for exploration
    'init_method', 'xavier', ... % Neuron initialization method
    'decoding_method', 'volatility', ... % Weight decoding method
    'adaptive_risk_aversion', true, ... % Enable adaptive risk aversion
    'lateral_inhibition', true, ... % Enable lateral inhibition
    'adaptive_tau', true, ... % Enable adaptive tau
    'transaction_cost', 0.0025 ... % Transaction cost rate
);

% Add sector constraints if available
if use_sector_constraints
    params.sector_info = sector_info;
    params.sector_weight = 0.2; % Weight for sector diversification
    
    % Define sector limits (equal allocation by default)
    n_sectors = max(sector_info);
    params.sector_limits = ones(1, n_sectors) / n_sectors * 1.5; % 50% more than equal allocation
end

%% Run SNN Portfolio Optimization
fprintf('Running SNN portfolio optimization...\n');
tic;
[optimal_weights, selected_idx, convergence_data] = snn_portfolio_solver(mean_ret, cov_mat, params);
optimization_time = toc;
fprintf('Optimization completed in %.2f seconds\n', optimization_time);

%% Map results back to original assets if clustering was used
if use_clustering
    full_weights = zeros(length(asset_mapping), 1);
    full_weights(asset_mapping(selected_idx)) = optimal_weights(selected_idx);
    optimal_weights = full_weights;
    selected_idx = asset_mapping(selected_idx);
end

%% Display Results
fprintf('\n=== Optimal Portfolio ===\n');
fprintf('Selected Stocks: %d\n', sum(optimal_weights > 0));
fprintf('Expected Return: %.2f%%\n', mean_ret' * optimal_weights * 100);
portfolio_risk = sqrt(optimal_weights' * cov_mat * optimal_weights) * 100;
fprintf('Portfolio Risk: %.2f%%\n', portfolio_risk);
sharpe_ratio = (mean_ret' * optimal_weights) / (sqrt(optimal_weights' * cov_mat * optimal_weights));
fprintf('Sharpe Ratio: %.4f\n', sharpe_ratio);

% Load stock symbols if available
try
    load('stock_symbols.mat'); % Should contain symbols cell array
catch
    % Generate dummy symbols if not available
    symbols = arrayfun(@(x) sprintf('STOCK%03d', x), 1:n_stocks, 'UniformOutput', false)';
end

% Display top stocks by weight
[sorted_weights, sort_idx] = sort(optimal_weights, 'descend');
n_display = min(30, sum(optimal_weights > 0));
top_stocks = symbols(sort_idx(1:n_display));
top_weights = optimal_weights(sort_idx(1:n_display)) * 100;

fprintf('\nTop %d Stocks:\n', n_display);
disp(table(top_stocks, top_weights, ...
    'VariableNames', {'Symbol', 'Weight_percent'}));

%% Portfolio Concentration Analysis
herfindahl = sum(optimal_weights.^2);
effective_n = 1 / herfindahl;
fprintf('Portfolio Concentration (Herfindahl): %.4f\n', herfindahl);
fprintf('Effective Number of Stocks: %.1f\n', effective_n);

% Sector allocation (if available)
if use_sector_constraints
    fprintf('\nSector Allocation:\n');
    for s = 1:max(sector_info)
        sector_mask = (sector_info == s);
        sector_weight = sum(optimal_weights(sector_mask)) * 100;
        fprintf('  Sector %d: %.2f%%\n', s, sector_weight);
    end
end

%% Visualization
figure('Name', 'SNN Portfolio Optimization Results', 'Position', [100, 100, 1200, 800]);

% Subplot 1: Portfolio weights distribution
subplot(2, 2, 1);
bar(optimal_weights(optimal_weights > 0) * 100);
title('Portfolio Weights Distribution');
xlabel('Selected Stocks'); 
ylabel('Weight (%)');
grid on;

% Subplot 2: Top holdings pie chart
subplot(2, 2, 2);
n_pie = min(10, sum(optimal_weights > 0));
if all(isfinite(top_weights(1:n_pie)))
    pie(top_weights(1:n_pie), top_stocks(1:n_pie));
    title('Top 10 Holdings');
else
    title('Top 10 Holdings: Data not finite');
end

% Subplot 3: Convergence plot
subplot(2, 2, 3);
plot(convergence_data.sharpe_history, 'b-', 'LineWidth', 1.5);
title('Sharpe Ratio Convergence');
xlabel('Epoch');
ylabel('Sharpe Ratio');
grid on;

% Subplot 4: Number of selected assets
subplot(2, 2, 4);
plot(convergence_data.n_selected_history, 'r-', 'LineWidth', 1.5);
title('Number of Selected Assets');
xlabel('Epoch');
ylabel('Count');
grid on;

%% Save Results
save('snn_portfolio_results.mat', 'optimal_weights', 'selected_idx', ...
    'convergence_data', 'params', 'sharpe_ratio', 'portfolio_risk');
fprintf('\nResults saved to snn_portfolio_results.mat\n');

%% Helper Functions (only used within this script)

function ch = calinski_harabasz(data, clusters)
    % Calculate Calinski-Harabasz index for cluster validation
    k = max(clusters);
    n = size(data, 1);
    
    % Calculate overall mean
    overall_mean = mean(data);
    
    % Calculate between-cluster variance
    between_var = 0;
    for i = 1:k
        cluster_data = data(clusters == i, :);
        n_i = size(cluster_data, 1);
        if n_i > 0
            cluster_mean = mean(cluster_data);
            between_var = between_var + n_i * sum((cluster_mean - overall_mean).^2);
        end
    end
    
    % Calculate within-cluster variance
    within_var = 0;
    for i = 1:k
        cluster_data = data(clusters == i, :);
        n_i = size(cluster_data, 1);
        if n_i > 0
            cluster_mean = mean(cluster_data);
            squared_diffs = sum((cluster_data - repmat(cluster_mean, n_i, 1)).^2, 2);
            within_var = within_var + sum(squared_diffs);
        end
    end
    
    % Calculate Calinski-Harabasz index
    if within_var == 0 || k == 1 || k == n
        ch = 0;
    else
        ch = (between_var / (k - 1)) / (within_var / (n - k));
    end
end