% SNN Portfolio Optimization - Updated Version 2.0 (June 2025)
clearvars; clc; close all;

%% 1. Load Data
fprintf('Loading and preparing data...\n');
load('portfolio_data.mat'); % Expects 'returns', 'mean_ret', 'cov_mat'
[n_days, n_stocks] = size(returns);
fprintf('Data loaded: %d days, %d stocks\n', n_days, n_stocks);

%% 2. Clean Data (Remove rows with NaN or Inf)
nan_count = sum(isnan(returns), 'all');
inf_count = sum(isinf(returns), 'all');
if nan_count > 0 || inf_count > 0
    fprintf('Cleaning returns: %d NaN, %d Inf values detected...\n', nan_count, inf_count);
    bad_rows = any(isnan(returns),2) | any(isinf(returns),2);
    returns = returns(~bad_rows, :);
    fprintf('Removed %d rows with NaN/Inf.\n', sum(bad_rows));
    n_days = size(returns,1);
    mean_ret = mean(returns, 1)';      % Recalculate mean returns
    cov_mat = cov(returns);            % Recalculate covariance matrix
end

% Check dimensions after cleaning
if size(mean_ret,1) ~= n_stocks || size(cov_mat,1) ~= n_stocks || size(cov_mat,2) ~= n_stocks
    error('Input data dimensions do not match after cleaning.');
end

%% 3. (Optional) Load Sector Information
try
    load('sector_data.mat'); % Expects 'sector_info'
    fprintf('Sector data loaded: %d sectors found\n', max(sector_info));
    use_sector_constraints = true;
catch
    fprintf('No sector data found. Proceeding without sector constraints.\n');
    use_sector_constraints = false;
    sector_info = [];
end

%% 4. (Optional) Pre-clustering for Dimensionality Reduction
use_clustering = false; % Set to true to enable clustering

if use_clustering
    fprintf('Applying hierarchical clustering for dimensionality reduction...\n');
    % Compute correlation and distance matrix for clustering
    corr_matrix = corr(returns);
    dist_matrix = 1 - abs(corr_matrix);
    % Perform hierarchical clustering (Ward linkage)
    Z = linkage(squareform(dist_matrix), 'ward');
    max_clusters = min(30, floor(n_stocks/5));
    ch_values = zeros(max_clusters-1, 1);
    % Evaluate Calinski-Harabasz index for cluster validity
    for k = 2:max_clusters
        clusters = cluster(Z, 'maxclust', k);
        ch_values(k-1) = calinski_harabasz(returns, clusters);
    end
    [~, idx] = max(ch_values);
    optimal_clusters = idx + 1;
    clusters = cluster(Z, 'maxclust', optimal_clusters);
    fprintf('Optimal number of clusters: %d\n', optimal_clusters);
    representative_assets = zeros(optimal_clusters, 1);
    % Select the best (highest Sharpe) asset from each cluster
    for i = 1:optimal_clusters
        cluster_assets = find(clusters == i);
        asset_sharpes = mean_ret(cluster_assets) ./ sqrt(diag(cov_mat(cluster_assets, cluster_assets)));
        [~, max_idx] = max(asset_sharpes);
        representative_assets(i) = cluster_assets(max_idx);
    end
    % Reduce returns, mean_ret, cov_mat to representatives only
    returns = returns(:, representative_assets);
    mean_ret = mean_ret(representative_assets);
    cov_mat = cov_mat(representative_assets, representative_assets);
    if use_sector_constraints
        sector_info = sector_info(representative_assets);
    end
    fprintf('Data reduced to %d representative assets\n', length(representative_assets));
    n_stocks = length(representative_assets);
    asset_mapping = representative_assets; % Mapping to original assets
else
    asset_mapping = 1:n_stocks;
end

%% 5. Set SNN Parameters
fprintf('Configuring SNN optimization parameters...\n');
params = struct(...
    'n_epochs', 200, ...                % Number of optimization epochs
    'pop_size', 100, ...                % Number of neuron populations
    'tau', 0.8, ...                     % Decay factor
    'threshold', 1.0, ...               % Initial firing threshold
    'threshold_decay', 0.98, ...        % Threshold decay per epoch
    'min_threshold', 0.2, ...           % Minimum threshold
    'cardinality', [30, 50], ...        % Min and max number of assets
    'risk_aversion', 0.85, ...          % Risk aversion parameter
    'learning_rate', 0.15, ...          % Learning rate
    'noise_factor', 0.05, ...           % Noise in neuron firing
    'init_method', 'xavier', ...        % Initialization method
    'decoding_method', 'volatility', ...% Decoding method for weights
    'adaptive_risk_aversion', true, ... % Adapt risk aversion over epochs
    'lateral_inhibition', true, ...     % Enable lateral inhibition
    'adaptive_tau', true, ...           % Adapt tau over epochs
    'transaction_cost', 0.0025 ...      % Transaction cost
);
if use_sector_constraints
    params.sector_info = sector_info;
    params.sector_weight = 0.2; % Penalty for sector concentration
    n_sectors = max(sector_info);
    params.sector_limits = ones(1, n_sectors) / n_sectors * 1.5;
end

%% 6. Run SNN Portfolio Optimization
fprintf('Running SNN portfolio optimization...\n');
tic;
[optimal_weights, selected_idx, convergence_data] = snn_portfolio_solver(mean_ret, cov_mat, params);
optimization_time = toc;
fprintf('Optimization completed in %.2f seconds\n', optimization_time);

%% 7. Map results back to original assets if clustering was used
if use_clustering
    full_weights = zeros(length(asset_mapping), 1);
    full_weights(asset_mapping(selected_idx)) = optimal_weights(selected_idx);
    optimal_weights = full_weights;
    selected_idx = asset_mapping(selected_idx);
end

%% 8. Display Results
fprintf('\n=== Optimal Portfolio ===\n');
fprintf('Selected Stocks: %d\n', sum(optimal_weights > 0));
fprintf('Expected Return: %.2f%%\n', mean_ret' * optimal_weights * 100);
portfolio_risk = sqrt(optimal_weights' * cov_mat * optimal_weights) * 100;
fprintf('Portfolio Risk: %.2f%%\n', portfolio_risk);
sharpe_ratio = (mean_ret' * optimal_weights) / (sqrt(optimal_weights' * cov_mat * optimal_weights));
fprintf('Sharpe Ratio: %.4f\n', sharpe_ratio);

% Print average activated neurons per epoch
avg_activated = mean(convergence_data.active_neurons_history);
fprintf('Average Activated Neurons per Epoch: %.2f\n', avg_activated);

% Load stock symbols if available, else use generic names
try
    load('stocksymbols.mat'); % Expects 'symbols' cell array
catch
    symbols = arrayfun(@(x) sprintf('STOCK%03d', x), 1:n_stocks, 'UniformOutput', false)';
end
[sorted_weights, sort_idx] = sort(optimal_weights, 'descend');
n_display = min(30, sum(optimal_weights > 0));
top_stocks = symbols(sort_idx(1:n_display));
top_weights = optimal_weights(sort_idx(1:n_display)) * 100;
fprintf('\nTop %d Stocks:\n', n_display);
disp(table(top_stocks, top_weights, 'VariableNames', {'Symbol', 'Weight_percent'}));

%% 9. Portfolio Concentration Analysis
herfindahl = sum(optimal_weights.^2);
effective_n = 1 / herfindahl;
fprintf('Portfolio Concentration (Herfindahl): %.4f\n', herfindahl);
fprintf('Effective Number of Stocks: %.1f\n', effective_n);

if use_sector_constraints
    fprintf('\nSector Allocation:\n');
    for s = 1:max(sector_info)
        sector_mask = (sector_info == s);
        sector_weight = sum(optimal_weights(sector_mask)) * 100;
        fprintf(' Sector %d: %.2f%%\n', s, sector_weight);
    end
end

%% 10. Visualization
% Main results dashboard
figure('Name', 'SNN Portfolio Optimization Results', 'Position', [100, 100, 1200, 800]);
subplot(2,2,1);
bar(optimal_weights(optimal_weights > 0) * 100);
title('Portfolio Weights Distribution');
xlabel('Selected Stocks'); ylabel('Weight (%)'); grid on;

subplot(2,2,2);
n_pie = min(10, sum(optimal_weights > 0));
if all(isfinite(top_weights(1:n_pie)))
    pie(top_weights(1:n_pie), top_stocks(1:n_pie));
    title('Top 10 Holdings');
else
    title('Top 10 Holdings: Data not finite');
end

subplot(2,2,3);
plot(convergence_data.sharpe_history, 'b-', 'LineWidth', 1.5);
title('Sharpe Ratio Convergence');
xlabel('Epoch'); ylabel('Sharpe Ratio'); grid on;

subplot(2,2,4);
plot(convergence_data.n_selected_history, 'r-', 'LineWidth', 1.5);
title('Number of Selected Assets');
xlabel('Epoch'); ylabel('Count'); grid on;

% Neuron activation tracking
figure('Name', 'Neuron Activation Tracking', 'Position', [200, 200, 800, 400]);
plot(convergence_data.active_neurons_history, 'LineWidth', 1.5, 'Color', [0.2 0.6 0.2]);
xlabel('Epoch');
ylabel('Total Activated Neurons');
title('Neuron Activation During Optimization');
grid on;

%% 11. Save Results
save('snn_portfolio_results.mat', 'optimal_weights', 'selected_idx', ... 
    'convergence_data', 'params', 'sharpe_ratio', 'portfolio_risk');

%% 12. (Optional) Helper Function for Clustering (if used)
function ch = calinski_harabasz(data, clusters)
    % Compute Calinski-Harabasz index for cluster validity
    k = max(clusters);
    n = size(data, 1);
    overall_mean = mean(data);
    between_var = 0;
    for i = 1:k
        cluster_data = data(clusters == i, :);
        n_i = size(cluster_data, 1);
        if n_i > 0
            cluster_mean = mean(cluster_data);
            between_var = between_var + n_i * sum((cluster_mean - overall_mean).^2);
        end
    end
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
    if within_var == 0 || k == 1 || k == n
        ch = 0;
    else
        ch = (between_var / (k - 1)) / (within_var / (n - k));
    end
end
