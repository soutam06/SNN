% SNN and ANN Portfolio Optimization - Updated Version 2.1 (June 2025)

clearvars; clc; close all;

%% 1. Load Data
fprintf('Loading and preparing data...\n');
load('portfolio_data.mat');
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
    mean_ret = mean(returns, 1)';
    cov_mat = cov(returns);
end
if size(mean_ret,1) ~= n_stocks || size(cov_mat,1) ~= n_stocks || size(cov_mat,2) ~= n_stocks
    error('Input data dimensions do not match after cleaning.');
end

%% 3. Load Sector Information
try
    load('sector_data.mat');
    fprintf('Sector data loaded: %d sectors found\n', max(sector_info));
    use_sector_constraints = true;
catch
    fprintf('No sector data found. Proceeding without sector constraints.\n');
    use_sector_constraints = false;
    sector_info = [];
end

%% 4. (Optional) Pre-clustering for Dimensionality Reduction
use_clustering = false;
if use_clustering
    fprintf('Applying hierarchical clustering for dimensionality reduction...\n');
    corr_matrix = corr(returns);
    dist_matrix = 1 - abs(corr_matrix);
    Z = linkage(squareform(dist_matrix), 'ward');
    max_clusters = min(30, floor(n_stocks/5));
    ch_values = zeros(max_clusters-1, 1);
    for k = 2:max_clusters
        clusters = cluster(Z, 'maxclust', k);
        ch_values(k-1) = calinski_harabasz(returns, clusters);
    end
    [~, idx] = max(ch_values);
    optimal_clusters = idx + 1;
    clusters = cluster(Z, 'maxclust', optimal_clusters);
    fprintf('Optimal number of clusters: %d\n', optimal_clusters);
    representative_assets = zeros(optimal_clusters, 1);
    for i = 1:optimal_clusters
        cluster_assets = find(clusters == i);
        asset_sharpes = mean_ret(cluster_assets) ./ sqrt(diag(cov_mat(cluster_assets, cluster_assets)));
        [~, max_idx] = max(asset_sharpes);
        representative_assets(i) = cluster_assets(max_idx);
    end
    returns = returns(:, representative_assets);
    mean_ret = mean_ret(representative_assets);
    cov_mat = cov_mat(representative_assets, representative_assets);
    if use_sector_constraints
        sector_info = sector_info(representative_assets);
    end
    fprintf('Data reduced to %d representative assets\n', length(representative_assets));
    n_stocks = length(representative_assets);
    asset_mapping = representative_assets;
else
    asset_mapping = 1:n_stocks;
end

%% 5. Set SNN Parameters
fprintf('Configuring SNN optimization parameters...\n');
params = struct(...
    'n_epochs', 200, ...
    'pop_size', 100, ...
    'tau', 0.8, ...
    'threshold', 1.0, ...
    'threshold_decay', 0.98, ...
    'min_threshold', 0.2, ...
    'cardinality', [30, 50], ...
    'risk_aversion', 0.85, ...
    'learning_rate', 0.15, ...
    'noise_factor', 0.05, ...
    'init_method', 'xavier', ...
    'decoding_method', 'volatility', ...
    'adaptive_risk_aversion', true, ...
    'lateral_inhibition', true, ...
    'adaptive_tau', true, ...
    'transaction_cost', 0.0025 ...
);
if use_sector_constraints
    params.sector_info = sector_info;
    params.sector_weight = 0.2;
    n_sectors = max(sector_info);
    params.sector_limits = ones(1, n_sectors) / n_sectors * 1.5;
end

%% 6. Run SNN Portfolio Optimization
fprintf('Running SNN portfolio optimization...\n');
tic;
[optimal_weights_snn, selected_idx_snn, convergence_data_snn] = snn_portfolio_solver(mean_ret, cov_mat, params);
snn_time = toc;
fprintf('SNN optimization completed in %.2f seconds\n', snn_time);

%% 7. Run ANN Portfolio Optimization (Standard ANN)
fprintf('Running ANN portfolio optimization (standard method)...\n');
ann_params = struct();
ann_params.n_epochs = 100;
ann_params.hidden_size = 16;
ann_params.learning_rate = 0.01;
ann_params.cardinality = [30, 50];
tic;
[optimal_weights_ann, selected_idx_ann, ann_info] = ann_portfolio_solver(mean_ret, cov_mat, ann_params);
ann_time = toc;
fprintf('ANN optimization completed in %.2f seconds\n', ann_time);

%% 8. Map results back to original assets if clustering was used
if use_clustering
    full_weights = zeros(length(asset_mapping), 1);
    full_weights(asset_mapping(selected_idx)) = optimal_weights(selected_idx);
    optimal_weights = full_weights;
    selected_idx = asset_mapping(selected_idx);
end
%% 8. Portfolio Risk and Return Calculations
portfolio_return_snn = mean_ret' * optimal_weights_snn;
portfolio_risk_snn = sqrt(optimal_weights_snn' * cov_mat * optimal_weights_snn);
portfolio_return_ann = mean_ret' * optimal_weights_ann;
portfolio_risk_ann = sqrt(optimal_weights_ann' * cov_mat * optimal_weights_ann);

%% Create Publication-Ready Plots (No Duplicate Plots)
create_portfolio_plots(optimal_weights_snn, optimal_weights_ann, convergence_data_snn, ann_info, mean_ret, returns);

%% Save Results
save('snn_vs_ann_portfolio_results.mat', ...
    'optimal_weights_snn', 'selected_idx_snn', 'convergence_data_snn', ...
    'optimal_weights_ann', 'selected_idx_ann', 'ann_info', ...
    'params', 'ann_params', ...
    'portfolio_return_snn', 'portfolio_return_ann', ...
    'portfolio_risk_snn', 'portfolio_risk_ann');
