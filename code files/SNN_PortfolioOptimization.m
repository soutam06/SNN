% SNN and ANN Portfolio Optimization - Updated Version 2.1 (June 2025)
% Main script to compare SNN and ANN portfolio optimization approaches

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
    mean_ret = mean(returns, 1)'; % Recalculate mean returns
    cov_mat = cov(returns);       % Recalculate covariance matrix
end
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
    asset_mapping = representative_assets; % Mapping to original assets
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
ann_params.n_epochs = 100;         % Fewer epochs for ANN
ann_params.hidden_size = 16;       % Small hidden layer
ann_params.learning_rate = 0.01;   % Conservative learning rate
ann_params.cardinality = [30, 50]; % Same as SNN
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

%% 9. Compare SNN and ANN Results
% Calculate SNN metrics for comparison
sharpe_ratio_snn = (mean_ret' * optimal_weights_snn) / (sqrt(optimal_weights_snn' * cov_mat * optimal_weights_snn) + 1e-8);
portfolio_return_snn = mean_ret' * optimal_weights_snn * 100;
portfolio_risk_snn = sqrt(optimal_weights_snn' * cov_mat * optimal_weights_snn) * 100;
effective_n_snn = 1 / sum(optimal_weights_snn.^2);

% Calculate ANN metrics
sharpe_ratio_ann = ann_info.final_sharpe;
portfolio_return_ann = ann_info.final_return * 100;
portfolio_risk_ann = ann_info.final_risk * 100;
effective_n_ann = ann_info.effective_n;

fprintf('\n=== Comparison: SNN vs ANN ===\n');
fprintf('Time taken (SNN): %.2f s | (ANN): %.2f s\n', snn_time, ann_time);
fprintf('Sharpe Ratio (SNN): %.4f | (ANN): %.4f\n', sharpe_ratio_snn, sharpe_ratio_ann);
fprintf('Expected Return (SNN): %.2f%% | (ANN): %.2f%%\n', portfolio_return_snn, portfolio_return_ann);
fprintf('Portfolio Risk (SNN): %.2f%% | (ANN): %.2f%%\n', portfolio_risk_snn, portfolio_risk_ann);
fprintf('Effective Number of Stocks (SNN): %.1f | (ANN): %.1f\n', effective_n_snn, effective_n_ann);

%% 10. Visualization: SNN vs ANN
figure('Name', 'SNN vs ANN Portfolio Weights', 'Position', [100, 100, 1200, 500]);
subplot(1,2,1);
bar(optimal_weights_snn(optimal_weights_snn > 0) * 100);
title('SNN Portfolio Weights');
xlabel('Selected Stocks'); ylabel('Weight (%)'); grid on;
subplot(1,2,2);
bar(optimal_weights_ann(optimal_weights_ann > 0) * 100, 'FaceColor', [0.8 0.4 0.2]);
title('ANN Portfolio Weights');
xlabel('Selected Stocks'); ylabel('Weight (%)'); grid on;

figure('Name', 'Sharpe Ratio Convergence', 'Position', [200, 200, 1000, 400]);
plot(convergence_data_snn.sharpe_history, 'b-', 'LineWidth', 1.5); hold on;
plot(ann_info.sharpe_history, 'r-', 'LineWidth', 1.5);
legend('SNN', 'ANN');
title('Sharpe Ratio Convergence');
xlabel('Epoch'); ylabel('Sharpe Ratio'); grid on;

figure('Name', 'Selected Assets Over Epochs', 'Position', [300, 300, 1000, 400]);
if isfield(convergence_data_snn, 'n_selected_history')
    plot(convergence_data_snn.n_selected_history, 'b-', 'LineWidth', 1.5); hold on;
else
    plot(sum(optimal_weights_snn > 1e-4) * ones(size(convergence_data_snn.sharpe_history)), 'b-', 'LineWidth', 1.5); hold on;
end
plot(sum(ann_info.weights_history > 1e-4, 2), 'r-', 'LineWidth', 1.5);
legend('SNN', 'ANN');
title('Number of Selected Assets Over Epochs');
xlabel('Epoch'); ylabel('Asset Count'); grid on;

%% 11. SNN Neuron Activation Analysis
figure('Name', 'SNN Neuron Activation Patterns', 'Position', [100, 100, 1400, 600]);

% Activation Heatmap
subplot(2,2,1);
imagesc(convergence_data_snn.neuron_activation_matrix');
xlabel('Epoch');
ylabel('Asset Index');
title('Neuron Activation Heatmap');
colorbar;
colormap(jet);

% Activation Rate Over Time
subplot(2,2,2);
plot(convergence_data_snn.activation_rate_history, 'LineWidth', 2);
xlabel('Epoch');
ylabel('Activation Rate');
title('Population-wide Activation Rate');
grid on;

% Cumulative Activations per Asset
subplot(2,2,3);
bar(convergence_data_snn.cumulative_activations, 'FaceColor', [0.4 0.6 0.8]);
xlabel('Asset Index');
ylabel('Total Activations');
title('Cumulative Activations per Asset');
grid on;

% Activation vs Portfolio Weight
subplot(2,2,4);
scatter(convergence_data_snn.cumulative_activations, optimal_weights_snn*100, ...
    'filled', 'MarkerFaceColor', [0.8 0.4 0.2]);
xlabel('Total Activations');
ylabel('Final Weight (%)');
title('Activation Count vs Final Portfolio Weight');
grid on;

fprintf('\n=== Neuron Activation Analysis ===\n');
fprintf('Total Activations: %d\n', sum(convergence_data_snn.active_neurons_history));
fprintf('Average Activations/Epoch: %.1f\n', mean(convergence_data_snn.active_neurons_history));
fprintf('Peak Activation Rate: %.2f%%\n', max(convergence_data_snn.activation_rate_history)*100);
[max_act, max_act_idx] = max(convergence_data_snn.cumulative_activations);
fprintf('Most Activated Asset Index: %d (%d activations)\n', max_act_idx, max_act);

%% 12. Save Results
save('snn_vs_ann_portfolio_results.mat', ...
    'optimal_weights_snn', 'selected_idx_snn', 'convergence_data_snn', ...
    'optimal_weights_ann', 'selected_idx_ann', 'ann_info', ...
    'params', 'ann_params', 'sharpe_ratio_snn', 'sharpe_ratio_ann', ...
    'portfolio_return_snn', 'portfolio_return_ann', ...
    'portfolio_risk_snn', 'portfolio_risk_ann', ...
    'effective_n_snn', 'effective_n_ann');

%% 13. (Optional) Helper Function for Clustering (if used)
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
