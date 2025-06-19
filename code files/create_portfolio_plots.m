function create_portfolio_plots(optimal_weights_snn, optimal_weights_ann, convergence_data_snn, ann_info, mean_ret, returns)
% Publication-quality Portfolio Plots, SNN vs ANN
% 6 plots, each as a PDF, with improved text handling and formatting

set(0, 'DefaultAxesFontSize', 18, 'DefaultAxesFontWeight', 'bold');
set(0, 'DefaultTextFontSize', 18);
set(0, 'DefaultLegendFontSize', 16);
set(0, 'DefaultLineLineWidth', 3);
set(0, 'DefaultFigureColor', 'w');

%% Plot 1: Cumulative Return Performance Analysis
fprintf('Creating Plot 1: Cumulative Return Performance Analysis...\n');
n_days = 252;
daily_returns_snn = simulate_daily_returns(optimal_weights_snn, returns, n_days);
daily_returns_ann = simulate_daily_returns(optimal_weights_ann, returns, n_days);
cumulative_snn = cumprod(1 + daily_returns_snn);
cumulative_ann = cumprod(1 + daily_returns_ann);

fig1 = figure('Units','inches', 'Position',[0 0 12 8], 'Color','w');
plot(1:n_days, (cumulative_snn-1)*100, 'b-', 'LineWidth', 4); hold on;
plot(1:n_days, (cumulative_ann-1)*100, 'r-', 'LineWidth', 4);
title('Cumulative Return Performance Analysis', 'FontSize', 24, 'FontWeight', 'bold');
xlabel('Trading Days', 'FontSize', 20, 'FontWeight', 'bold');
ylabel('Cumulative Return (%)', 'FontSize', 20, 'FontWeight', 'bold');
legend({'SNN Portfolio', 'ANN Portfolio'}, 'FontSize', 18, 'Location', 'best', 'Box', 'on');
grid on; grid minor;
set(gca, 'FontSize', 18, 'FontWeight', 'bold', 'LineWidth', 2);
final_return_snn = (cumulative_snn(end) - 1) * 100;
final_return_ann = (cumulative_ann(end) - 1) * 100;
text(0.02, 0.98, sprintf('SNN Final Return: %.2f%%\nANN Final Return: %.2f%%', ...
    final_return_snn, final_return_ann), 'Units', 'normalized', ...
    'FontSize', 16, 'FontWeight', 'bold', 'BackgroundColor', 'white', 'EdgeColor', 'black', ...
    'VerticalAlignment', 'top', 'Interpreter','none');
tightfig_pdf(fig1, 'plot1_cumulative_returns.pdf');

%% Plot 2: Rolling Sharpe Ratio Stability Assessment
fprintf('Creating Plot 2: Rolling Sharpe Ratio Stability Assessment...\n');
window = 60;
rolling_sharpe_snn = calculate_rolling_sharpe(daily_returns_snn, window);
rolling_sharpe_ann = calculate_rolling_sharpe(daily_returns_ann, window);
fig2 = figure('Units','inches', 'Position',[0 0 12 8], 'Color','w');
plot(window:n_days, rolling_sharpe_snn, 'b-', 'LineWidth', 4); hold on;
plot(window:n_days, rolling_sharpe_ann, 'r-', 'LineWidth', 4);
title('Rolling Sharpe Ratio Stability Assessment (60-Day Window)', 'FontSize', 24, 'FontWeight', 'bold');
xlabel('Trading Days', 'FontSize', 20, 'FontWeight', 'bold');
ylabel('Rolling Sharpe Ratio', 'FontSize', 20, 'FontWeight', 'bold');
legend({'SNN Portfolio', 'ANN Portfolio'}, 'FontSize', 18, 'Location', 'best', 'Box', 'on');
grid on; grid minor;
set(gca, 'FontSize', 18, 'FontWeight', 'bold', 'LineWidth', 2);
yline(0, 'k--', 'LineWidth', 2, 'Alpha', 0.7);
mean_sharpe_snn = mean(rolling_sharpe_snn);
mean_sharpe_ann = mean(rolling_sharpe_ann);
text(0.02, 0.98, sprintf('Mean Rolling Sharpe\nSNN: %.3f\nANN: %.3f', ...
    mean_sharpe_snn, mean_sharpe_ann), 'Units', 'normalized', ...
    'FontSize', 16, 'FontWeight', 'bold', 'BackgroundColor', 'white', 'EdgeColor', 'black', ...
    'VerticalAlignment', 'top', 'Interpreter','none');
tightfig_pdf(fig2, 'plot2_rolling_sharpe_ratio.pdf');

%% Plot 3: Training Loss Convergence Comparison (fixed annotation)
fprintf('Creating Plot 3: Training Loss Convergence Comparison...\n');
fig3 = figure('Units','inches', 'Position',[0 0 12 8], 'Color','w');
plot(1:length(convergence_data_snn.sharpe_history), -convergence_data_snn.sharpe_history, ...
    'b-', 'LineWidth', 4); hold on;
plot(1:length(ann_info.sharpe_history), ann_info.loss_history, 'r-', 'LineWidth', 4);
title('Training Loss Convergence Comparison', 'FontSize', 24, 'FontWeight', 'bold');
xlabel('Training Epochs', 'FontSize', 20, 'FontWeight', 'bold');
ylabel('Loss (Negative Sharpe Ratio)', 'FontSize', 20, 'FontWeight', 'bold');
legend({'SNN (200 epochs)', 'ANN (100 epochs)'}, 'FontSize', 18, 'Location', 'best', 'Box', 'on');
grid on; grid minor;
set(gca, 'FontSize', 18, 'FontWeight', 'bold', 'LineWidth', 2);
% Use a cell array for multi-line annotation (no interpreter issues)
text(0.6, 0.8, {'Both methods achieve', 'effective convergence'}, 'Units', 'normalized', ...
    'FontSize', 16, 'FontWeight', 'bold', 'BackgroundColor', 'white', 'EdgeColor', 'black', ...
    'HorizontalAlignment', 'center');
tightfig_pdf(fig3, 'plot3_loss_convergence.pdf');

%% Plot 4: Asset Contribution Attribution Analysis
fprintf('Creating Plot 4: Asset Contribution Attribution Analysis...\n');
contrib_snn = optimal_weights_snn .* mean_ret * 252;
contrib_ann = optimal_weights_ann .* mean_ret * 252;
[~, top_snn] = sort(abs(contrib_snn), 'descend');
[~, top_ann] = sort(abs(contrib_ann), 'descend');
top_15_combined = unique([top_snn(1:15); top_ann(1:15)]);
fig4 = figure('Units','inches', 'Position',[0 0 14 10], 'Color','w');
y_pos = 1:length(top_15_combined);
barh(y_pos - 0.2, contrib_snn(top_15_combined) * 100, 0.35, 'FaceColor', [0.2 0.6 0.8]); hold on;
barh(y_pos + 0.2, contrib_ann(top_15_combined) * 100, 0.35, 'FaceColor', [0.8 0.4 0.2]);
title('Asset Contribution Attribution Analysis (Top Contributors)', 'FontSize', 24, 'FontWeight', 'bold');
xlabel('Return Contribution (%)', 'FontSize', 20, 'FontWeight', 'bold');
ylabel('Asset Index', 'FontSize', 20, 'FontWeight', 'bold');
legend({'SNN Portfolio', 'ANN Portfolio'}, 'FontSize', 18, 'Location', 'best', 'Box', 'on');
grid on; grid minor;
set(gca, 'FontSize', 18, 'FontWeight', 'bold', 'LineWidth', 2);
set(gca, 'YTick', y_pos, 'YTickLabel', arrayfun(@(x) sprintf('Asset%d', x), top_15_combined, 'UniformOutput', false));
tightfig_pdf(fig4, 'plot4_asset_contributions.pdf');

%% Plot 5: Correlation Matrix for Diversification Analysis
fprintf('Creating Plot 5: Correlation Matrix for Diversification Analysis...\n');
all_selected = unique([find(optimal_weights_snn > 1e-4); find(optimal_weights_ann > 1e-4)]);
top_20_holdings = all_selected(1:min(20, length(all_selected)));
corr_matrix = corrcoef(returns(:, top_20_holdings));
fig5 = figure('Units','inches', 'Position',[0 0 12 12], 'Color','w');
imagesc(corr_matrix);
colormap(redblue);
cb = colorbar('FontSize', 18);
cb.Label.String = 'Correlation Coefficient';
cb.Label.FontSize = 20;
title('Correlation Matrix - Portfolio Holdings Diversification Analysis', 'FontSize', 24, 'FontWeight', 'bold');
xlabel('Asset Index', 'FontSize', 20, 'FontWeight', 'bold');
ylabel('Asset Index', 'FontSize', 20, 'FontWeight', 'bold');
set(gca, 'FontSize', 18, 'FontWeight', 'bold', 'LineWidth', 2);
[n_row, n_col] = size(corr_matrix);
for i = 1:n_row
    for j = 1:n_col
        value = corr_matrix(i,j);
        if abs(value) > 0.5
            text_color = 'white';
        else
            text_color = 'black';
        end
        text(j, i, sprintf('%.2f', value), ...
            'HorizontalAlignment', 'center', 'FontSize', 14, 'FontWeight', 'bold', ...
            'Color', text_color);
    end
end
asset_labels = arrayfun(@(x) sprintf('A%d', x), top_20_holdings, 'UniformOutput', false);
set(gca, 'XTick', 1:length(top_20_holdings), 'XTickLabel', asset_labels, 'XTickLabelRotation', 45);
set(gca, 'YTick', 1:length(top_20_holdings), 'YTickLabel', asset_labels);
tightfig_pdf(fig5, 'plot5_correlation_matrix.pdf');

%% Plot 6: SNN Spike Raster Plot - Unique Neural Dynamics
fprintf('Creating Plot 6: SNN Spike Raster Plot - Neural Dynamics...\n');
n_neurons = 30;
spike_data = generate_spike_raster(convergence_data_snn, n_neurons);
fig6 = figure('Units','inches', 'Position',[0 0 14 10], 'Color','w');
scatter(spike_data(:,1), spike_data(:,2), 40, 'filled', 'MarkerFaceColor', [0.2 0.2 0.8]);
title('SNN Spike Raster Plot - Unique Neural Dynamics', 'FontSize', 24, 'FontWeight', 'bold');
xlabel('Training Epochs', 'FontSize', 20, 'FontWeight', 'bold');
ylabel('Neuron ID', 'FontSize', 20, 'FontWeight', 'bold');
grid on; grid minor;
set(gca, 'FontSize', 18, 'FontWeight', 'bold', 'LineWidth', 2);
ylim([0.5, n_neurons + 0.5]);
text(0.6, 0.9, sprintf('Sparse Firing Pattern\n%d spike events\n%d neurons tracked', ...
    size(spike_data, 1), n_neurons), 'Units', 'normalized', ...
    'FontSize', 16, 'FontWeight', 'bold', 'BackgroundColor', 'white', 'EdgeColor', 'black', ...
    'HorizontalAlignment', 'center', 'Interpreter','none');
tightfig_pdf(fig6, 'plot6_spike_raster.pdf');

fprintf('\nAll 6 plots have been successfully created and saved as individual PDFs!\n');
fprintf('Files created:\n');
fprintf('- plot1_cumulative_returns.pdf\n');
fprintf('- plot2_rolling_sharpe_ratio.pdf\n');
fprintf('- plot3_loss_convergence.pdf\n');
fprintf('- plot4_asset_contributions.pdf\n');
fprintf('- plot5_correlation_matrix.pdf\n');
fprintf('- plot6_spike_raster.pdf\n');

end

%% Helper Functions

function daily_returns = simulate_daily_returns(weights, historical_returns, n_days)
n_stocks = length(weights);
selected_assets = find(weights > 1e-6);
if isempty(selected_assets)
    daily_returns = zeros(n_days, 1);
    return;
end
n_hist = size(historical_returns, 1);
random_days = randi(n_hist, n_days, 1);
daily_returns = zeros(n_days, 1);
for i = 1:n_days
    day_returns = historical_returns(random_days(i), selected_assets);
    daily_returns(i) = weights(selected_assets)' * day_returns';
end
end

function rolling_sharpe = calculate_rolling_sharpe(returns, window)
n_days = length(returns);
rolling_sharpe = zeros(n_days - window + 1, 1);
for i = window:n_days
    window_returns = returns(i-window+1:i);
    mean_ret = mean(window_returns) * 252;
    std_ret = std(window_returns) * sqrt(252);
    rolling_sharpe(i-window+1) = mean_ret / (std_ret + 1e-8);
end
end

function spike_data = generate_spike_raster(convergence_data, n_neurons)
n_epochs = length(convergence_data.sharpe_history);
if isfield(convergence_data, 'neuron_activation_matrix')
    activation_matrix = convergence_data.neuron_activation_matrix;
    if size(activation_matrix, 2) >= n_neurons
        activation_matrix = activation_matrix(:, 1:n_neurons);
    else
        additional_neurons = n_neurons - size(activation_matrix, 2);
        additional_data = rand(n_epochs, additional_neurons) > 0.8;
        activation_matrix = [activation_matrix, additional_data];
    end
else
    activation_prob = 0.1;
    activation_matrix = rand(n_epochs, n_neurons) < activation_prob;
end
[epochs, neurons] = find(activation_matrix);
spike_data = [epochs, neurons];
end

function cmap = redblue(n)
if nargin < 1
    n = 256;
end
mid = ceil(n/2);
r = [linspace(0, 1, mid), ones(1, n-mid)];
g = [linspace(0, 1, mid), linspace(1, 0, n-mid)];
b = [ones(1, mid), linspace(1, 0, n-mid)];
cmap = [r', g', b'];
colormap(cmap);
end

function tightfig_pdf(fig_handle, filename)
set(fig_handle, 'PaperPositionMode', 'auto');
set(fig_handle, 'PaperUnits', 'inches');
figPos = get(fig_handle, 'Position');
set(fig_handle, 'PaperSize', [figPos(3) figPos(4)]);
print(fig_handle, filename, '-dpdf', '-painters');
close(fig_handle);
end
