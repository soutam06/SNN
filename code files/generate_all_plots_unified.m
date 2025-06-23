function [generated_plots, success] = generate_all_plots_unified(plot_data)
%% GENERATE_ALL_PLOTS_UNIFIED - Unified Plot Generation with PDF Compilation
%
% This function generates all publication-quality plots for SNN vs ANN analysis
% and compiles them into a single PDF file with comprehensive coverage.
%
% INPUTS:
%   plot_data - Comprehensive data structure with all plot information
%
% OUTPUTS:
%   generated_plots - Cell array of generated plot filenames
%   success - Boolean indicating successful PDF compilation

fprintf('Generating unified publication-quality plots...\n');

%% SETUP PUBLICATION DEFAULTS
set(0, 'DefaultAxesFontSize', 16, 'DefaultAxesFontWeight', 'bold');
set(0, 'DefaultTextFontSize', 16, 'DefaultLegendFontSize', 14);
set(0, 'DefaultLineLineWidth', 3, 'DefaultFigureColor', 'w');
set(0, 'DefaultAxesLineWidth', 1.5);

% Colors
snn_color = [0.2, 0.6, 0.8];
ann_color = [0.8, 0.4, 0.2];
efficiency_color = [0.2, 0.7, 0.3];

% Output configuration
output_dir = plot_data.config.output_dir;
analysis_name = plot_data.config.analysis_name;
individual_plots = {};

%% PLOT 1: SNN ARCHITECTURE VISUALIZATION
fprintf('  Creating Plot 1: SNN Architecture Visualization...\n');
fig1 = create_snn_architecture_plot(plot_data.snn_architecture);
plot1_file = fullfile(output_dir, '01_SNN_Architecture_Visualization.pdf');
exportgraphics(fig1, plot1_file, 'ContentType', 'vector', 'Resolution', 300);
individual_plots{end+1} = plot1_file;
close(fig1);

%% PLOT 2: ENERGY COMPARISON PER EPOCH (KEY REQUIREMENT)
fprintf('  Creating Plot 2: Energy Comparison per Epoch (SNN vs ANN)...\n');
fig2 = create_energy_per_epoch_plot(plot_data.energy_analysis, snn_color, ann_color);
plot2_file = fullfile(output_dir, '02_Energy_Comparison_Per_Epoch.pdf');
exportgraphics(fig2, plot2_file, 'ContentType', 'vector', 'Resolution', 300);
individual_plots{end+1} = plot2_file;
close(fig2);

%% PLOT 3: PORTFOLIO PERFORMANCE COMPARISON
fprintf('  Creating Plot 3: Portfolio Performance Comparison...\n');
fig3 = create_performance_comparison_plot(plot_data.performance, snn_color, ann_color);
plot3_file = fullfile(output_dir, '03_Portfolio_Performance_Comparison.pdf');
exportgraphics(fig3, plot3_file, 'ContentType', 'vector', 'Resolution', 300);
individual_plots{end+1} = plot3_file;
close(fig3);

%% PLOT 4: LAYER-WISE NEURON ACTIVATION ANALYSIS
fprintf('  Creating Plot 4: Layer-wise Neuron Activation Analysis...\n');
fig4 = create_layer_activation_analysis_plot(plot_data.snn_architecture, plot_data.ann_architecture);
plot4_file = fullfile(output_dir, '04_Layer_Activation_Analysis.pdf');
exportgraphics(fig4, plot4_file, 'ContentType', 'vector', 'Resolution', 300);
individual_plots{end+1} = plot4_file;
close(fig4);

%% PLOT 5: COMPREHENSIVE ENERGY BREAKDOWN
fprintf('  Creating Plot 5: Comprehensive Energy Breakdown...\n');
fig5 = create_energy_breakdown_plot(plot_data.energy_analysis, snn_color, ann_color, efficiency_color);
plot5_file = fullfile(output_dir, '05_Comprehensive_Energy_Breakdown.pdf');
exportgraphics(fig5, plot5_file, 'ContentType', 'vector', 'Resolution', 300);
individual_plots{end+1} = plot5_file;
close(fig5);

%% PLOT 6: SPARSITY EVOLUTION AND EFFICIENCY
fprintf('  Creating Plot 6: Sparsity Evolution and Efficiency...\n');
fig6 = create_sparsity_efficiency_plot(plot_data.snn_results, plot_data.energy_analysis, snn_color, efficiency_color);
plot6_file = fullfile(output_dir, '06_Sparsity_Evolution_Efficiency.pdf');
exportgraphics(fig6, plot6_file, 'ContentType', 'vector', 'Resolution', 300);
individual_plots{end+1} = plot6_file;
close(fig6);

%% PLOT 7: CUMULATIVE ENERGY CONSUMPTION
fprintf('  Creating Plot 7: Cumulative Energy Consumption...\n');
fig7 = create_cumulative_energy_plot(plot_data.energy_analysis, snn_color, ann_color);
plot7_file = fullfile(output_dir, '07_Cumulative_Energy_Consumption.pdf');
exportgraphics(fig7, plot7_file, 'ContentType', 'vector', 'Resolution', 300);
individual_plots{end+1} = plot7_file;
close(fig7);

%% PLOT 8: CONVERGENCE ANALYSIS
fprintf('  Creating Plot 8: Training Convergence Analysis...\n');
fig8 = create_convergence_analysis_plot(plot_data.snn_results, plot_data.ann_results, snn_color, ann_color);
plot8_file = fullfile(output_dir, '08_Training_Convergence_Analysis.pdf');
exportgraphics(fig8, plot8_file, 'ContentType', 'vector', 'Resolution', 300);
individual_plots{end+1} = plot8_file;
close(fig8);

%% COMPILE INTO SINGLE PDF
fprintf('  Compiling all plots into single PDF...\n');
output_pdf = fullfile(output_dir, [analysis_name '.pdf']);

try
    % Try using Ghostscript for PDF compilation
    cmd = sprintf('gs -dBATCH -dNOPAUSE -q -sDEVICE=pdfwrite -sOutputFile="%s" %s', ...
        output_pdf, strjoin(individual_plots, ' '));
    [status, ~] = system(cmd);
    
    if status == 0
        fprintf('  ✓ PDF compilation successful using Ghostscript\n');
        success = true;
        generated_plots = {output_pdf};
        
        % Clean up individual files
        for i = 1:length(individual_plots)
            delete(individual_plots{i});
        end
    else
        error('Ghostscript compilation failed');
    end
catch
    fprintf('  ⚠ PDF compilation failed, keeping individual plots\n');
    success = false;
    generated_plots = individual_plots;
end

% Reset plotting defaults
reset_plotting_defaults();

fprintf('✓ Plot generation completed: %d plots created\n', length(generated_plots));

end

%% SUPPORTING PLOT FUNCTIONS

function fig = create_snn_architecture_plot(snn_arch)
% Create SNN architecture visualization

fig = figure('Units', 'inches', 'Position', [0 0 16 10], 'Color', 'w');

layer_names = fieldnames(snn_arch);
n_layers = length(layer_names);
layer_x = linspace(2, 14, n_layers);

for i = 1:n_layers
    layer = snn_arch.(layer_names{i});
    n_neurons = layer.neurons;
    sparsity = layer.sparsity;
    
    % Visual representation
    max_visual = 15;
    visual_neurons = min(n_neurons, max_visual);
    y_pos = linspace(2, 8, visual_neurons);
    
    % Draw neurons
    for j = 1:visual_neurons
        if j <= visual_neurons * (1 - sparsity)
            % Active neuron
            scatter(layer_x(i), y_pos(j), 120, [0.2 0.6 0.8], 'filled', 'MarkerEdgeColor', 'k');
        else
            % Inactive neuron
            scatter(layer_x(i), y_pos(j), 60, [0.9 0.9 0.9], 'filled', 'MarkerEdgeColor', [0.7 0.7 0.7]);
        end
    end
    
    % Layer labels
    text(layer_x(i), 1, strrep(layer_names{i}, '_', ' '), 'HorizontalAlignment', 'center', ...
        'FontSize', 14, 'FontWeight', 'bold');
    text(layer_x(i), 0.3, sprintf('%d neurons\n%.0f%% sparse', n_neurons, sparsity*100), ...
        'HorizontalAlignment', 'center', 'FontSize', 12);
end

title('SNN Architecture: Event-Driven Sparse Neural Computing', 'FontSize', 18, 'FontWeight', 'bold');
xlabel('Network Layers', 'FontSize', 16);
ylabel('Neuron Organization', 'FontSize', 16);
xlim([0, 16]); ylim([0, 9]);
set(gca, 'XTick', [], 'YTick', []);

drawnow;
end

function fig = create_energy_per_epoch_plot(energy_analysis, snn_color, ann_color)
% Create energy comparison per epoch plot (KEY REQUIREMENT)

fig = figure('Units', 'inches', 'Position', [0 0 14 10], 'Color', 'w');

subplot(2,2,1);
% Energy per epoch comparison
epochs_snn = 1:length(energy_analysis.snn_energy_per_epoch);
epochs_ann = 1:length(energy_analysis.ann_energy_per_epoch);

plot(epochs_snn, energy_analysis.snn_energy_per_epoch/1000, 'Color', snn_color, 'LineWidth', 3);
hold on;
plot(epochs_ann, energy_analysis.ann_energy_per_epoch/1000, 'Color', ann_color, 'LineWidth', 3);
title('Energy Consumption per Epoch', 'FontSize', 16, 'FontWeight', 'bold');
xlabel('Training Epoch', 'FontSize', 14);
ylabel('Energy (nJ)', 'FontSize', 14);
legend({'SNN', 'ANN'}, 'Location', 'best');
grid on;

subplot(2,2,2);
% Cumulative energy
cumulative_snn = cumsum(energy_analysis.snn_energy_per_epoch)/1e6;
cumulative_ann = cumsum(energy_analysis.ann_energy_per_epoch)/1e6;

plot(epochs_snn, cumulative_snn, 'Color', snn_color, 'LineWidth', 3);
hold on;
plot(epochs_ann, cumulative_ann, 'Color', ann_color, 'LineWidth', 3);
title('Cumulative Energy Consumption', 'FontSize', 16, 'FontWeight', 'bold');
xlabel('Training Epoch', 'FontSize', 14);
ylabel('Cumulative Energy (µJ)', 'FontSize', 14);
legend({'SNN', 'ANN'}, 'Location', 'best');
grid on;

subplot(2,2,3);
% Energy efficiency ratio over epochs
if length(energy_analysis.snn_energy_per_epoch) == length(energy_analysis.ann_energy_per_epoch)
    efficiency_per_epoch = energy_analysis.snn_energy_per_epoch ./ energy_analysis.ann_energy_per_epoch;
else
    min_epochs = min(length(energy_analysis.snn_energy_per_epoch), length(energy_analysis.ann_energy_per_epoch));
    efficiency_per_epoch = energy_analysis.snn_energy_per_epoch(1:min_epochs) ./ energy_analysis.ann_energy_per_epoch(1:min_epochs);
end

plot(1:length(efficiency_per_epoch), efficiency_per_epoch, 'Color', [0.2 0.7 0.3], 'LineWidth', 3);
yline(1, 'k--', 'Energy Parity', 'LineWidth', 2);
title('SNN Energy Efficiency Ratio (SNN/ANN)', 'FontSize', 16, 'FontWeight', 'bold');
xlabel('Training Epoch', 'FontSize', 14);
ylabel('Efficiency Ratio', 'FontSize', 14);
grid on;

subplot(2,2,4);
% Total energy comparison bar chart
total_energies = [energy_analysis.snn_total_energy_uJ, energy_analysis.ann_total_energy_uJ];
bar(total_energies, 'FaceColor', 'flat', 'CData', [snn_color; ann_color]);
title('Total Energy Comparison', 'FontSize', 16, 'FontWeight', 'bold');
ylabel('Total Energy (µJ)', 'FontSize', 14);
set(gca, 'XTickLabel', {'SNN', 'ANN'});
grid on;

% Add efficiency annotation
efficiency_advantage = 1/energy_analysis.efficiency_ratio;
text(0.5, 0.9, sprintf('SNN: %.1fx more efficient', efficiency_advantage), ...
    'Units', 'normalized', 'HorizontalAlignment', 'center', ...
    'FontSize', 14, 'FontWeight', 'bold', 'BackgroundColor', 'yellow');

sgtitle('Energy Comparison Analysis: SNN vs ANN per Epoch', 'FontSize', 20, 'FontWeight', 'bold');

drawnow;
end

function fig = create_performance_comparison_plot(performance, snn_color, ann_color)
% Create portfolio performance comparison

fig = figure('Units', 'inches', 'Position', [0 0 12 8], 'Color', 'w');

subplot(2,2,1);
returns = [performance.snn_return*100, performance.ann_return*100];
bar(returns, 'FaceColor', 'flat', 'CData', [snn_color; ann_color]);
title('Portfolio Returns', 'FontSize', 16, 'FontWeight', 'bold');
ylabel('Annual Return (%)', 'FontSize', 14);
set(gca, 'XTickLabel', {'SNN', 'ANN'});
grid on;

subplot(2,2,2);
risks = [performance.snn_risk*100, performance.ann_risk*100];
bar(risks, 'FaceColor', 'flat', 'CData', [snn_color; ann_color]);
title('Portfolio Risk', 'FontSize', 16, 'FontWeight', 'bold');
ylabel('Annual Volatility (%)', 'FontSize', 14);
set(gca, 'XTickLabel', {'SNN', 'ANN'});
grid on;

subplot(2,2,3);
sharpes = [performance.snn_sharpe, performance.ann_sharpe];
bar(sharpes, 'FaceColor', 'flat', 'CData', [snn_color; ann_color]);
title('Sharpe Ratio', 'FontSize', 16, 'FontWeight', 'bold');
ylabel('Risk-Adjusted Return', 'FontSize', 14);
set(gca, 'XTickLabel', {'SNN', 'ANN'});
grid on;

subplot(2,2,4);
scatter(performance.snn_risk*100, performance.snn_return*100, 250, snn_color, 'filled', 'MarkerEdgeColor', 'k');
hold on;
scatter(performance.ann_risk*100, performance.ann_return*100, 250, ann_color, 'filled', 'MarkerEdgeColor', 'k');
title('Risk-Return Profile', 'FontSize', 16, 'FontWeight', 'bold');
xlabel('Risk (% p.a.)', 'FontSize', 14);
ylabel('Return (% p.a.)', 'FontSize', 14);
legend({'SNN', 'ANN'}, 'Location', 'best');
grid on;

sgtitle('Portfolio Performance Comparison: SNN vs ANN', 'FontSize', 18, 'FontWeight', 'bold');

drawnow;
end

function fig = create_layer_activation_analysis_plot(snn_arch, ann_arch)
% Create layer-wise activation analysis

fig = figure('Units', 'inches', 'Position', [0 0 16 10], 'Color', 'w');

% Extract layer data
snn_layers = fieldnames(snn_arch);
ann_layers = fieldnames(ann_arch);

snn_neurons = zeros(length(snn_layers), 1);
snn_active = zeros(length(snn_layers), 1);
for i = 1:length(snn_layers)
    layer = snn_arch.(snn_layers{i});
    snn_neurons(i) = layer.neurons;
    snn_active(i) = layer.neurons * (1 - layer.sparsity);
end

ann_neurons = zeros(length(ann_layers), 1);
for i = 1:length(ann_layers)
    layer = ann_arch.(ann_layers{i});
    ann_neurons(i) = layer.neurons;
end

subplot(2,2,1);
% Total neurons per layer
max_layers = max(length(snn_layers), length(ann_layers));
neuron_data = zeros(max_layers, 2);
neuron_data(1:length(snn_layers), 1) = snn_neurons;
neuron_data(1:length(ann_layers), 2) = ann_neurons;

bar(neuron_data);
title('Total Neurons per Layer', 'FontSize', 16, 'FontWeight', 'bold');
xlabel('Layer Index', 'FontSize', 14);
ylabel('Number of Neurons', 'FontSize', 14);
legend({'SNN', 'ANN'}, 'Location', 'best');
grid on;

subplot(2,2,2);
% Active neurons per layer
active_data = zeros(max_layers, 2);
active_data(1:length(snn_layers), 1) = snn_active;
active_data(1:length(ann_layers), 2) = ann_neurons; % ANN all active

bar(active_data);
title('Active Neurons per Layer', 'FontSize', 16, 'FontWeight', 'bold');
xlabel('Layer Index', 'FontSize', 14);
ylabel('Active Neurons', 'FontSize', 14);
legend({'SNN', 'ANN'}, 'Location', 'best');
grid on;

subplot(2,2,3);
% Activation percentage
activation_pct = zeros(max_layers, 2);
activation_pct(1:length(snn_layers), 1) = (snn_active ./ snn_neurons) * 100;
activation_pct(1:length(ann_layers), 2) = 100; % ANN always 100%

bar(activation_pct);
title('Activation Percentage', 'FontSize', 16, 'FontWeight', 'bold');
xlabel('Layer Index', 'FontSize', 14);
ylabel('Activation (%)', 'FontSize', 14);
legend({'SNN', 'ANN'}, 'Location', 'best');
grid on;

subplot(2,2,4);
% Summary statistics
total_snn = sum(snn_neurons);
total_active_snn = sum(snn_active);
total_ann = sum(ann_neurons);

summary_data = [total_active_snn/total_snn*100, 100; % Activation rates
               total_snn, total_ann]; % Total neurons

bar(summary_data);
title('Architecture Summary', 'FontSize', 16, 'FontWeight', 'bold');
ylabel('Percentage (%) / Neuron Count', 'FontSize', 14);
set(gca, 'XTickLabel', {'Activation Rate', 'Total Neurons'});
legend({'SNN', 'ANN'}, 'Location', 'best');
grid on;

sgtitle('Layer-wise Neuron Activation Analysis: SNN vs ANN', 'FontSize', 18, 'FontWeight', 'bold');

drawnow;
end

function fig = create_energy_breakdown_plot(energy_analysis, snn_color, ann_color, efficiency_color)
% Create comprehensive energy breakdown

fig = figure('Units', 'inches', 'Position', [0 0 14 8], 'Color', 'w');

subplot(1,3,1);
% SNN energy breakdown
snn_breakdown = energy_analysis.snn_breakdown;
snn_values = [snn_breakdown.spike, snn_breakdown.synaptic, snn_breakdown.active_neuron, ...
             snn_breakdown.idle_neuron, snn_breakdown.membrane]/1000;
snn_labels = {'Spike', 'Synaptic', 'Active Neuron', 'Idle Neuron', 'Membrane'};

pie(snn_values, snn_labels);
title('SNN Energy Breakdown', 'FontSize', 16, 'FontWeight', 'bold');

subplot(1,3,2);
% ANN energy breakdown
ann_breakdown = energy_analysis.ann_breakdown;
ann_values = [ann_breakdown.mac, ann_breakdown.activation, ann_breakdown.memory]/1000;
ann_labels = {'MAC Operations', 'Activations', 'Memory Access'};

pie(ann_values, ann_labels);
title('ANN Energy Breakdown', 'FontSize', 16, 'FontWeight', 'bold');

subplot(1,3,3);
% Efficiency comparison
efficiency_metrics = [1/energy_analysis.efficiency_ratio, energy_analysis.energy_savings_percent/20, ...
                     energy_analysis.avg_snn_sparsity*100/20];
efficiency_labels = {'Efficiency Advantage', 'Energy Savings (/20)', 'Sparsity Level (/20)'};

bar(efficiency_metrics, 'FaceColor', efficiency_color);
title('SNN Efficiency Metrics', 'FontSize', 16, 'FontWeight', 'bold');
ylabel('Relative Value', 'FontSize', 14);
set(gca, 'XTickLabel', efficiency_labels);
grid on;

sgtitle('Comprehensive Energy Analysis Breakdown', 'FontSize', 18, 'FontWeight', 'bold');

drawnow;
end

function fig = create_sparsity_efficiency_plot(snn_results, energy_analysis, snn_color, efficiency_color)
% Create sparsity evolution and efficiency plot

fig = figure('Units', 'inches', 'Position', [0 0 12 8], 'Color', 'w');

subplot(1,2,1);
% Sparsity evolution
plot(1:length(snn_results.sparsity_evolution), snn_results.sparsity_evolution*100, ...
    'Color', snn_color, 'LineWidth', 3);
yline(95, 'r--', 'Target (95%)', 'LineWidth', 2);
title('SNN Sparsity Evolution', 'FontSize', 16, 'FontWeight', 'bold');
xlabel('Training Epoch', 'FontSize', 14);
ylabel('Sparsity Level (%)', 'FontSize', 14);
grid on;

subplot(1,2,2);
% Energy efficiency evolution
if isfield(energy_analysis, 'efficiency_evolution')
    plot(1:length(energy_analysis.efficiency_evolution), energy_analysis.efficiency_evolution, ...
        'Color', efficiency_color, 'LineWidth', 3);
else
    % Fallback: calculate simple efficiency metric
    efficiency_metric = cumsum(snn_results.energy_per_epoch) ./ max(cumsum(snn_results.energy_per_epoch));
    plot(1:length(efficiency_metric), efficiency_metric, 'Color', efficiency_color, 'LineWidth', 3);
end
title('Energy Efficiency Evolution', 'FontSize', 16, 'FontWeight', 'bold');
xlabel('Training Epoch', 'FontSize', 14);
ylabel('Efficiency Metric', 'FontSize', 14);
grid on;

sgtitle('Sparsity and Energy Efficiency Analysis', 'FontSize', 18, 'FontWeight', 'bold');

drawnow;
end

function fig = create_cumulative_energy_plot(energy_analysis, snn_color, ann_color)
% Create cumulative energy consumption plot

fig = figure('Units', 'inches', 'Position', [0 0 12 6], 'Color', 'w');

plot(1:length(energy_analysis.snn_cumulative_energy), energy_analysis.snn_cumulative_energy/1e6, ...
    'Color', snn_color, 'LineWidth', 3);
hold on;
plot(1:length(energy_analysis.ann_cumulative_energy), energy_analysis.ann_cumulative_energy/1e6, ...
    'Color', ann_color, 'LineWidth', 3);

title('Cumulative Energy Consumption: SNN vs ANN', 'FontSize', 16, 'FontWeight', 'bold');
xlabel('Training Epoch', 'FontSize', 14);
ylabel('Cumulative Energy (µJ)', 'FontSize', 14);
legend({'SNN', 'ANN'}, 'Location', 'best');
grid on;

% Add final savings annotation
final_savings = energy_analysis.energy_savings_percent;
text(0.7, 0.8, sprintf('Final Energy Savings: %.1f%%', final_savings), ...
    'Units', 'normalized', 'FontSize', 14, 'FontWeight', 'bold', ...
    'BackgroundColor', 'yellow');

drawnow;
end

function fig = create_convergence_analysis_plot(snn_results, ann_results, snn_color, ann_color)
% Create convergence analysis plot

fig = figure('Units', 'inches', 'Position', [0 0 12 8], 'Color', 'w');

subplot(1,2,1);
% Sharpe ratio convergence
plot(1:length(snn_results.sharpe_evolution), snn_results.sharpe_evolution, ...
    'Color', snn_color, 'LineWidth', 3);
hold on;
plot(1:length(ann_results.sharpe_evolution), ann_results.sharpe_evolution, ...
    'Color', ann_color, 'LineWidth', 3);
title('Sharpe Ratio Convergence', 'FontSize', 16, 'FontWeight', 'bold');
xlabel('Training Epoch', 'FontSize', 14);
ylabel('Sharpe Ratio', 'FontSize', 14);
legend({'SNN', 'ANN'}, 'Location', 'best');
grid on;

subplot(1,2,2);
% Energy vs Performance trade-off
scatter(snn_results.total_energy/1e6, snn_results.final_sharpe, 200, snn_color, 'filled', 'MarkerEdgeColor', 'k');
hold on;
scatter(ann_results.total_energy/1e6, ann_results.final_sharpe, 200, ann_color, 'filled', 'MarkerEdgeColor', 'k');
title('Energy vs Performance Trade-off', 'FontSize', 16, 'FontWeight', 'bold');
xlabel('Total Energy (µJ)', 'FontSize', 14);
ylabel('Final Sharpe Ratio', 'FontSize', 14);
legend({'SNN', 'ANN'}, 'Location', 'best');
grid on;

sgtitle('Training Convergence and Trade-off Analysis', 'FontSize', 18, 'FontWeight', 'bold');

drawnow;
end

function reset_plotting_defaults()
% Reset plotting defaults safely

try
    set(0, 'DefaultAxesFontSize', 'remove');
    set(0, 'DefaultTextFontSize', 'remove');
    set(0, 'DefaultLegendFontSize', 'remove');
    set(0, 'DefaultLineLineWidth', 'remove');
    set(0, 'DefaultFigureColor', 'remove');
    set(0, 'DefaultAxesLineWidth', 'remove');
catch
    % Silent failure for default reset
end
end
