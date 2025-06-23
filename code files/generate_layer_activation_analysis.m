%% GENERATE_LAYER_ACTIVATION_ANALYSIS.m
% Script to generate layer-wise neuron activation comparison between SNN and ANN
%
% This script loads the existing optimization results and creates a comprehensive
% visualization showing how many neurons are activated in each layer for energy analysis.

clearvars; clc; close all;

fprintf('==========================================================\n');
fprintf('  LAYER-WISE NEURON ACTIVATION COMPARISON: SNN vs ANN\n');
fprintf('  For Energy Analysis and Architecture Understanding\n');
fprintf('==========================================================\n\n');

%% LOAD EXISTING DATA

fprintf('1. LOADING OPTIMIZATION RESULTS\n');
fprintf('===============================\n');

try
    % Load the comprehensive results from previous optimization
    load('complete_optimization_results.mat');
    fprintf('✓ Optimization results loaded successfully\n');
catch
    fprintf('⚠ Optimization results not found. Generating sample data...\n');
    % Create sample architectures for demonstration
    n_stocks = 100; % Example portfolio size
    
    % SNN Architecture (Event-driven, Sparse)
    snn_architecture = struct();
    snn_architecture.input_layer = struct('neurons', n_stocks, 'type', 'encoding', 'sparsity', 0.95);
    snn_architecture.hidden_layer1 = struct('neurons', n_stocks*2, 'type', 'leaky_integrate_fire', 'sparsity', 0.98);
    snn_architecture.hidden_layer2 = struct('neurons', n_stocks, 'type', 'leaky_integrate_fire', 'sparsity', 0.97);
    snn_architecture.output_layer = struct('neurons', n_stocks, 'type', 'readout', 'sparsity', 0.70);
    
    % ANN Architecture (Dense computation)
    ann_architecture = struct();
    ann_architecture.input_layer = struct('neurons', n_stocks, 'activation', 'linear');
    ann_architecture.hidden_layer1 = struct('neurons', 64, 'activation', 'relu');
    ann_architecture.hidden_layer2 = struct('neurons', 32, 'activation', 'relu');
    ann_architecture.output_layer = struct('neurons', n_stocks, 'activation', 'softmax');
    
    % Create sample results structures
    snn_results = struct();
    snn_results.final_sparsity = 0.95;
    snn_results.total_energy = 150000; % pJ
    
    ann_results = struct();
    ann_results.total_energy = 650000; % pJ
    
    fprintf('✓ Sample data generated\n');
end

%% DISPLAY ARCHITECTURE INFORMATION

fprintf('\n2. ARCHITECTURE SUMMARY\n');
fprintf('========================\n');

% SNN Architecture Summary
snn_layer_names = fieldnames(snn_architecture);
fprintf('SNN Architecture (Sparse, Event-driven):\n');
total_snn_neurons = 0;
weighted_sparsity = 0;

for i = 1:length(snn_layer_names)
    layer = snn_architecture.(snn_layer_names{i});
    total_snn_neurons = total_snn_neurons + layer.neurons;
    weighted_sparsity = weighted_sparsity + layer.neurons * layer.sparsity;
    fprintf('  Layer %d (%s): %d neurons, %.0f%% sparse (%.0f%% active)\n', ...
        i, snn_layer_names{i}, layer.neurons, layer.sparsity*100, (1-layer.sparsity)*100);
end

avg_snn_sparsity = weighted_sparsity / total_snn_neurons;
fprintf('  → Total: %d neurons, Average sparsity: %.1f%%\n', total_snn_neurons, avg_snn_sparsity*100);

% ANN Architecture Summary
ann_layer_names = fieldnames(ann_architecture);
fprintf('\nANN Architecture (Dense computation):\n');
total_ann_neurons = 0;

for i = 1:length(ann_layer_names)
    layer = ann_architecture.(ann_layer_names{i});
    total_ann_neurons = total_ann_neurons + layer.neurons;
    fprintf('  Layer %d (%s): %d neurons, 100%% active\n', ...
        i, ann_layer_names{i}, layer.neurons);
end

fprintf('  → Total: %d neurons, No sparsity (100%% dense)\n', total_ann_neurons);

%% CALCULATE DETAILED ACTIVATION STATISTICS

fprintf('\n3. ACTIVATION STATISTICS CALCULATION\n');
fprintf('=====================================\n');

% SNN Detailed Statistics
fprintf('SNN Layer-by-Layer Activation Analysis:\n');
snn_total_active = 0;

for i = 1:length(snn_layer_names)
    layer = snn_architecture.(snn_layer_names{i});
    active_neurons = layer.neurons * (1 - layer.sparsity);
    snn_total_active = snn_total_active + active_neurons;
    fprintf('  %s: %.0f active / %d total\n', ...
        snn_layer_names{i}, active_neurons, layer.neurons);
end

% ANN Detailed Statistics
fprintf('\nANN Layer-by-Layer Activation Analysis:\n');
ann_total_active = 0;

for i = 1:length(ann_layer_names)
    layer = ann_architecture.(ann_layer_names{i});
    ann_total_active = ann_total_active + layer.neurons;
    fprintf('  %s: %d active / %d total\n', ...
        ann_layer_names{i}, layer.neurons, layer.neurons);
end

fprintf('\nCOMPARISON SUMMARY:\n');
fprintf('  SNN Total Active: %.0f / %d (%.1f%%)\n', ...
    snn_total_active, total_snn_neurons, snn_total_active/total_snn_neurons*100);
fprintf('  ANN Total Active: %d / %d (100.0%%)\n', ...
    ann_total_active, total_ann_neurons);
fprintf('  Activation Efficiency: SNN uses %.1fx fewer active neurons\n', ...
    ann_total_active/snn_total_active);

%% GENERATE LAYER ACTIVATION COMPARISON PLOT

fprintf('\n4. GENERATING LAYER ACTIVATION COMPARISON PLOT\n');
fprintf('===============================================\n');

% Prepare plot data structure
plot_data = struct();
plot_data.config = struct('output_directory', 'results_complete');

% Ensure output directory exists
if ~exist(plot_data.config.output_directory, 'dir')
    mkdir(plot_data.config.output_directory);
end

try
    % Generate the comprehensive layer activation comparison
    fig = create_layer_activation_comparison(snn_architecture, ann_architecture, ...
        snn_results, ann_results);
    
    % Save the plot as high-quality PDF
    output_filename = fullfile(plot_data.config.output_directory, ...
        'Layer_Activation_Comparison_SNN_vs_ANN.pdf');
    fprintf('Saving plot to: %s\n', output_filename);
    exportgraphics(fig, output_filename, 'ContentType', 'vector', 'Resolution', 300);
    
    % Also save as PNG for easy viewing
    png_filename = strrep(output_filename, '.pdf', '.png');
    exportgraphics(fig, png_filename, 'Resolution', 300);
    
    fprintf('✓ Layer activation comparison plot generated successfully!\n');
    fprintf('Files created:\n');
    fprintf('  • %s\n', output_filename);
    fprintf('  • %s\n', png_filename);
    
    close(fig);
    
catch ME
    fprintf('✗ Error generating plot: %s\n', ME.message);
    fprintf('Please ensure create_layer_activation_comparison.m is in the path\n');
end

%% ENERGY ANALYSIS SUMMARY

fprintf('\n5. ENERGY IMPLICATIONS SUMMARY\n');
fprintf('===============================\n');

% Calculate energy implications based on activation patterns
fprintf('Energy Analysis Based on Neuron Activation:\n');

% Energy constants (realistic hardware values)
NEUROMORPHIC_SPIKE_ENERGY = 1.7; % pJ per spike (Loihi)
NEUROMORPHIC_IDLE_ENERGY = 52.0; % pJ per idle neuron
DIGITAL_MAC_ENERGY = 4.6; % pJ per MAC operation (GPU)
DIGITAL_ACTIVATION_ENERGY = 0.1; % pJ per activation

% Estimate SNN energy
snn_estimated_energy = snn_total_active * NEUROMORPHIC_SPIKE_ENERGY + ...
    (total_snn_neurons - snn_total_active) * NEUROMORPHIC_IDLE_ENERGY;

% Estimate ANN energy (simplified - based on neurons and connections)
ann_estimated_connections = 0;
for i = 1:length(ann_layer_names)-1
    current_layer = ann_architecture.(ann_layer_names{i});
    next_layer = ann_architecture.(ann_layer_names{i+1});
    ann_estimated_connections = ann_estimated_connections + current_layer.neurons * next_layer.neurons;
end

ann_estimated_energy = ann_estimated_connections * DIGITAL_MAC_ENERGY + ...
    ann_total_active * DIGITAL_ACTIVATION_ENERGY;

fprintf('  SNN Estimated Energy: %.2f µJ\n', snn_estimated_energy/1e6);
fprintf('  ANN Estimated Energy: %.2f µJ\n', ann_estimated_energy/1e6);
fprintf('  Energy Efficiency Ratio: %.2f (SNN/ANN)\n', snn_estimated_energy/ann_estimated_energy);

if snn_estimated_energy < ann_estimated_energy
    energy_savings = (1 - snn_estimated_energy/ann_estimated_energy) * 100;
    fprintf('  ✓ SNN achieves %.1f%% energy savings\n', energy_savings);
    fprintf('  ✓ SNN is %.1fx more energy efficient\n', ann_estimated_energy/snn_estimated_energy);
else
    fprintf('  ⚠ Unexpected: ANN appears more efficient\n');
end

fprintf('\n🎯 KEY INSIGHTS FOR ENERGY OPTIMIZATION:\n');
fprintf('  • SNN sparsity directly reduces energy consumption\n');
fprintf('  • Each layer''s sparsity level contributes to overall efficiency\n');
fprintf('  • Event-driven computation minimizes unnecessary calculations\n');
fprintf('  • Neuromorphic hardware amplifies the energy advantage\n');

fprintf('\n==========================================================\n');
fprintf('  LAYER ACTIVATION ANALYSIS COMPLETED SUCCESSFULLY\n');
fprintf('==========================================================\n');

%% SUPPORTING FUNCTION
function fig = create_layer_activation_comparison(snn_arch, ann_arch, snn_results, ann_results)
% Create comprehensive layer activation comparison plot

fig = figure('Units', 'inches', 'Position', [0 0 16 12], 'Color', 'w');

% Extract layer data
snn_layers = fieldnames(snn_arch);
ann_layers = fieldnames(ann_arch);

% SNN data
snn_neurons = zeros(length(snn_layers), 1);
snn_active = zeros(length(snn_layers), 1);
snn_energy = zeros(length(snn_layers), 1);

for i = 1:length(snn_layers)
    layer = snn_arch.(snn_layers{i});
    snn_neurons(i) = layer.neurons;
    snn_active(i) = layer.neurons * (1 - layer.sparsity);
    % Energy per layer (simplified)
    snn_energy(i) = snn_active(i) * 1.7 + (snn_neurons(i) - snn_active(i)) * 0.052; % pJ
end

% ANN data
ann_neurons = zeros(length(ann_layers), 1);
ann_energy = zeros(length(ann_layers), 1);

for i = 1:length(ann_layers)
    layer = ann_arch.(ann_layers{i});
    ann_neurons(i) = layer.neurons;
    % Energy per layer (simplified)
    ann_energy(i) = ann_neurons(i) * 4.6; % pJ
end

% Subplot 1: Total neurons per layer
subplot(2,3,1);
max_layers = max(length(snn_layers), length(ann_layers));
neuron_data = zeros(max_layers, 2);
neuron_data(1:length(snn_layers), 1) = snn_neurons;
neuron_data(1:length(ann_layers), 2) = ann_neurons;

bar(neuron_data);
title('Total Neurons per Layer', 'FontSize', 14, 'FontWeight', 'bold');
xlabel('Layer Index');
ylabel('Number of Neurons');
legend({'SNN', 'ANN'}, 'Location', 'best');
grid on;

% Subplot 2: Active neurons per layer
subplot(2,3,2);
active_data = zeros(max_layers, 2);
active_data(1:length(snn_layers), 1) = snn_active;
active_data(1:length(ann_layers), 2) = ann_neurons; % ANN all active

bar(active_data);
title('Active Neurons per Layer', 'FontSize', 14, 'FontWeight', 'bold');
xlabel('Layer Index');
ylabel('Active Neurons');
legend({'SNN', 'ANN'}, 'Location', 'best');
grid on;

% Subplot 3: Activation percentage
subplot(2,3,3);
activation_pct = zeros(max_layers, 2);
activation_pct(1:length(snn_layers), 1) = (snn_active ./ snn_neurons) * 100;
activation_pct(1:length(ann_layers), 2) = 100; % ANN always 100%

bar(activation_pct);
title('Activation Percentage', 'FontSize', 14, 'FontWeight', 'bold');
xlabel('Layer Index');
ylabel('Activation (%)');
legend({'SNN', 'ANN'}, 'Location', 'best');
grid on;

% Subplot 4: Energy consumption per layer
subplot(2,3,4);
energy_data = zeros(max_layers, 2);
energy_data(1:length(snn_layers), 1) = snn_energy / 1000; % Convert to nJ
energy_data(1:length(ann_layers), 2) = ann_energy / 1000; % Convert to nJ

bar(energy_data);
title('Energy per Layer', 'FontSize', 14, 'FontWeight', 'bold');
xlabel('Layer Index');
ylabel('Energy (nJ)');
legend({'SNN', 'ANN'}, 'Location', 'best');
grid on;

% Subplot 5: Cumulative energy
subplot(2,3,5);
cumulative_snn = cumsum(snn_energy) / 1000;
cumulative_ann = cumsum(ann_energy) / 1000;

plot(1:length(cumulative_snn), cumulative_snn, 'b-o', 'LineWidth', 2, 'MarkerSize', 6);
hold on;
plot(1:length(cumulative_ann), cumulative_ann, 'r-s', 'LineWidth', 2, 'MarkerSize', 6);
title('Cumulative Energy', 'FontSize', 14, 'FontWeight', 'bold');
xlabel('Layer Index');
ylabel('Cumulative Energy (nJ)');
legend({'SNN', 'ANN'}, 'Location', 'best');
grid on;

% Subplot 6: Summary statistics
subplot(2,3,6);
total_snn = sum(snn_neurons);
total_active_snn = sum(snn_active);
total_ann = sum(ann_neurons);

summary_data = [total_active_snn/total_snn*100, 100; % Activation rates
               sum(snn_energy)/1000, sum(ann_energy)/1000]; % Total energy

bar(summary_data);
title('Architecture Summary', 'FontSize', 14, 'FontWeight', 'bold');
ylabel('Percentage (%) / Energy (nJ)');
set(gca, 'XTickLabel', {'Activation Rate', 'Total Energy'});
legend({'SNN', 'ANN'}, 'Location', 'best');
grid on;

sgtitle('Layer-wise Neuron Activation and Energy Analysis: SNN vs ANN', ...
    'FontSize', 16, 'FontWeight', 'bold');

drawnow;
end