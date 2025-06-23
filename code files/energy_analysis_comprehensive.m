function energy_analysis = energy_analysis_comprehensive(snn_results, ann_results, snn_architecture, ann_architecture)
%% ENERGY_ANALYSIS_COMPREHENSIVE - Centralized Energy Comparison Function
%
% This function performs comprehensive energy analysis comparing SNN and ANN
% implementations using realistic hardware energy constants and sparsity considerations.
%
% INPUTS:
%   snn_results - SNN optimization results with energy data
%   ann_results - ANN optimization results with energy data
%   snn_architecture - SNN architecture specification
%   ann_architecture - ANN architecture specification
%
% OUTPUTS:
%   energy_analysis - Comprehensive energy comparison structure
%
% FEATURES:
%   - Hardware-specific energy constants (Loihi vs GPU/TPU)
%   - Per-epoch energy tracking and analysis
%   - Sparsity-aware energy calculations
%   - Efficiency ratio computations
%   - Energy savings quantification

fprintf('Performing comprehensive energy analysis...\n');

%% EXTRACT BASIC ENERGY DATA
snn_total_energy = snn_results.total_energy; % pJ
ann_total_energy = ann_results.total_energy; % pJ
snn_energy_per_epoch = snn_results.energy_per_epoch; % pJ per epoch
ann_energy_per_epoch = ann_results.energy_per_epoch; % pJ per epoch

%% ARCHITECTURE ANALYSIS
% SNN architecture complexity
snn_layers = fieldnames(snn_architecture);
snn_total_neurons = 0;
snn_total_active = 0;
snn_weighted_sparsity = 0;

for i = 1:length(snn_layers)
    layer = snn_architecture.(snn_layers{i});
    snn_total_neurons = snn_total_neurons + layer.neurons;
    active_neurons = layer.neurons * (1 - layer.sparsity);
    snn_total_active = snn_total_active + active_neurons;
    snn_weighted_sparsity = snn_weighted_sparsity + layer.neurons * layer.sparsity;
end

avg_snn_sparsity = snn_weighted_sparsity / snn_total_neurons;

% ANN architecture complexity
ann_layers = fieldnames(ann_architecture);
ann_total_neurons = 0;
ann_total_parameters = 0;

for i = 1:length(ann_layers)
    layer = ann_architecture.(ann_layers{i});
    ann_total_neurons = ann_total_neurons + layer.neurons;
end

% Estimate parameters (simplified)
ann_total_parameters = ann_total_neurons * 10; % Rough estimate

%% HARDWARE-SPECIFIC ENERGY CONSTANTS

% Neuromorphic hardware (Intel Loihi) constants
LOIHI_SPIKE_ENERGY = 1.7;      % pJ per spike
LOIHI_SYNAPTIC_OP = 24.0;      % pJ per synaptic operation
LOIHI_NEURON_ACTIVE = 81.0;    % pJ per active neuron
LOIHI_NEURON_IDLE = 52.0;      % pJ per idle neuron
LOIHI_MEMBRANE_LEAK = 12.0;    % pJ per membrane leak

% Digital hardware (GPU/TPU) constants
DIGITAL_MAC = 4.6;             % pJ per multiply-accumulate
DIGITAL_ADD = 0.03;            % pJ per addition
DIGITAL_ACTIVATION = 0.1;      % pJ per activation function
DIGITAL_L1_CACHE = 0.5;        % pJ per L1 cache access
DIGITAL_L2_CACHE = 2.3;        % pJ per L2 cache access
DIGITAL_DRAM = 640.0;          % pJ per DRAM access

%% DETAILED ENERGY BREAKDOWN

% SNN energy breakdown
snn_spike_operations = snn_total_active * length(snn_energy_per_epoch) * 0.05; % 5% activation
snn_synaptic_operations = snn_spike_operations * 1.5; % Synaptic fanout
snn_membrane_operations = snn_total_neurons * length(snn_energy_per_epoch);

snn_spike_energy = snn_spike_operations * LOIHI_SPIKE_ENERGY;
snn_synaptic_energy = snn_synaptic_operations * LOIHI_SYNAPTIC_OP;
snn_neuron_energy = snn_total_active * length(snn_energy_per_epoch) * LOIHI_NEURON_ACTIVE;
snn_idle_energy = (snn_total_neurons - snn_total_active) * length(snn_energy_per_epoch) * LOIHI_NEURON_IDLE;
snn_membrane_energy = snn_membrane_operations * LOIHI_MEMBRANE_LEAK;

snn_theoretical_energy = snn_spike_energy + snn_synaptic_energy + snn_neuron_energy + ...
                        snn_idle_energy + snn_membrane_energy;

% ANN energy breakdown
ann_mac_operations = ann_total_parameters * length(ann_energy_per_epoch) * 0.7; % 70% utilization
ann_activation_operations = ann_total_neurons * length(ann_energy_per_epoch);
ann_memory_operations = ann_total_parameters * length(ann_energy_per_epoch) * 2; % Read + write

ann_mac_energy = ann_mac_operations * DIGITAL_MAC;
ann_activation_energy = ann_activation_operations * DIGITAL_ACTIVATION;
ann_memory_energy = ann_memory_operations * DIGITAL_L2_CACHE;

ann_theoretical_energy = ann_mac_energy + ann_activation_energy + ann_memory_energy;

%% EFFICIENCY CALCULATIONS

% Primary efficiency ratio (lower is better for SNN)
efficiency_ratio = snn_total_energy / ann_total_energy;

% Energy savings percentage
energy_savings_percent = (1 - efficiency_ratio) * 100;

% Energy per operation
snn_energy_per_op = snn_total_energy / (snn_total_active * length(snn_energy_per_epoch));
ann_energy_per_op = ann_total_energy / (ann_total_neurons * length(ann_energy_per_epoch));

% Sparsity impact analysis
sparsity_advantage = (1 - avg_snn_sparsity) / 1.0; % SNN vs ANN (100% dense)
theoretical_efficiency_advantage = 1 / sparsity_advantage;

%% TEMPORAL ANALYSIS

% Energy evolution over training
snn_cumulative_energy = cumsum(snn_energy_per_epoch);
ann_cumulative_energy = cumsum(ann_energy_per_epoch);

% Efficiency evolution
efficiency_evolution = snn_energy_per_epoch ./ ann_energy_per_epoch(1); % Normalized to first ANN epoch

% Average energy per epoch
snn_avg_epoch_energy = mean(snn_energy_per_epoch);
ann_avg_epoch_energy = mean(ann_energy_per_epoch);

%% COMPREHENSIVE RESULTS STRUCTURE

energy_analysis = struct();

% Basic energy metrics
energy_analysis.snn_total_energy_pJ = snn_total_energy;
energy_analysis.ann_total_energy_pJ = ann_total_energy;
energy_analysis.snn_total_energy_uJ = snn_total_energy / 1e6;
energy_analysis.ann_total_energy_uJ = ann_total_energy / 1e6;

% Per-epoch data
energy_analysis.snn_energy_per_epoch = snn_energy_per_epoch;
energy_analysis.ann_energy_per_epoch = ann_energy_per_epoch;
energy_analysis.snn_avg_epoch_energy = snn_avg_epoch_energy;
energy_analysis.ann_avg_epoch_energy = ann_avg_epoch_energy;

% Efficiency metrics
energy_analysis.efficiency_ratio = efficiency_ratio;
energy_analysis.energy_savings_percent = energy_savings_percent;
energy_analysis.snn_energy_per_op = snn_energy_per_op;
energy_analysis.ann_energy_per_op = ann_energy_per_op;

% Sparsity analysis
energy_analysis.avg_snn_sparsity = avg_snn_sparsity;
energy_analysis.sparsity_advantage = sparsity_advantage;
energy_analysis.theoretical_efficiency_advantage = theoretical_efficiency_advantage;

% Architecture metrics
energy_analysis.snn_total_neurons = snn_total_neurons;
energy_analysis.ann_total_neurons = ann_total_neurons;
energy_analysis.snn_active_neurons = snn_total_active;
energy_analysis.ann_active_neurons = ann_total_neurons; % 100% active

% Temporal analysis
energy_analysis.snn_cumulative_energy = snn_cumulative_energy;
energy_analysis.ann_cumulative_energy = ann_cumulative_energy;
energy_analysis.efficiency_evolution = efficiency_evolution;

% Hardware comparison
energy_analysis.snn_theoretical_energy = snn_theoretical_energy;
energy_analysis.ann_theoretical_energy = ann_theoretical_energy;
energy_analysis.hardware_advantage = ann_theoretical_energy / snn_theoretical_energy;

% Energy breakdown
energy_analysis.snn_breakdown = struct('spike', snn_spike_energy, 'synaptic', snn_synaptic_energy, ...
    'active_neuron', snn_neuron_energy, 'idle_neuron', snn_idle_energy, 'membrane', snn_membrane_energy);
energy_analysis.ann_breakdown = struct('mac', ann_mac_energy, 'activation', ann_activation_energy, ...
    'memory', ann_memory_energy);

%% VALIDATION AND ASSERTIONS

% Ensure SNN is more efficient (basic validation)
if efficiency_ratio >= 1.0
    fprintf('⚠ Warning: SNN efficiency ratio %.3f indicates ANN may be more efficient\n', efficiency_ratio);
    fprintf('  This could indicate parameter issues or insufficient sparsity\n');
else
    fprintf('✓ SNN efficiency validated: %.1fx more efficient than ANN\n', 1/efficiency_ratio);
end

% Sparsity validation
if avg_snn_sparsity < 0.90
    fprintf('⚠ Warning: SNN sparsity %.1f%% is lower than expected (>90%%)\n', avg_snn_sparsity*100);
else
    fprintf('✓ SNN sparsity validated: %.1f%% sparse operation\n', avg_snn_sparsity*100);
end

fprintf('Energy analysis completed: SNN=%.2f µJ, ANN=%.2f µJ, Efficiency=%.3f\n', ...
    energy_analysis.snn_total_energy_uJ, energy_analysis.ann_total_energy_uJ, efficiency_ratio);

end