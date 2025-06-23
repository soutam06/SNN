function [weights, selected_assets, snn_results] = snn_portfolio_solver_optimized(mean_ret, cov_mat, params, architecture)
%% SNN_PORTFOLIO_SOLVER_OPTIMIZED - Optimized Spiking Neural Network Portfolio Solver
%
% This function implements an energy-efficient SNN for portfolio optimization
% using realistic neuromorphic computing principles with optimized variable usage.
%
% INPUTS:
%   mean_ret - Vector of expected returns (n_stocks x 1)
%   cov_mat - Covariance matrix (n_stocks x n_stocks)
%   params - SNN parameters structure
%   architecture - SNN architecture definition
%
% OUTPUTS:
%   weights - Optimal portfolio weights (n_stocks x 1)
%   selected_assets - Indices of selected assets
%   snn_results - Comprehensive results structure
%
% OPTIMIZATION FEATURES:
%   - Removed unnecessary variables (population_temp, extra_arrays)
%   - Event-driven sparse computation (95%+ sparsity)
%   - Realistic neuromorphic energy constants
%   - Proper memory management
%   - Energy-aware optimization

n_stocks = length(mean_ret);
fprintf('Initializing optimized SNN solver with %d assets...\n', n_stocks);

%% INITIALIZE OPTIMIZED SNN POPULATION (removed unnecessary variables)
population = abs(randn(params.pop_size, n_stocks) * 0.1);
membrane_potentials = zeros(params.pop_size, n_stocks);

% Essential tracking arrays only (removed redundant variables)
sharpe_evolution = zeros(params.n_epochs, 1);
sparsity_evolution = zeros(params.n_epochs, 1);
energy_per_epoch = zeros(params.n_epochs, 1);

best_sharpe = -inf;
best_weights = zeros(n_stocks, 1);

%% NEUROMORPHIC ENERGY CONSTANTS (Intel Loihi-based)
SPIKE_ENERGY = 1.7;        % pJ per spike
SYNAPTIC_ENERGY = 24.0;    % pJ per synaptic operation
NEURON_ACTIVE_ENERGY = 81.0; % pJ per active neuron
NEURON_IDLE_ENERGY = 52.0;   % pJ per idle neuron
MEMBRANE_LEAK_ENERGY = 12.0; % pJ per membrane leak

%% ADAPTIVE PARAMETERS (streamlined)
current_threshold = params.threshold_initial;

%% MAIN SNN OPTIMIZATION LOOP
fprintf('Starting optimized SNN training (%d epochs)...\n', params.n_epochs);

for epoch = 1:params.n_epochs
    epoch_energy = 0;
    total_spikes = 0;
    total_active_neurons = 0;
    
    % Time window simulation for temporal dynamics
    for t = 1:params.time_window
        for i = 1:params.pop_size
            % Calculate input current based on portfolio theory
            input_current = mean_ret' - 0.5 * diag(cov_mat)' + ...
                params.learning_rate * randn(1, n_stocks) * 0.05;
            
            % Membrane potential update (leaky integrate-and-fire)
            membrane_potentials(i, :) = params.tau_membrane * membrane_potentials(i, :) + input_current;
            
            % Spike generation with sparsity enforcement
            spike_mask = membrane_potentials(i, :) > current_threshold;
            
            % Enforce target sparsity (95% sparsity = 5% activation)
            n_spikes = sum(spike_mask);
            max_spikes = ceil(n_stocks * params.target_sparsity);
            
            if n_spikes > max_spikes
                [~, spike_indices] = sort(membrane_potentials(i, :), 'descend');
                spike_mask(:) = false;
                spike_mask(spike_indices(1:max_spikes)) = true;
                n_spikes = max_spikes;
            end
            
            % Energy calculation (per time step)
            active_neurons = n_spikes;
            idle_neurons = n_stocks - active_neurons;
            
            % Spike and synaptic energy
            spike_energy = n_spikes * SPIKE_ENERGY;
            synaptic_energy = n_spikes * SYNAPTIC_ENERGY;
            
            % Neuron maintenance energy
            active_energy = active_neurons * NEURON_ACTIVE_ENERGY;
            idle_energy = idle_neurons * NEURON_IDLE_ENERGY;
            
            % Membrane leak energy (all neurons)
            leak_energy = n_stocks * MEMBRANE_LEAK_ENERGY;
            
            total_step_energy = spike_energy + synaptic_energy + active_energy + idle_energy + leak_energy;
            epoch_energy = epoch_energy + total_step_energy;
            
            % Update population weights based on spikes
            if any(spike_mask)
                population(i, spike_mask) = population(i, spike_mask) + ...
                    params.learning_rate * input_current(spike_mask);
                population(i, spike_mask) = max(0, population(i, spike_mask));
            end
            
            % Reset spiking neurons
            membrane_potentials(i, spike_mask) = 0;
            
            % Track statistics
            total_spikes = total_spikes + n_spikes;
            total_active_neurons = total_active_neurons + active_neurons;
        end
    end
    
    % Evaluate current population
    epoch_weights = evaluate_population(population, mean_ret, cov_mat, params.cardinality_range);
    epoch_return = mean_ret' * epoch_weights;
    epoch_risk = sqrt(epoch_weights' * cov_mat * epoch_weights);
    epoch_sharpe = epoch_return / (epoch_risk + 1e-8);
    
    % Update best solution
    if epoch_sharpe > best_sharpe
        best_sharpe = epoch_sharpe;
        best_weights = epoch_weights;
    end
    
    % Calculate sparsity
    total_possible_operations = params.pop_size * n_stocks * params.time_window;
    current_sparsity = 1 - (total_active_neurons / total_possible_operations);
    
    % Store results
    sharpe_evolution(epoch) = epoch_sharpe;
    sparsity_evolution(epoch) = current_sparsity;
    energy_per_epoch(epoch) = epoch_energy;
    
    % Adaptive threshold decay
    current_threshold = max(params.threshold_min, current_threshold * params.threshold_decay);
    
    % Progress reporting
    if mod(epoch, 25) == 0 || epoch == params.n_epochs
        fprintf('Epoch %d: Sharpe=%.4f, Sparsity=%.1f%%, Energy=%.2f nJ\n', ...
            epoch, epoch_sharpe, current_sparsity*100, epoch_energy/1000);
    end
end

%% FINALIZE RESULTS
weights = best_weights;
selected_assets = find(weights > 1e-4);

% Comprehensive results structure (optimized)
snn_results = struct();
snn_results.sharpe_evolution = sharpe_evolution;
snn_results.sparsity_evolution = sparsity_evolution;
snn_results.energy_per_epoch = energy_per_epoch;
snn_results.total_energy = sum(energy_per_epoch);
snn_results.final_sparsity = mean(sparsity_evolution(end-10:end));
snn_results.avg_energy_per_epoch = mean(energy_per_epoch);
snn_results.architecture_complexity = sum(structfun(@(x) x.neurons, architecture));
snn_results.final_sharpe = best_sharpe;

fprintf('✓ SNN optimization completed: Sharpe=%.4f, Sparsity=%.1f%%, Energy=%.2f µJ\n', ...
    best_sharpe, snn_results.final_sparsity*100, snn_results.total_energy/1e6);

end

%% SUPPORTING FUNCTIONS

function weights = evaluate_population(population, mean_ret, cov_mat, cardinality_range)
% Evaluate population and return best portfolio weights

pop_size = size(population, 1);
n_stocks = size(population, 2);
sharpe_scores = zeros(pop_size, 1);

for i = 1:pop_size
    % Normalize weights
    w = population(i, :)';
    w = w / (sum(w) + eps);
    
    % Apply cardinality constraints
    [sorted_w, indices] = sort(w, 'descend');
    n_select = min(cardinality_range(2), max(cardinality_range(1), sum(w > 1e-4)));
    
    final_w = zeros(n_stocks, 1);
    final_w(indices(1:n_select)) = sorted_w(1:n_select);
    final_w = final_w / (sum(final_w) + eps);
    
    % Calculate Sharpe ratio
    ret = mean_ret' * final_w;
    risk = sqrt(final_w' * cov_mat * final_w);
    sharpe_scores(i) = ret / (risk + 1e-8);
    
    population(i, :) = final_w';
end

% Return best weights
[~, best_idx] = max(sharpe_scores);
weights = population(best_idx, :)';
end
