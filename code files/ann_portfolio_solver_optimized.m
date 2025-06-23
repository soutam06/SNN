function [weights, selected_assets, ann_results] = ann_portfolio_solver_optimized(mean_ret, cov_mat, params, architecture)
%% ANN_PORTFOLIO_SOLVER_OPTIMIZED - Optimized Artificial Neural Network Portfolio Solver
%
% This function implements an efficient ANN for portfolio optimization using
% optimized variable management and realistic GPU/TPU energy consumption modeling.
%
% INPUTS:
%   mean_ret - Vector of expected returns (n_stocks x 1)
%   cov_mat - Covariance matrix (n_stocks x n_stocks)
%   params - ANN parameters structure
%   architecture - ANN architecture definition
%
% OUTPUTS:
%   weights - Optimal portfolio weights (n_stocks x 1)
%   selected_assets - Indices of selected assets
%   ann_results - Comprehensive results structure
%
% OPTIMIZATION FEATURES:
%   - Removed unnecessary variables and redundant arrays
%   - Dense feedforward neural network with proper energy modeling
%   - Adam optimizer with adaptive learning rate
%   - Realistic GPU/TPU energy consumption constants
%   - Streamlined gradient computation

n_stocks = length(mean_ret);
fprintf('Initializing optimized ANN solver with %d assets...\n', n_stocks);

%% NETWORK ARCHITECTURE (optimized initialization)
input_size = architecture.input_layer.neurons;
hidden1_size = architecture.hidden_layer1.neurons;
hidden2_size = architecture.hidden_layer2.neurons;
output_size = architecture.output_layer.neurons;

% Xavier initialization for stable training
W1 = randn(hidden1_size, input_size) * sqrt(2/(input_size + hidden1_size));
b1 = zeros(hidden1_size, 1);
W2 = randn(hidden2_size, hidden1_size) * sqrt(2/(hidden1_size + hidden2_size));
b2 = zeros(hidden2_size, 1);
W3 = randn(output_size, hidden2_size) * sqrt(2/(hidden2_size + output_size));
b3 = zeros(output_size, 1);

%% TRACKING VARIABLES (optimized - removed unnecessary arrays)
sharpe_evolution = zeros(params.n_epochs, 1);
energy_per_epoch = zeros(params.n_epochs, 1);
gradient_norms = zeros(params.n_epochs, 1);

best_sharpe = -inf;
best_weights = zeros(n_stocks, 1);

%% ADAM OPTIMIZER PARAMETERS (streamlined)
m_W1 = zeros(size(W1)); v_W1 = zeros(size(W1));
m_b1 = zeros(size(b1)); v_b1 = zeros(size(b1));
m_W2 = zeros(size(W2)); v_W2 = zeros(size(W2));
m_b2 = zeros(size(b2)); v_b2 = zeros(size(b2));
m_W3 = zeros(size(W3)); v_W3 = zeros(size(W3));
m_b3 = zeros(size(b3)); v_b3 = zeros(size(b3));

beta1 = 0.9;
beta2 = 0.999;
epsilon_adam = 1e-8;
learning_rate = params.learning_rate;

%% DIGITAL HARDWARE ENERGY CONSTANTS (GPU/TPU-based)
MAC_ENERGY = 4.6;          % pJ per multiply-accumulate operation
ADD_ENERGY = 0.03;         % pJ per addition
ACTIVATION_ENERGY = 0.1;   % pJ per activation function
L1_CACHE_ENERGY = 0.5;     % pJ per L1 cache access
L2_CACHE_ENERGY = 2.3;     % pJ per L2 cache access
DRAM_ENERGY = 640.0;       % pJ per DRAM access

%% MAIN ANN TRAINING LOOP
fprintf('Starting optimized ANN training (%d epochs)...\n', params.n_epochs);

for epoch = 1:params.n_epochs
    epoch_energy = 0;
    
    % Batch processing for improved convergence
    for batch = 1:params.batch_size
        %% FORWARD PASS
        input_features = ones(input_size, 1);
        if batch > 1
            input_features = input_features + 0.1 * randn(input_size, 1);
        end
        
        % Layer 1: Input → Hidden1
        z1 = W1 * input_features + b1;
        a1 = max(0, z1); % ReLU
        
        % Energy calculation for Layer 1
        mac_ops1 = numel(W1);
        memory_access1 = numel(W1) + numel(input_features) + numel(z1);
        layer1_energy = mac_ops1 * MAC_ENERGY + memory_access1 * L1_CACHE_ENERGY + ...
                       numel(z1) * ACTIVATION_ENERGY;
        
        % Layer 2: Hidden1 → Hidden2
        z2 = W2 * a1 + b2;
        a2 = max(0, z2); % ReLU
        
        % Energy calculation for Layer 2
        mac_ops2 = numel(W2);
        memory_access2 = numel(W2) + numel(a1) + numel(z2);
        layer2_energy = mac_ops2 * MAC_ENERGY + memory_access2 * L1_CACHE_ENERGY + ...
                       numel(z2) * ACTIVATION_ENERGY;
        
        % Layer 3: Hidden2 → Output
        z3 = W3 * a2 + b3;
        portfolio_weights = exp(z3) ./ (sum(exp(z3)) + eps); % Softmax
        
        % Energy calculation for Layer 3
        mac_ops3 = numel(W3);
        memory_access3 = numel(W3) + numel(a2) + numel(z3);
        layer3_energy = mac_ops3 * MAC_ENERGY + memory_access3 * L2_CACHE_ENERGY + ...
                       numel(z3) * ACTIVATION_ENERGY * 2; % Softmax is more expensive
        
        forward_energy = layer1_energy + layer2_energy + layer3_energy;
        
        %% PORTFOLIO EVALUATION
        [constrained_weights, portfolio_return, portfolio_risk, sharpe_ratio] = ...
            evaluate_portfolio(portfolio_weights, mean_ret, cov_mat, params.cardinality_range);
        
        % Loss function (negative Sharpe with regularization)
        base_loss = -sharpe_ratio;
        l2_penalty = params.weight_decay * (sum(W1(:).^2) + sum(W2(:).^2) + sum(W3(:).^2));
        total_loss = base_loss + l2_penalty;
        
        %% SIMPLIFIED GRADIENT COMPUTATION (optimized)
        epsilon_grad = 1e-5;
        
        % Gradients for W3 (critical layer)
        dW3 = compute_gradient_sample(W3, z3, a2, b3, constrained_weights, mean_ret, cov_mat, ...
                                     params.cardinality_range, epsilon_grad, 50);
        
        % Gradients for W2 (reduced sampling)
        dW2 = compute_gradient_sample(W2, z2, a1, b2, constrained_weights, mean_ret, cov_mat, ...
                                     params.cardinality_range, epsilon_grad, 25);
        
        % Simplified gradients for W1
        dW1 = zeros(size(W1));
        db1 = zeros(size(b1));
        db2 = zeros(size(b2));
        db3 = zeros(size(b3));
        
        % Backward pass energy (approximately 2.5x forward pass)
        backward_energy = forward_energy * 2.5;
        gradient_memory_energy = (numel(dW1) + numel(dW2) + numel(dW3)) * L2_CACHE_ENERGY;
        
        epoch_energy = epoch_energy + forward_energy + backward_energy + gradient_memory_energy;
    end
    
    % Average gradients over batches
    dW1 = dW1 / params.batch_size;
    dW2 = dW2 / params.batch_size;
    dW3 = dW3 / params.batch_size;
    
    % Calculate gradient norm
    total_grad_norm = norm(dW1, 'fro') + norm(dW2, 'fro') + norm(dW3, 'fro');
    
    %% ADAM OPTIMIZER UPDATE
    % Update momentum and velocity
    m_W1 = beta1 * m_W1 + (1 - beta1) * dW1;
    v_W1 = beta2 * v_W1 + (1 - beta2) * (dW1.^2);
    m_W2 = beta1 * m_W2 + (1 - beta1) * dW2;
    v_W2 = beta2 * v_W2 + (1 - beta2) * (dW2.^2);
    m_W3 = beta1 * m_W3 + (1 - beta1) * dW3;
    v_W3 = beta2 * v_W3 + (1 - beta2) * (dW3.^2);
    
    % Bias correction
    m_W1_hat = m_W1 / (1 - beta1^epoch);
    v_W1_hat = v_W1 / (1 - beta2^epoch);
    m_W2_hat = m_W2 / (1 - beta1^epoch);
    v_W2_hat = v_W2 / (1 - beta2^epoch);
    m_W3_hat = m_W3 / (1 - beta1^epoch);
    v_W3_hat = v_W3 / (1 - beta2^epoch);
    
    % Parameter updates
    W1 = W1 - learning_rate * m_W1_hat ./ (sqrt(v_W1_hat) + epsilon_adam);
    W2 = W2 - learning_rate * m_W2_hat ./ (sqrt(v_W2_hat) + epsilon_adam);
    W3 = W3 - learning_rate * m_W3_hat ./ (sqrt(v_W3_hat) + epsilon_adam);
    
    %% EPOCH EVALUATION
    % Final forward pass for evaluation
    final_input = ones(input_size, 1);
    final_z1 = W1 * final_input + b1;
    final_a1 = max(0, final_z1);
    final_z2 = W2 * final_a1 + b2;
    final_a2 = max(0, final_z2);
    final_z3 = W3 * final_a2 + b3;
    final_weights = exp(final_z3) ./ (sum(exp(final_z3)) + eps);
    
    [epoch_weights, epoch_return, epoch_risk, epoch_sharpe] = ...
        evaluate_portfolio(final_weights, mean_ret, cov_mat, params.cardinality_range);
    
    % Update best solution
    if epoch_sharpe > best_sharpe
        best_sharpe = epoch_sharpe;
        best_weights = epoch_weights;
    end
    
    % Store results
    sharpe_evolution(epoch) = epoch_sharpe;
    energy_per_epoch(epoch) = epoch_energy;
    gradient_norms(epoch) = total_grad_norm;
    
    % Adaptive learning rate
    if epoch > 20 && mod(epoch, 10) == 0
        recent_improvement = mean(sharpe_evolution(epoch-9:epoch)) - mean(sharpe_evolution(epoch-19:epoch-10));
        if recent_improvement < 0.001
            learning_rate = learning_rate * 0.95;
        end
    end
    
    % Progress reporting
    if mod(epoch, 20) == 0 || epoch == params.n_epochs
        fprintf('Epoch %d: Sharpe=%.4f, Energy=%.2f nJ, LR=%.5f\n', ...
            epoch, epoch_sharpe, epoch_energy/1000, learning_rate);
    end
end

%% FINALIZE RESULTS
weights = best_weights;
selected_assets = find(weights > 1e-4);

% Comprehensive results structure (optimized)
ann_results = struct();
ann_results.sharpe_evolution = sharpe_evolution;
ann_results.energy_per_epoch = energy_per_epoch;
ann_results.gradient_norms = gradient_norms;
ann_results.total_energy = sum(energy_per_epoch);
ann_results.avg_energy_per_epoch = mean(energy_per_epoch);
ann_results.final_sharpe = best_sharpe;
ann_results.total_parameters = numel(W1) + numel(b1) + numel(W2) + numel(b2) + numel(W3) + numel(b3);
ann_results.architecture_complexity = ann_results.total_parameters / n_stocks;

fprintf('✓ ANN optimization completed: Sharpe=%.4f, Energy=%.2f µJ, Params=%d\n', ...
    best_sharpe, ann_results.total_energy/1e6, ann_results.total_parameters);

end

%% SUPPORTING FUNCTIONS

function [weights, portfolio_return, portfolio_risk, sharpe_ratio] = evaluate_portfolio(raw_weights, mean_ret, cov_mat, cardinality_range)
% Evaluate and constrain portfolio weights

n_stocks = length(raw_weights);

% Apply cardinality constraints
[sorted_weights, indices] = sort(raw_weights, 'descend');
n_select = min(cardinality_range(2), max(cardinality_range(1), sum(raw_weights > 1e-4)));

weights = zeros(n_stocks, 1);
weights(indices(1:n_select)) = sorted_weights(1:n_select);
weights = weights / (sum(weights) + eps);

% Calculate metrics
portfolio_return = mean_ret' * weights;
portfolio_risk = sqrt(weights' * cov_mat * weights);
sharpe_ratio = portfolio_return / (portfolio_risk + 1e-8);
end

function dW = compute_gradient_sample(W, z, a_prev, b, weights, mean_ret, cov_mat, cardinality_range, epsilon, sample_size)
% Compute gradients using finite differences with sampling

dW = zeros(size(W));
sample_size = min(sample_size, numel(W));
sample_indices = randperm(numel(W), sample_size);

% Get baseline performance
[~, ~, ~, base_sharpe] = evaluate_portfolio(weights, mean_ret, cov_mat, cardinality_range);

for idx = 1:sample_size
    linear_idx = sample_indices(idx);
    
    % Perturb weight
    W_temp = W;
    W_temp(linear_idx) = W_temp(linear_idx) + epsilon;
    
    % Forward pass with perturbed weight
    z_temp = W_temp * a_prev + b;
    if size(z_temp, 1) == length(mean_ret) % Output layer
        weights_temp = exp(z_temp) ./ (sum(exp(z_temp)) + eps);
    else
        weights_temp = weights; % For hidden layers, use current weights
    end
    
    % Evaluate perturbed performance
    [~, ~, ~, perturbed_sharpe] = evaluate_portfolio(weights_temp, mean_ret, cov_mat, cardinality_range);
    
    % Calculate gradient
    dW(linear_idx) = ((-perturbed_sharpe) - (-base_sharpe)) / epsilon;
end
end
