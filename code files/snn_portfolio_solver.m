function [weights, selected_idx, convergence_data] = snn_portfolio_solver(mean_ret, cov_mat, params)
% SNN Portfolio Solver for Asset Allocation
% Uses a spiking neural network-inspired approach to select portfolio weights
% Tracks neuron activation during optimization

n_stocks = length(mean_ret); % Number of assets
% Initialize population of neuron "potentials" for each asset
population = init_population(n_stocks, params.pop_size, params.init_method);

best_sharpe = -inf;                % Track best Sharpe ratio found
best_weights = zeros(n_stocks, 1); % Track weights for best Sharpe
% Store convergence and tracking info for analysis/plots
convergence_data = struct( ...
    'sharpe_history', zeros(params.n_epochs, 1), ...
    'weights_history', zeros(params.n_epochs, n_stocks), ...
    'n_selected_history', zeros(params.n_epochs, 1), ...
    'active_neurons_history', zeros(params.n_epochs, 1) ... % Track neuron activations
);

current_threshold = params.threshold; % Initial neuron firing threshold
current_tau = params.tau;             % Initial decay parameter

for epoch = 1:params.n_epochs
    potentials = zeros(params.pop_size, n_stocks); % Store neuron potentials for this epoch
    total_activated = 0;                           % Track total neuron activations this epoch

    for i = 1:params.pop_size
        % Optionally adapt risk aversion over epochs
        if isfield(params, 'adaptive_risk_aversion') && params.adaptive_risk_aversion
            risk_factor = max(0.1, min(0.9, 1 - epoch/params.n_epochs));
        else
            risk_factor = params.risk_aversion;
        end
        % Calculate input current: reward (mean return) minus risk (variance)
        input_current = risk_factor * mean_ret' - (1 - risk_factor) * diag(cov_mat)';
        % Penalize sector concentration if sector info is present
        if isfield(params, 'sector_info') && ~isempty(params.sector_info)
            sector_penalty = calculate_sector_concentration(population(i,:), params.sector_info);
            input_current = input_current - params.sector_weight * sector_penalty;
        end
        % Update neuron potentials with decay, input, and noise
        potentials(i,:) = current_tau * abs(population(i,:)) + input_current + params.noise_factor * randn(1, n_stocks);
        % Neuron "fires" if potential exceeds threshold
        spikes = potentials(i,:) > current_threshold;
        total_activated = total_activated + sum(spikes); % Count activations
        % Update population based on spikes
        if any(spikes)
            population(i,spikes) = abs(population(i,spikes) + params.learning_rate * input_current(spikes));
            % Lateral inhibition: suppress non-spiking neurons
            if isfield(params, 'lateral_inhibition') && params.lateral_inhibition
                non_spikes = ~spikes;
                population(i,non_spikes) = population(i,non_spikes) * 0.95;
            end
        end
    end

    % Store the total number of neuron activations this epoch
    convergence_data.active_neurons_history(epoch) = total_activated;

    % Decode population into portfolio weights
    epoch_weights = decode_weights(population, cov_mat, params.decoding_method);
    % Apply transaction cost if previous weights are available
    if isfield(params, 'prev_weights') && ~isempty(params.prev_weights) && isfield(params, 'transaction_cost')
        [epoch_weights, trans_cost] = apply_transaction_cost(epoch_weights, params.prev_weights, params.transaction_cost);
    else
        trans_cost = 0;
    end
    % Evaluate portfolio: compute Sharpe ratio and number selected
    [sharpe, n_selected] = evaluate_portfolio(epoch_weights, mean_ret, cov_mat, trans_cost);
    % Enforce minimum cardinality (number of assets)
    if n_selected < params.cardinality(1)
        epoch_weights = enforce_min_selection(epoch_weights, params.cardinality(1));
        [sharpe, n_selected] = evaluate_portfolio(epoch_weights, mean_ret, cov_mat, trans_cost);
    end
    % Track best portfolio found so far
    if sharpe > best_sharpe
        best_sharpe = sharpe;
        best_weights = epoch_weights;
    end
    % Store convergence data for plots/analysis
    convergence_data.sharpe_history(epoch) = sharpe;
    convergence_data.weights_history(epoch,:) = epoch_weights';
    convergence_data.n_selected_history(epoch) = n_selected;
    % Decay the neuron firing threshold over epochs
    current_threshold = max(params.min_threshold, current_threshold * params.threshold_decay);
    % Optionally adapt tau (decay) over epochs
    if isfield(params, 'adaptive_tau') && params.adaptive_tau
        current_tau = params.tau * (1 - 0.5 * epoch/params.n_epochs);
    end
end

% Enforce cardinality constraint on the best weights found
[weights, selected_idx] = apply_cardinality(best_weights, params.cardinality);
% Enforce sector constraints if required
if isfield(params, 'sector_info') && isfield(params, 'sector_limits') && ~isempty(params.sector_limits)
    weights = apply_sector_constraints(weights, params.sector_info, params.sector_limits);
end
% Normalize weights to sum to 1
weights = weights / sum(weights);
end

%% === Helper Functions ===

function pop = init_population(n_stocks, pop_size, method)
% Initialize the neuron population matrix
switch lower(method)
    case 'uniform'
        pop = rand(pop_size, n_stocks) * 0.1;
    case 'normal'
        pop = abs(randn(pop_size, n_stocks) * 0.1);
    case 'xavier'
        scale = sqrt(2 / (n_stocks + pop_size));
        pop = abs(randn(pop_size, n_stocks) * scale);
    otherwise
        pop = abs(randn(pop_size, n_stocks) * 0.1);
end
end

function weights = decode_weights(population, cov_mat, method)
% Convert neuron population to portfolio weights
raw_weights = mean(population, 1)';
switch lower(method)
    case 'simple'
        weights = abs(raw_weights);
        weights = weights / sum(weights + eps);
    case 'volatility'
        vol_scaling = 1 ./ sqrt(diag(cov_mat));
        weights = abs(raw_weights) .* vol_scaling;
        weights = weights / sum(weights + eps);
    case 'softmax'
        temp = 2.0;
        exp_weights = exp(abs(raw_weights) / temp);
        weights = exp_weights / sum(exp_weights + eps);
    otherwise
        vol_scaling = 1 ./ sqrt(diag(cov_mat));
        weights = abs(raw_weights) .* vol_scaling;
        weights = weights / sum(weights + eps);
end
end

function [weights, selected_idx] = apply_cardinality(weights, card_range)
% Enforce minimum and (optionally) maximum number of assets
[sorted_weights, rank] = sort(weights, 'descend');
min_card = card_range(1);
if length(card_range) > 1
    max_card = card_range(2);
    best_k = min_card;
    best_sharpe = -inf;
    % Try all cardinalities in range and pick the best by diversification
    for k = min_card:max_card
        test_idx = rank(1:k);
        test_weights = zeros(size(weights));
        test_weights(test_idx) = sorted_weights(1:k);
        test_weights = test_weights / sum(test_weights + eps);
        herfindahl = sum(test_weights.^2);
        effective_k = 1 / herfindahl;
        if effective_k > best_sharpe
            best_sharpe = effective_k;
            best_k = k;
        end
    end
    target = best_k;
else
    target = min_card;
end
selected_idx = rank(1:target);
final_weights = zeros(size(weights));
final_weights(selected_idx) = weights(selected_idx);
weights = final_weights / (sum(final_weights) + eps);
end

function [sharpe, n_selected] = evaluate_portfolio(weights, mean_ret, cov_mat, trans_cost)
% Compute Sharpe ratio and number of selected assets
valid = weights > 1e-4;
n_selected = sum(valid);
if n_selected < 1
    sharpe = -inf;
    return
end
ret_vector = mean_ret(valid);
weight_vector = weights(valid);
portfolio_return = ret_vector' * weight_vector - trans_cost;
portfolio_risk = sqrt(weight_vector' * cov_mat(valid,valid) * weight_vector);
sharpe = portfolio_return / (portfolio_risk + 1e-6);
end

function weights = enforce_min_selection(weights, min_stocks)
% Ensure at least min_stocks assets are present in the portfolio
[~, idx] = sort(weights, 'descend');
min_weight = 1e-3;
weights(idx(1:min_stocks)) = max(weights(idx(1:min_stocks)), min_weight);
weights = weights / sum(weights + eps);
end

function sector_penalty = calculate_sector_concentration(neuron_values, sector_info)
% Penalize over-concentration in any sector (for sector constraints)
n_sectors = max(sector_info);
sector_totals = zeros(1, n_sectors);
for s = 1:n_sectors
    sector_mask = (sector_info == s);
    sector_totals(s) = sum(neuron_values(sector_mask));
end
sector_weights = sector_totals / (sum(sector_totals) + eps);
% Distribute sector penalty back to neuron vector
sector_penalty = zeros(size(neuron_values));
for s = 1:n_sectors
    sector_mask = (sector_info == s);
    sector_penalty(sector_mask) = sector_weights(s);
end
end

function weights = apply_sector_constraints(weights, sector_info, sector_limits)
% Enforce sector allocation limits
n_sectors = max(sector_info);
sector_weights = zeros(1, n_sectors);
for s = 1:n_sectors
    sector_mask = (sector_info == s);
    sector_weights(s) = sum(weights(sector_mask));
end
% If no violation, return
any_violation = false;
for s = 1:n_sectors
    if sector_weights(s) > sector_limits(s)
        any_violation = true;
        break;
    end
end
if ~any_violation
    return;
end
% Scale down sectors that exceed limits
for s = 1:n_sectors
    if sector_weights(s) > sector_limits(s)
        sector_mask = (sector_info == s);
        scale_factor = sector_limits(s) / sector_weights(s);
        weights(sector_mask) = weights(sector_mask) * scale_factor;
    end
end
% Redistribute leftover weight to underweight sectors
remaining = 1 - sum(weights);
if remaining > 0
    underweight_sectors = find(sector_weights < sector_limits);
    available_capacity = sum(sector_limits(underweight_sectors)) - sum(sector_weights(underweight_sectors));
    if available_capacity > 0
        for s = underweight_sectors
            sector_mask = (sector_info == s);
            capacity = sector_limits(s) - sector_weights(s);
            proportion = capacity / available_capacity;
            sector_stocks = find(sector_mask);
            if ~isempty(sector_stocks)
                sector_distribution = weights(sector_stocks) / (sum(weights(sector_stocks)) + eps);
                weights(sector_stocks) = weights(sector_stocks) + remaining * proportion * sector_distribution;
            end
        end
    end
end
weights = weights / sum(weights + eps);
end

function [adjusted_weights, cost] = apply_transaction_cost(new_weights, prev_weights, cost_rate)
% Penalize excessive turnover by applying transaction costs
turnover = sum(abs(new_weights - prev_weights));
cost = cost_rate * turnover;
max_turnover = 0.5; % Limit max turnover per period
if turnover > max_turnover
    adjustment_factor = max_turnover / turnover;
    adjusted_weights = prev_weights + adjustment_factor * (new_weights - prev_weights);
    cost = cost_rate * max_turnover;
else
    adjusted_weights = new_weights;
end
end
