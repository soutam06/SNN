%% snn_portfolio_solver.m
% Spiking Neural Network Portfolio Optimization with Cardinality Constraints

function [weights, selected_idx] = snn_portfolio_solver(mean_ret, cov_mat, params)
n_stocks = length(mean_ret);
population = init_population(n_stocks, params.pop_size);

best_sharpe = -inf;
best_weights = zeros(n_stocks, 1);

for epoch = 1:params.n_epochs
    potentials = zeros(params.pop_size, n_stocks);

    for i = 1:params.pop_size
        input_current = params.risk_aversion * mean_ret' - ...
            (1 - params.risk_aversion) * diag(cov_mat)';
        potentials(i,:) = params.tau * abs(population(i,:)) + ...
            input_current + 0.05 * randn(1, n_stocks);
        spikes = potentials(i,:) > params.threshold;
        population(i,spikes) = abs(population(i,spikes) + ...
            0.1 * input_current(spikes));
    end

    epoch_weights = decode_weights(population, cov_mat);

    [sharpe, n_selected] = evaluate_portfolio(epoch_weights, mean_ret, cov_mat);
    if n_selected < params.cardinality(1)
        epoch_weights = enforce_min_selection(epoch_weights, params.cardinality(1));
    end

    if sharpe > best_sharpe
        best_sharpe = sharpe;
        best_weights = epoch_weights;
    end

    params.threshold = max(0.2, params.threshold * 0.95); % Floor threshold
end

[weights, selected_idx] = apply_cardinality(best_weights, params.cardinality);
end

%% Helper functions

function [final_weights, selected_idx] = apply_cardinality(weights, card_range)
    [~, rank] = sort(weights, 'descend');
    target = card_range(1); % Minimum cardinality (e.g., 30)
    selected_idx = rank(1:target);
    final_weights = zeros(size(weights));
    final_weights(selected_idx) = weights(selected_idx);
    final_weights = final_weights / (sum(final_weights) + eps); % Prevent division by zero
end

function pop = init_population(n_stocks, pop_size)
    pop = abs(randn(pop_size, n_stocks) * 0.1); % Positive initialization
end

function weights = decode_weights(population, cov_mat)
    raw_weights = mean(population, 1)'; % ensure column vector
    vol_scaling = 1 ./ sqrt(diag(cov_mat)); % column vector
    weights = abs(raw_weights) .* vol_scaling;
    weights = weights / sum(weights + eps);
end

function [sharpe, n_selected] = evaluate_portfolio(weights, mean_ret, cov_mat)
    transaction_cost = 0.0025 * sum(abs(weights));
    valid = weights > 1e-4;
    n_selected = sum(valid);

    if n_selected < 1
        sharpe = -inf;
        return
    end

    % Ensure column vectors for dot product
    ret_vector = mean_ret(valid);
    weight_vector = weights(valid);

    ret = ret_vector' * weight_vector - transaction_cost;
    risk = sqrt(weight_vector' * cov_mat(valid,valid) * weight_vector);
    sharpe = ret / (risk + 1e-6);
end

function weights = enforce_min_selection(weights, min_stocks)
    [~, idx] = sort(weights, 'descend');
    weights(idx(1:min_stocks)) = max(weights(idx(1:min_stocks)), 1e-3);
    weights = weights / sum(weights + eps);
end
