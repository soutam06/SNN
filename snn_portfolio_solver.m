function [weights, selected_idx, convergence_data] = snn_portfolio_solver(mean_ret, cov_mat, params)
    % Input:
    %   mean_ret - Expected returns vector (n_stocks x 1)
    %   cov_mat - Covariance matrix (n_stocks x n_stocks)
    %   params - Structure with algorithm parameters
    %
    % Output:
    %   weights - Optimal portfolio weights
    %   selected_idx - Indices of selected assets
    %   convergence_data - Structure with convergence information

    % Initialize variables
    n_stocks = length(mean_ret);

    % Initialize neuron population (population-based approach)
    population = init_population(n_stocks, params.pop_size, params.init_method);

    % Initialize tracking variables
    best_sharpe = -inf;
    best_weights = zeros(n_stocks, 1);
    convergence_data = struct('sharpe_history', zeros(params.n_epochs, 1), ...
                              'weights_history', zeros(params.n_epochs, n_stocks), ...
                              'n_selected_history', zeros(params.n_epochs, 1));

    % Define adaptive parameters
    current_threshold = params.threshold;
    current_tau = params.tau;

    % Run for specified number of epochs
    for epoch = 1:params.n_epochs
        % Initialize potentials for this epoch
        potentials = zeros(params.pop_size, n_stocks);

        % Process each neuron in the population
        for i = 1:params.pop_size
            % Calculate input current based on risk and return
            % Apply adaptive risk aversion if enabled
            if isfield(params, 'adaptive_risk_aversion') && params.adaptive_risk_aversion
                risk_factor = max(0.1, min(0.9, 1 - epoch/params.n_epochs)); % Decrease risk aversion over time
            else
                risk_factor = params.risk_aversion;
            end

            % Enhanced input current calculation with sector diversification
            input_current = risk_factor * mean_ret' - ...
                (1 - risk_factor) * diag(cov_mat)';

            % If sector information is provided, add sector diversification factor
            if isfield(params, 'sector_info') && ~isempty(params.sector_info)
                sector_penalty = calculate_sector_concentration(population(i,:), params.sector_info);
                input_current = input_current - params.sector_weight * sector_penalty;
            end

            % Calculate membrane potentials with temporal dynamics
            potentials(i,:) = current_tau * abs(population(i,:)) + ...
                input_current + params.noise_factor * randn(1, n_stocks);

            % Generate spikes based on threshold
            spikes = potentials(i,:) > current_threshold;

            % Update neuron values based on spikes (Hebbian-like learning)
            if any(spikes)
                population(i,spikes) = abs(population(i,spikes) + ...
                    params.learning_rate * input_current(spikes));

                % Apply lateral inhibition (optional)
                if isfield(params, 'lateral_inhibition') && params.lateral_inhibition
                    non_spikes = ~spikes;
                    population(i,non_spikes) = population(i,non_spikes) * 0.95;
                end
            end
        end

        % Decode neuron population to portfolio weights
        epoch_weights = decode_weights(population, cov_mat, params.decoding_method);

        % Apply transaction cost constraints if provided
        if isfield(params, 'prev_weights') && ~isempty(params.prev_weights) && isfield(params, 'transaction_cost')
            [epoch_weights, trans_cost] = apply_transaction_cost(epoch_weights, params.prev_weights, params.transaction_cost);
        else
            trans_cost = 0;
        end

        % Evaluate portfolio performance
        [sharpe, n_selected] = evaluate_portfolio(epoch_weights, mean_ret, cov_mat, trans_cost);

        % Enforce minimum cardinality if needed
        if n_selected < params.cardinality(1)
            epoch_weights = enforce_min_selection(epoch_weights, params.cardinality(1));
            [sharpe, n_selected] = evaluate_portfolio(epoch_weights, mean_ret, cov_mat, trans_cost);
        end

        % Update best solution if improved
        if sharpe > best_sharpe
            best_sharpe = sharpe;
            best_weights = epoch_weights;
        end

        % Store convergence data
        convergence_data.sharpe_history(epoch) = sharpe;
        convergence_data.weights_history(epoch,:) = epoch_weights';
        convergence_data.n_selected_history(epoch) = n_selected;

        % Adaptive parameter updates
        % Decay threshold (annealing schedule)
        current_threshold = max(params.min_threshold, current_threshold * params.threshold_decay);

        % Adaptive tau parameter (optional)
        if isfield(params, 'adaptive_tau') && params.adaptive_tau
            current_tau = params.tau * (1 - 0.5 * epoch/params.n_epochs);
        end
    end

    % Apply final cardinality constraints
    [weights, selected_idx] = apply_cardinality(best_weights, params.cardinality);

    % Apply sector constraints (if needed)
    if isfield(params, 'sector_info') && isfield(params, 'sector_limits') && ~isempty(params.sector_limits)
        weights = apply_sector_constraints(weights, params.sector_info, params.sector_limits);
    end

    % Normalize weights to sum to 1
    weights = weights / sum(weights);
end

%% Helper functions

function pop = init_population(n_stocks, pop_size, method)
    % Initialize neuron population based on specified method
    switch lower(method)
        case 'uniform'
            pop = rand(pop_size, n_stocks) * 0.1;
        case 'normal'
            pop = abs(randn(pop_size, n_stocks) * 0.1); % Positive initialization
        case 'xavier'
            % Xavier/Glorot initialization
            scale = sqrt(2 / (n_stocks + pop_size));
            pop = abs(randn(pop_size, n_stocks) * scale);
        otherwise
            % Default to normal initialization
            pop = abs(randn(pop_size, n_stocks) * 0.1);
    end
end

function weights = decode_weights(population, cov_mat, method)
    % Convert neuron activity to portfolio weights using specified method

    % Get mean activation across population
    raw_weights = mean(population, 1)'; % Ensure column vector

    switch lower(method)
        case 'simple'
            % Simple normalization
            weights = abs(raw_weights);
            weights = weights / sum(weights + eps);

        case 'volatility'
            % Volatility-scaled weights (default method)
            vol_scaling = 1 ./ sqrt(diag(cov_mat)); % Column vector
            weights = abs(raw_weights) .* vol_scaling;
            weights = weights / sum(weights + eps);

        case 'softmax'
            % Softmax-based weighting
            temp = 2.0; % Temperature parameter
            exp_weights = exp(abs(raw_weights) / temp);
            weights = exp_weights / sum(exp_weights + eps);

        otherwise
            % Default to volatility scaling
            vol_scaling = 1 ./ sqrt(diag(cov_mat));
            weights = abs(raw_weights) .* vol_scaling;
            weights = weights / sum(weights + eps);
    end
end

function [weights, selected_idx] = apply_cardinality(weights, card_range)
    % Apply cardinality constraints to select top K assets
    [sorted_weights, rank] = sort(weights, 'descend');

    % Determine target cardinality
    min_card = card_range(1);

    % If max cardinality is specified, use it
    if length(card_range) > 1
        max_card = card_range(2);
        % Select optimal K within range by evaluating Sharpe
        best_k = min_card;
        best_sharpe = -inf;

        for k = min_card:max_card
            test_idx = rank(1:k);
            test_weights = zeros(size(weights));
            test_weights(test_idx) = sorted_weights(1:k);
            test_weights = test_weights / sum(test_weights + eps);

            % Use concentration index as a proxy for diversification
            herfindahl = sum(test_weights.^2);
            effective_k = 1 / herfindahl;

            % Prefer solutions with higher effective K
            if effective_k > best_sharpe
                best_sharpe = effective_k;
                best_k = k;
            end
        end

        target = best_k;
    else
        % Just use minimum cardinality
        target = min_card;
    end

    % Select top K assets
    selected_idx = rank(1:target);
    final_weights = zeros(size(weights));
    final_weights(selected_idx) = weights(selected_idx);

    % Normalize to sum to 1
    weights = final_weights / (sum(final_weights) + eps);
end

function [sharpe, n_selected] = evaluate_portfolio(weights, mean_ret, cov_mat, trans_cost)
    % Calculate Sharpe ratio and count selected assets

    % Count assets with non-zero weights
    valid = weights > 1e-4;
    n_selected = sum(valid);

    % Return negative infinity if no assets selected
    if n_selected < 1
        sharpe = -inf;
        return
    end

    % Calculate expected return
    ret_vector = mean_ret(valid);
    weight_vector = weights(valid);
    portfolio_return = ret_vector' * weight_vector - trans_cost;

    % Calculate portfolio risk
    portfolio_risk = sqrt(weight_vector' * cov_mat(valid,valid) * weight_vector);

    % Calculate Sharpe ratio (with small constant to prevent division by zero)
    sharpe = portfolio_return / (portfolio_risk + 1e-6);
end

function weights = enforce_min_selection(weights, min_stocks)
    % Ensure minimum number of assets are selected
    [~, idx] = sort(weights, 'descend');
    min_weight = 1e-3;

    % Set minimum weight for top K assets
    weights(idx(1:min_stocks)) = max(weights(idx(1:min_stocks)), min_weight);

    % Normalize weights
    weights = weights / sum(weights + eps);
end

function sector_penalty = calculate_sector_concentration(neuron_values, sector_info)
    % Calculate sector concentration penalty to encourage diversification
    n_sectors = max(sector_info);
    sector_totals = zeros(1, n_sectors);

    % Sum neuron values by sector
    for s = 1:n_sectors
        sector_mask = (sector_info == s);
        sector_totals(s) = sum(neuron_values(sector_mask));
    end

    % Normalize sector totals
    sector_weights = sector_totals / (sum(sector_totals) + eps);

    % Calculate Herfindahl index for sector concentration
    herfindahl = sum(sector_weights.^2);

    % Create penalty vector - higher values for assets in overrepresented sectors
    sector_penalty = zeros(size(neuron_values));
    for s = 1:n_sectors
        sector_mask = (sector_info == s);
        sector_penalty(sector_mask) = sector_weights(s);
    end
end

function weights = apply_sector_constraints(weights, sector_info, sector_limits)
    % Apply sector limits to portfolio weights
    n_sectors = max(sector_info);
    n_stocks = length(weights);

    % Calculate current sector weights
    sector_weights = zeros(1, n_sectors);
    for s = 1:n_sectors
        sector_mask = (sector_info == s);
        sector_weights(s) = sum(weights(sector_mask));
    end

    % Check if any sector exceeds its limit
    any_violation = false;
    for s = 1:n_sectors
        if sector_weights(s) > sector_limits(s)
            any_violation = true;
            break;
        end
    end

    % If no violations, return original weights
    if ~any_violation
        return;
    end

    % Otherwise, apply quadratic programming to enforce sector constraints
    % (simplified approach here - in practice, use quadprog)

    % Scale down overweight sectors proportionally
    for s = 1:n_sectors
        if sector_weights(s) > sector_limits(s)
            sector_mask = (sector_info == s);
            scale_factor = sector_limits(s) / sector_weights(s);
            weights(sector_mask) = weights(sector_mask) * scale_factor;
        end
    end

    % Redistribute to underweight sectors
    remaining = 1 - sum(weights);
    if remaining > 0
        underweight_sectors = find(sector_weights < sector_limits);
        available_capacity = sum(sector_limits(underweight_sectors)) - sum(sector_weights(underweight_sectors));

        if available_capacity > 0
            for s = underweight_sectors
                sector_mask = (sector_info == s);
                capacity = sector_limits(s) - sector_weights(s);
                proportion = capacity / available_capacity;

                % Distribute proportionally to existing weights within sector
                sector_stocks = find(sector_mask);
                if ~isempty(sector_stocks)
                    sector_distribution = weights(sector_stocks) / (sum(weights(sector_stocks)) + eps);
                    weights(sector_stocks) = weights(sector_stocks) + remaining * proportion * sector_distribution;
                end
            end
        end
    end

    % Final normalization
    weights = weights / sum(weights + eps);
end

function [adjusted_weights, cost] = apply_transaction_cost(new_weights, prev_weights, cost_rate)
    % Apply transaction cost penalty and adjustment
    turnover = sum(abs(new_weights - prev_weights));
    cost = cost_rate * turnover;

    % If turnover is too high, reduce it by moving partially toward new weights
    max_turnover = 0.5; % Example maximum turnover constraint
    if turnover > max_turnover
        adjustment_factor = max_turnover / turnover;
        adjusted_weights = prev_weights + adjustment_factor * (new_weights - prev_weights);
        cost = cost_rate * max_turnover;
    else
        adjusted_weights = new_weights;
    end
end