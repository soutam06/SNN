function [weights, selected_idx, ann_info] = ann_portfolio_solver(mean_ret, cov_mat, params)
% ANN_PORTFOLIO_SOLVER: Feedforward ANN for portfolio optimization
%
% Inputs:
%   mean_ret   - Column vector of annualized mean returns
%   cov_mat    - Covariance matrix of returns
%   params     - Struct with fields:
%                  .n_epochs      (number of training epochs)
%                  .hidden_size   (number of hidden units)
%                  .learning_rate (gradient step size)
%                  .cardinality   ([min, max] assets)
%
% Outputs:
%   weights      - Optimal portfolio weights (sum to 1, cardinality enforced)
%   selected_idx - Indices of selected assets
%   ann_info     - Struct with training loss, Sharpe history, etc.

n_stocks = length(mean_ret);
rng(42); % For reproducibility

% --- Network Architecture ---
input_size = n_stocks;
hidden_size = params.hidden_size;
output_size = n_stocks;

% Xavier initialization for weights and zero biases
W1 = randn(hidden_size, input_size) * sqrt(2/(input_size + hidden_size));
b1 = zeros(hidden_size, 1);
W2 = randn(output_size, hidden_size) * sqrt(2/(hidden_size + output_size));
b2 = zeros(output_size, 1);

% Trackers for convergence and best solution
sharpe_history = zeros(params.n_epochs, 1);
loss_history = zeros(params.n_epochs, 1);
weights_history = zeros(params.n_epochs, n_stocks);

target_sharpe = -inf; % Best Sharpe so far
best_weights = zeros(n_stocks, 1);

for epoch = 1:params.n_epochs
    % --- Forward Pass ---
    x = ones(input_size, 1); % Dummy input (all ones)
    h = max(0, W1 * x + b1); % ReLU activation
    out = W2 * h + b2;       % Linear output

    % Non-negative weights, normalized to sum to 1
    raw_weights = max(0, out);
    weights = raw_weights / (sum(raw_weights) + eps);

    % Enforce cardinality: keep only top-K weights
    [~, idx] = sort(weights, 'descend');
    minK = params.cardinality(1);
    maxK = params.cardinality(2);
    k = min(maxK, max(minK, sum(weights > 1e-4))); % Number of assets to keep
    selected_idx = idx(1:k);

    final_weights = zeros(n_stocks, 1);
    final_weights(selected_idx) = weights(selected_idx);
    weights = final_weights / (sum(final_weights) + eps);

    % --- Portfolio Metrics ---
    port_return = mean_ret' * weights;
    port_risk = sqrt(weights' * cov_mat * weights);
    sharpe = port_return / (port_risk + 1e-8);

    % --- Loss Function ---
    loss = -sharpe; % Negative Sharpe for maximization

    % --- Numerical Gradient (W1 only for demonstration) ---
    grad = zeros(size(W1));
    epsilon = 1e-5;
    for i = 1:numel(W1)
        W1_try = W1;
        W1_try(i) = W1_try(i) + epsilon;
        h_try = max(0, W1_try * x + b1);
        out_try = W2 * h_try + b2;
        raw_weights_try = max(0, out_try);
        weights_try = raw_weights_try / (sum(raw_weights_try) + eps);

        [~, idx_try] = sort(weights_try, 'descend');
        final_weights_try = zeros(n_stocks, 1);
        final_weights_try(idx_try(1:k)) = weights_try(idx_try(1:k));
        weights_try = final_weights_try / (sum(final_weights_try) + eps);

        port_return_try = mean_ret' * weights_try;
        port_risk_try = sqrt(weights_try' * cov_mat * weights_try);
        sharpe_try = port_return_try / (port_risk_try + 1e-8);

        grad(i) = (sharpe_try - sharpe) / epsilon;
    end

    % Gradient ascent step
    W1 = W1 + params.learning_rate * grad;

    % --- Trackers ---
    sharpe_history(epoch) = sharpe;
    loss_history(epoch) = loss;
    weights_history(epoch, :) = weights';

    % Save best solution so far
    if sharpe > target_sharpe
        target_sharpe = sharpe;
        best_weights = weights;
        best_selected_idx = selected_idx;
    end
end

% Output best solution found
weights = best_weights;
selected_idx = best_selected_idx;

% Package info
ann_info = struct();
ann_info.sharpe_history = sharpe_history;
ann_info.loss_history = loss_history;
ann_info.weights_history = weights_history;
ann_info.final_sharpe = target_sharpe;
ann_info.final_return = mean_ret' * weights;
ann_info.final_risk = sqrt(weights' * cov_mat * weights);
ann_info.effective_n = 1 / sum(weights.^2);
ann_info.selected_count = sum(weights > 0);

end
