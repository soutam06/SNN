% SNN.m — Main script for QIBAS-based Portfolio Optimization

% Load data
mu = readNPY('data/mu.npy');         % [n_assets x 1]
sigma = readNPY('data/sigma.npy');   % [n_assets x n_assets]
n_assets = length(mu);

% QIBAS/Markowitz parameters
alpha = 0.7;                         % Risk aversion parameter
A = 2 * alpha * sigma;               % Quadratic term
b = -(1 - alpha) * mu;               % Linear term

% Constraints
Aeq = ones(1, n_assets);
beq = 1;
lb = zeros(n_assets, 1);
ub = ones(n_assets, 1);
m = 40; % Cardinality (number of assets in portfolio)

% Constraint matrices (Cx + d <= 0)
C = [Aeq; -Aeq; eye(n_assets); -eye(n_assets)];
d = [-beq; beq; -ub; lb]; % *** Corrected bounds ***

% Initial point with projection
x0 = ones(n_assets,1)/n_assets;
x0 = max(lb, min(x0, ub));  % Clamp to [0,1]
x0 = x0 / sum(x0);          % Renormalize

% Check feasibility with tolerance
tolerance = 1e-10;
violations = C*x0 + d;
if any(violations > tolerance)
    fprintf('Maximum constraint violation: %.2e\n', max(violations));
    error('Initial point violates constraints beyond tolerance');
end

% SNN solver parameters
t_end = 200;
k0 = 0.01;
k1 = 0.01;

% In SNN.m
% Replace snn_solver call with:
[t, X] = snn_solver(A, b, C, d, t_end, x0, k0, k1);

% Remove post-processing cardinality enforcement (already handled in solver)
w_snn = X(end, :)';

% Remove negligible weights for cardinality
threshold = 1e-4;
w_snn(w_snn < threshold) = 0;

% Enforce cardinality: keep only the largest m weights, renormalize
[~, idx] = sort(w_snn, 'descend');
w_card = zeros(size(w_snn));
w_card(idx(1:min(m, sum(w_snn>0)))) = w_snn(idx(1:min(m, sum(w_snn>0))));
w_card = w_card / sum(w_card);

% Portfolio metrics
port_return = mu' * w_card;
port_risk = sqrt(w_card' * sigma * w_card);
sharpe_ratio = port_return / port_risk;

fprintf('Portfolio Return: %.4f\n', port_return);
fprintf('Portfolio Risk:   %.4f\n', port_risk);
fprintf('Sharpe Ratio:     %.4f\n', sharpe_ratio);

% Plot weight trajectories
figure;
plot(t, X, 'LineWidth', 1.1);
xlabel('Time');
ylabel('Portfolio Weights');
title('SNN Portfolio Optimization Trajectory');
grid on;

% Plot final portfolio weights
figure;
bar(w_card, 'FaceColor', [0.2 0.6 0.8]);
xlabel('Asset Index');
ylabel('Weight');
title('Optimized Portfolio Weights (SNN)');
grid on;

% Plot cumulative sum of weights (should be 1)
figure;
plot(cumsum(w_card), 'r', 'LineWidth', 1.2);
xlabel('Asset Index (sorted)');
ylabel('Cumulative Weight');
title('Cumulative Sum of Portfolio Weights');
grid on;

fprintf('Plot generated with %d time points\n', length(t));
