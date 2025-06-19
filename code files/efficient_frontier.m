function [risk, ret] = efficient_frontier(mean_ret, cov_mat, n_points)
% Compute efficient frontier using projected gradient descent (no toolboxes)
n_assets = length(mean_ret);
lb = zeros(n_assets, 1);
ub = ones(n_assets, 1);
ret = linspace(min(mean_ret), max(mean_ret), n_points);
risk = zeros(size(ret));
for i = 1:n_points
    target_ret = ret(i);
    w = ones(n_assets, 1) / n_assets;
    lr = 0.01;
    for iter = 1:500
        grad = 2 * cov_mat * w;
        w = w - lr * grad;
        w(w<0) = 0; w(w>1) = 1;
        w = w / sum(w + 1e-12);
        curr_ret = mean_ret' * w;
        if abs(curr_ret - target_ret) > 1e-4
            w = w + (target_ret - curr_ret) * (mean_ret / sum(mean_ret.^2));
            w(w<0) = 0; w(w>1) = 1;
            w = w / sum(w + 1e-12);
        end
    end
    risk(i) = sqrt(w' * cov_mat * w);
end
end
