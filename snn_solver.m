function [t, X] = qibas_solver(A, b, C, d, t_end, x0, k0, k1)
    % QIBAS parameters from paper
    delta0 = 1; 
    d0 = 0.99;
    min_sgn = -0.5; 
    max_sgn = 0.5;
    
    % Initialize best solution
    x_min = x0;
    f_min = 0.5*x0'*A*x0 + b'*x0; % Paper's Eq. 49
    
    for iter = 1:10000
        % Generate random direction vector
        b_dir = 2*rand(size(x0)) - 1;
        b_dir = b_dir/norm(b_dir);
        
        % Quadratic interpolation for each dimension
        for i = 1:length(x0)
            x_r = x0 + d0*b_dir;
            x_l = x0 - d0*b_dir;
            
            f_r = 0.5*x_r'*A*x_r + b'*x_r;
            f_l = 0.5*x_l'*A*x_l + b'*x_l;
            f_c = 0.5*x0'*A*x0 + b'*x0;
            
            % Quadratic coefficients (Eq. 40-41)
            numerator = f_r*(x0(i)-x_l(i)) - f_l*(x0(i)-x_r(i)) - f_c*(x_r(i)-x_l(i));
            denominator = (x_l(i)-x_r(i))*(x0(i)^2 + x0(i)*x_l(i) + x0(i)*x_r(i) - x_l(i)*x_r(i));
            a0 = numerator/denominator;
            
            if a0 > 0
                x0(i) = -b_dir(i)/(2*a0); % QI update
            else
                % BAS update with activation
                sgn_val = max(min(f_r - f_l, max_sgn), min_sgn);
                x0(i) = x0(i) + delta0*b_dir(i)*sign(sgn_val);
            end
        end
        
        % Project to constraints (paper's Eq. 17-22)
        x0 = max(0, min(x0, 1)); % Bounds
        x0 = x0/sum(x0); % Budget constraint
        [~,idx] = sort(x0,'descend');
        x0(idx(41:end)) = 0; % Cardinality m=40
        x0 = x0/sum(x0);
        
        % Update best solution
        current_f = 0.5*x0'*A*x0 + b'*x0;
        if current_f < f_min
            x_min = x0;
            f_min = current_f;
        end
        
        % Parameter decay (Eq. 29-30)
        delta0 = 0.95*delta0;
        d0 = 0.95*d0 + 0.01;
    end
    t = 1:10000; % Dummy time vector
    X = repmat(x_min',10000,1); % Best solution trajectory
end
