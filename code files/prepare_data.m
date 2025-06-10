%% prepare_data.m - Updated Version 2.0 (June 2025)
% Data preparation for SNN Portfolio Optimization
% This script loads stock price data, calculates returns,
% computes mean and covariance, and handles data cleaning

clearvars; clc; close all;

%% Configuration
% Set data parameters
start_date = '2017-01-01';
end_date = '2025-05-31';  % Current date
window_years = 8;         % 8-year analysis window
min_history_days = 252;   % Minimum days of data required (1 trading year)

% Set file paths
input_prices_file = 'data\processed_prices.csv';
output_file = 'portfolio_data.mat';

%% Data Loading
fprintf('Loading price data from %s...\n', input_prices_file);

try
    % Attempt to load CSV with prices
    price_data = readtable(input_prices_file, 'VariableNamingRule', 'preserve');  % Added parameter
    
    % Extract date column and convert to datetime if needed
    dates = price_data{:,1};
    if ~isa(dates, 'datetime')
        if iscell(dates)
            dates = datetime(dates, 'InputFormat', 'yyyy-MM-dd');
        elseif isnumeric(dates)
            %dates = datetime(datestr(dates));
            dates = datetime(dates, 'ConvertFrom', 'datenum');
        end
    end
    
    % Extract prices (all columns except the first date column)
    prices = price_data{:, 2:end};
    symbols = price_data.Properties.VariableNames(2:end);
    
    fprintf('Successfully loaded data with %d days and %d stocks\n', size(prices, 1), size(prices, 2));
catch exception
    fprintf('Error loading CSV: %s\n', exception.message);
    error('Failed to load price data. Please check the file format.');
end

%% Data Cleaning
fprintf('Cleaning price data...\n');

% Check for negative or zero prices (invalid values)
invalid_prices = (prices <= 0);
if any(invalid_prices(:))
    fprintf('Warning: Found %d negative or zero prices. Replacing with NaN.\n', sum(invalid_prices(:)));
    prices(invalid_prices) = NaN;
end

% Fill missing values using forward fill
prices_filled = fillmissing(prices, 'previous');

% Check if any columns still have NaNs at the beginning
nan_cols = any(isnan(prices_filled), 1);
if any(nan_cols)
    fprintf('Warning: %d stocks have missing values at the beginning. Applying backward fill.\n', sum(nan_cols));
    prices_filled = fillmissing(prices_filled, 'nearest');
end

% Check for stocks with too few valid days
valid_days_per_stock = sum(~isnan(prices), 1);
insufficient_history = valid_days_per_stock < min_history_days;

if any(insufficient_history)
    fprintf('Warning: Removing %d stocks with insufficient history (<252 days)\n', sum(insufficient_history));
    prices_filled = prices_filled(:, ~insufficient_history);
    symbols = symbols(~insufficient_history);
end

%% Calculate Returns
fprintf('Calculating daily returns...\n');

% Compute percentage returns
returns = diff(prices_filled) ./ prices_filled(1:end-1, :);

% Check for extreme returns (potential data errors)
extreme_returns_threshold = 0.5; % 50% daily change is suspicious
extreme_returns = abs(returns) > extreme_returns_threshold;

% Improved extreme returns handling
if any(extreme_returns(:))
    fprintf('Warning: Found %d extreme daily returns (>50%%).\n', sum(extreme_returns(:)));
    fprintf('Positions: ');
    [row, col] = find(extreme_returns);
    disp([row, col, returns(sub2ind(size(returns), row, col))]);
    
    % Uncomment below to auto-handle extremes
    % returns(extreme_returns) = NaN;
    % returns = fillmissing(returns, 'movmedian', 5);
end

%% Statistical Analysis
fprintf('Computing statistics...\n');

% Calculate mean returns (annualized)
mean_ret = mean(returns, 1, 'omitnan')' * 252;

% Calculate covariance matrix (annualized)
cov_mat = cov(returns, 'partialrows') * 252;

%% Save processed data
fprintf('Saving processed data to %s...\n', output_file);

% Create and save metadata
metadata = struct();
metadata.start_date = dates(1);
metadata.end_date = dates(end);
metadata.n_days = size(returns, 1);
metadata.n_stocks = size(returns, 2);
metadata.symbols = symbols;
metadata.processing_date = datetime('now');
metadata.window_years = window_years;

% Save data to MAT file
save(output_file, 'returns', 'mean_ret', 'cov_mat', 'symbols', 'metadata');

fprintf('Data processing complete! Ready for SNN portfolio optimization.\n');

%% Optional: Create sector data (if available)
try
    % Try to load sector classification data
    % This is optional and depends on data availability
    fprintf('Attempting to load sector classification data...\n');
    
    % Example: Load from CSV with columns: Symbol, SectorID, SectorName
    sector_table = readtable('data\sector_classification.csv');
    
    % Match sectors to our symbols
    [is_member, symbol_idx] = ismember(symbols, sector_table.Symbol);
    
    if sum(is_member) < length(symbols)
        fprintf('Warning: Sector information missing for %d stocks\n', length(symbols) - sum(is_member));
    end
    
    % Create sector ID vector (numeric codes)
    sector_info = zeros(length(symbols), 1);
    sector_info(is_member) = sector_table.SectorID(symbol_idx(is_member));
    
    % Create sector names cell array
    sector_names = cell(max(sector_table.SectorID), 1);
    for i = 1:max(sector_table.SectorID)
        mask = sector_table.SectorID == i;
        if any(mask)
            sector_names{i} = sector_table.SectorName{find(mask, 1)};
        else
            sector_names{i} = sprintf('Unknown Sector %d', i);
        end
    end
    
    % Save sector data
    save('sector_data.mat', 'sector_info', 'sector_names');
    fprintf('Sector data saved to sector_data.mat\n');
catch exception
    fprintf('Note: No sector classification data available. Proceeding without sector constraints.\n');
    fprintf('Error details: %s\n', exception.message);
end

%% Run this in command window to inspect problematic stocks
load('portfolio_data.mat');
problem_stocks = symbols(unique(col));
disp('Stocks with extreme returns:');
disp(problem_stocks');

%% Optional: Data Visualization
visualize_data = true;

if visualize_data
    fprintf('Generating data visualizations...\n');
    
    % Calculate correlation matrix if not already present
    if ~exist('corr_mat', 'var')
        stddev = sqrt(diag(cov_mat));
        corr_mat = cov_mat ./ (stddev * stddev');
    end

    % Create figure for returns distribution
    figure('Name', 'Returns Distribution', 'Position', [100, 100, 1200, 800]);
    
    % Subplot 1: Histogram of mean returns
    subplot(2, 2, 1);
    histogram(mean_ret, 30);
    title('Distribution of Annualized Mean Returns');
    xlabel('Mean Return');
    ylabel('Frequency');
    grid on;
    
    % Subplot 2: Histogram of volatilities
    subplot(2, 2, 2);
    volatilities = sqrt(diag(cov_mat)) * 100;  % Convert to percentage
    histogram(volatilities, 30);
    title('Distribution of Annualized Volatilities (%)');
    xlabel('Volatility (%)');
    ylabel('Frequency');
    grid on;
    
    % Subplot 3: Return vs. Volatility Scatter Plot
    subplot(2, 2, 3);
    scatter(volatilities, mean_ret * 100, 25, 'filled');
    title('Return vs. Risk Profile');
    xlabel('Volatility (%)');
    ylabel('Mean Return (%)');
    grid on;
    
    % Add linear fit line
    hold on;
    p = polyfit(volatilities, mean_ret * 100, 1);
    x_range = linspace(min(volatilities), max(volatilities), 100);
    y_fit = polyval(p, x_range);
    plot(x_range, y_fit, 'r-', 'LineWidth', 2);
    text(min(volatilities) + 2, max(mean_ret * 100) - 2, ...
        sprintf('Slope: %.4f', p(1)), 'FontSize', 10);
    hold off;
    
    % Subplot 4: Correlation Heatmap
    subplot(2, 2, 4);
    imagesc(corr_mat);  % Now using pre-computed corr_mat
    colorbar;
    title('Correlation Matrix Heatmap');
    axis square;
    colormap('jet');
    
    % Save figure
    saveas(gcf, 'data_statistics.png');
    fprintf('Data visualization saved to data_statistics.png\n');
end
