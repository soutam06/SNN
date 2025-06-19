%% prepare_data.m - Updated Version 2.0 (June 2025)
% Data preparation for SNN Portfolio Optimization

clearvars; clc; close all;

%% Configuration
start_date = '2017-01-01';
end_date = '2025-05-31';
window_years = 8;
min_history_days = 252;
input_prices_file = 'data\processed_prices.csv';
output_file = 'portfolio_data.mat';

%% Data Loading
fprintf('Loading price data from %s...\n', input_prices_file);
try
    price_data = readtable(input_prices_file, 'VariableNamingRule', 'preserve');
    dates = price_data{:,1};
    if ~isa(dates, 'datetime')
        if iscell(dates)
            dates = datetime(dates, 'InputFormat', 'yyyy-MM-dd');
        elseif isnumeric(dates)
            dates = datetime(dates, 'ConvertFrom', 'datenum');
        end
    end
    prices = price_data{:, 2:end};
    symbols = price_data.Properties.VariableNames(2:end);
    fprintf('Successfully loaded data with %d days and %d stocks\n', size(prices, 1), size(prices, 2));
catch exception
    fprintf('Error loading CSV: %s\n', exception.message);
    error('Failed to load price data. Please check the file format.');
end

%% Data Cleaning
fprintf('Cleaning price data...\n');
invalid_prices = (prices <= 0);
if any(invalid_prices(:))
    fprintf('Warning: Found %d negative or zero prices. Replacing with NaN.\n', sum(invalid_prices(:)));
    prices(invalid_prices) = NaN;
end
prices_filled = fillmissing(prices, 'previous');
nan_cols = any(isnan(prices_filled), 1);
if any(nan_cols)
    fprintf('Warning: %d stocks have missing values at the beginning. Applying backward fill.\n', sum(nan_cols));
    prices_filled = fillmissing(prices_filled, 'nearest');
end
valid_days_per_stock = sum(~isnan(prices), 1);
insufficient_history = valid_days_per_stock < min_history_days;
if any(insufficient_history)
    fprintf('Warning: Removing %d stocks with insufficient history (<252 days)\n', sum(insufficient_history));
    prices_filled = prices_filled(:, ~insufficient_history);
    symbols = symbols(~insufficient_history);
end

%% Calculate Returns
fprintf('Calculating daily returns...\n');
returns = diff(prices_filled) ./ prices_filled(1:end-1, :);
extreme_returns_threshold = 0.5;
extreme_returns = abs(returns) > extreme_returns_threshold;
if any(extreme_returns(:))
    fprintf('Warning: Found %d extreme daily returns (>50%%).\n', sum(extreme_returns(:)));
    [row, col] = find(extreme_returns);
    disp([row, col, returns(sub2ind(size(returns), row, col))]);
end

%% Statistical Analysis
fprintf('Computing statistics...\n');
mean_ret = mean(returns, 1, 'omitnan')' * 252;
cov_mat = cov(returns, 'partialrows') * 252;

%% Save processed data
fprintf('Saving processed data to %s...\n', output_file);
metadata = struct();
metadata.start_date = dates(1);
metadata.end_date = dates(end);
metadata.n_days = size(returns, 1);
metadata.n_stocks = size(returns, 2);
metadata.symbols = symbols;
metadata.processing_date = datetime('now');
metadata.window_years = window_years;
save(output_file, 'returns', 'mean_ret', 'cov_mat', 'symbols', 'metadata');
fprintf('Data processing complete! Ready for SNN portfolio optimization.\n');

%% Create sector data
try
    fprintf('Attempting to load sector classification data...\n');
    sector_table = readtable('data\sector_classification.csv');
    [is_member, symbol_idx] = ismember(symbols, sector_table.Symbol);
    if sum(is_member) < length(symbols)
        fprintf('Warning: Sector information missing for %d stocks\n', length(symbols) - sum(is_member));
    end
    sector_info = zeros(length(symbols), 1);
    sector_info(is_member) = sector_table.SectorID(symbol_idx(is_member));
    sector_names = cell(max(sector_table.SectorID), 1);
    for i = 1:max(sector_table.SectorID)
        mask = sector_table.SectorID == i;
        if any(mask)
            sector_names{i} = sector_table.SectorName{find(mask, 1)};
        else
            sector_names{i} = sprintf('Unknown Sector %d', i);
        end
    end
    save('sector_data.mat', 'sector_info', 'sector_names');
    fprintf('Sector data saved to sector_data.mat\n');
catch exception
    fprintf('Note: No sector classification data available. Proceeding without sector constraints.\n');
    fprintf('Error details: %s\n', exception.message);
end

%% Enhanced Data Visualization
visualize_data = true;
if visualize_data
    fprintf('Generating publication-quality data visualizations...\n');
    fig = figure('Units','inches', 'Position',[0 0 12 9], 'Color','w');
    set(fig, 'PaperPositionMode', 'auto');
    subplot(2,1,1)
    histogram(mean_ret*100, 30, 'FaceColor',[0.2 0.6 0.8], 'EdgeColor','none')
    title('Annualized Mean Returns','FontWeight','bold','FontSize',16)
    xlabel('Return (%)','FontSize',14)
    ylabel('Frequency','FontSize',14)
    set(gca, 'FontSize', 14)
    grid on
    subplot(2,1,2)
    volatilities = sqrt(diag(cov_mat)) * 100;
    histogram(volatilities, 30, 'FaceColor',[0.8 0.4 0.2], 'EdgeColor','none')
    title('Annualized Volatilities','FontWeight','bold','FontSize',16)
    xlabel('Volatility (%)','FontSize',14)
    ylabel('Frequency','FontSize',14)
    set(gca, 'FontSize', 14)
    grid on
    print(fig, 'data_statistics.pdf','-dpdf','-bestfit')
end
