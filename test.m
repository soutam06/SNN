% Run this in command window to inspect problematic stocks
load('portfolio_data.mat');
problem_stocks = symbols(unique(col));
disp('Stocks with extreme returns:');
disp(problem_stocks');
