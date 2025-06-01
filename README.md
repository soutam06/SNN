# SNN Portfolio Optimization Framework
**Version 2.0 (June 2025)**

This repository contains a complete implementation of portfolio optimization using Spiking Neural Networks (SNNs) with cardinality constraints. The framework is designed for optimizing large-scale stock portfolios while enforcing practical constraints like cardinality limits, sector diversification, and transaction costs.

## Overview

Spiking Neural Networks (SNNs) are biologically-inspired computing models that process information through discrete events (spikes) rather than continuous values. This project applies SNNs to the challenging problem of portfolio optimization with cardinality constraints, which is NP-hard and difficult to solve using traditional methods.

The framework includes:

1. **Data Preparation** - Tools to download, clean, and prepare financial market data
2. **SNN Optimization** - A MATLAB implementation of portfolio optimization using SNNs
3. **Visualization** - Analysis and visualization of optimization results

## Key Features

- **Cardinality Constraints** - Limit the number of assets in your portfolio
- **Sector Diversification** - Enforce sector limits to avoid concentration risk
- **Transaction Costs** - Model realistic trading costs and turnover constraints
- **Adaptive Parameters** - Dynamic adjustment of risk aversion and neuron parameters
- **Multiple Initialization Methods** - Various techniques for initializing the neural population
- **Comprehensive Visualization** - Detailed portfolio analysis and optimization convergence plots

## Requirements

### MATLAB
- MATLAB R2021b or newer
- Financial Toolbox (optional, for benchmarking)
- Statistics and Machine Learning Toolbox

### Python (for data preparation)
- Python 3.8+
- pandas
- numpy
- yfinance
- matplotlib
- seaborn

## Usage

### 1. Data Preparation

#### Option 1: Using Python

```python
# Install dependencies
pip install pandas numpy yfinance matplotlib seaborn

# Run the data preparation script
python data_preparation.py
```

#### Option 2: Using MATLAB

```matlab
% Run the data preparation script
prepare_data
```

### 2. Portfolio Optimization

```matlab
% Run the main optimization script
SNN_PortfolioOptimization
```

### 3. Customizing Parameters

Edit the parameters in `SNN_PortfolioOptimization.m` to customize the optimization:

```matlab
params = struct(...
    'n_epochs', 200, ... % Number of training iterations
    'pop_size', 100, ... % Population size
    'tau', 0.8, ... % Decay rate for spike potentials
    'threshold', 1.0, ... % Initial firing threshold
    'threshold_decay', 0.98, ... % Threshold decay rate
    'min_threshold', 0.2, ... % Minimum threshold
    'cardinality', [30, 50], ... % Target stock selection range [min, max]
    'risk_aversion', 0.85, ... % Risk-reward balance [0-1]
    'learning_rate', 0.15, ... % Learning rate for neuron updates
    'noise_factor', 0.05, ... % Noise factor for exploration
    'init_method', 'xavier', ... % Neuron initialization method
    'decoding_method', 'volatility', ... % Weight decoding method
    'adaptive_risk_aversion', true, ... % Enable adaptive risk aversion
    'lateral_inhibition', true, ... % Enable lateral inhibition
    'adaptive_tau', true, ... % Enable adaptive tau
    'transaction_cost', 0.0025 ... % Transaction cost rate
);
```

## Implementation Details

### SNN Portfolio Solver Algorithm

The core of this framework is the `snn_portfolio_solver` function, which implements a Spiking Neural Network approach to portfolio optimization. Here's how it works:

1. **Initialization**: A population of spiking neurons is created, with each neuron representing a potential portfolio allocation.

2. **Iterative Optimization**: For each epoch:
   - Calculate neuron potentials based on expected returns and risks
   - Generate spikes when potentials exceed a threshold
   - Update neuron values based on spikes (Hebbian-like learning)
   - Decode the neuron population to portfolio weights
   - Evaluate portfolio performance (Sharpe ratio)
   - Apply cardinality and other constraints
   - Update the best solution if improved

3. **Adaptive Parameters**:
   - The threshold decays over time to allow more assets to be selected
   - Risk aversion can adapt dynamically during optimization
   - The tau parameter controlling temporal dynamics can also adapt

4. **Constraint Handling**:
   - Cardinality constraints limit the number of assets
   - Sector constraints ensure proper diversification
   - Transaction costs model realistic trading limitations

### Parameter Tuning Guidelines

- **Risk Aversion (0-1)**: Higher values prioritize return, lower values prioritize risk reduction
- **Population Size**: Larger populations explore more solutions but take longer
- **Number of Epochs**: More epochs allow better convergence but increase runtime
- **Learning Rate**: Controls how quickly neurons respond to spikes
- **Noise Factor**: Adds randomness to help escape local optima

## Files and Functions

- `data_preparation.py` - Python script for downloading and preparing stock data
- `prepare_data.m` - MATLAB script for data preparation
- `snn_portfolio_solver.m` - Core SNN implementation for portfolio optimization
- `SNN_PortfolioOptimization.m` - Main script that orchestrates the optimization process

## Input Data Format

### For CSV Price Data (`processed_prices.csv`)
- First column: Dates (YYYY-MM-DD format)
- Subsequent columns: Daily closing prices for each stock
- Column headers should be stock symbols

### For Sector Classification Data (`sector_classification.csv`)
- Must contain columns: 'Symbol', 'SectorID', 'Industry'
- 'SectorID' should be numeric values starting from 1
- 'Symbol' should match the symbols in your price data

## Troubleshooting

### Common Issues

1. **Missing Data Error**
   - Ensure your CSV files have the correct format
   - Check that date formats are consistent

2. **Dimension Mismatch**
   - This often occurs when there are NaN values in your data
   - Try running the data cleaning functions separately

3. **Out of Memory Error**
   - Reduce the number of stocks or the time period
   - Increase MATLAB's memory allocation

## References

1. Markowitz, H. (1952). Portfolio Selection. The Journal of Finance, 7(1), 77-91.
2. Chang, T.J., Meade, N., Beasley, J.E., & Sharaiha, Y.M. (2000). Heuristics for cardinality constrained portfolio optimisation. Computers & Operations Research, 27(13), 1271-1302.
3. Maass, W. (1997). Networks of spiking neurons: The third generation of neural network models. Neural Networks, 10(9), 1659-1671.
4. Reid, D., Hussain, A.J., & Tawfik, H. (2014). Financial time series prediction using spiking neural networks. PLoS ONE, 9(8).

## License

MIT License

## Contact

For questions or suggestions, please contact the author.

---

*Last updated: June 1, 2025*