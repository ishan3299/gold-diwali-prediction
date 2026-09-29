import numpy as np

def run_monte_carlo(current_price, days_to_diwali, drift, vol, num_simulations=10000):
    """
    Run Geometric Brownian Motion monte carlo simulation to get price distribution.
    """
    dt = 1/252
    
    # Calculate daily drift and vol
    daily_drift = drift / 252
    daily_vol = vol / np.sqrt(252)
    
    simulations = np.zeros((num_simulations, int(days_to_diwali) + 1))
    simulations[:, 0] = current_price
    
    for t in range(1, int(days_to_diwali) + 1):
        Z = np.random.standard_normal(num_simulations)
        simulations[:, t] = simulations[:, t-1] * np.exp((daily_drift - 0.5 * daily_vol**2) * 1 + daily_vol * Z)
        
    final_prices = simulations[:, -1]
    
    results = {
        'P10': np.percentile(final_prices, 10),
        'P25': np.percentile(final_prices, 25),
        'P50': np.percentile(final_prices, 50),
        'P75': np.percentile(final_prices, 75),
        'P90': np.percentile(final_prices, 90),
        'simulations': simulations
    }
    
    return results
