import numpy as np

def demand_forecasting(historical_sales, window_size=3):
    """
    Forecast future demand using a moving average method.

    Parameters:
    - historical_sales (list or numpy array): Historical sales data.
    - window_size (int): Size of the moving average window.

    Returns:
    - forecast (numpy array): Forecasted demand for future periods.
    """
    forecast = []
    num_periods = len(historical_sales)

    # Calculate moving average for each period
    for i in range(num_periods - window_size + 1):
        window_sales = historical_sales[i : i + window_size]
        average_sales = np.mean(window_sales)
        forecast.append(average_sales)

    # Fill in remaining forecast periods with the last observed value
    last_observed_value = historical_sales[-1]
    remaining_periods = num_periods - len(forecast)
    forecast.extend([last_observed_value] * remaining_periods)

    return np.array(forecast)

# Example usage:
historical_sales = [10, 15, 20, 25, 30]
forecast = demand_forecasting(historical_sales, window_size=3)
print("Forecasted demand:", forecast)
