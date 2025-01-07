import numpy as np
import matplotlib.pyplot as plt

# Generate random numbers for Monte Carlo simulation
def generate_random_numbers(n_simulations):
    Z = np.random.randn(n_simulations)
    return Z

# Simulate stock prices at maturity using Geometric Brownian Motion
def simulate_stock_prices(s0, r, sigma, T, Z):
    drift_term = (r - 0.5 * sigma ** 2) * T  # Risk-free growth and volatility adjustment
    random_term = sigma * np.sqrt(T) * Z     # Random price movement
    sT = s0 * np.exp(drift_term + random_term)
    return sT

# Simulate stock price paths over multiple time steps
def simulate_stock_price_paths(s0, r, sigma, T, n_simulations, n_steps):
    dt = T / n_steps
    paths = np.zeros((n_simulations, n_steps + 1))
    paths[:, 0] = s0

    for t in range(1, n_steps + 1):
        Z = np.random.randn(n_simulations)
        paths[:, t] = paths[:, t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z)

    return paths

# Calculate option payoffs
def calculate_payoffs(sT, K, option_type):
    option_type = option_type.lower()
    if option_type == 'call':
        return np.maximum(sT - K, 0)
    elif option_type == 'put':
        return np.maximum(K - sT, 0)
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")

# Discount payoffs to present value
def discount_payoff(payoffs, r, T):
    average_payoff = np.mean(payoffs)
    discounted_payoff = average_payoff * np.exp(-r * T)
    return discounted_payoff

# Validate input parameters
def validate_inputs(s0, K, T, r, sigma, n_simulations):
    if s0 <= 0 or K <= 0 or T <= 0 or r < 0 or sigma < 0 or n_simulations <= 0:
        raise ValueError("All input parameters must be positive and valid.")

# Monte Carlo pricing for options
def monte_carlo_pricing(s0, K, T, r, sigma, n_simulations, option_type):
    validate_inputs(s0, K, T, r, sigma, n_simulations)

    # Generate random numbers
    Z = generate_random_numbers(n_simulations)

    # Simulate stock prices at maturity
    stock_prices_at_maturity = simulate_stock_prices(s0, r, sigma, T, Z)

    # Calculate payoffs
    payoffs = calculate_payoffs(stock_prices_at_maturity, K, option_type)

    # Discount payoffs
    option_price = discount_payoff(payoffs, r, T)

    return option_price

# Plot histogram of stock price distribution at maturity
def plot_stock_price_distribution(sT):
    plt.hist(sT, bins=50, alpha=0.75)
    plt.title("Simulated Stock Prices at Maturity")
    plt.xlabel("Stock Price")
    plt.ylabel("Frequency")
    plt.show()

# Plot stock price paths
def plot_sample_paths(paths, n_samples=5):
    for i in range(min(n_samples, paths.shape[0])):
        plt.plot(paths[i])
    plt.title(f"Sample Stock Price Paths ({n_samples} paths)")
    plt.xlabel("Time Steps")
    plt.ylabel("Stock Price")
    plt.show()

# Plot payoff distribution
def plot_payoff_distribution(payoffs):
    plt.hist(payoffs, bins=50, alpha=0.75)
    plt.title("Payoff Distribution")
    plt.xlabel("Payoff")
    plt.ylabel("Frequency")
    plt.show()

# Validate put-call parity
def validate_put_call_parity(call_price, put_price, s0, K, r, T):
    parity_difference = (call_price - put_price) - (s0 - K * np.exp(-r * T))
    print(f"Put-Call Parity Difference: {parity_difference:.6f}")

# # Example usage
# if __name__ == "__main__":
#     # Parameters
#     s0 = 100       # Initial stock price
#     K = 100        # Strike price
#     T = 1          # Time to maturity (in years)
#     r = 0.05       # Risk-free rate (5%)
#     sigma = 0.2    # Volatility (20%)
#     n_simulations = 100000  # Number of simulations
#     n_steps = 252  # Number of time steps for paths

#     # Pricing call and put options
#     call_price = monte_carlo_pricing(s0, K, T, r, sigma, n_simulations, "call")
#     put_price = monte_carlo_pricing(s0, K, T, r, sigma, n_simulations, "put")
#     print(f"Call Option Price: ${call_price:.2f}")
#     print(f"Put Option Price: ${put_price:.2f}")

#     # Validate put-call parity
#     validate_put_call_parity(call_price, put_price, s0, K, r, T)

#     # Simulate stock prices
#     paths = simulate_stock_price_paths(s0, r, sigma, T, n_simulations, n_steps)
#     stock_prices_at_maturity = paths[:, -1]

#     # Visualizations
#     plot_stock_price_distribution(stock_prices_at_maturity)
#     plot_sample_paths(paths, n_samples=5)
#     payoffs = calculate_payoffs(stock_prices_at_maturity, K, "call")
#     plot_payoff_distribution(payoffs)

