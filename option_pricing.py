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

from app import (
    generate_random_numbers,
    simulate_stock_prices,
    simulate_stock_price_paths,
    calculate_payoffs,
    discount_payoff,
    validate_inputs,
    monte_carlo_pricing,
    plot_stock_price_distribution,
    plot_sample_paths,
    plot_payoff_distribution,
    validate_put_call_parity
)

import streamlit as st
import matplotlib.pyplot as plt

# Title and Description for Option Pricing using Monte Carlo
st.title('Monte Carlo Option Pricing')
st.markdown('This app allows you to calculate the price of European call and put options using the Monte Carlo simulation method. You can also visualize stock price paths and distributions.')

# User Inputs for Option Parameters
st.sidebar.header("Option Parameters")
s0 = st.sidebar.number_input("Initial Stock Price (S0)", value=100.0, min_value=0.01)
K = st.sidebar.number_input("Strike Price (K)", value=100.0, min_value=0.01)
T = st.sidebar.number_input("Time to Maturity (T, in years)", value=1.0, min_value=0.01)
r = st.sidebar.number_input("Risk-Free Rate (r)", value=0.05, min_value=0.0, max_value=1.0)
sigma = st.sidebar.number_input("Volatility (sigma)", value=0.2, min_value=0.0, max_value=1.0)
n_simulations = st.sidebar.number_input("Number of Simulations", value=100000, min_value=1, step=1000)
option_type = st.sidebar.selectbox("Option Type", options=["Call", "Put"]).lower()

# Button to Trigger the Monte Carlo Simulation
if st.sidebar.button("Run Simulation"):
    try:
        # Run Monte Carlo Pricing
        price = monte_carlo_pricing(s0, K, T, r, sigma, n_simulations, option_type)
        st.success(f"The estimated {option_type} option price is: ${price:.2f}")

        # Simulate Stock Price Paths
        paths = simulate_stock_price_paths(s0, r, sigma, T, n_simulations, n_steps=252)

        # Visualizations
        st.subheader("Stock Price Distribution at Maturity")
        fig1, ax1 = plt.subplots()
        plot_stock_price_distribution(paths[:, -1])
        st.pyplot(fig1)

        st.subheader("Sample Stock Price Paths")
        fig2, ax2 = plt.subplots()
        plot_sample_paths(paths, n_samples=5)
        st.pyplot(fig2)

    except Exception as e:
        st.error(f"An error occurred: {e}")
