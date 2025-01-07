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
