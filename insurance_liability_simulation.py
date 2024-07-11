import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

# Load historical claims data
def load_historical_data():
    np.random.seed(42)
    years = 10
    claims_per_year = 100
    historical_claims = np.random.exponential(scale=1000, size=(years, claims_per_year))
    return historical_claims

# Preprocess data
def preprocess_data(historical_claims):
    claims_df = pd.DataFrame(historical_claims, columns=[f'Claim_{i+1}' for i in range(historical_claims.shape[1])])
    return claims_df

# Monte Carlo simulation for insurance liabilities
def monte_carlo_liability_simulation(claims_df, num_simulations, num_years):
    annual_claims = claims_df.mean(axis=1)
    mean_claims = annual_claims.mean()
    std_claims = annual_claims.std()
    
    # Debug prints
    print(f"Mean annual claims: {mean_claims}")
    print(f"Standard deviation of annual claims: {std_claims}")
    
    simulated_liabilities = np.zeros((num_years, num_simulations))
    for t in range(num_years):
        simulated_liabilities[t] = np.random.normal(mean_claims, std_claims, num_simulations)
    
    return simulated_liabilities

# Analyze simulation results
def analyze_simulation_results(simulated_liabilities, threshold_liability):
    liabilities_df = pd.DataFrame(simulated_liabilities)
    probability_above_threshold_liability = (liabilities_df > threshold_liability).mean(axis=1)
    
    # Debug prints
    print(f"Simulated liabilities:\n{liabilities_df}")
    print(f"Probability of exceeding threshold each year:\n{probability_above_threshold_liability}")
    
    return probability_above_threshold_liability

# Visualize simulation results
def visualize_simulation_results(simulated_liabilities, probability_above_threshold_liability):
    plt.figure(figsize=(15, 6))
    plt.plot(simulated_liabilities)
    plt.title('Monte Carlo Simulation of Future Liabilities')
    plt.xlabel('Years')
    plt.ylabel('Liability Amount')
    plt.show()
    
    plt.figure(figsize=(10, 6))
    plt.plot(probability_above_threshold_liability)
    plt.title('Probability of Annual Liabilities Exceeding Threshold')
    plt.xlabel('Years')
    plt.ylabel('Probability')
    plt.show()

def main():
    historical_claims = load_historical_data()
    claims_df = preprocess_data(historical_claims)
    
    num_simulations = 1000
    num_years = 10
    threshold_liability = 120000  # Adjust this threshold as needed
    
    simulated_liabilities = monte_carlo_liability_simulation(claims_df, num_simulations, num_years)
    
    probability_above_threshold_liability = analyze_simulation_results(simulated_liabilities, threshold_liability)
    
    visualize_simulation_results(simulated_liabilities, probability_above_threshold_liability)
    
    print(f"Probability of liabilities exceeding threshold in any given year: {probability_above_threshold_liability.mean() * 100:.2f}%")

if __name__ == "__main__":
    main()
