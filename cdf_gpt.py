import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

def load_historical_data(seed: int = 42, years: int = 10, claims_per_year: int = 100, scale: float = 1000.0) -> np.ndarray:
    """Generate historical claims data using an exponential distribution."""
    np.random.seed(seed)
    historical_claims = np.random.exponential(scale=scale, size=(years, claims_per_year))
    return historical_claims

def preprocess_data(historical_claims: np.ndarray) -> pd.DataFrame:
    """Convert historical claims data to a Pandas DataFrame."""
    claims_df = pd.DataFrame(historical_claims, columns=[f'Claim_{i+1}' for i in range(historical_claims.shape[1])])
    return claims_df

def calculate_cdf_threshold(mean_claims: float, std_claims: float, target_probability: float = 0.05) -> float:
    """Calculate the threshold using the CDF of the normal distribution."""
    z_score = norm.ppf(1 - target_probability)
    threshold = mean_claims + z_score * std_claims
    return threshold

def visualize_results(mean_claims: float, std_claims: float, threshold_liability: float, simulated_liabilities: np.ndarray):
    """Visualize the simulation results."""
    plt.figure(figsize=(15, 6))
    plt.plot(simulated_liabilities)
    plt.axhline(y=threshold_liability, color='r', linestyle='--', label=f'Threshold: {threshold_liability:.2f}')
    plt.title('Simulation of Future Liabilities with CDF-Based Threshold')
    plt.xlabel('Years')
    plt.ylabel('Liability Amount')
    plt.legend()
    plt.show()

def main():
    historical_claims = load_historical_data()
    claims_df = preprocess_data(historical_claims)
    
    annual_claims = claims_df.mean(axis=1)
    mean_claims = annual_claims.mean()
    std_claims = annual_claims.std()
    
    # Calculate the threshold using the CDF method
    threshold_liability = calculate_cdf_threshold(mean_claims, std_claims, target_probability=0.05)
    
    # Simulate liabilities for visualization purposes (using the same method as Monte Carlo for comparison)
    num_simulations = 1000
    num_years = 10
    simulated_liabilities = np.zeros((num_years, num_simulations))
    for t in range(num_years):
        simulated_liabilities[t] = np.random.normal(mean_claims, std_claims, num_simulations)
    
    visualize_results(mean_claims, std_claims, threshold_liability, simulated_liabilities)
    
    print(f"Determined CDF-based threshold for 5% exceedance probability: {threshold_liability:.2f}")

if __name__ == "__main__":
    main()
