# watsonxChallenge

# Insurance Liability Simulation

This repository contains a Python script for simulating future insurance liabilities using Monte Carlo simulations. The script is designed for a code challenge and demonstrates the use of stochastic modeling to predict future liabilities based on historical claims data.

## Project Structure

- `insurance_liability_simulation.py`: The main Python script for simulating insurance liabilities.
- `README.md`: This file, providing an overview of the project and instructions for setup.

## Setup Instructions

### Prerequisites

- Python 3.6 or higher
- NumPy
- Pandas
- Matplotlib
- SciPy

### Installation

1. Clone the repository to your local machine using SSH:

    ```sh
    git clone git@github.com:joe24ibm/watsonxChallenge.git
    cd watsonxChallenge
    ```

2. (Optional) Create a virtual environment and activate it:

    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:

    ```sh
    pip install numpy pandas matplotlib scipy
    ```

## Running the Simulation

1. Navigate to the project directory:

    ```sh
    cd ~/coding_projects/watsonxChallenge
    ```

2. Run the simulation script:

    ```sh
    python insurance_liability_simulation.py
    ```

The script will generate plots showing the simulated future liabilities and the probability of annual liabilities exceeding a specified threshold.

## Explanation of `insurance_liability_simulation.py`

The script includes the following functions:

- `load_historical_data()`: Generates random historical claims data.
- `preprocess_data()`: Converts the historical claims data into a Pandas DataFrame.
- `monte_carlo_liability_simulation()`: Simulates future liabilities using Monte Carlo simulation based on historical claims data.
- `analyze_simulation_results()`: Calculates the probability of future liabilities exceeding a specified threshold.
- `visualize_simulation_results()`: Generates plots to visualize the simulated future liabilities and the probability of exceeding the threshold.
- `main()`: Orchestrates the entire process from loading data to visualizing results and printing probabilities.

## License

This project is for demonstration purposes and does not include a specific license.

## Contact

For any questions or issues, please contact [your email or GitHub username].
