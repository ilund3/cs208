# Starter code for Homework 2.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Problem setup

# Update to point to the dataset on your machine
data: pd.DataFrame = pd.read_csv("https://raw.githubusercontent.com/opendp/cs208/refs/heads/main/spring2025/data/fake_healthcare_dataset_sample100.csv")

# names of public identifier columns
pub = ["age", "sex", "blood", "admission"]

# variable to reconstruct
target = "result"

# Gaussian Elimination Solver Import
from LinAlg import gaussian_solve

def execute_subsetsums_exact(predicates):
    """Count the number of patients that satisfy each predicate.
    Resembles a public query interface on a sequestered dataset.
    Computed as in equation (1).

    :param predicates: a list of predicates on the public variables
    :returns a 1-d np.ndarray of exact answers the subset sum queries"""
    return data[target].values @ np.stack([pred(data) for pred in predicates], axis=1)

# defense functions
def make_random_predicate():
    prime = 2003
    desc = np.random.randint(prime, size=len(pub))
    def predicate(data):
        return ((data[pub].values @ desc) % (prime + 1) % 2).astype(bool)
    return predicate

def execute_subsetsums_round(R, predicates):
    sums_exact = execute_subsetsums_exact(predicates)
    return np.floor(sums_exact / R) * R

def execute_subsetsums_noise(sigma, predicates):
    sums_exact = execute_subsetsums_exact(predicates)
    return sums_exact + np.random.normal(loc=0.0, scale=sigma, size=sums_exact.shape)

def execute_subsetsums_sample(t, predicates):
    subsample_indices = np.random.choice(len(data), size=t, replace=False)
    subsample_data = data.iloc[subsample_indices]
    
    result = []
    for func in predicates:
        mask = func(subsample_data).astype(bool)
        subset_sum = subsample_data[target].values[mask].sum()
        result.append((subset_sum * (len(data) / t)))
    return np.array(result)


# TODO: Write the reconstruction function!
def reconstruction_attack(data_pub, predicates, answers):
    """Reconstructs a target column based on the `answers` to queries about `data`.

    :param data_pub: data of length n consisting of public identifiers
    :param predicates: a list of k predicate functions
    :param answers: a list of k answers to a query on data filtered by the k predicates
    :return 1-dimensional boolean ndarray"""
    n = len(data_pub)
    k = len(predicates)

    # Build T (k x n) with T[j,i] = 1 if row i is included by predicate j
    T_bool = [pred(data_pub).astype(int) for pred in predicates]
    T = np.stack(T_bool, axis=0)  # shape (k, n)

    y_mod2 = (np.round(answers) % 2).astype(int)

    # Solve T x = y (mod 2)
    x_hat = gaussian_solve(T, y_mod2)
    return x_hat if x_hat is not None else np.zeros(n, dtype=int)

if __name__ == "__main__":
    # Run reconstruction with exact answers
    random_preds = [make_random_predicate() for _ in range(2 * len(data))]
    exact_answers = execute_subsetsums_exact(random_preds)
    x_hat = reconstruction_attack(data[pub], random_preds, exact_answers)
    x_true = data[target].values
    print("No defense reconstruction success: ",
          np.mean(x_hat == x_true))
    
    # Initial Testing of Rounding Defense
    R = 1
    round_answers = execute_subsetsums_round(R, random_preds)
    print("Using R=", R)
    print("RMSE:", np.sqrt(np.mean((round_answers - exact_answers)**2)))

    x_hat_rounded = reconstruction_attack(data[pub], random_preds, round_answers)
    print("Reconstruction success under rounding:",
          np.mean(x_hat_rounded == x_true))
    
    # Testing Noise Defense
    sigma = 0.0
    noise_answers = execute_subsetsums_noise(sigma, random_preds)
    print("Noise under sigma=", sigma)
    print("RMSE:", np.sqrt(np.mean((noise_answers - exact_answers)**2)))

    x_hat_noisy = reconstruction_attack(data[pub], random_preds, noise_answers)
    print("Reconstruction success under noise:",
          np.mean(x_hat_noisy == x_true))
    
    # Testing Sampling Defense
    t = 100
    sample_answers = execute_subsetsums_sample(t, random_preds)
    print("Sampling under t =", t)
    print("RMSE:", np.sqrt(np.mean((sample_answers - exact_answers)**2)))

    x_hat_sample = reconstruction_attack(data[pub], random_preds, sample_answers)
    print("Reconstruction success under sampling:",
          np.mean(x_hat_sample == x_true))
    
semi_blue = (0/255, 90/255, 239/255, 200/255)
semi_red = (239/255, 90/255, 0/255, 200/255)

def run_experiment(R, sigma, t, num_trials=10):
    # Runs the test for multiple trials to be used in graph
    rmse_list = []
    success_list = []
    n = len(data)

    for _ in range(num_trials):
        random_preds = [make_random_predicate() for _ in range(n)]
        exact_answers = execute_subsetsums_exact(random_preds)

        # Apply defense
        if R is not None:
            answers = execute_subsetsums_round(R, random_preds)
        elif sigma is not None:
            answers = execute_subsetsums_noise(sigma, random_preds)
        elif t is not None:
            answers = execute_subsetsums_sample(t, random_preds)

        # Find RMSE
        rmse = np.sqrt(np.mean((answers - exact_answers) ** 2))
        rmse_list.append(rmse)

        # Run attack and get reconstruction success
        x_hat = reconstruction_attack(data[pub], random_preds, answers)
        x_true = data[target].values
        success = np.mean(x_hat == x_true)
        success_list.append(success)

    return np.mean(rmse_list), np.mean(success_list)

# Function to vary parameter and plot results
def vary_parameter(param_name, values):
    """
    Varies a single parameter (R, σ, or t) and plots RMSE & reconstruction success.
    """
    rmse_results = []
    success_results = []

    for value in values:
        if param_name == "R":
            rmse, success = run_experiment(R=value, sigma=None, t=None)
        elif param_name == "sigma":
            rmse, success = run_experiment(R=None, sigma=value, t=None)
        elif param_name == "t":
            rmse, success = run_experiment(R=None, sigma=None, t=value)
        else:
            raise ValueError("Invalid parameter name.")

        rmse_results.append(rmse)
        success_results.append(success)

    fig, ax1 = plt.subplots()

    ax1.set_xlabel(param_name)
    ax1.set_ylabel("Reconstruction Success", color=semi_blue)
    ax1.plot(values, success_results, color=semi_blue, linewidth=1.5, label="Reconstruction Success")

    ax2 = ax1.twinx()
    ax2.set_ylabel("RMSE", color=semi_red)
    ax2.plot(values, rmse_results, color=semi_red, linewidth=1.5, label="RMSE")

    ax1.axhline(y=1, color="black", linestyle="dashed")

    leg_x = values[int(len(values) * 0.7)]
    ax1.plot([leg_x, leg_x * 0.9], [0.2, 0.2], color=semi_red, linewidth=2)
    ax1.plot([leg_x, leg_x * 0.9], [0.3, 0.3], color=semi_blue, linewidth=2)
    ax1.text(leg_x, 0.2, "RMSE", verticalalignment="bottom", fontsize=10, color="black")
    ax1.text(leg_x, 0.3, "Reconstruction Success", verticalalignment="bottom", fontsize=10, color="black")

    fig.tight_layout()
    plt.title(f"Impact of {param_name} on Attack Performance")
    plt.show()

# Run experiments and plot for each parameter
vary_parameter("R", values=range(1, 101, 5))
vary_parameter("sigma", values=np.linspace(1, 101, 5))
vary_parameter("t", values=range(5, 101, 5))