import numpy as np
import matplotlib.pyplot as plt

def generate_dp_synthetic(bound, data_arr, privacy, grid_bins, synth_count=None):
    # data_arr is assumed to be an array of shape (N,2)
    N = data_arr.shape[0]
    if synth_count is None:
        synth_count = N

    # Create bin edges manually using a loop
    step = (2 * bound) / grid_bins
    bin_edges = []
    for i in range(grid_bins + 1):
        bin_edges.append(-bound + i * step)

    # Build histogram: use nested loops to assign each point to a bin
    hist_matrix = [[0 for _ in range(grid_bins)] for _ in range(grid_bins)]
    for pt in data_arr:
        x_val, y_val = pt[0], pt[1]
        # Determine bin index for x
        ix = 0
        while ix < grid_bins and not (bin_edges[ix] <= x_val < bin_edges[ix+1]):
            ix += 1
        if ix >= grid_bins:
            ix = grid_bins - 1
        # Determine bin index for y
        iy = 0
        while iy < grid_bins and not (bin_edges[iy] <= y_val < bin_edges[iy+1]):
            iy += 1
        if iy >= grid_bins:
            iy = grid_bins - 1
        hist_matrix[ix][iy] += 1

    # Add Laplace noise to each bin for differential privacy
    noisy_hist = []
    for i in range(grid_bins):
        row_noisy = []
        for j in range(grid_bins):
            noise = np.random.laplace(0, 1/privacy)
            noisy_val = hist_matrix[i][j] + noise
            if noisy_val < 0:
                noisy_val = 0
            row_noisy.append(noisy_val)
        noisy_hist.append(row_noisy)

    # Normalize the noisy histogram into probabilities manually
    tot = 0
    for i in range(grid_bins):
        for j in range(grid_bins):
            tot += noisy_hist[i][j]
    probs = []
    if tot > 0:
        for i in range(grid_bins):
            for j in range(grid_bins):
                probs.append(noisy_hist[i][j] / tot)
    else:
        for i in range(grid_bins * grid_bins):
            probs.append(1 / (grid_bins * grid_bins))

    # Build a cumulative distribution (CDF) for sampling bins
    cdf = []
    running = 0
    for p in probs:
        running += p
        cdf.append(running)

    # Sample bins for synthetic points using the CDF (inefficient but explicit)
    chosen_bins = []
    for _ in range(synth_count):
        r = np.random.rand()
        idx = 0
        for j, cp in enumerate(cdf):
            if r < cp:
                idx = j
                break
        chosen_bins.append(idx)

    # Generate synthetic data points uniformly within the chosen bin boundaries
    synthetic_pts = []
    for bin_idx in chosen_bins:
        # Determine grid position from the flattened index
        row_idx = bin_idx // grid_bins
        col_idx = bin_idx % grid_bins
        low_x = bin_edges[row_idx]
        high_x = bin_edges[row_idx+1]
        low_y = bin_edges[col_idx]
        high_y = bin_edges[col_idx+1]
        new_x = low_x + (high_x - low_x) * np.random.rand()
        new_y = low_y + (high_y - low_y) * np.random.rand()
        synthetic_pts.append([new_x, new_y])
        
    return np.array(synthetic_pts)

def compute_ols_slope(points):
    numerator = 0
    denominator = 0
    for pt in points:
        numerator += pt[0] * pt[1]
        denominator += pt[0] * pt[0]
    if denominator == 0:
        return 0
    return numerator / denominator

# Monte Carlo simulation to compare non-private OLS vs. DP synthetic-data OLS
sample_sizes = list(range(100, 5001, 100))
num_trials = 200

ols_biases = []
dp_biases = []
ols_stds = []
dp_stds = []

for n in sample_sizes:
    ols_slopes = []
    dp_slopes = []
    for _ in range(num_trials):
        # Generate dataset: x from Uniform[-0.5, 0.5] and y = x + noise, clipped to [-1,1]
        orig_points = []
        for _ in range(n):
            xi = np.random.uniform(-0.5, 0.5)
            yi = xi + np.random.normal(0, np.sqrt(0.02))
            if yi > 1:
                yi = 1
            if yi < -1:
                yi = -1
            orig_points.append([xi, yi])
        orig_points = np.array(orig_points)
        
        # Compute OLS slope on original data
        ols_val = compute_ols_slope(orig_points)
        ols_slopes.append(ols_val)
        
        # Generate DP synthetic dataset (using m = n) and compute its OLS slope
        dp_data = generate_dp_synthetic(1, orig_points, 0.1, 20, synth_count=n)
        dp_val = compute_ols_slope(dp_data)
        dp_slopes.append(dp_val)
    
    # True beta is 1 because y = x + noise (with mean 0)
    true_beta = 1.0
    mean_ols = sum(ols_slopes) / len(ols_slopes)
    mean_dp = sum(dp_slopes) / len(dp_slopes)
    ols_biases.append(mean_ols - true_beta)
    dp_biases.append(mean_dp - true_beta)
    
    # Calculate standard deviations manually
    var_ols = 0
    var_dp = 0
    for val in ols_slopes:
        var_ols += (val - mean_ols) ** 2
    for val in dp_slopes:
        var_dp += (val - mean_dp) ** 2
    ols_stds.append(np.sqrt(var_ols / len(ols_slopes)))
    dp_stds.append(np.sqrt(var_dp / len(dp_slopes)))

# Plotting bias and standard deviation vs. sample size
plt.figure(figsize=(10, 10))

# Bias plot
plt.subplot(2, 1, 1)
plt.plot(sample_sizes, ols_biases, 'bo-', label="OLS Bias")
plt.plot(sample_sizes, dp_biases, 'rx-', label="DP Bias")
plt.xlabel("Sample Size (n)")
plt.ylabel("Bias")
plt.ylim([-1, 1])
plt.title("Bias vs. Sample Size")
plt.legend()
plt.grid(True)

# Standard Deviation plot
plt.subplot(2, 1, 2)
plt.plot(sample_sizes, ols_stds, 'bo-', label="OLS Std")
plt.plot(sample_sizes, dp_stds, 'rx-', label="DP Std")
plt.xlabel("Sample Size (n)")
plt.ylabel("Standard Deviation")
plt.ylim([0, 1])
plt.title("Std Dev vs. Sample Size")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()