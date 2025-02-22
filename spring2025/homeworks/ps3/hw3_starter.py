import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import struct

data: pd.DataFrame = pd.read_csv("https://raw.githubusercontent.com/opendp/cs208/refs/heads/main/spring2025/data/fake_patient_dataset.csv")

# names of public identifier columns
pub = ["id", "age", "sex", "blood"]

# variable to reconstruct
target = "invoice"

epsilon = 0.1

# Returns the ULP in base-2 of a floating-point number
def get_ulp_base2(x):
    return math.log2(math.ulp(x))

# Release a DP count over the invoice column for patients in the dataset with ID = id
def release_dp_count(query):
    sensitivity = 1
    scale = sensitivity/epsilon
    sensitive_count = query(data)
    return sensitive_count + np.random.laplace(loc=0.0, scale=scale)

# Helper to be used in bit extraction for positive "litmus test"
def extract_lsb(x):
    # Convert to 8-byte
    bytes_repr = struct.pack("!d", x)
    # Convert to integer
    int_val = int.from_bytes(bytes_repr)
    # Return last bit
    return int_val & 0x1

# Implement the hypothesis test!
def hypothesis_test(observed):
    if extract_lsb(observed - 1.0) == 0:
        return 1  # predict 50000
    else:
        return 0  # predict 1000

num_experiments = 10000
true_pos, false_pos, true_neg, false_neg = 0, 0, 0, 0

# Run the experiments
for _ in range(num_experiments):

    # Randomly choose a patient ID
    id = np.random.randint(0, len(data))
    
    # Counting query that counts 1 if invoice = 50000 (alternative hypothesis)
    observed = release_dp_count(
        lambda data: (data.loc[data['id'] == id, 'invoice'] == 50000).sum()
    )
    # Retrieve the actual true invoice to compare the query
    real_invoice = data.loc[data['id'] == id, 'invoice'].iloc[0]
    
    # 1 = reject null, 0 otherwise
    decision = hypothesis_test(observed)

    # estimate the TPR and FPR of your attack
    if decision == 1 and real_invoice == 50000:
        true_pos += 1
    elif decision == 1 and real_invoice != 50000:
        false_pos += 1
    elif decision != 1 and real_invoice == 50000:
        false_neg += 1
    else:
        true_neg += 1
print(true_neg)
print(false_neg)
print("True Positive Rate (TPR):", true_pos / (true_pos + false_neg))
print("False Positive Rate (FPR):", false_pos / (false_pos + true_neg))