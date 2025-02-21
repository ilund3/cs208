import numpy as np
import math
import pandas as pd
import struct

# Load the fake patient dataset.
data = pd.read_csv("https://raw.githubusercontent.com/opendp/cs208/refs/heads/main/spring2025/data/fake_patient_dataset.csv")

epsilon = 0.1

# Laplace mechanism for a counting query (sensitivity = 1)
def release_dp_count(query):
    sensitivity = 1
    scale = sensitivity / epsilon
    sensitive_count = query(data)
    return sensitive_count + np.random.laplace(loc=0.0, scale=scale)

# Extract the least-significant bit of the stored significand.
# In IEEE 754, a normalized double is stored with a 52-bit fraction (with an implicit leading 1).
# If a value is computed as 1 ⊕ Z (with 1's ULP=2^-52), the floating-point addition property implies
# that (observed - 1) will be an exact multiple of 2^-53 and hence its least-significant bit will be 0.
def get_last_significand_bit(x):
    # Pack x into 64 bits and unpack as an unsigned 64-bit integer.
    bits = struct.unpack('>Q', struct.pack('>d', x))[0]
    # The stored fraction is in the lower 52 bits; we check the least-significant bit of that field.
    # (This bit is expected to be 0 when the output came from 1 ⊕ Z.)
    return bits & 1

def hypothesis_test(observed):
    # Compute the difference from 1.
    diff = observed - 1.0
    # If the least-significant bit is 0, predict tier 1 (invoice = 50,000).
    if get_last_significand_bit(diff) == 0:
        return 1
    else:
        return 0

# Run experiments to estimate the True Positive Rate (TPR) and False Positive Rate (FPR)
num_experiments = 10000
true_positives = 0
false_positives = 0
true_negatives = 0
false_negatives = 0

for _ in range(num_experiments):
    # Randomly select a patient id.
    pid = np.random.randint(0, len(data))
    # Retrieve the patient's invoice (tier 0: \$1,000, tier 1: \$50,000)
    true_invoice = data.loc[data['id'] == pid, 'invoice'].iloc[0]
    
    # Define the predicate: count is 1 if the invoice equals 50,000, else 0.
    query = lambda df: (df.loc[df['id'] == pid, 'invoice'] == 50000).sum()
    
    # Release the noisy count.
    observed = release_dp_count(query)
    decision = hypothesis_test(observed)
    
    # decision 1 means we predict tier 1 (invoice = 50,000); 0 means tier 0.
    if decision == 1:
        if true_invoice == 50000:
            true_positives += 1
        else:
            false_positives += 1
    else:
        if true_invoice == 50000:
            false_negatives += 1
        else:
            true_negatives += 1

TPR = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
FPR = false_positives / (false_positives + true_negatives) if (false_positives + true_negatives) > 0 else 0

print("True Positive Rate (TPR):", TPR)
print("False Positive Rate (FPR):", FPR)