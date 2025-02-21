import numpy as np

def gaussian_solve(T, y):
    """
    Solve T x = y in GF(2) using Gaussian elimination with NumPy.
    Faster than SymPy's `rref()`.
    """
    T = np.array(T, dtype=np.uint8) % 2  # Ensure binary field GF(2)
    y = np.array(y, dtype=np.uint8) % 2  # Ensure binary field GF(2)

    k, n = T.shape

    # Forward elimination (convert to row echelon form)
    for col in range(min(k, n)):
        pivot_row = np.where(T[col:, col] == 1)[0]
        if len(pivot_row) == 0:
            continue  # No pivot found, move to next column
        pivot_row = pivot_row[0] + col

        # Swap rows if necessary
        if pivot_row != col:
            T[[col, pivot_row]] = T[[pivot_row, col]]
            y[[col, pivot_row]] = y[[pivot_row, col]]

        # Eliminate lower rows
        for row in range(col + 1, k):
            if T[row, col] == 1:
                T[row] ^= T[col]  # XOR row operation
                y[row] ^= y[col]  # XOR y values

    # Back-substitution
    x = np.zeros(n, dtype=np.uint8)
    for col in reversed(range(n)):
        nonzero_rows = np.where(T[:, col] == 1)[0]
        if len(nonzero_rows) == 0:
            continue

        pivot_row = nonzero_rows[0]
        x[col] = y[pivot_row]

        # Subtract row contribution from other equations
        for row in nonzero_rows[1:]:
            y[row] ^= x[col]  # XOR solution into RHS

    return x