# Importing necessary libraries
import numpy as np
import time
from scipy.linalg import lu, lu_factor, lu_solve


# -----------------------------------------------------------
# LU FACTORIZATION WITH PARTIAL PIVOTING (CUSTOM)
# -----------------------------------------------------------
"""Performing LU factorization of A, return P,L,U"""

def lu_factorization(A):
    A = A.copy()
    n = A.shape[0]

    P = np.eye(n)
    L = np.zeros((n, n))
    U = A.copy()

    for k in range(n):
        # Finding the pivot
        pivot = np.argmax(np.abs(U[k:, k])) + k

        # Checks if any non zero pivots exist, if not raising singularity of matrix
        if np.isclose(U[pivot, k], 0):
            raise ValueError("Matrix is singular!")

        # Swapping rows in U
        U[[k, pivot]] = U[[pivot, k]]
        # Swapping rows in P
        P[[k, pivot]] = P[[pivot, k]]
        # Swapping rows in L (only the previous columns 0 to k-1)
        if k > 0:
            L[[k, pivot], :k] = L[[pivot, k], :k]

        # Elimination step
        for i in range(k + 1, n):
            L[i, k] = U[i, k] / U[k, k]
            U[i] = U[i] - L[i, k] * U[k]

    np.fill_diagonal(L, 1)
    return P, L, U


# -----------------------------------------------------------
# FORWARD SUBSTITUTION
# -----------------------------------------------------------
def forward_substitution(L, b):
    """
    Solving Ly = b , L is lower triangular.
    """
    n = L.shape[0]
    y = np.zeros(n)

    for i in range(n):
        y[i] = b[i] - np.dot(L[i, :i], y[:i])
        
    return y


# -----------------------------------------------------------
# BACKWARD SUBSTITUTION
# -----------------------------------------------------------
def backward_substitution(U, y):
    """
    Solving Ux = y, U is upper triangular.
    """
    n = U.shape[0]
    x = np.zeros(n)

    for i in reversed(range(n)):
        x[i] = (y[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i, i]

    return x


# -----------------------------------------------------------
# SOLVER USING LU (CUSTOM)
# -----------------------------------------------------------
def lu_solve_custom(P, L, U, b):
    """
    Solving Ax=b using previously computed P, L, U.
    """
    Pb = P @ b
    y = forward_substitution(L, Pb)
    x = backward_substitution(U, y)
    return x


# -----------------------------------------------------------
# ERROR METRICS
# -----------------------------------------------------------
def backward_error(A, P, L, U):
    """
    Computes ||PA - LU|| / ||A||
    """
    numerator = np.linalg.norm(P @ A - L @ U)
    denominator = np.linalg.norm(A)
    return numerator / denominator

def residual_error(A, x, b):
    """
    Computes ||Ax - b|| / ||b||
    """
    numerator = np.linalg.norm(A @ x - b)
    denominator = np.linalg.norm(b)
    return numerator / denominator


# -----------------------------------------------------------
# TIMING FUNCTIONS
# -----------------------------------------------------------
def time_factorization(A):
    """
    Timing only the LU factorization (custom).
    """
    start = time.perf_counter()
    P, L, U = lu_factorization(A)
    end = time.perf_counter()
    return P, L, U, end - start

def time_solution(P, L, U, b):
    """
    Timing (custom solver), forward + backward substitution.
    """
    start = time.perf_counter()
    x = lu_solve_custom(P, L, U, b)
    end = time.perf_counter()
    return x, end - start


# -----------------------------------------------------------
# SCIPY MODULE FOR COMPARISON
# -----------------------------------------------------------
def scipy_lu(A, b):
    """
    Performs LU using SciPy and returns P, L, U, solution and timings.
    """
    # ---- Factorization timings ----
    start_fact = time.perf_counter()
    P, L, U = lu(A)     # Explicit LU decomposition
    end_fact = time.perf_counter()

    # ---- Solver timings ----
    start_solve = time.perf_counter()
    x = np.linalg.solve(A, b)   # Fair solve comparison
    end_solve = time.perf_counter()

    return P, L, U, x, end_fact - start_fact, end_solve - start_solve

