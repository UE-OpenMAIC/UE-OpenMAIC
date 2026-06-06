import numpy as np

def all_normalize(X, eps=1e-8):

    X = np.asarray(X, dtype=float).copy()

    mean = X.mean(axis=2, keepdims=True)

    std = X.std(axis=2, keepdims=True)

    std[std < eps] = 1.0

    return (X - mean) / std

def calculate_scalar_velocity_list(*args, **kwargs):

    return None
