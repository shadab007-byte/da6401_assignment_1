## Activation functions and their derivatives used in the MLP

import numpy as np

# Sigmoid func

def sigmoid(z):
    z = np.clip(z, -500, 500)  # prevent overflow

    return 1.0/(1.0+np.exp(-z))

def sigmoid_derivative(z):
    d = sigmoid(z)
    return d*(1.0 - d)

# tanh(Z)
def tanh(z):   
    return np.tanh(z)


def tanh_derivative(z):
    d =np.tanh(z) ** 2
    return 1.0 - d

# ReLU: max(0, z)

def relu(z):  
    return np.maximum(0, z)


def relu_derivative(z):
    return (z > 0).astype(float)


def softmax(z):
    z_stable = z - np.max(z, axis=1, keepdims=True)     #Subtracts max per row for numerical stability.
    exp_z = np.exp(z_stable)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


# Easy lookup by string name
ACTIVATIONS = {
    "sigmoid": (sigmoid, sigmoid_derivative),
    "tanh":    (tanh,    tanh_derivative),
    "relu":    (relu,    relu_derivative),
}


def get_activation(name):
    """Returns (activation_fn, derivative_fn) tuple by name string."""
    name = name.lower()
    if name not in ACTIVATIONS:
        raise ValueError(f"Unknown activation '{name}'. Choose: {list(ACTIVATIONS.keys())}")
    return ACTIVATIONS[name]







