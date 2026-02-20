"""
Neural Layer Implementation
Handles weight initialization, forward pass and gradient computation
"""

import numpy as np
from .activations import get_activation, softmax


class Layer:
    """
      One dense layer. Contains:
      W        : weight matrix shape (input_size, output_size)
      b        : bias vector  shape (1, output_size)
      grad_W   : gradient of loss w.r.t. W  
      grad_b   : gradient of loss w.r.t. b  
    """

    def __init__(self, input_size, output_size, activation="relu", weight_init="xavier"):
        """
        Args:
            input_size  (int): number of inputs to this layer
            output_size (int): number of neurons in this layer
            activation  (str): 'sigmoid', 'tanh', 'relu' or 'softmax'
            weight_init (str): 'random' or 'xavier'
        """
        self.input_size  = input_size
        self.output_size = output_size
        self.activation_name = activation

        #weight initialization
        if weight_init == "xavier":
            limit = np.sqrt(6.0 / (input_size + output_size))
            self.W = np.random.uniform(-limit, limit, (input_size, output_size))
        elif weight_init == "random":
            self.W = np.random.randn(input_size, output_size) * 0.01
        else:
            raise ValueError(f"Unknown weight_init '{weight_init}'. Use 'random' or 'xavier'.")

        self.b = np.zeros((1, output_size))  # biases start at 0

        #activation 
        if activation == "softmax":
            self.activation_fn    = softmax
            self.activation_deriv = None  # handled at loss level
        else:
            self.activation_fn, self.activation_deriv = get_activation(activation)

        #forward pass 
        self.input = None   # input  coming into this layer
        self.z     = None   # pre-activation  (W*input + b)
        self.a     = None   # post-activation (activation(z))

        #gradients 
        self.grad_W = np.zeros_like(self.W)
        self.grad_b = np.zeros_like(self.b)

        #optimizer state
        self.v_W = np.zeros_like(self.W)   # velocity / 1st moment
        self.v_b = np.zeros_like(self.b)
        self.m_W = np.zeros_like(self.W)   # 2nd moment (Adam/RMSProp)
        self.m_b = np.zeros_like(self.b)

    def forward(self, a_prev):
        """
        forward pass: z = a_prev @ W + b,  a = activation(z)
        args:
            a_prev: array shape (batch_size, input_size)
        returns:
            a: array shape (batch_size, output_size)
        """
        self.input = a_prev
        self.z = a_prev @ self.W + self.b
        self.a = self.activation_fn(self.z)
        return self.a

    def backward(self, delta, weight_decay=0.0):
        """
        args:
            delta        : gradient from next layer, shape (batch_size, output_size)
            weight_decay : L2 regularization lambda
        Returns:
            delta_prev: gradient to pass back, shape (batch_size, input_size)
        """
        batch_size = self.input.shape[0]

        # Gradient w.r.t. W (averaged over batch) + L2 penalty
        self.grad_W = (self.input.T @ delta) / batch_size + weight_decay * self.W

        # Gradient w.r.t. b (averaged over batch)
        self.grad_b = np.sum(delta, axis=0, keepdims=True) / batch_size

        # Pass gradient to previous layer
        delta_prev = delta @ self.W.T
        return delta_prev