
import numpy as np
import os
import json
from .neural_layer import Layer
from .objective_functions import get_loss
from .activations import get_activation


class NeuralNetwork:
    def __init__(self, cli_args):
        """
        Args:
            cli_args: Command-line arguments (argparse Namespace) containing:
                      - num_layers   : number of hidden layers
                      - hidden_size  : neurons per hidden layer (list or int)
                      - activation   : 'relu', 'sigmoid', or 'tanh'
                      - weight_init  : 'xavier' or 'random'
                      - loss         : 'cross_entropy' or 'mse'
                      - weight_decay : L2 regularization coefficient
                      - optimizer    : optimizer name string
                      - learning_rate: float
        """
        self.cli_args    = cli_args
        self.weight_decay = getattr(cli_args, 'weight_decay', 0.0)

        hidden_size_arg = cli_args.hidden_size
        num_layers      = cli_args.num_layers

        if isinstance(hidden_size_arg, list):
            if len(hidden_size_arg) == 1:
                self.hidden_sizes = [hidden_size_arg[0]] * num_layers
            elif len(hidden_size_arg) == num_layers:
                self.hidden_sizes = hidden_size_arg
            else:
                raise ValueError("hidden_size must have 1 value or num_layers values.")
        else:
            self.hidden_sizes = [hidden_size_arg] * num_layers

        self.input_size  = 784    #MNIST/Fashion-MNIST flattened
        self.output_size = 10     #10 classes
        self.activation  = cli_args.activation
        self.weight_init = cli_args.weight_init
        self.loss_fn     = get_loss(cli_args.loss)

        #build layer stack: input → hidden layers → output (softmax)
        self.layers = []
        sizes = [self.input_size] + self.hidden_sizes + [self.output_size]

        for i in range(len(sizes) - 1):
            act = "softmax" if i == len(sizes) - 2 else self.activation
            self.layers.append(
                Layer(
                    input_size=sizes[i],
                    output_size=sizes[i + 1],
                    activation=act,
                    weight_init=self.weight_init,
                )
            )

   
    def forward(self, X):
        
        a = X
        for layer in self.layers:
            a = layer.forward(a)
       
        return self.layers[-1].z   #shape: (batch_size, 10)

    def backward(self, y_true, y_pred):
        
        #compute loss and initial gradient at output logits
        loss, delta = self.loss_fn(y_pred, y_true)
        self._last_loss = loss

        #backprop through layers (last → first)
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            if i == len(self.layers) - 1:
                #output layer: delta already combines softmax + loss derivative
                delta = layer.backward(delta, self.weight_decay)
            else:
                #hidden layers: multiply by activation derivative
                act_deriv  = layer.activation_deriv(layer.z)
                delta_hidden = delta * act_deriv
                delta = layer.backward(delta_hidden, self.weight_decay)

        #return gradients as lists (matches skeleton return signature)
        grad_w = [layer.grad_W for layer in self.layers]
        grad_b = [layer.grad_b for layer in self.layers]
        return grad_w, grad_b

    
    def update_weights(self):
        self.optimizer.update(self.layers)

    def train(self, X_train, y_train, epochs, batch_size):
       
        history = {'train_loss': [], 'train_acc': []}
        N = X_train.shape[0]

        for epoch in range(1, epochs + 1):
            # Shuffle data each epoch
            idx = np.random.permutation(N)
            X_shuf, y_shuf = X_train[idx], y_train[idx]

            epoch_losses = []

            for start in range(0, N, batch_size):
                end   = min(start + batch_size, N)
                X_b   = X_shuf[start:end]
                y_b   = y_shuf[start:end]

                #forward
                logits = self.forward(X_b)
                #backward
                self.backward(y_b, logits)
                #update
                self.update_weights()

                epoch_losses.append(self._last_loss)

            avg_loss = float(np.mean(epoch_losses))
            preds    = np.argmax(self.forward(X_train), axis=1)

            
            if y_train.ndim == 2:
                labels = np.argmax(y_train, axis=1)
            else:
                labels = y_train
            acc = float(np.mean(preds == labels))

            history['train_loss'].append(avg_loss)
            history['train_acc'].append(acc)

        return history

    
    def evaluate(self, X, y):
        
        logits = self.forward(X)

        if y.ndim == 2:
            y_onehot = y
            labels   = np.argmax(y, axis=1)
        else:
            labels   = y
            y_onehot = np.zeros((len(y), 10))
            y_onehot[np.arange(len(y)), y] = 1.0

        loss, _ = self.loss_fn(logits, y_onehot)
        preds   = np.argmax(logits, axis=1)
        acc     = float(np.mean(preds == labels))

        return {
            'accuracy':    acc,
            'loss':        float(loss),
            'predictions': preds,
        }

   
    def save_weights(self, filepath):
        
        weights = {}
        for i, layer in enumerate(self.layers):
            weights[f"layer_{i}_W"] = layer.W
            weights[f"layer_{i}_b"] = layer.b
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        np.save(filepath, weights, allow_pickle=True)

    def load_weights(self, filepath):
        
        weights = np.load(filepath, allow_pickle=True).item()
        for i, layer in enumerate(self.layers):
            layer.W = weights[f"layer_{i}_W"]
            layer.b = weights[f"layer_{i}_b"]

    def predict(self, X):
        logits = self.forward(X)
        return np.argmax(logits, axis=1)

    def get_gradient_norms(self):
        return [(f"layer_{i+1}_grad_norm", float(np.linalg.norm(l.grad_W)))
                for i, l in enumerate(self.layers)]

    def get_activation_stats(self):
        stats = {}
        for i, layer in enumerate(self.layers[:-1]):
            if layer.a is not None:
                stats[f"layer_{i+1}_dead_frac"] = float(np.mean(layer.a == 0))
                stats[f"layer_{i+1}_act_mean"]  = float(np.mean(np.abs(layer.a)))
        return stats