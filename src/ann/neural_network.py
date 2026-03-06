
import numpy as np
import os
import json

from .neural_layer import Layer
from .objective_functions import get_loss
from .activations import get_activation


class NeuralNetwork:
    

    def __init__(self, cli_args):
       
        self.cli_args     = cli_args
        self.weight_decay = getattr(cli_args, 'weight_decay', 0.0)
        self.optimizer    = None  

        #build hidden_sizes list
        raw    = cli_args.hidden_size
        n_lay  = cli_args.num_layers

        if isinstance(raw, list):
            if len(raw) == 1:
                self.hidden_sizes = [raw[0]] * n_lay
            elif len(raw) == n_lay:
                self.hidden_sizes = raw
            else:
                raise ValueError(
                    f"hidden_size has {len(raw)} values but num_layers={n_lay}. "
                    "Provide 1 value (same for all) or exactly num_layers values."
                )
        else:
            self.hidden_sizes = [raw] * n_lay

        self.input_size  = 784
        self.output_size = 10
        self.loss_fn     = get_loss(cli_args.loss)

        #build layer stack
        # sizes = [784, h1, h2, ..., hN, 10]
        sizes = [self.input_size] + self.hidden_sizes + [self.output_size]

        self.layers = []
        for i in range(len(sizes) - 1):
            is_output = (i == len(sizes) - 2)
            act = "softmax" if is_output else cli_args.activation
            self.layers.append(
                Layer(
                    input_size  = sizes[i],
                    output_size = sizes[i + 1],
                    activation  = act,
                    weight_init = cli_args.weight_init,
                )
            )

    def forward(self, X):

        a = X
        for layer in self.layers:
            a = layer.forward(a)
        # Return pre-activation logits of output layer (not softmax probabilities)
        return self.layers[-1].z

    def backward(self, y_true, y_pred):
       
        #compute loss + initial delta at output logits
        loss, delta = self.loss_fn(y_pred, y_true)
        self._last_loss = float(loss)

        #backprop: last layer → first layer
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            if i == len(self.layers) - 1:
                
                # delta from loss_fn 
                delta = layer.backward(delta)
            else:
                #hidden layer: chain rule through activation
                act_deriv    = layer.activation_deriv(layer.z)
                delta_hidden = delta * act_deriv
                delta        = layer.backward(delta_hidden)

        grad_w = [layer.grad_W for layer in self.layers]
        grad_b = [layer.grad_b for layer in self.layers]
        return grad_w, grad_b


    def update_weights(self):
        self.optimizer.update(self.layers)


    def train(self, X_train, y_train_oh, epochs, batch_size):
        
        if self.optimizer is None:
            raise ValueError("Set model.optimizer before calling train()!")

        history = {'train_loss': [], 'train_acc': []}
        N = X_train.shape[0]
        use_nag = getattr(self.optimizer, 'is_nag', False)

        for epoch in range(1, epochs + 1):
            #shuffle training data each epoch
            idx    = np.random.permutation(N)
            X_shuf = X_train[idx]
            y_shuf = y_train_oh[idx]

            epoch_losses = []

            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                Xb  = X_shuf[start:end]
                yb  = y_shuf[start:end]

                if use_nag:
                    #NAG Step A: shift to lookahead BEFORE forward
                    self.optimizer.apply_lookahead(self.layers)

                #forward at (lookahead for NAG, current W for others)
                logits = self.forward(Xb)

                #backward: gradient computed at current W (= lookahead for NAG)
                self.backward(yb, logits)

                #update weights (for NAG: uses W_original + new velocity)
                self.update_weights()

                epoch_losses.append(self._last_loss)

            avg_loss = float(np.mean(epoch_losses))

            #accuracy on full training set (using current weights)
            preds  = np.argmax(self.forward(X_train), axis=1)
            labels = np.argmax(y_train_oh, axis=1)
            acc    = float(np.mean(preds == labels))

            history['train_loss'].append(avg_loss)
            history['train_acc'].append(acc)

        return history
 
    #EVALUATE
    
    def evaluate(self, X, y):
        
        logits = self.forward(X)

        if y.ndim == 2:
            y_onehot = y
            labels   = np.argmax(y, axis=1)
        else:
            labels   = y
            y_onehot = np.zeros((len(y), self.output_size), dtype=np.float32)
            y_onehot[np.arange(len(y)), y] = 1.0

        loss, _ = self.loss_fn(logits, y_onehot)
        preds   = np.argmax(logits, axis=1)
        acc     = float(np.mean(preds == labels))

        return {
            'accuracy':    acc,
            'loss':        float(loss),
            'predictions': preds,
        }

    
    #PREDICT
    

    def predict(self, X):
        """Return predicted class indices for input X."""
        logits = self.forward(X)
        return np.argmax(logits, axis=1)

    
    #SAVE / LOAD
    

    def get_weights(self):
        
        weights = {}
        for i, layer in enumerate(self.layers):
            weights[f"layer_{i}_W"] = layer.W.copy()
            weights[f"layer_{i}_b"] = layer.b.copy()
        return weights

    def set_weights(self, weights):
        
        # If weights is a numpy array, convert to dict
        if isinstance(weights, np.ndarray):
            weights = weights.item()
        
        for i, layer in enumerate(self.layers):
            key_W = f"layer_{i}_W"
            key_b = f"layer_{i}_b"
            if key_W in weights:
                layer.W = weights[key_W].copy()
                layer.b = weights[key_b].copy()
            else:
                print(f"[WARN] Key {key_W} not found in weights dict")

    def save_weights(self, filepath):
        """Save all layer W and b to a .npy file."""
        weights = {}
        for i, layer in enumerate(self.layers):
            weights[f"layer_{i}_W"] = layer.W
            weights[f"layer_{i}_b"] = layer.b
        dirpath = os.path.dirname(filepath)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        np.save(filepath, weights, allow_pickle=True)
        print(f"[INFO] Saved weights → {filepath}")

    def load_weights(self, filepath):
        """Load weights from a .npy file produced by save_weights()."""
        weights = np.load(filepath, allow_pickle=True).item()
        for i, layer in enumerate(self.layers):
            layer.W = weights[f"layer_{i}_W"]
            layer.b = weights[f"layer_{i}_b"]
        print(f"[INFO] Loaded weights ← {filepath}")

    
    #W&B LOGGING HELPERS
    

    def get_gradient_norms(self):
       
        return [
            (f"layer_{i+1}_grad_norm", float(np.linalg.norm(layer.grad_W)))
            for i, layer in enumerate(self.layers)
        ]

    def get_activation_stats(self):
        
        stats = {}
        for i, layer in enumerate(self.layers[:-1]):   # skip output layer
            if layer.a is not None:
                stats[f"layer_{i+1}_dead_frac"] = float(np.mean(layer.a == 0))
                stats[f"layer_{i+1}_act_mean"]  = float(np.mean(np.abs(layer.a)))
        return stats