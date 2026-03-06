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

        raw   = cli_args.hidden_size
        n_lay = cli_args.num_layers

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
        return self.layers[-1].z

    def backward(self, y_true, y_pred):
        loss, delta = self.loss_fn(y_pred, y_true)
        self._last_loss = float(loss)

        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            if i == len(self.layers) - 1:
                delta = layer.backward(delta)
            else:
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
            idx    = np.random.permutation(N)
            X_shuf = X_train[idx]
            y_shuf = y_train_oh[idx]

            epoch_losses = []
            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                Xb  = X_shuf[start:end]
                yb  = y_shuf[start:end]

                if use_nag:
                    self.optimizer.apply_lookahead(self.layers)

                logits = self.forward(Xb)
                self.backward(yb, logits)
                self.update_weights()
                epoch_losses.append(self._last_loss)

            avg_loss = float(np.mean(epoch_losses))
            preds    = np.argmax(self.forward(X_train), axis=1)
            labels   = np.argmax(y_train_oh, axis=1)
            acc      = float(np.mean(preds == labels))

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

    def predict(self, X):
        """Return predicted class indices for input X."""
        logits = self.forward(X)
        return np.argmax(logits, axis=1)

    # ── SAVE / LOAD ──────────────────────────────────────────────────────────

    def get_weights(self):
        """Return weights dict with keys W0, b0, W1, b1, ..."""
        d = {}
        for i, layer in enumerate(self.layers):
            d[f"W{i}"] = layer.W.copy()
            d[f"b{i}"] = layer.b.copy()
        return d

    def set_weights(self, weight_dict):
        """Set weights from dict with keys W0, b0, W1, b1, ..."""
        for i, layer in enumerate(self.layers):
            w_key = f"W{i}"
            b_key = f"b{i}"
            if w_key in weight_dict:
                layer.W = weight_dict[w_key].copy()
            if b_key in weight_dict:
                layer.b = weight_dict[b_key].copy()

    def save_weights(self, filepath):
        """Save weights to .npy file."""
        weights = self.get_weights()
        dirpath = os.path.dirname(filepath)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        np.save(filepath, weights, allow_pickle=True)
        print(f"[INFO] Saved weights → {filepath}")

    def load_weights(self, filepath):
        """Load weights from .npy file."""
        weights = np.load(filepath, allow_pickle=True)
        weights = weights.tolist()  # tolist() handles both array and dict formats
        self.set_weights(weights)
        print(f"[INFO] Loaded weights ← {filepath}")

    # ── W&B LOGGING HELPERS ──────────────────────────────────────────────────

    def get_gradient_norms(self):
        return [
            (f"layer_{i+1}_grad_norm", float(np.linalg.norm(layer.grad_W)))
            for i, layer in enumerate(self.layers)
        ]

    def get_activation_stats(self):
        stats = {}
        for i, layer in enumerate(self.layers[:-1]):
            if layer.a is not None:
                stats[f"layer_{i+1}_dead_frac"] = float(np.mean(layer.a == 0))
                stats[f"layer_{i+1}_act_mean"]  = float(np.mean(np.abs(layer.a)))
        return stats
