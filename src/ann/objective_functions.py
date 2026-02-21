
import numpy as np
from .activations import softmax


def cross_entropy_loss(logits, y_true_onehot):
    
    batch_size = logits.shape[0]
    probs = softmax(logits)
    probs_clipped = np.clip(probs, 1e-12, 1.0)        

    loss = -np.sum(y_true_onehot * np.log(probs_clipped)) / batch_size

    #d(CE)/d(logit) = softmax_output - one_hot
    delta = (probs - y_true_onehot)
    return loss, delta


def mse_loss(logits, y_true_onehot):
    
    probs = softmax(logits)
    num_classes = probs.shape[1]

    loss = np.mean(np.sum((probs - y_true_onehot) ** 2, axis=1))

    #chain rule
    diff = probs - y_true_onehot
    dot  = np.sum(diff * probs, axis=1, keepdims=True)
    delta = (2.0 / num_classes) * probs * (diff - dot)

    return loss, delta


LOSSES = {
    "cross_entropy": cross_entropy_loss,
    "mse":           mse_loss,
}


def get_loss(name):
    """Return loss function by string name."""
    name = name.lower()
    if name not in LOSSES:
        raise ValueError(f"Unknown loss '{name}'. Choose: {list(LOSSES.keys())}")
    return LOSSES[name]