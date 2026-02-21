import numpy as np

class SGD:
    """
    Vanilla Stochastic Gradient Descent.
    W_new = W - lr * grad_W
    """

    def __init__(self, learning_rate=0.01, weight_decay=0.0, **kwargs):
        self.lr     = learning_rate
        self.wd     = weight_decay
        self.is_nag = False   # flag checked by training loop

    def apply_lookahead(self, layers):
        """SGD has no lookahead step."""
        pass

    def undo_lookahead(self, layers):
        pass

    def update(self, layers):
        for layer in layers:
            # L2 weight decay: effective gradient = grad_W + wd * W  , prevents overfitting
            # Bias is NOT decayed
            grad_W_eff = layer.grad_W + self.wd * layer.W
            layer.W -= self.lr * grad_W_eff
            layer.b -= self.lr * layer.grad_b


class Momentum:

    def __init__(self, learning_rate=0.01, beta=0.9, weight_decay=0.0, **kwargs):
        self.lr     = learning_rate
        self.beta   = beta
        self.wd     = weight_decay
        self.is_nag = False

    def apply_lookahead(self, layers):
        pass

    def undo_lookahead(self, layers):
        pass

    def update(self, layers):
        for layer in layers:
            grad_W_eff = layer.grad_W + self.wd * layer.W   # L2 weight decay
            layer.v_W = self.beta * layer.v_W - self.lr * grad_W_eff
            layer.v_b = self.beta * layer.v_b - self.lr * layer.grad_b
            layer.W  += layer.v_W
            layer.b  += layer.v_b


class NAG:
    
    def __init__(self, learning_rate=0.01, beta=0.9, weight_decay=0.0, **kwargs):
        self.lr     = learning_rate
        self.beta   = beta
        self.wd     = weight_decay
        self.is_nag = True   # tells training loop to call apply_lookahead

    def apply_lookahead(self, layers):
        """
        BEFORE forward pass: shift weights to lookahead position.
        W_look = W + beta * v
        """
        for layer in layers:
            #save current weights
            layer.W_original = layer.W.copy()
            layer.b_original = layer.b.copy()
            #shift to lookahead position
            layer.W = layer.W + self.beta * layer.v_W
            layer.b = layer.b + self.beta * layer.v_b

    def undo_lookahead(self, layers):
        """restore original weights"""
        for layer in layers:
            layer.W = layer.W_original
            layer.b = layer.b_original

    def update(self, layers):
        for layer in layers:
            # grad_W was computed at W_look — this is the correct NAG gradient
            # Weight decay applied to ORIGINAL W (not lookahead W)
            grad_W_eff = layer.grad_W + self.wd * layer.W_original
            layer.v_W = self.beta * layer.v_W - self.lr * grad_W_eff
            layer.v_b = self.beta * layer.v_b - self.lr * layer.grad_b
            # Update from ORIGINAL W (not from lookahead W)
            layer.W = layer.W_original + layer.v_W
            layer.b = layer.b_original + layer.v_b


class RMSProp:
    def __init__(self, learning_rate=0.001, beta=0.9, eps=1e-8,
                 weight_decay=0.0, **kwargs):
        self.lr     = learning_rate
        self.beta   = beta
        self.eps    = eps
        self.wd     = weight_decay
        self.is_nag = False

    def apply_lookahead(self, layers):
        pass

    def undo_lookahead(self, layers):
        pass

    def update(self, layers):
        for layer in layers:
            grad_W_eff = layer.grad_W + self.wd * layer.W   # L2 weight decay
            #update running mean of squared gradients using effective gradient
            layer.m_W = self.beta * layer.m_W + (1.0 - self.beta) * (grad_W_eff ** 2)
            layer.m_b = self.beta * layer.m_b + (1.0 - self.beta) * (layer.grad_b ** 2)
            #adaptive lr update
            layer.W -= self.lr * grad_W_eff / (np.sqrt(layer.m_W) + self.eps)
            layer.b -= self.lr * layer.grad_b / (np.sqrt(layer.m_b) + self.eps)


class Adam:

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999,
                 eps=1e-8, weight_decay=0.0, **kwargs):
        self.lr     = learning_rate
        self.beta1  = beta1
        self.beta2  = beta2
        self.eps    = eps
        self.wd     = weight_decay
        self.t      = 0       # timestep for bias correction
        self.is_nag = False

    def apply_lookahead(self, layers):
        pass

    def undo_lookahead(self, layers):
        pass

    def update(self, layers):
        self.t += 1
        bc1 = 1.0 - self.beta1 ** self.t
        bc2 = 1.0 - self.beta2 ** self.t

        for layer in layers:
            grad_W_eff = layer.grad_W + self.wd * layer.W   # L2 weight decay

            #1st moment: exponential moving average of effective gradients
            layer.v_W = self.beta1 * layer.v_W + (1.0 - self.beta1) * grad_W_eff
            layer.v_b = self.beta1 * layer.v_b + (1.0 - self.beta1) * layer.grad_b

            #2nd moment: exponential moving average of squared effective gradients
            layer.m_W = self.beta2 * layer.m_W + (1.0 - self.beta2) * (grad_W_eff ** 2)
            layer.m_b = self.beta2 * layer.m_b + (1.0 - self.beta2) * (layer.grad_b ** 2)

            #bias-corrected estimates
            m_hat_W = layer.v_W / bc1
            m_hat_b = layer.v_b / bc1
            v_hat_W = layer.m_W / bc2
            v_hat_b = layer.m_b / bc2

            #weight update
            layer.W -= self.lr * m_hat_W / (np.sqrt(v_hat_W) + self.eps)
            layer.b -= self.lr * m_hat_b / (np.sqrt(v_hat_b) + self.eps)


class Nadam:

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999,
                 eps=1e-8, weight_decay=0.0, **kwargs):
        self.lr     = learning_rate
        self.beta1  = beta1
        self.beta2  = beta2
        self.eps    = eps
        self.wd     = weight_decay
        self.t      = 0
        self.is_nag = False

    def apply_lookahead(self, layers):
        pass

    def undo_lookahead(self, layers):
        pass

    def update(self, layers):
        self.t += 1
        bc1      = 1.0 - self.beta1 ** self.t
        bc2      = 1.0 - self.beta2 ** self.t
        bc1_next = 1.0 - self.beta1 ** (self.t + 1)

        for layer in layers:
            grad_W_eff = layer.grad_W + self.wd * layer.W   # L2 weight decay

            #1st moment using effective gradient
            layer.v_W = self.beta1 * layer.v_W + (1.0 - self.beta1) * grad_W_eff
            layer.v_b = self.beta1 * layer.v_b + (1.0 - self.beta1) * layer.grad_b

            #2nd moment using effective gradient
            layer.m_W = self.beta2 * layer.m_W + (1.0 - self.beta2) * (grad_W_eff ** 2)
            layer.m_b = self.beta2 * layer.m_b + (1.0 - self.beta2) * (layer.grad_b ** 2)

            #bias-corrected 2nd moment
            v_hat_W = layer.m_W / bc2
            v_hat_b = layer.m_b / bc2

            #Nesterov-corrected 1st moment 
            nesterov_W = (self.beta1 * layer.v_W / bc1_next) + \
                         ((1.0 - self.beta1) * grad_W_eff / bc1)
            nesterov_b = (self.beta1 * layer.v_b / bc1_next) + \
                         ((1.0 - self.beta1) * layer.grad_b / bc1)

            #weight update
            layer.W -= self.lr * nesterov_W / (np.sqrt(v_hat_W) + self.eps)
            layer.b -= self.lr * nesterov_b / (np.sqrt(v_hat_b) + self.eps)


OPTIMIZERS = {
    "sgd":      SGD,
    "momentum": Momentum,
    "nag":      NAG,
    "rmsprop":  RMSProp,
    "adam":     Adam,
    "nadam":    Nadam,
}


def get_optimizer(name, **kwargs):
    """Return an instantiated optimizer by string name."""
    name = name.lower()
    if name not in OPTIMIZERS:
        raise ValueError(
            f"Unknown optimizer '{name}'. Choose from: {list(OPTIMIZERS.keys())}"
        )
    return OPTIMIZERS[name](**kwargs)