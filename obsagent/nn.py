"""A tiny neural network written from scratch in NumPy.

No ML framework and no pretrained weights: every number in the network is
learned from our own bench data. The network is a plain multilayer
perceptron (fully connected layers + ReLU) trained with Adam.
"""
import numpy as np


class MLP:
    """Fully connected network: input -> hidden (ReLU) ... -> linear output."""

    def __init__(self, sizes, seed=0):
        rng = np.random.default_rng(seed)
        self.sizes = list(sizes)
        self.params = []
        for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
            # He initialisation keeps ReLU activations at a sane scale.
            w = rng.normal(0.0, np.sqrt(2.0 / fan_in), (fan_in, fan_out))
            self.params.append([w, np.zeros(fan_out)])

    def forward(self, x, keep=False):
        """Return outputs; with keep=True also the activations for backprop."""
        acts = [x]
        h = x
        last = len(self.params) - 1
        for i, (w, b) in enumerate(self.params):
            z = h @ w + b
            h = z if i == last else np.maximum(z, 0.0)
            acts.append(h)
        return (h, acts) if keep else h

    def backward(self, acts, grad_out):
        """Gradients of the loss w.r.t. every weight, given dLoss/dOutput."""
        grads = [None] * len(self.params)
        g = grad_out
        for i in reversed(range(len(self.params))):
            w, _ = self.params[i]
            grads[i] = [acts[i].T @ g, g.sum(axis=0)]
            if i > 0:
                # acts[i] is the ReLU output of the previous layer.
                g = (g @ w.T) * (acts[i] > 0.0)
        return grads

    def copy_from(self, other):
        self.params = [[w.copy(), b.copy()] for w, b in other.params]

    def state(self):
        out = {"sizes": np.array(self.sizes)}
        for i, (w, b) in enumerate(self.params):
            out[f"w{i}"] = w
            out[f"b{i}"] = b
        return out

    @classmethod
    def from_state(cls, state):
        net = cls([int(s) for s in state["sizes"]])
        net.params = [[state[f"w{i}"].copy(), state[f"b{i}"].copy()]
                      for i in range(len(net.params))]
        return net


class Adam:
    """Adam optimiser with global-norm gradient clipping."""

    def __init__(self, net, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, clip=1.0):
        self.net, self.lr, self.b1, self.b2, self.eps, self.clip = net, lr, beta1, beta2, eps, clip
        self.m = [[np.zeros_like(w), np.zeros_like(b)] for w, b in net.params]
        self.v = [[np.zeros_like(w), np.zeros_like(b)] for w, b in net.params]
        self.t = 0

    def step(self, grads):
        norm = np.sqrt(sum(float((g * g).sum()) for pair in grads for g in pair))
        scale = min(1.0, self.clip / (norm + 1e-12))
        self.t += 1
        for layer, pair in enumerate(grads):
            for k, g in enumerate(pair):
                g = g * scale
                self.m[layer][k] = self.b1 * self.m[layer][k] + (1 - self.b1) * g
                self.v[layer][k] = self.b2 * self.v[layer][k] + (1 - self.b2) * g * g
                m_hat = self.m[layer][k] / (1 - self.b1 ** self.t)
                v_hat = self.v[layer][k] / (1 - self.b2 ** self.t)
                self.net.params[layer][k] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        return norm
