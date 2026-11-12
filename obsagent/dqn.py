"""Deep Q-Network agent, same recipe as the dissertation's scaling agent:
experience replay, epsilon-greedy exploration, a target network and a
Huber TD loss. Only the inputs, actions and reward are new.
"""
import os

import numpy as np

from .nn import MLP, Adam

ACTIONS = (-2, -1, 0, +1, +2)  # change of telemetry level, like SCALE_DOWN_2 .. SCALE_UP_2


class ReplayBuffer:
    def __init__(self, capacity, state_dim, seed=0):
        self.cap = capacity
        self.s = np.zeros((capacity, state_dim))
        self.a = np.zeros(capacity, dtype=np.int64)
        self.r = np.zeros(capacity)
        self.s2 = np.zeros((capacity, state_dim))
        self.done = np.zeros(capacity)
        self.n = 0
        self.i = 0
        self.rng = np.random.default_rng(seed)

    def add(self, s, a, r, s2, done):
        self.s[self.i], self.a[self.i], self.r[self.i] = s, a, r
        self.s2[self.i], self.done[self.i] = s2, done
        self.i = (self.i + 1) % self.cap
        self.n = min(self.n + 1, self.cap)

    def sample(self, batch):
        idx = self.rng.integers(0, self.n, size=batch)
        return self.s[idx], self.a[idx], self.r[idx], self.s2[idx], self.done[idx]


class DQNAgent:
    def __init__(self, state_dim, hidden=(64, 32), gamma=0.95, lr=1e-3,
                 buffer=50_000, batch=64, target_every=500, huber=1.0, seed=0, floor=0, double=False):
        self.floor = floor    # lowest level the agent may choose; saved with the weights
        self.double = double  # Double DQN: the online net picks the next action, the target net scores it
        sizes = [state_dim, *hidden, len(ACTIONS)]
        self.q = MLP(sizes, seed=seed)
        self.target = MLP(sizes, seed=seed)
        self.target.copy_from(self.q)
        self.opt = Adam(self.q, lr=lr)
        self.mem = ReplayBuffer(buffer, state_dim, seed=seed)
        self.gamma, self.batch, self.target_every, self.huber = gamma, batch, target_every, huber
        self.rng = np.random.default_rng(seed + 1)
        self.steps = 0

    def act(self, state, epsilon=0.0):
        if self.rng.random() < epsilon:
            return int(self.rng.integers(len(ACTIONS)))
        return int(np.argmax(self.q.forward(state[None, :])[0]))

    def q_values(self, state):
        return self.q.forward(state[None, :])[0]

    def learn(self, batch=None):
        """One gradient step on a random minibatch from replay (or on the
        given one, e.g. mixed bench + simulator data). Returns the loss."""
        if batch is None:
            if self.mem.n < self.batch * 10:
                return None
            batch = self.mem.sample(self.batch)
        s, a, r, s2, done = batch
        if self.double:
            best = np.argmax(self.q.forward(s2), axis=1)
            q_next = self.target.forward(s2)[np.arange(len(best)), best]
        else:
            q_next = self.target.forward(s2).max(axis=1)
        y = r + self.gamma * (1.0 - done) * q_next

        q_all, acts = self.q.forward(s, keep=True)
        rows = np.arange(len(a))
        err = q_all[rows, a] - y
        # Huber loss: quadratic for small errors, linear for big ones.
        grad_q = np.clip(err, -self.huber, self.huber) / len(a)
        grad_out = np.zeros_like(q_all)
        grad_out[rows, a] = grad_q
        self.opt.step(self.q.backward(acts, grad_out))

        self.steps += 1
        if self.steps % self.target_every == 0:
            self.target.copy_from(self.q)
        absr = np.abs(err)
        return float(np.where(absr < self.huber, 0.5 * err ** 2, self.huber * (absr - 0.5 * self.huber)).mean())

    def save(self, path):
        np.savez(path, floor=np.array(self.floor), **self.q.state())

    @staticmethod
    def load_policy_net(path):
        """One weight file, or several joined with '+': their Q-values are averaged."""
        if "+" in os.path.basename(path):
            folder = os.path.dirname(path)
            return Ensemble([DQNAgent.load_policy_net(os.path.join(folder, p))
                             for p in os.path.basename(path).split("+")])
        with np.load(path) as data:
            net = MLP.from_state({k: data[k] for k in data.files})
            net.floor = int(data["floor"]) if "floor" in data.files else 0   # older files: no floor
            return net


class Ensemble:
    """Several independently trained networks vote by averaging Q-values.
    Training runs that differ only in their random seed end up at different
    points of the cost/coverage trade-off; the average is steadier than any one."""

    def __init__(self, nets):
        self.nets = nets
        self.floor = max(getattr(n, "floor", 0) for n in nets)

    def forward(self, x):
        return np.mean([n.forward(x) for n in self.nets], axis=0)
