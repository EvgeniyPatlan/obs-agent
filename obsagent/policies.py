"""The DQN agent and the baselines it is compared against."""
import numpy as np

from .agent_core import apply_action, window_shares
from .config import DIAGNOSABLE, MAX_LEVEL, SLA_ERR, SLA_P99_S
from .dqn import ACTIONS


class StaticPolicy:
    def __init__(self, level, name):
        self.level, self.name = level, name

    def reset(self):
        pass

    def decide(self, svc, state, obs, mem):
        return self.level, None


class RulePolicy:
    """Typical hand-written rule: when the SLA is visibly broken, switch to
    diagnostic detail for 2 minutes, otherwise stay lean."""
    name = "rules"

    def __init__(self, hold_ticks=24, lean=1, high=DIAGNOSABLE):
        self.hold_ticks, self.lean, self.high = hold_ticks, lean, high

    def reset(self):
        self.hold = {}

    def decide(self, svc, state, obs, mem):
        if obs["p99"] > SLA_P99_S or obs["err"] > SLA_ERR:
            self.hold[svc] = self.hold_ticks
        if self.hold.get(svc, 0) > 0:
            self.hold[svc] -= 1
            return self.high, None
        return self.lean, None


class BurnRatePolicy(RulePolicy):
    """A stronger hand-written rule, after multi-window burn-rate alerting:
    switch to diagnostic detail only when the latest reading is over the SLA
    and at least 30% of the last minute was too. At a 15 s scrape that means
    two bad scrapes in a row, so a single spike is ignored."""
    name = "rules-burn"

    def __init__(self, share=0.3, **kw):
        super().__init__(**kw)
        self.share = share

    def decide(self, svc, state, obs, mem):
        one_min = window_shares(mem, obs)[0]
        if (obs["p99"] > SLA_P99_S or obs["err"] > SLA_ERR) and one_min >= self.share:
            self.hold[svc] = self.hold_ticks
        if self.hold.get(svc, 0) > 0:
            self.hold[svc] -= 1
            return self.high, None
        return self.lean, None


def input_dim(net):
    first = net.nets[0] if hasattr(net, "nets") else net
    return first.sizes[0]


class DQNPolicy:
    """The agent. Optional guard rail "up fast, down slow": it may always
    raise detail, but lowers it only after holding a level for dwell_down ticks.
    It never goes below its floor (stored with the weights; floor=... overrides)."""

    def __init__(self, net=None, agent=None, epsilon=0.0, rng=None, dwell_down=0, floor=None):
        self.net, self.agent, self.epsilon, self.dwell_down = net, agent, epsilon, dwell_down
        self.floor = floor if floor is not None else getattr(agent or net, "floor", 0)
        self.name = "dqn" if dwell_down == 0 else "dqn-guarded"
        self.rng = rng or np.random.default_rng(0)

    def reset(self):
        pass

    def decide(self, svc, state, obs, mem):
        if self.agent is not None:
            a = self.agent.act(state, self.epsilon)
        elif self.rng.random() < self.epsilon:
            a = int(self.rng.integers(len(ACTIONS)))
        else:
            x = state[:input_dim(self.net)]   # older agents read only the first 12 inputs
            a = int(np.argmax(self.net.forward(x[None, :])[0]))
        new = apply_action(mem.level, ACTIONS[a])
        if new < mem.level and mem.since_change < self.dwell_down:
            new = mem.level
        return max(new, self.floor), a


def baselines():
    return [StaticPolicy(MAX_LEVEL, "static-full"), StaticPolicy(1, "static-lean"),
            RulePolicy(), BurnRatePolicy()]
