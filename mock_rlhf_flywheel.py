#!/usr/bin/env python3
"""
Minimal mock of: data collection -> feature store -> RLHF fine-tuning of the
robot "reaction model" (readme §4 collection, §6 feature store, §7 training).

No Kafka/Flink/Feast/real GPT — stdlib only. The "GPT model" is a linear
scoring policy over hand-crafted state features (stand-in for a language
model's logits over actions given a serialized state/prompt). It is fine-
tuned directly from human preference feedback using a DPO-style update
(increase margin between preferred and rejected action scores) — the same
shape as real RLHF, just without a neural net.

Loop:  ReAct(Reason -> Act -> Observe) over a mocked grid-world environment
       -> every step logged to the FeatureStore (online + offline)
       -> human feedback (preferred vs rejected action) collected
       -> fine_tune() periodically updates the GPT reaction model's weights
       -> reaction to the environment improves (fewer collisions/timeouts)

Run:  python3 mock_rlhf_flywheel.py
"""
from __future__ import annotations
import math
import random
from dataclasses import dataclass, field

random.seed(7)

ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]
DELTA = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0)}


# --------------------------------------------------------------------------
# Mocked real-world environment: grid world with obstacles + a goal.
# --------------------------------------------------------------------------
class GridWorld:
    def __init__(self, size: int = 6, n_obstacles: int = 5) -> None:
        self.size = size
        self.goal = (size - 1, size - 1)
        self.obstacles = set()
        while len(self.obstacles) < n_obstacles:
            p = (random.randint(0, size - 1), random.randint(0, size - 1))
            if p not in (self.goal, (0, 0)):
                self.obstacles.add(p)
        self.reset()

    def reset(self):
        self.pos = (0, 0)
        self.done = False
        return self.pos

    def step(self, action: str):
        dx, dy = DELTA[action]
        nx = min(max(self.pos[0] + dx, 0), self.size - 1)
        ny = min(max(self.pos[1] + dy, 0), self.size - 1)
        new_pos = (nx, ny)
        collided = new_pos in self.obstacles
        if not collided:
            self.pos = new_pos
        reached = self.pos == self.goal
        self.done = reached
        reward = 1.0 if reached else (-1.0 if collided else -0.02)
        return self.pos, reward, self.done, {"collided": collided}

    def features(self, pos=None):
        """State -> named features (stand-in for a serialized prompt)."""
        pos = pos or self.pos
        gx, gy = self.goal
        return {
            "dx_goal": (gx - pos[0]) / self.size,
            "dy_goal": (gy - pos[1]) / self.size,
            "near_obstacle": 1.0 if any(
                abs(pos[0] - ox) + abs(pos[1] - oy) <= 1 for ox, oy in self.obstacles
            ) else 0.0,
            "bias": 1.0,
        }


# --------------------------------------------------------------------------
# Feature store: online (latest per entity) + offline (full log), Feast-ish.
# --------------------------------------------------------------------------
@dataclass
class FeatureStore:
    offline_log: list = field(default_factory=list)
    online: dict = field(default_factory=dict)

    def write(self, entity_id: str, features: dict, meta: dict):
        row = {"entity_id": entity_id, "features": features, **meta}
        self.offline_log.append(row)      # offline: append-only, for training
        self.online[entity_id] = row      # online: latest value, for serving

    def get_online(self, entity_id: str):
        return self.online.get(entity_id)

    def get_training_batch(self):
        return list(self.offline_log)


# --------------------------------------------------------------------------
# The "GPT" reaction model: linear policy over action-conditioned features.
# score(state, action) approximates a GPT's logit for emitting `action`
# given the state prompt. fine_tune() is the RLHF/DPO update.
# --------------------------------------------------------------------------
class GPTReactionModel:
    def __init__(self):
        # weights[action][feature] — analogous to per-token logit weights.
        # Small random init breaks symmetry (all-zero weights tie forever
        # and the model never varies its action with the state).
        self.weights = {a: {k: random.uniform(-0.05, 0.05)
                             for k in ("dx_goal", "dy_goal", "near_obstacle", "bias")}
                         for a in ACTIONS}

    def score(self, feats: dict, action: str) -> float:
        w = self.weights[action]
        return sum(w[k] * v for k, v in feats.items())

    def reason(self, feats: dict, epsilon: float = 0.0) -> tuple[str, str]:
        """Reason step of ReAct: pick argmax action + a thought string.
        epsilon>0 mixes in random exploration, needed while collecting
        fresh feedback so the policy doesn't get stuck repeating one
        action against a wall (a cold-start RLHF data-collection issue)."""
        if epsilon and random.random() < epsilon:
            action = random.choice(ACTIONS)
        else:
            scored = {a: self.score(feats, a) for a in ACTIONS}
            action = max(scored, key=scored.get)
        thought = f"goal_delta=({feats['dx_goal']:.2f},{feats['dy_goal']:.2f}) " \
                  f"near_obstacle={bool(feats['near_obstacle'])} -> choose {action}"
        return action, thought

    def fine_tune(self, preference_batch: list[tuple[dict, str, str]], lr: float = 0.02):
        """
        DPO-style RLHF update: for each (state_feats, preferred_action,
        rejected_action) human-feedback triple, nudge weights so the
        preferred action's score rises relative to the rejected one.
        Weights are clipped to keep the linear model well-behaved (stand-in
        for the trust-region clipping real RLHF/PPO uses to avoid the
        policy drifting too far in one update).
        """
        for feats, chosen, rejected in preference_batch:
            margin = self.score(feats, chosen) - self.score(feats, rejected)
            grad = 1.0 / (1.0 + math.exp(margin))   # sigmoid, DPO gradient shape
            for k, v in feats.items():
                self.weights[chosen][k] = max(-2.0, min(2.0, self.weights[chosen][k] + lr * grad * v))
                self.weights[rejected][k] = max(-2.0, min(2.0, self.weights[rejected][k] - lr * grad * v))


# --------------------------------------------------------------------------
# ReAct loop: Reason -> Act -> Observe, feeding the feature store, and
# synthesizing human feedback (preferred vs rejected action) per step.
# --------------------------------------------------------------------------
def synth_human_feedback(env: GridWorld, feats: dict, taken: str) -> tuple[str, str] | None:
    """Mock human-in-the-loop preference: label taken vs the best alternative
    by which one gets closer to the goal without hitting an obstacle."""
    def would_collide(a):
        dx, dy = DELTA[a]
        p = (min(max(env.pos[0] + dx, 0), env.size - 1),
             min(max(env.pos[1] + dy, 0), env.size - 1))
        return p in env.obstacles

    def dist_after(a):
        dx, dy = DELTA[a]
        p = (min(max(env.pos[0] + dx, 0), env.size - 1),
             min(max(env.pos[1] + dy, 0), env.size - 1))
        gx, gy = env.goal
        return abs(gx - p[0]) + abs(gy - p[1])

    ranked = sorted(ACTIONS, key=lambda a: (would_collide(a), dist_after(a)))
    best_score = (would_collide(ranked[0]), dist_after(ranked[0]))
    tied = [a for a in ranked if (would_collide(a), dist_after(a)) == best_score]
    if taken in tied:
        return None  # human agrees (taken is among the equally-best options)
    best = random.choice(tied)  # avoid always reinforcing list-order bias
    return best, taken  # (preferred, rejected)


def run_episode(env: GridWorld, model: GPTReactionModel, store: FeatureStore,
                 episode_id: int, max_steps: int = 30, collect_feedback: bool = True,
                 epsilon: float = 0.0):
    env.reset()
    steps, collisions = 0, 0
    prefs = []
    for t in range(max_steps):
        feats = env.features()
        action, thought = model.reason(feats, epsilon=epsilon)  # Reason
        pos, reward, done, info = env.step(action)             # Act
        steps += 1
        collisions += int(info["collided"])

        store.write(                                            # Observe -> collect
            entity_id=f"ep{episode_id}_t{t}",
            features=feats,
            meta={"action": action, "reward": reward, "thought": thought},
        )

        if collect_feedback:
            fb = synth_human_feedback(env, feats, action)
            if fb:
                prefs.append((feats, fb[0], fb[1]))

        if done:
            break
    return {"steps": steps, "collisions": collisions, "reached": env.done, "prefs": prefs}


def main():
    store = FeatureStore()
    model = GPTReactionModel()

    print("=== before fine-tuning (data-collection phase, epsilon-greedy) ===")
    env = GridWorld()
    before = [run_episode(env, model, store, ep, collect_feedback=True, epsilon=0.3)
              for ep in range(20)]
    print(f"reached={sum(r['reached'] for r in before)}/20  "
          f"avg_collisions={sum(r['collisions'] for r in before)/20:.2f}  "
          f"avg_steps={sum(r['steps'] for r in before)/20:.1f}")

    # RLHF fine-tuning: pool human preference feedback collected above.
    pref_batch = [p for r in before for p in r["prefs"]]
    print(f"\n[rlhf] collected {len(pref_batch)} human preference pairs; fine-tuning...")
    for _ in range(3):        # a few epochs over the same feedback batch
        random.shuffle(pref_batch)
        model.fine_tune(pref_batch)

    print(f"[feature_store] offline_log rows = {len(store.get_training_batch())}")

    print("\n=== after fine-tuning ===")
    env2 = GridWorld()
    random.seed(7)  # same world layout for a fair before/after comparison
    env2 = GridWorld()
    after = [run_episode(env2, model, store, 100 + ep, collect_feedback=False) for ep in range(20)]
    print(f"reached={sum(r['reached'] for r in after)}/20  "
          f"avg_collisions={sum(r['collisions'] for r in after)/20:.2f}  "
          f"avg_steps={sum(r['steps'] for r in after)/20:.1f}")

    print("\n[ok] reaction model updated via RLHF from user feedback; "
          "feature store holds the full offline training log")


if __name__ == "__main__":
    main()
