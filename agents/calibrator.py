from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from agents.networks import ActorCritic


@dataclass
class RolloutBatch:
    obs: list
    actions: list
    log_probs: list
    rewards: list
    dones: list
    values: list


class PPOAgent:
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        entropy_coef: float = 0.02,
        value_coef: float = 0.5,
        device: str | None = None,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef

        self.model = ActorCritic(obs_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def act(self, obs: np.ndarray):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits, value = self.model(obs_t)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return int(action.item()), float(log_prob.item()), float(value.item())

    def evaluate_actions(self, obs_t: torch.Tensor, actions_t: torch.Tensor):
        logits, values = self.model(obs_t)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions_t)
        entropy = dist.entropy().mean()
        return log_probs, values.squeeze(-1), entropy

    def compute_gae(self, rewards, dones, values, next_value=0.0):
        advantages = []
        gae = 0.0
        values = values + [next_value]

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1.0 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1.0 - dones[t]) * gae
            advantages.insert(0, gae)

        returns = [a + v for a, v in zip(advantages, values[:-1])]
        return np.array(advantages, dtype=np.float32), np.array(returns, dtype=np.float32)

    def update(self, batch: RolloutBatch, epochs: int = 4, minibatch_size: int = 32) -> dict:
        advantages, returns = self.compute_gae(batch.rewards, batch.dones, batch.values)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        obs_t = torch.tensor(np.array(batch.obs), dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(batch.actions, dtype=torch.long, device=self.device)
        old_log_probs_t = torch.tensor(batch.log_probs, dtype=torch.float32, device=self.device)
        advantages_t = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)

        n = len(batch.obs)
        idxs = np.arange(n)
        last_stats = {}

        for _ in range(epochs):
            np.random.shuffle(idxs)

            for start in range(0, n, minibatch_size):
                mb_idx = idxs[start:start + minibatch_size]

                new_log_probs, values_pred, entropy = self.evaluate_actions(
                    obs_t[mb_idx],
                    actions_t[mb_idx]
                )

                ratio = torch.exp(new_log_probs - old_log_probs_t[mb_idx])

                surr1 = ratio * advantages_t[mb_idx]
                surr2 = torch.clamp(
                    ratio,
                    1.0 - self.clip_ratio,
                    1.0 + self.clip_ratio
                ) * advantages_t[mb_idx]

                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = nn.functional.mse_loss(values_pred, returns_t[mb_idx])
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()

                last_stats = {
                    "loss": float(loss.item()),
                    "policy_loss": float(policy_loss.item()),
                    "value_loss": float(value_loss.item()),
                    "entropy": float(entropy.item()),
                }

        return last_stats

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.model.state_dict()}, path)

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        self.model.load_state_dict(state_dict)
        self.model.eval()