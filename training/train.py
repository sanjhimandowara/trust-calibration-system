import numpy as np

from envs.trust_env import TrustCalibrationEnv
from agents.calibrator import PPOAgent, RolloutBatch


def train(
    episodes: int = 300,
    rollout_steps: int = 200,
    save_path: str = "results/checkpoints/ppo_model.pth",
):
    env = TrustCalibrationEnv(difficulty="medium", max_steps=20)

    obs_dim = len(env.reset()[0])
    action_dim = 6

    agent = PPOAgent(obs_dim, action_dim)

    all_rewards = []

    for episode in range(1, episodes + 1):
        obs, _ = env.reset(seed=42 + episode)

        batch = RolloutBatch(
            obs=[],
            actions=[],
            log_probs=[],
            rewards=[],
            dones=[],
            values=[],
        )

        total_reward = 0.0
        steps = 0

        while steps < rollout_steps:
            action, log_prob, value = agent.act(obs)

            next_obs, reward, terminated, truncated, _ = env.step(action)

            done = terminated or truncated

            batch.obs.append(obs)
            batch.actions.append(action)
            batch.log_probs.append(log_prob)
            batch.rewards.append(reward)
            batch.dones.append(float(done))
            batch.values.append(value)

            total_reward += reward
            obs = next_obs
            steps += 1

            if done:
                obs, _ = env.reset()

        stats = agent.update(batch)

        all_rewards.append(total_reward)

        if episode % 10 == 0:
            avg_reward = np.mean(all_rewards[-10:])
            print(
                f"Episode {episode} | "
                f"AvgReward: {avg_reward:.2f} | "
                f"Loss: {stats.get('loss', 0):.3f}"
            )

    agent.save(save_path)
    print(f"\nModel saved to {save_path}")


if __name__ == "__main__":
    train()