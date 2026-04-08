import numpy as np
from envs.trust_env import TrustCalibrationEnv


def simple_policy(observation):
    """
    Deliberately simple, reproducible baseline.

    Uses only the raw average signal value.
    No confidence weighting.
    No conflict awareness.
    No adaptive suppression logic.
    """

    obs = np.array(observation, dtype=np.float32)
    avg_signal = float(np.mean(obs[:4]))

    if avg_signal < 0.35:
        return 0  # equal trust
    elif avg_signal < 0.70:
        return 1  # favor anomaly + network
    else:
        return 2  # favor rule + anomaly


def run_episode(env, seed=42):
    obs, _ = env.reset(seed=seed)

    total_reward = 0.0
    steps = 0
    done = False
    final_info = {}

    while not done:
        action = simple_policy(obs)
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        steps += 1
        done = terminated or truncated
        final_info = info

    return {
        "total_reward": round(float(total_reward), 2),
        "steps": steps,
        "final_info": final_info,
    }


def run_baseline(episodes=5, difficulty="medium"):
    env = TrustCalibrationEnv(difficulty=difficulty, max_steps=20)
    results = []

    for i in range(episodes):
        result = run_episode(env, seed=42 + i)
        results.append(result)

    avg_reward = np.mean([r["total_reward"] for r in results])

    summary = {
        "episodes": episodes,
        "difficulty": difficulty,
        "avg_reward": round(float(avg_reward), 2),
        "runs": results,
    }

    return summary


if __name__ == "__main__":
    result = run_baseline(episodes=5)

    print("\n=== BASELINE RESULTS ===")
    print(f"Difficulty: {result['difficulty']}")
    print(f"Average Reward: {result['avg_reward']}")
    print("========================\n")

    for i, run in enumerate(result["runs"]):
        print(f"Run {i+1}: Reward={run['total_reward']} Steps={run['steps']}")