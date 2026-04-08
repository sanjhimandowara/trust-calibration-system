import json
import os

import matplotlib.pyplot as plt
import numpy as np

from envs.trust_env import TrustCalibrationEnv
from agents.calibrator import PPOAgent
from training.baseline import simple_policy
from envs.grader import grade_easy, grade_medium, grade_hard


def _run_policy_on_task(env, policy_fn, episodes=10, seed_start=100):
    rewards = []
    correct = 0
    total = 0
    missed_escalate = 0
    false_escalate = 0

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed_start + ep)
        done = False
        ep_reward = 0.0

        while not done:
            action = policy_fn(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated

            if info.get("correct", False):
                correct += 1

            if info.get("true_label") == "ESCALATE" and info.get("decision") != "ESCALATE":
                missed_escalate += 1

            if info.get("decision") == "ESCALATE" and info.get("true_label") != "ESCALATE":
                false_escalate += 1

            total += 1

        rewards.append(float(ep_reward))

    metrics = {
        "correct": correct,
        "total": total,
        "missed_escalate": missed_escalate,
        "false_escalate": false_escalate,
    }

    return {
        "episodes": episodes,
        "avg_reward": round(float(np.mean(rewards)), 2),
        "runs": [round(float(r), 2) for r in rewards],
        "metrics": metrics,
    }


def evaluate_baseline(episodes=10):
    results = {}

    task_map = {
        "easy": grade_easy,
        "medium": grade_medium,
        "hard": grade_hard,
    }

    for difficulty, grader_fn in task_map.items():
        env = TrustCalibrationEnv(difficulty=difficulty, max_steps=20)
        task_result = _run_policy_on_task(env, simple_policy, episodes=episodes, seed_start=100)
        task_result["score"] = grader_fn(task_result["metrics"])
        results[difficulty] = task_result

    all_rewards = []
    for task_name in ["easy", "medium", "hard"]:
        all_rewards.extend(results[task_name]["runs"])

    results["overall"] = {
        "avg_reward": round(float(np.mean(all_rewards)), 2),
        "avg_score": round(float(np.mean([results["easy"]["score"], results["medium"]["score"], results["hard"]["score"]])), 3),
    }

    return results


def evaluate_ppo(model_path="results/checkpoints/ppo_model.pth", episodes=10):
    env_probe = TrustCalibrationEnv(difficulty="medium", max_steps=20)
    obs_dim = len(env_probe.reset()[0])
    action_dim = 6

    agent = PPOAgent(obs_dim, action_dim)
    agent.load(model_path)

    def ppo_policy(obs):
        action, _, _ = agent.act(obs)
        return action

    results = {}

    task_map = {
        "easy": grade_easy,
        "medium": grade_medium,
        "hard": grade_hard,
    }

    for difficulty, grader_fn in task_map.items():
        env = TrustCalibrationEnv(difficulty=difficulty, max_steps=20)
        task_result = _run_policy_on_task(env, ppo_policy, episodes=episodes, seed_start=500)
        task_result["score"] = grader_fn(task_result["metrics"])
        results[difficulty] = task_result

    all_rewards = []
    for task_name in ["easy", "medium", "hard"]:
        all_rewards.extend(results[task_name]["runs"])

    results["overall"] = {
        "avg_reward": round(float(np.mean(all_rewards)), 2),
        "avg_score": round(float(np.mean([results["easy"]["score"], results["medium"]["score"], results["hard"]["score"]])), 3),
    }

    return results


def save_comparison_plot(baseline_result: dict, ppo_result: dict):
    os.makedirs("outputs", exist_ok=True)

    baseline_runs = baseline_result["hard"]["runs"]
    ppo_runs = ppo_result["hard"]["runs"]

    baseline_avg = baseline_result["overall"]["avg_reward"]
    ppo_avg = ppo_result["overall"]["avg_reward"]

    plt.figure(figsize=(8, 5))
    plt.plot(baseline_runs, label="Baseline Runs", marker="o")
    plt.plot(ppo_runs, label="PPO Runs", marker="o")

    plt.axhline(baseline_avg, linestyle="--", label=f"Baseline Avg ({baseline_avg})")
    plt.axhline(ppo_avg, linestyle="--", label=f"PPO Avg ({ppo_avg})")

    plt.title("Baseline vs PPO Performance")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("outputs/baseline_vs_ppo.png", dpi=300)
    plt.close()


def main():
    baseline_result = evaluate_baseline(episodes=10)
    ppo_result = evaluate_ppo(episodes=10)

    save_comparison_plot(baseline_result, ppo_result)

    summary = {
        "baseline": baseline_result,
        "ppo": ppo_result,
    }

    os.makedirs("results/logs", exist_ok=True)
    with open("results/logs/evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== EVALUATION SUMMARY ===")
    print(f"Baseline Overall Avg Reward: {baseline_result['overall']['avg_reward']}")
    print(f"PPO Overall Avg Reward: {ppo_result['overall']['avg_reward']}")
    print(f"Baseline Easy Score: {baseline_result['easy']['score']}")
    print(f"Baseline Medium Score: {baseline_result['medium']['score']}")
    print(f"Baseline Hard Score: {baseline_result['hard']['score']}")
    print(f"PPO Easy Score: {ppo_result['easy']['score']}")
    print(f"PPO Medium Score: {ppo_result['medium']['score']}")
    print(f"PPO Hard Score: {ppo_result['hard']['score']}")
    print("Plot saved to outputs/baseline_vs_ppo.png")
    print("Summary saved to results/logs/evaluation_summary.json")
    print("==========================\n")


if __name__ == "__main__":
    main()