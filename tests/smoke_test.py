from envs.trust_env import TrustCalibrationEnv


def run_smoke_test():
    env = TrustCalibrationEnv()

    print("Running smoke test...\n")

    obs, _ = env.reset(seed=42)

    print("Initial observation shape:", len(obs))

    done = False
    steps = 0

    while not done and steps < 5:
        action = steps % 6

        obs, reward, terminated, truncated, info = env.step(action)

        print(f"\nStep {steps+1}")
        print("Action:", action)
        print("Reward:", round(reward, 3))
        print("Decision:", info["decision"])
        print("Conflict:", round(info["conflict"], 3))
        print("Uncertainty:", round(info["uncertainty"], 3))

        done = terminated or truncated
        steps += 1

    print("\nSmoke test completed.\n")


if __name__ == "__main__":
    run_smoke_test()