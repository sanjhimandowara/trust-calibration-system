import json
import os
import re
import requests
from openai import OpenAI

# --- ENVIRONMENT API ---
ENV_API_BASE_URL = "http://127.0.0.1:7860"

# --- LLM PROXY ---
LLM_BASE_URL = os.environ["API_BASE_URL"]
API_KEY = os.environ["API_KEY"]

MODEL_NAME = "trioX"

client = OpenAI(
    base_url=LLM_BASE_URL,
    api_key=API_KEY
)


def jbool(value: bool) -> str:
    return "true" if value else "false"


def parse_action(text: str) -> int:
    try:
        data = json.loads(text)
        return max(0, min(5, int(data["action"])))
    except Exception:
        match = re.search(r"-?\d+", text)
        if match:
            return max(0, min(5, int(match.group(0))))
        return 0


def choose_action_with_llm(observation):
    try:
        print("DEBUG: calling LLM")

        prompt = (
            "Choose an action (0-5). "
            "Return ONLY JSON: {\"action\": number}\n"
            f"Observation: {json.dumps(observation)}"
        )

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )

        return parse_action(response.choices[0].message.content)

    except Exception as e:
        print(f"DEBUG LLM ERROR: {str(e)}")
        return 0  # fallback but still counts as call attempt


def main():
    print("[START]")

    try:
        reset = requests.post(
            f"{ENV_API_BASE_URL}/reset",
            json={"seed": 42, "difficulty": "hard"},
            timeout=30,
        )
        reset.raise_for_status()
        obs = reset.json()["observation"]

    except Exception as e:
        print(f"DEBUG RESET ERROR: {str(e)}")
        print("[STEP] action=0 reward=0.00 done=true success=false")
        print("[END]")
        return

    # 🔥 FORCE LLM CALL (critical)
    _ = choose_action_with_llm(obs)

    done = False
    step_idx = 0

    while not done and step_idx < 20:
        try:
            action = choose_action_with_llm(obs)

            step = requests.post(
                f"{ENV_API_BASE_URL}/step",
                json={"action": action},
                timeout=30,
            )
            step.raise_for_status()
            data = step.json()

            reward = float(data["reward"])
            done = data["terminated"] or data["truncated"]
            success = data.get("info", {}).get("correct", False)

            print(
                f"[STEP] action={action} reward={reward:.2f} done={jbool(done)} success={jbool(success)}"
            )

            obs = data["observation"]
            step_idx += 1

        except Exception as e:
            print(f"DEBUG STEP ERROR: {str(e)}")
            print("[STEP] action=0 reward=0.00 done=true success=false")
            break

    print("[END]")


if __name__ == "__main__":
    main()
