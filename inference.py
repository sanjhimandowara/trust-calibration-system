import json
import os
import re
import requests
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:7860")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")

HF_TOKEN = os.getenv("HF_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

API_KEY = OPENAI_API_KEY or HF_TOKEN

if not API_KEY:
    raise EnvironmentError("Either OPENAI_API_KEY or HF_TOKEN is required.")


def jbool(value: bool) -> str:
    return "true" if value else "false"


def parse_action(text: str) -> int:
    try:
        data = json.loads(text)
        action = int(data["action"])
        return max(0, min(5, action))
    except Exception:
        match = re.search(r"-?\d+", text)
        if match:
            return max(0, min(5, int(match.group(0))))
        return 0


def choose_action_with_llm(observation: list[float]) -> int:
    if OPENAI_API_KEY:
        client = OpenAI(api_key=OPENAI_API_KEY)
    else:
        client = OpenAI(
            api_key=HF_TOKEN,
            base_url="https://router.huggingface.co/v1",
        )

    prompt = (
        "You are choosing exactly one action for a trust calibration RL environment. "
        "Valid actions are integers 0 to 5 only. "
        "Return ONLY strict JSON in exactly this format: "
        "{\"action\": <integer between 0 and 5>}.\n"
        f"Observation: {json.dumps(observation)}"
    )

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": "Return ONLY JSON. No explanation. Format: {\"action\": integer between 0 and 5}.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0.0,
    )

    text = response.choices[0].message.content.strip()
    return parse_action(text)


def main() -> None:
    print("[START]")
    try:
        reset_resp = requests.post(
            f"{API_BASE_URL}/reset",
            json={"seed": 42, "difficulty": "hard"},
            timeout=30,
        )
        reset_resp.raise_for_status()
        reset_payload = reset_resp.json()
        observation = reset_payload["observation"]

        done = False
        step_idx = 0

        while not done and step_idx < 20:
            action = choose_action_with_llm(observation)

            step_resp = requests.post(
                f"{API_BASE_URL}/step",
                json={"action": action},
                timeout=30,
            )
            step_resp.raise_for_status()
            step_payload = step_resp.json()

            reward = float(step_payload["reward"])
            terminated = bool(step_payload["terminated"])
            truncated = bool(step_payload["truncated"])
            done = terminated or truncated

            info = step_payload.get("info", {})
            success = bool(info.get("correct", False))

            print(
                f"[STEP] action={action} reward={reward:.2f} done={jbool(done)} success={jbool(success)}"
            )

            observation = step_payload["observation"]
            step_idx += 1

    except Exception:
        print("[STEP] action=0 reward=0.00 done=true success=false")
    finally:
        print("[END]")


if __name__ == "__main__":
    main()