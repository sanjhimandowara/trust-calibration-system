import numpy as np
import random
from models.schemas import SignalBundle, N_SOURCES


ADVERSARIAL_MODES = [
    "none",
    "spoof_high",
    "spoof_low",
    "lagging",
    "mirror"
]


def generate_base_truth():
    """
    0 = IGNORE, 1 = INVESTIGATE, 2 = ESCALATE
    """
    return np.random.choice([0, 1, 2], p=[0.5, 0.3, 0.2])


def generate_clean_signals(true_label):
    """
    Generate base signals aligned with ground truth.
    """
    base_values = np.zeros(N_SOURCES, dtype=np.float32)

    for i in range(N_SOURCES):
        if true_label == 0:
            base_values[i] = np.random.uniform(0.0, 0.3)
        elif true_label == 1:
            base_values[i] = np.random.uniform(0.3, 0.7)
        else:
            base_values[i] = np.random.uniform(0.7, 1.0)

    return base_values


def apply_noise(values, scale=0.05):
    noise = np.random.normal(0.0, scale, size=N_SOURCES)
    return np.clip(values + noise, 0.0, 1.0)


def generate_confidences(values):
    """
    Confidence correlates with distance from 0.5.
    More extreme values often appear more 'certain'.
    """
    return np.clip(1.0 - np.abs(values - 0.5), 0.3, 1.0).astype(np.float32)


def apply_adversarial_behavior(values, confidences, mode, step_count):
    """
    Modify only the unreliable_feed (index 3).
    """
    v = values.copy()
    c = confidences.copy()

    if mode == "spoof_high":
        v[3] = np.random.uniform(0.9, 1.0)
        c[3] = 0.95

    elif mode == "spoof_low":
        v[3] = np.random.uniform(0.0, 0.1)
        c[3] = 0.95

    elif mode == "lagging":
        # Appears stale / ambiguous
        if step_count % 2 == 0:
            v[3] = np.random.uniform(0.2, 0.8)
            c[3] = 0.60

    elif mode == "mirror":
        v[3] = v[random.randint(0, 2)]
        c[3] = 0.80

    return np.clip(v, 0.0, 1.0), np.clip(c, 0.3, 1.0)


def generate_signal_bundle(step_count=0, difficulty="medium"):
    """
    Main signal generator.
    Easy   -> cleaner, low conflict
    Medium -> some conflict and lagging
    Hard   -> stronger adversarial behavior + more noise
    """
    true_label = generate_base_truth()

    values = generate_clean_signals(true_label)

    if difficulty == "easy":
        values = apply_noise(values, scale=0.03)
        confidences = generate_confidences(values)
        mode = "none"

    elif difficulty == "medium":
        values = apply_noise(values, scale=0.06)
        confidences = generate_confidences(values)
        mode = random.choice(["none", "lagging"])
        values, confidences = apply_adversarial_behavior(values, confidences, mode, step_count)

    elif difficulty == "hard":
        values = apply_noise(values, scale=0.12)
        confidences = generate_confidences(values)

        # Stronger adversarial behavior on hard mode
        if np.random.rand() < 0.75:
            mode = random.choice(["spoof_high", "spoof_low", "mirror", "lagging"])
        else:
            mode = "none"

        values, confidences = apply_adversarial_behavior(values, confidences, mode, step_count)

        # Add extra instability across all sources so hard mode isn't trivially solved
        extra_noise = np.random.uniform(-0.18, 0.18, size=N_SOURCES)
        values = np.clip(values + extra_noise, 0.0, 1.0).astype(np.float32)
        confidences = generate_confidences(values)

        # Keep unreliable source extra deceptive if adversarial
        if mode in ["spoof_high", "spoof_low"]:
            confidences[3] = 0.95
        elif mode == "mirror":
            confidences[3] = 0.80
        elif mode == "lagging":
            confidences[3] = 0.60

    else:
        raise ValueError(f"Unknown difficulty: {difficulty}")

    return SignalBundle(
        values=np.array(values, dtype=np.float32),
        confidences=np.array(confidences, dtype=np.float32),
        true_label=int(true_label),
        adversarial_mode=mode
    )