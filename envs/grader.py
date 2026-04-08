def clamp_score(value: float) -> float:
    value = round(float(value), 2)
    if value <= 0.0:
        return 0.01
    if value >= 1.0:
        return 0.99
    return value


def compute_continuous_score(metrics: dict, task: str) -> float:
    total = max(1, int(metrics.get("total", 0)))
    correct = float(metrics.get("correct", 0)) / total
    missed = float(metrics.get("missed_escalate", 0)) / total
    false_escalate = float(metrics.get("false_escalate", 0)) / total
    conflict = float(metrics.get("avg_conflict", 0.0))
    uncertainty = float(metrics.get("avg_uncertainty", 0.0))

    score = (
        0.70 * correct
        + 0.15 * (1.0 - min(1.0, conflict))
        + 0.15 * (1.0 - min(1.0, uncertainty))
        - 0.20 * missed
        - 0.10 * false_escalate
    )

    if task == "easy":
        score += 0.03
    elif task == "hard":
        score -= 0.03

    return clamp_score(score)


def grade_easy(metrics: dict) -> float:
    return compute_continuous_score(metrics, "easy")


def grade_medium(metrics: dict) -> float:
    return compute_continuous_score(metrics, "medium")


def grade_hard(metrics: dict) -> float:
    return compute_continuous_score(metrics, "hard")
