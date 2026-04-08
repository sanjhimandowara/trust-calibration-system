def _safe_ratio(num: int, den: int) -> float:
    return float(num) / float(max(den, 1))


def compute_score(metrics: dict) -> float:
    """
    Deterministic normalized grader score in [0.0, 1.0].

    metrics expected:
    - correct
    - total
    - missed_escalate
    - false_escalate
    """
    correct = int(metrics.get("correct", 0))
    total = int(metrics.get("total", 0))
    missed_escalate = int(metrics.get("missed_escalate", 0))
    false_escalate = int(metrics.get("false_escalate", 0))

    accuracy = _safe_ratio(correct, total)

    missed_penalty = 0.15 * _safe_ratio(missed_escalate, total)
    false_penalty = 0.08 * _safe_ratio(false_escalate, total)

    score = accuracy - missed_penalty - false_penalty
    return round(max(0.0, min(1.0, score)), 3)


def grade_easy(metrics: dict) -> float:
    """
    Easy task: lighter penalties because the setting is cleaner.
    """
    correct = int(metrics.get("correct", 0))
    total = int(metrics.get("total", 0))
    missed_escalate = int(metrics.get("missed_escalate", 0))
    false_escalate = int(metrics.get("false_escalate", 0))

    accuracy = _safe_ratio(correct, total)
    score = (
        accuracy
        - 0.10 * _safe_ratio(missed_escalate, total)
        - 0.05 * _safe_ratio(false_escalate, total)
    )
    return round(max(0.0, min(1.0, score)), 3)


def grade_medium(metrics: dict) -> float:
    """
    Medium task: balanced penalties.
    """
    return compute_score(metrics)


def grade_hard(metrics: dict) -> float:
    """
    Hard task: much stricter penalties.
    Missing critical escalations should hurt a lot more here.
    """
    correct = int(metrics.get("correct", 0))
    total = int(metrics.get("total", 0))
    missed_escalate = int(metrics.get("missed_escalate", 0))
    false_escalate = int(metrics.get("false_escalate", 0))

    accuracy = _safe_ratio(correct, total)

    score = (
        accuracy
        - 0.35 * _safe_ratio(missed_escalate, total)
        - 0.20 * _safe_ratio(false_escalate, total)
    )

    return round(max(0.0, min(1.0, score)), 3)