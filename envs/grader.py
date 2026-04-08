def _safe_ratio(num: int, den: int) -> float:
    return float(num) / float(max(den, 1))


def _strict_unit_interval(value: float) -> float:
    """
    Force score strictly inside (0, 1), never exactly 0.0 or 1.0.
    """
    if value <= 0.0:
        return 0.001
    if value >= 1.0:
        return 0.999
    return round(float(value), 3)


def _score_from_metrics(metrics: dict, missed_weight: float, false_weight: float) -> float:
    correct = int(metrics.get("correct", 0))
    total = int(metrics.get("total", 0))
    missed_escalate = int(metrics.get("missed_escalate", 0))
    false_escalate = int(metrics.get("false_escalate", 0))

    # If no steps yet, return a valid in-range neutral score.
    if total <= 0:
        return 0.500

    accuracy = _safe_ratio(correct, total)
    missed_penalty = missed_weight * _safe_ratio(missed_escalate, total)
    false_penalty = false_weight * _safe_ratio(false_escalate, total)

    score = accuracy - missed_penalty - false_penalty
    score = max(0.0, min(1.0, score))
    return _strict_unit_interval(score)


def grade_easy(metrics: dict) -> float:
    return _score_from_metrics(metrics, missed_weight=0.10, false_weight=0.05)


def grade_medium(metrics: dict) -> float:
    return _score_from_metrics(metrics, missed_weight=0.15, false_weight=0.08)


def grade_hard(metrics: dict) -> float:
    return _score_from_metrics(metrics, missed_weight=0.35, false_weight=0.20)
