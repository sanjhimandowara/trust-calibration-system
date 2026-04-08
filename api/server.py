from typing import Optional

from fastapi import FastAPI, Query

from envs.trust_env import TrustCalibrationEnv
from models.schemas import (
    HealthResponse,
    TaskInfo,
    ResetRequest,
    ResetResponse,
    StepRequest,
    StepResponse,
    StateResponse,
)
from envs.grader import grade_easy, grade_medium, grade_hard

app = FastAPI(title="Trust Calibration OpenEnv API")

ENV = TrustCalibrationEnv(difficulty="medium", max_steps=20)
CURRENT_OBS = None
CURRENT_TASK = "medium"

TASK_METRICS = {
    "correct": 0,
    "total": 0,
    "missed_escalate": 0,
    "false_escalate": 0,
}


def _reset_metrics():
    global TASK_METRICS
    TASK_METRICS = {
        "correct": 0,
        "total": 0,
        "missed_escalate": 0,
        "false_escalate": 0,
    }


def _normalize_task_name(task_name: Optional[str]) -> str:
    if not task_name:
        return CURRENT_TASK

    task_name = str(task_name).strip().lower()

    mapping = {
        "easy": "easy",
        "medium": "medium",
        "hard": "hard",
        "task_easy_stable": "easy",
        "task_medium_conflict": "medium",
        "task_hard_adversarial": "hard",
    }

    return mapping.get(task_name, "medium")


def _grade_for_task(task_name: str) -> float:
    task_name = _normalize_task_name(task_name)
    if task_name == "easy":
        return grade_easy(TASK_METRICS)
    if task_name == "medium":
        return grade_medium(TASK_METRICS)
    return grade_hard(TASK_METRICS)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        message="Trust Calibration API is running"
    )


@app.get("/tasks", response_model=list[TaskInfo])
def tasks() -> list[TaskInfo]:
    return [
        TaskInfo(
            id="task_easy_stable",
            title="Stable Low-Conflict Trust Calibration",
            difficulty="easy",
            description="Stable low-conflict signals with minimal ambiguity.",
            grader_endpoint="/grader?task=task_easy_stable",
            score_min=0.001,
            score_max=0.999,
        ),
        TaskInfo(
            id="task_medium_conflict",
            title="Mixed Conflict Decision-Making",
            difficulty="medium",
            description="Mixed conflict with moderate uncertainty.",
            grader_endpoint="/grader?task=task_medium_conflict",
            score_min=0.001,
            score_max=0.999,
        ),
        TaskInfo(
            id="task_hard_adversarial",
            title="Adversarial Signal Suppression",
            difficulty="hard",
            description="Adversarial unreliable signals requiring suppression.",
            grader_endpoint="/grader?task=task_hard_adversarial",
            score_min=0.001,
            score_max=0.999,
        ),
    ]


@app.get("/grader")
def grader(
    task: Optional[str] = Query(default=None),
    task_id: Optional[str] = Query(default=None),
    id: Optional[str] = Query(default=None),
):
    effective_task = _normalize_task_name(task or task_id or id)
    score = _grade_for_task(effective_task)

    return {
        "task": effective_task,
        "score": score,
        "metrics": TASK_METRICS,
    }


@app.post("/reset", response_model=ResetResponse)
def reset(req: Optional[ResetRequest] = None) -> ResetResponse:
    global ENV, CURRENT_OBS, CURRENT_TASK

    req = req or ResetRequest()
    difficulty = _normalize_task_name(getattr(req, "difficulty", "medium"))

    CURRENT_TASK = difficulty
    _reset_metrics()

    ENV = TrustCalibrationEnv(
        difficulty=difficulty,
        max_steps=20
    )

    CURRENT_OBS, info = ENV.reset(seed=req.seed)

    info["task"] = CURRENT_TASK
    info["score"] = _grade_for_task(CURRENT_TASK)

    return ResetResponse(
        observation=[float(x) for x in CURRENT_OBS.tolist()],
        info=info
    )


@app.post("/step", response_model=StepResponse)
def step(req: StepRequest) -> StepResponse:
    global CURRENT_OBS, TASK_METRICS

    if CURRENT_OBS is None:
        CURRENT_OBS, _ = ENV.reset(seed=42)

    obs, reward, terminated, truncated, info = ENV.step(req.action)
    CURRENT_OBS = obs

    TASK_METRICS["total"] += 1

    if info.get("correct", False):
        TASK_METRICS["correct"] += 1

    if info.get("true_label") == "ESCALATE" and info.get("decision") != "ESCALATE":
        TASK_METRICS["missed_escalate"] += 1

    if info.get("decision") == "ESCALATE" and info.get("true_label") != "ESCALATE":
        TASK_METRICS["false_escalate"] += 1

    info["task"] = CURRENT_TASK
    info["score"] = _grade_for_task(CURRENT_TASK)

    return StepResponse(
        observation=[float(x) for x in obs.tolist()],
        reward=float(reward),
        terminated=bool(terminated),
        truncated=bool(truncated),
        info=info
    )


@app.get("/state", response_model=StateResponse)
def state() -> StateResponse:
    adversarial_mode = "none"
    if ENV.last_bundle is not None:
        adversarial_mode = str(ENV.last_bundle.adversarial_mode)

    return StateResponse(
        step_count=int(ENV.current_step),
        max_steps=int(ENV.max_steps),
        difficulty=str(ENV.difficulty),
        done=bool(ENV.done),
        weights=[float(x) for x in ENV.state.weights.tolist()],
        suppressed=[bool(x) for x in ENV.state.suppressed.tolist()],
        adversarial_mode=adversarial_mode
    )
