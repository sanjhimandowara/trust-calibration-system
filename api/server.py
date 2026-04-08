from typing import Optional

from fastapi import FastAPI, Query

from envs.trust_env import TrustCalibrationEnv
from models.schemas import (
    HealthResponse,
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


def _grade_current_task() -> float:
    if CURRENT_TASK == "easy":
        return grade_easy(TASK_METRICS)
    if CURRENT_TASK == "medium":
        return grade_medium(TASK_METRICS)
    return grade_hard(TASK_METRICS)


@app.get("/")
def root():
    return {
        "message": "Trust Calibration API Running",
        "health": "/health",
        "tasks": "/tasks",
        "grader": "/grader",
        "state": "/state",
    }


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        message="Trust Calibration API is running"
    )


@app.get("/tasks")
def tasks():
    # Keep it simple. Validator likely wants explicit task ids.
    return ["easy", "medium", "hard"]


@app.get("/grader")
def grader(task: Optional[str] = Query(default=None)):
    effective_task = task or CURRENT_TASK

    if effective_task == "easy":
        score = grade_easy(TASK_METRICS)
    elif effective_task == "medium":
        score = grade_medium(TASK_METRICS)
    else:
        score = grade_hard(TASK_METRICS)

    return {
        "task": effective_task,
        "score": score,
        "metrics": TASK_METRICS,
    }


@app.post("/reset", response_model=ResetResponse)
def reset(req: Optional[ResetRequest] = None) -> ResetResponse:
    global ENV, CURRENT_OBS, CURRENT_TASK

    req = req or ResetRequest()
    difficulty = getattr(req, "difficulty", "medium") or "medium"

    if difficulty not in ["easy", "medium", "hard"]:
        difficulty = "medium"

    CURRENT_TASK = difficulty
    _reset_metrics()

    ENV = TrustCalibrationEnv(
        difficulty=difficulty,
        max_steps=20
    )

    CURRENT_OBS, info = ENV.reset(seed=req.seed)

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

    # Update task metrics for grader
    TASK_METRICS["total"] += 1

    if info.get("correct", False):
        TASK_METRICS["correct"] += 1

    if info.get("true_label") == "ESCALATE" and info.get("decision") != "ESCALATE":
        TASK_METRICS["missed_escalate"] += 1

    if info.get("decision") == "ESCALATE" and info.get("true_label") != "ESCALATE":
        TASK_METRICS["false_escalate"] += 1

    # Expose current score in step info too
    info["task"] = CURRENT_TASK
    info["score"] = _grade_current_task()

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
