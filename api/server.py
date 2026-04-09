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
    "avg_conflict": 0.0,
    "avg_uncertainty": 0.0,
}


def reset_metrics():
    global TASK_METRICS
    TASK_METRICS = {
        "correct": 0,
        "total": 0,
        "missed_escalate": 0,
        "false_escalate": 0,
        "avg_conflict": 0.0,
        "avg_uncertainty": 0.0,
    }


def normalize_task_name(task_name: Optional[str]) -> str:
    if not task_name:
        return CURRENT_TASK

    task_name = str(task_name).strip().lower()

    mapping = {
        "easy": "easy",
        "medium": "medium",
        "hard": "hard",
    }

    return mapping.get(task_name, "medium")


def grade_for_task(task_name: str) -> float:
    task_name = normalize_task_name(task_name)

    if task_name == "easy":
        score = grade_easy(TASK_METRICS)
    elif task_name == "medium":
        score = grade_medium(TASK_METRICS)
    else:
        score = grade_hard(TASK_METRICS)

    if score <= 0.0:
        score = 0.01
    if score >= 1.0:
        score = 0.99

    return round(float(score), 2)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        message="Trust Calibration API is running"
    )


@app.get("/tasks")
def tasks():
    return {
        "tasks": [
            {"id": "easy"},
            {"id": "medium"},
            {"id": "hard"},
        ]
    }


@app.get("/grader")
def grader(task: str = Query(...)):
    score = grade_for_task(task)
    return {"score": score}


@app.post("/reset", response_model=ResetResponse)
def reset(req: Optional[ResetRequest] = None) -> ResetResponse:
    global ENV, CURRENT_OBS, CURRENT_TASK

    req = req or ResetRequest()
    difficulty = normalize_task_name(getattr(req, "difficulty", "medium"))

    CURRENT_TASK = difficulty
    reset_metrics()

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

    TASK_METRICS["total"] += 1

    if info.get("correct", False):
        TASK_METRICS["correct"] += 1

    if info.get("true_label") == "ESCALATE" and info.get("decision") != "ESCALATE":
        TASK_METRICS["missed_escalate"] += 1

    if info.get("decision") == "ESCALATE" and info.get("true_label") != "ESCALATE":
        TASK_METRICS["false_escalate"] += 1

    current_total = TASK_METRICS["total"]

    TASK_METRICS["avg_conflict"] = (
        ((current_total - 1) * TASK_METRICS["avg_conflict"])
        + float(info.get("conflict", 0.0))
    ) / current_total

    TASK_METRICS["avg_uncertainty"] = (
        ((current_total - 1) * TASK_METRICS["avg_uncertainty"])
        + float(info.get("uncertainty", 0.0))
    ) / current_total

    info["task"] = CURRENT_TASK
    info["score"] = grade_for_task(CURRENT_TASK)

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
