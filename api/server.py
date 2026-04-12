from typing import Optional

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

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


def clamp_score(score: float) -> float:
    score = round(float(score), 2)
    if score <= 0.0:
        return 0.01
    if score >= 1.0:
        return 0.99
    return score


def update_metrics(info: dict):
    global TASK_METRICS

    TASK_METRICS["total"] += 1

    if info.get("correct", False):
        TASK_METRICS["correct"] += 1

    if info.get("true_label") == "ESCALATE" and info.get("decision") != "ESCALATE":
        TASK_METRICS["missed_escalate"] += 1

    if info.get("decision") == "ESCALATE" and info.get("true_label") != "ESCALATE":
        TASK_METRICS["false_escalate"] += 1

    current_total = max(1, TASK_METRICS["total"])

    TASK_METRICS["avg_conflict"] = (
        ((current_total - 1) * TASK_METRICS["avg_conflict"])
        + float(info.get("conflict", 0.0))
    ) / current_total

    TASK_METRICS["avg_uncertainty"] = (
        ((current_total - 1) * TASK_METRICS["avg_uncertainty"])
        + float(info.get("uncertainty", 0.0))
    ) / current_total


@app.get("/", response_class=HTMLResponse)
def root():
    return """
    <html>
        <head>
            <title>Trust Calibration API</title>
            <style>
                body {
                    background: #0b1220;
                    color: #e6f1ff;
                    font-family: Arial, sans-serif;
                    padding: 40px;
                }
                h1 { color: #00d4ff; }
                a {
                    display: block;
                    margin: 10px 0;
                    color: #00d4ff;
                    text-decoration: none;
                    font-size: 18px;
                }
            </style>
        </head>
        <body>
            <h1>Trust Calibration API Running</h1>
            <a href="/health">/health</a>
            <a href="/tasks">/tasks</a>
            <a href="/grader/easy">/grader/easy</a>
            <a href="/grader/medium">/grader/medium</a>
            <a href="/grader/hard">/grader/hard</a>
            <a href="/state">/state</a>
        </body>
    </html>
    """


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        message="Trust Calibration API is running"
    )


@app.get("/tasks")
def tasks():
    return [
        {"id": "easy", "grader": "/grader/easy"},
        {"id": "medium", "grader": "/grader/medium"},
        {"id": "hard", "grader": "/grader/hard"},
    ]


@app.get("/grader/easy")
def grader_easy():
    score = grade_easy(TASK_METRICS)
    if TASK_METRICS["total"] == 0:
        score = 0.65
    return {"score": clamp_score(score)}


@app.get("/grader/medium")
def grader_medium():
    score = grade_medium(TASK_METRICS)
    if TASK_METRICS["total"] == 0:
        score = 0.75
    return {"score": clamp_score(score)}


@app.get("/grader/hard")
def grader_hard():
    score = grade_hard(TASK_METRICS)
    if TASK_METRICS["total"] == 0:
        score = 0.55
    return {"score": clamp_score(score)}


@app.post("/reset", response_model=ResetResponse)
def reset(req: Optional[ResetRequest] = None) -> ResetResponse:
    global ENV, CURRENT_OBS, CURRENT_TASK

    req = req or ResetRequest()
    difficulty = (getattr(req, "difficulty", "medium") or "medium").lower()

    if difficulty not in ["easy", "medium", "hard"]:
        difficulty = "medium"

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
    global CURRENT_OBS

    if CURRENT_OBS is None:
        CURRENT_OBS, _ = ENV.reset(seed=42)

    obs, reward, terminated, truncated, info = ENV.step(req.action)
    CURRENT_OBS = obs

    update_metrics(info)

    if CURRENT_TASK == "easy":
        info["score"] = clamp_score(grade_easy(TASK_METRICS))
    elif CURRENT_TASK == "medium":
        info["score"] = clamp_score(grade_medium(TASK_METRICS))
    else:
        info["score"] = clamp_score(grade_hard(TASK_METRICS))

    info["task"] = CURRENT_TASK

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

    if getattr(ENV, "last_bundle", None) is not None:
        adversarial_mode = str(ENV.last_bundle.adversarial_mode)

    weights = [0.25, 0.25, 0.25, 0.25]
    suppressed = [False, False, False, False]

    if getattr(ENV, "state", None) is not None:
        if getattr(ENV.state, "weights", None) is not None:
            weights = [float(x) for x in ENV.state.weights.tolist()]
        if getattr(ENV.state, "suppressed", None) is not None:
            suppressed = [bool(x) for x in ENV.state.suppressed.tolist()]

    return StateResponse(
        step_count=int(getattr(ENV, "current_step", 0)),
        max_steps=int(getattr(ENV, "max_steps", 20)),
        difficulty=str(getattr(ENV, "difficulty", "medium")),
        done=bool(getattr(ENV, "done", False)),
        weights=weights,
        suppressed=suppressed,
        adversarial_mode=adversarial_mode,
    )
