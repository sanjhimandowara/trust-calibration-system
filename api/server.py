from typing import Optional

from fastapi import FastAPI

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

app = FastAPI(title="Trust Calibration OpenEnv API")

ENV = TrustCalibrationEnv(difficulty="medium", max_steps=20)
CURRENT_OBS = None


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
            description="Mostly aligned signals with low ambiguity and minimal adversarial behavior.",
            reward_focus="Basic decision accuracy and clean trust fusion.",
            grader="grade_easy",
            score_min=0.001,
            score_max=0.999,
        ),
        TaskInfo(
            id="task_medium_conflict",
            title="Mixed Conflict Decision-Making",
            difficulty="medium",
            description="Moderate conflict, uncertainty, and partial source disagreement.",
            reward_focus="Balanced investigation decisions under uncertainty.",
            grader="grade_medium",
            score_min=0.001,
            score_max=0.999,
        ),
        TaskInfo(
            id="task_hard_adversarial",
            title="Adversarial Signal Suppression",
            difficulty="hard",
            description="The unreliable feed may spoof, lag, or mirror other signals and should be handled safely.",
            reward_focus="Robust suppression and safe escalation behavior.",
            grader="grade_hard",
            score_min=0.001,
            score_max=0.999,
        ),
    ]


@app.post("/reset", response_model=ResetResponse)
def reset(req: Optional[ResetRequest] = None) -> ResetResponse:
    global ENV, CURRENT_OBS

    req = req or ResetRequest()

    ENV = TrustCalibrationEnv(
        difficulty=req.difficulty,
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
