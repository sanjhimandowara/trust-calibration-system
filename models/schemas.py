from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np
from pydantic import BaseModel, Field

# ---- CONSTANTS ----

SOURCE_NAMES = [
    "anomaly_detector",
    "network_monitor",
    "rule_engine",
    "unreliable_feed",
]

DECISIONS = ["IGNORE", "INVESTIGATE", "ESCALATE"]

N_SOURCES = 4
N_ACTIONS = 6
OBS_DIM = 22

# ---- INTERNAL DATA STRUCTURES ----

@dataclass
class SignalBundle:
    values: np.ndarray
    confidences: np.ndarray
    true_label: int
    adversarial_mode: str


@dataclass
class TrustState:
    weights: np.ndarray
    suppressed: np.ndarray
    history_correct: List[List[bool]] = field(
        default_factory=lambda: [[] for _ in range(N_SOURCES)]
    )


@dataclass
class DecisionOutput:
    decision: int
    source_consensus: float
    ensemble_assessment: float
    agreement_score: float

# ---- API / OPENENV-FRIENDLY PYDANTIC MODELS ----

class HealthResponse(BaseModel):
    status: str
    message: str


class TaskInfo(BaseModel):
    id: str
    title: str
    difficulty: str
    description: str
    reward_focus: str
    grader: str
    score_min: float
    score_max: float


class ResetRequest(BaseModel):
    seed: Optional[int] = 42
    difficulty: str = "medium"


class ResetResponse(BaseModel):
    observation: List[float]
    info: Dict[str, Any]


class StepRequest(BaseModel):
    action: int = Field(ge=0, le=5)


class StepResponse(BaseModel):
    observation: List[float]
    reward: float
    terminated: bool
    truncated: bool
    info: Dict[str, Any]


class StateResponse(BaseModel):
    step_count: int
    max_steps: int
    difficulty: str
    done: bool
    weights: List[float]
    suppressed: List[bool]
    adversarial_mode: str
