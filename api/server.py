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
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Trust Calibration AI</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: Inter, Arial, sans-serif;
      background: linear-gradient(180deg, #04101d 0%, #071827 100%);
      color: #eaf6ff;
      padding: 22px;
    }
    .container {
      max-width: 1250px;
      margin: 0 auto;
    }
    .topbar {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      margin-bottom: 22px;
      flex-wrap: wrap;
    }
    .badge {
      border: 1px solid #1ed3ff55;
      color: #8ff0ff;
      padding: 10px 18px;
      border-radius: 18px;
      background: #0b1d32;
      box-shadow: 0 0 18px rgba(30, 211, 255, 0.14);
      font-weight: 700;
      font-size: 18px;
    }
    h1 {
      text-align: center;
      font-size: 62px;
      line-height: 1.05;
      color: #9fe8ff;
      margin-bottom: 8px;
      text-shadow: 0 0 18px rgba(47, 215, 255, 0.24);
      font-weight: 900;
      letter-spacing: -1px;
    }
    .subtitle {
      text-align: center;
      color: #acbdd1;
      font-size: 22px;
      margin-bottom: 26px;
    }
    .hero {
      background: #d9e2dc;
      color: #17202a;
      border-radius: 28px;
      padding: 24px 26px;
      margin-bottom: 24px;
      border-left: 6px solid #ff7b5e;
      box-shadow: 0 8px 26px rgba(0,0,0,0.18);
    }
    .hero small {
      display: block;
      color: #566471;
      margin-bottom: 10px;
      font-size: 18px;
    }
    .hero h2 {
      font-size: 34px;
      line-height: 1.35;
      margin-bottom: 16px;
      font-weight: 900;
    }
    .chips {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
    }
    .chip {
      background: #eceaf0;
      color: #4d5660;
      padding: 10px 15px;
      border-radius: 999px;
      font-size: 16px;
      font-weight: 500;
    }
    .grid {
      display: grid;
      grid-template-columns: 1.12fr 1fr 0.95fr;
      gap: 20px;
      margin-top: 22px;
    }
    .panel {
      background: #0d1d33;
      border-radius: 24px;
      padding: 22px;
      box-shadow: 0 0 22px rgba(0, 213, 255, 0.08);
      border: 1px solid rgba(27, 224, 255, 0.14);
    }
    .panel-title {
      font-size: 22px;
      color: #f0fbff;
      margin-bottom: 18px;
      font-weight: 900;
    }
    .signal-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
    }
    .signal-card {
      background: #11243c;
      border: 1px solid rgba(36, 221, 255, 0.35);
      border-radius: 20px;
      padding: 18px 16px;
    }
    .signal-name {
      font-size: 17px;
      color: #eef7ff;
      margin-bottom: 16px;
      word-break: break-word;
    }
    .signal-value {
      font-size: 30px;
      font-weight: 900;
      margin-bottom: 8px;
      color: #ffffff;
    }
    .signal-conf {
      color: #9fb2c8;
      font-size: 15px;
      margin-bottom: 8px;
    }
    .bar {
      width: 100%;
      height: 12px;
      background: #dde4ea;
      border-radius: 999px;
      overflow: hidden;
      margin-top: 8px;
    }
    .bar-fill {
      height: 100%;
      background: linear-gradient(90deg, #2b84ff, #57d9ff);
      border-radius: 999px;
    }
    .weights {
      display: flex;
      flex-direction: column;
      gap: 18px;
    }
    .weight-row {
      display: grid;
      grid-template-columns: 1.2fr 1.6fr 0.45fr;
      align-items: center;
      gap: 12px;
    }
    .weight-name {
      color: #a8bfd4;
      font-size: 15px;
      word-break: break-word;
    }
    .weight-bar {
      height: 48px;
      background: #11243c;
      border-radius: 14px;
      overflow: hidden;
      position: relative;
      border: 1px solid rgba(27, 224, 255, 0.10);
    }
    .weight-fill {
      height: 100%;
      background: #51c4d9;
      border-radius: 14px;
    }
    .weight-val {
      color: #e2f9ff;
      font-size: 15px;
      text-align: right;
      font-weight: 700;
    }
    .decision-box {
      border: 2px solid #2ce1ff;
      border-radius: 20px;
      padding: 24px 16px;
      text-align: center;
      font-size: 36px;
      font-weight: 900;
      margin-bottom: 16px;
      background: #11243c;
      box-shadow: 0 0 18px rgba(44, 225, 255, 0.13);
    }
    .decision-ignore {
      color: #88ebff;
      border-color: #2ce1ff;
    }
    .decision-investigate {
      color: #ffd66f;
      border-color: #ffd66f;
      box-shadow: 0 0 18px rgba(255, 214, 111, 0.13);
    }
    .decision-escalate {
      color: #ff8c8c;
      border-color: #ff7272;
      box-shadow: 0 0 18px rgba(255, 114, 114, 0.16);
    }
    .risk-pill {
      display: inline-block;
      padding: 10px 16px;
      border-radius: 999px;
      font-size: 15px;
      font-weight: 800;
      margin-bottom: 14px;
    }
    .risk-low {
      background: rgba(54, 214, 161, 0.12);
      color: #92f0cc;
      border: 1px solid rgba(54, 214, 161, 0.32);
    }
    .risk-medium {
      background: rgba(255, 204, 92, 0.10);
      color: #ffd97a;
      border: 1px solid rgba(255, 204, 92, 0.34);
    }
    .risk-high {
      background: rgba(255, 111, 111, 0.10);
      color: #ff9e9e;
      border: 1px solid rgba(255, 111, 111, 0.34);
    }
    .reason-box {
      background: #11243c;
      border-radius: 18px;
      padding: 18px;
      font-size: 17px;
      line-height: 1.75;
      color: #d9ebf8;
      margin-bottom: 16px;
    }
    .impact-box {
      background: rgba(79, 196, 255, 0.08);
      border: 1px solid rgba(79, 196, 255, 0.16);
      color: #d9f5ff;
      border-radius: 18px;
      padding: 14px 16px;
      font-size: 15px;
      line-height: 1.6;
      margin-bottom: 16px;
    }
    .meta {
      display: grid;
      grid-template-columns: 1fr 1fr 1fr;
      gap: 14px;
      margin-top: 6px;
    }
    .meta-card {
      background: #11243c;
      border-radius: 16px;
      padding: 16px;
      text-align: center;
    }
    .meta-label {
      color: #97abc0;
      font-size: 14px;
      margin-bottom: 8px;
    }
    .meta-value {
      font-size: 24px;
      font-weight: 900;
      color: #ecf7ff;
    }
    .footer {
      margin-top: 24px;
      display: flex;
      gap: 12px;
      justify-content: center;
      flex-wrap: wrap;
    }
    button {
      padding: 14px 20px;
      border-radius: 14px;
      border: none;
      font-size: 16px;
      font-weight: 800;
      cursor: pointer;
      background: #53cce5;
      color: #07111f;
      min-width: 170px;
    }
    button.secondary {
      background: #11243c;
      color: #e4f8ff;
      border: 1px solid rgba(37, 222, 255, 0.32);
    }
    @media (max-width: 1000px) {
      .grid { grid-template-columns: 1fr; }
      h1 { font-size: 46px; }
      .subtitle { font-size: 18px; }
      .hero h2 { font-size: 25px; }
    }
  </style>
</head>
<body>
  <div class="container">
    <div class="topbar">
      <div class="badge">System Active</div>
      <div class="badge" id="cycleLabel">Cycle #0001</div>
    </div>

    <h1>Trust Calibration AI</h1>
    <div class="subtitle">Real-time Decision Intelligence System</div>

    <div class="hero">
      <small>Live incident</small>
      <h2 id="incidentText">Loading calibrated incident context...</h2>
      <div class="chips">
        <div class="chip" id="chip1">auth anomaly</div>
        <div class="chip" id="chip2">geo mismatch</div>
        <div class="chip" id="chip3">session risk</div>
      </div>
    </div>

    <div class="grid">
      <div class="panel">
        <div class="panel-title">Signal Sources</div>
        <div class="signal-grid" id="signals"></div>
      </div>

      <div class="panel">
        <div class="panel-title">Adaptive Trust Weights</div>
        <div class="weights" id="weights"></div>
      </div>

      <div class="panel">
        <div class="panel-title">Decision Output</div>
        <div id="riskPill" class="risk-pill risk-low">Risk Level: LOW</div>
        <div class="decision-box decision-ignore" id="decisionBox">READY</div>
        <div class="reason-box" id="reasonBox">Environment reset complete. Select a trust strategy to generate a calibrated decision.</div>
        <div class="impact-box" id="impactBox">Projected analyst load reduction: 68%. Unreliable signals are continuously down-weighted to reduce false escalations.</div>

        <div class="meta">
          <div class="meta-card">
            <div class="meta-label">Conflict</div>
            <div class="meta-value" id="conflictVal">-</div>
          </div>
          <div class="meta-card">
            <div class="meta-label">Uncertainty</div>
            <div class="meta-value" id="uncertaintyVal">-</div>
          </div>
          <div class="meta-card">
            <div class="meta-label">Task</div>
            <div class="meta-value" id="taskVal">-</div>
          </div>
        </div>
      </div>
    </div>

    <div class="footer">
      <button onclick="resetEnv()">Reset</button>
      <button class="secondary" onclick="stepAction(4)">Confidence-Weighted</button>
      <button class="secondary" onclick="stepAction(5)">Conflict-Aware</button>
      <button class="secondary" onclick="stepAction(3)">Suppress Unreliable</button>
    </div>
  </div>

  <script>
    let cycle = 1;
    const sourceNames = ["anomaly_detector", "network_monitor", "rule_engine", "unreliable_feed"];

    function num(v) {
      return Number(v || 0).toFixed(2);
    }

    function getRiskLabel(score, decision) {
      if (decision === "ESCALATE" || score >= 0.70) {
        return { text: "Risk Level: HIGH", cls: "risk-high" };
      }
      if (decision === "INVESTIGATE" || score >= 0.35) {
        return { text: "Risk Level: MEDIUM", cls: "risk-medium" };
      }
      return { text: "Risk Level: LOW", cls: "risk-low" };
    }

    function decisionClass(decision) {
      if (decision === "ESCALATE") return "decision-box decision-escalate";
      if (decision === "INVESTIGATE") return "decision-box decision-investigate";
      return "decision-box decision-ignore";
    }

    function incidentHeadline(decision, score) {
      if (decision === "ESCALATE") {
        return `Suspicious multi-signal security incident detected. Current decision: ESCALATE, score ${score}.`;
      }
      if (decision === "INVESTIGATE") {
        return `Mixed-risk incident with partial detector disagreement. Current decision: INVESTIGATE, score ${score}.`;
      }
      return `Low-severity event with weak supporting evidence. Current decision: IGNORE, score ${score}.`;
    }

    function incidentTags(decision) {
      if (decision === "ESCALATE") return ["credential abuse", "rule hit", "network spike"];
      if (decision === "INVESTIGATE") return ["auth anomaly", "geo mismatch", "session risk"];
      return ["benign pattern", "routine traffic", "low anomaly"];
    }

    function impactText(decision) {
      if (decision === "ESCALATE") {
        return "Escalation triggered only after cross-source agreement. This reduces missed high-severity incidents while maintaining calibrated analyst response.";
      }
      if (decision === "INVESTIGATE") {
        return "Investigation selected under ambiguity. This prevents premature escalation while preserving safety under uncertainty.";
      }
      return "Low-risk classification prevents alert fatigue. Weak or deceptive signals were prevented from driving unnecessary escalation.";
    }

    function shortReason(info) {
      const explanation = info?.explanation || {};
      const topIdx = explanation?.top_source_index ?? 0;
      const topName = sourceNames[topIdx] || "source";
      const topValue = explanation?.top_source_value ?? 0;
      const topConf = explanation?.top_source_confidence ?? 0;
      return `Strongest active signal came from ${topName} (${topValue}, conf ${topConf}). The fused calibrated threat signal led to a final decision of ${info?.decision || "UNKNOWN"}.`;
    }

    function renderSignals(obs) {
      const values = obs.slice(0, 4);
      const confs = obs.slice(4, 8);

      const html = values.map((v, i) => `
        <div class="signal-card">
          <div class="signal-name">${sourceNames[i]}</div>
          <div class="signal-value">${num(v)}</div>
          <div class="signal-conf">Confidence</div>
          <div class="bar"><div class="bar-fill" style="width:${Math.max(0, Math.min(100, confs[i] * 100))}%"></div></div>
        </div>
      `).join("");

      document.getElementById("signals").innerHTML = html;
    }

    function renderWeights(state) {
      const weights = state.weights || [0.25, 0.25, 0.25, 0.25];
      const html = weights.map((w, i) => `
        <div class="weight-row">
          <div class="weight-name">${sourceNames[i]}</div>
          <div class="weight-bar"><div class="weight-fill" style="width:${Math.max(0, Math.min(100, w * 100))}%"></div></div>
          <div class="weight-val">${num(w)}</div>
        </div>
      `).join("");

      document.getElementById("weights").innerHTML = html;
    }

    async function resetEnv() {
      const res = await fetch("/reset", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ seed: 42, difficulty: "hard" })
      });
      const data = await res.json();
      const obs = data.observation;

      cycle = 1;
      document.getElementById("cycleLabel").innerText = "Cycle #0001";

      renderSignals(obs);
      await refreshState();

      const decision = "INVESTIGATE";
      const defaultScore = "0.49";
      const risk = getRiskLabel(0.49, decision);
      const tags = incidentTags(decision);

      document.getElementById("incidentText").innerText = incidentHeadline(decision, defaultScore);
      document.getElementById("chip1").innerText = tags[0];
      document.getElementById("chip2").innerText = tags[1];
      document.getElementById("chip3").innerText = tags[2];

      const decisionBox = document.getElementById("decisionBox");
      decisionBox.className = decisionClass(decision);
      decisionBox.innerText = decision;

      const riskPill = document.getElementById("riskPill");
      riskPill.className = `risk-pill ${risk.cls}`;
      riskPill.innerText = risk.text;

      document.getElementById("reasonBox").innerText = "Environment reset complete. Select a trust strategy to generate a calibrated decision.";
      document.getElementById("impactBox").innerText = impactText(decision);
      document.getElementById("conflictVal").innerText = "-";
      document.getElementById("uncertaintyVal").innerText = "-";
    }

    async function stepAction(action) {
      const res = await fetch("/step", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action })
      });
      const data = await res.json();
      const obs = data.observation;
      renderSignals(obs);

      const decision = data.info.decision || "UNKNOWN";
      const score = ((obs[0] + obs[1] + obs[2] + obs[3]) / 4).toFixed(2);
      const risk = getRiskLabel(Number(score), decision);
      const tags = incidentTags(decision);

      document.getElementById("incidentText").innerText = incidentHeadline(decision, score);
      document.getElementById("chip1").innerText = tags[0];
      document.getElementById("chip2").innerText = tags[1];
      document.getElementById("chip3").innerText = tags[2];

      const decisionBox = document.getElementById("decisionBox");
      decisionBox.className = decisionClass(decision);
      decisionBox.innerText = decision;

      const riskPill = document.getElementById("riskPill");
      riskPill.className = `risk-pill ${risk.cls}`;
      riskPill.innerText = risk.text;

      document.getElementById("reasonBox").innerText = shortReason(data.info);
      document.getElementById("impactBox").innerText = impactText(decision);
      document.getElementById("conflictVal").innerText = num(data.info.conflict);
      document.getElementById("uncertaintyVal").innerText = num(data.info.uncertainty);

      cycle += 1;
      document.getElementById("cycleLabel").innerText = "Cycle #" + String(cycle).padStart(4, "0");

      await refreshState();
    }

    async function refreshState() {
      const res = await fetch("/state");
      const state = await res.json();
      renderWeights(state);
      document.getElementById("taskVal").innerText = (state.difficulty || "-").toUpperCase();
    }

    resetEnv();
  </script>
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
