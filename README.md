# Trust Calibration OpenEnv  
Learning What to Trust Under Uncertainty and Adversarial Signals

---

## Overview

Modern AI systems do not fail because they lack data — they fail because they trust the wrong data at the wrong time.

This project introduces a reinforcement learning environment where an agent learns how to dynamically allocate trust across multiple noisy, conflicting, and adversarial information sources before making a decision.

Instead of directly predicting outcomes, the system learns:

"Which signals should I trust right now?"

The final decisions are:

- IGNORE (low risk)  
- INVESTIGATE (uncertain)  
- ESCALATE (critical)  

---

## Real-World Motivation

In real-world systems such as:

- cybersecurity alert triage  
- fraud detection pipelines  
- monitoring and anomaly detection systems  

multiple signals must be combined under uncertainty.

These systems face:

- conflicting inputs from different detectors  
- unreliable or adversarial sources  
- delayed or misleading signals  
- high-cost decision errors  

The real bottleneck is not prediction accuracy — it is trust calibration.

---

## Core Idea

This environment reformulates decision-making as a trust allocation problem.

At every step:

1. multiple signals are generated  
2. signals may conflict or behave adversarially  
3. the agent selects a trust strategy  
4. unreliable sources may be suppressed  
5. signals are fused into a final decision  

This creates a separation between:

- information processing (trust weighting and suppression)  
- decision-making (final action selection)  

---

## Environment Design

### Signal Sources

The environment simulates four sources:

- anomaly_detector  
- network_monitor  
- rule_engine  
- unreliable_feed  

The unreliable feed may behave adversarially through:

- spoofing high values  
- spoofing low values  
- mirroring other signals  
- delayed response patterns  

---

### Action Space (Trust Strategies)

The agent selects one of six trust strategies:

| Action | Strategy |
|--------|----------|
| 0 | Uniform trust |
| 1 | Prioritize anomaly + network |
| 2 | Prioritize rule-based signals |
| 3 | Suppress unreliable source |
| 4 | Confidence-weighted fusion |
| 5 | Conflict-aware adaptive balancing |

---

### Decision Output

After applying the selected trust strategy:

- IGNORE → low-risk case  
- INVESTIGATE → uncertain case  
- ESCALATE → critical case  

---

## Observation Space

Each step provides a 22-dimensional observation vector including:

- signal values (4)  
- signal confidences (4)  
- trust weights (4)  
- suppression flags (4)  
- conflict score  
- uncertainty score  
- consensus measure  
- ensemble assessment  
- specialist agreement score  
- step ratio  

This allows the agent to reason about reliability, disagreement, and uncertainty.

---

## Tasks and Difficulty Levels

The environment defines three deterministic tasks:

### Easy
- aligned signals  
- minimal noise  
- low uncertainty  

### Medium
- moderate disagreement  
- partial conflict  
- occasional unreliable behavior  

### Hard
- adversarial interference  
- deceptive unreliable signals  
- high uncertainty and conflict  
- increased decision risk  

Each task uses deterministic graders producing scores between 0.0 and 1.0.

---

## Reward Design

The reward function reflects real-world operational cost:

- reward for correct decisions  
- higher reward for correct ESCALATE  
- heavy penalty for missed ESCALATE  
- penalty for false ESCALATE  
- reward for INVESTIGATE under uncertainty  
- penalty for unsafe cascades  
- reward for suppressing unreliable sources  

This ensures continuous learning signals instead of sparse outcomes.

---

## Learning Method

The agent is trained using PPO (Proximal Policy Optimization):

- actor-critic architecture  
- generalized advantage estimation  
- clipped objective  
- entropy regularization  

The policy operates on trust strategies rather than direct labels.

---

## Evaluation Results

### Overall Reward

| Model | Average Reward |
|------|----------------|
| Baseline | 16.59 |
| PPO | 17.50 |

### Task Scores

| Task | Baseline | PPO |
|------|----------|-----|
| Easy | 1.000 | 1.000 |
| Medium | 0.995 | 1.000 |
| Hard | 0.921 | 0.930 |

PPO demonstrates improved robustness under adversarial conditions.

---

## Visualization

Generated artifacts:

- outputs/baseline_vs_ppo.png  
- results/checkpoints/ppo_model.pth  
- results/logs/evaluation_summary.json  

The graph highlights improved stability and performance of PPO over baseline.

---

## API Interface

- GET /health  
- GET /tasks  
- POST /reset  
- POST /step  
- GET /state  

---

## Inference Format

[START]  
[STEP] action=... reward=... done=true/false success=true/false  
[END]

---

## How to Run

Install dependencies:

pip install -r requirements.txt  

Run API:

PYTHONPATH=. uvicorn api.server:app --reload --port 7860  

Run tests:

PYTHONPATH=. python tests/smoke_test.py  

Run baseline:

PYTHONPATH=. python training/baseline.py  

Train PPO:

PYTHONPATH=. python training/train.py  

Evaluate:

PYTHONPATH=. python training/evaluate.py  

Run inference:

export HF_TOKEN=your_token  
export API_BASE_URL=http://127.0.0.1:7860  
python inference.py  

or

export OPENAI_API_KEY=your_key  
export API_BASE_URL=http://127.0.0.1:7860  
python inference.py  

---

## Key Contributions

- Trust-centric decision modeling  
- Adversarial signal simulation  
- Conflict-aware reasoning  
- Dynamic source suppression  
- Cascade risk modeling  
- Full RL training + evaluation pipeline  

---

## Future Work

This environment opens several high-impact directions:

- Human-in-the-loop trust correction for real-world deployment  
- Meta-learning for adaptive trust policies across domains  
- Multi-agent trust negotiation systems  
- Explainable trust attribution for decision transparency  
- Integration with real cybersecurity or fraud datasets  
- Continual learning under evolving adversarial behavior  
- Trust calibration benchmarking for foundation models  

---

## Applications

- cybersecurity alert triage  
- fraud detection systems  
- monitoring pipelines  
- sensor fusion systems  
- decision systems under uncertainty  

---

## Conclusion

This project shifts AI from asking:

"What is correct?"

to asking:

"What should be trusted?"

which is the core challenge in real-world intelligent systems.