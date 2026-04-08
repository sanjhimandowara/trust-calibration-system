import numpy as np

from models.schemas import (
    TrustState,
    N_SOURCES,
    DECISIONS
)

from envs.signal_generator import generate_signal_bundle
from envs.conflict_detector import (
    compute_conflict,
    compute_uncertainty,
    consensus_score
)
from envs.specialist_agents import (
    compute_ensemble_assessment,
    compute_specialist_agreement
)
from envs.cascade_simulator import cascade_penalty
from envs.explainer import generate_explanation


class TrustCalibrationEnv:

    def __init__(self, max_steps=20, difficulty="medium"):
        self.max_steps = max_steps
        self.difficulty = difficulty

        self.current_step = 0
        self.done = False

        self.state = None
        self.last_bundle = None

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

        self.current_step = 0
        self.done = False

        self.state = TrustState(
            weights=np.ones(N_SOURCES) / N_SOURCES,
            suppressed=np.zeros(N_SOURCES, dtype=bool)
        )

        bundle = generate_signal_bundle(
            step_count=0,
            difficulty=self.difficulty
        )

        self.last_bundle = bundle

        obs = self._build_observation(bundle)

        return obs, {}

    def _build_observation(self, bundle):
        values = bundle.values
        confidences = bundle.confidences
        weights = self.state.weights.astype(np.float32)
        suppressed = self.state.suppressed.astype(np.float32)

        conflict = compute_conflict(values, weights)
        uncertainty = compute_uncertainty(confidences, conflict)
        consensus = consensus_score(values)
        ensemble = compute_ensemble_assessment(values, confidences)
        agreement = compute_specialist_agreement(values, confidences)

        step_ratio = self.current_step / max(1, self.max_steps)

        observation = np.concatenate([
            values,                           # 4
            confidences,                      # 4
            weights,                          # 4
            suppressed,                       # 4
            np.array([conflict], dtype=np.float32),
            np.array([uncertainty], dtype=np.float32),
            np.array([consensus], dtype=np.float32),
            np.array([ensemble], dtype=np.float32),
            np.array([agreement], dtype=np.float32),
            np.array([step_ratio], dtype=np.float32),
        ]).astype(np.float32)

        return observation

    def _apply_action_strategy(self, action, values, confidences):

        if action == 0:  # equal_trust
            weights = np.ones(N_SOURCES, dtype=np.float32) / N_SOURCES

        elif action == 1:  # favor_anomaly_network
            weights = np.array([0.35, 0.35, 0.20, 0.10], dtype=np.float32)

        elif action == 2:  # favor_rule_anomaly
            weights = np.array([0.30, 0.15, 0.40, 0.15], dtype=np.float32)

        elif action == 3:  # downweight_unreliable
            weights = np.array([0.30, 0.25, 0.30, 0.15], dtype=np.float32)

        elif action == 4:  # confidence_weighted
            weights = np.array(confidences, dtype=np.float32)

        elif action == 5:  # conflict_sensitive_balance
            weights = np.array([0.25, 0.25, 0.30, 0.20], dtype=np.float32)

        else:
            raise ValueError(f"Invalid action: {action}")

        weights = np.clip(weights, 1e-6, None)
        weights = weights / np.sum(weights)

        return weights

    def _make_decision(self, fused_score):
        if fused_score < 0.35:
            return 0
        elif fused_score < 0.70:
            return 1
        return 2

    def step(self, action):

        if self.done:
            raise RuntimeError("Episode is done. Call reset() before step().")

        action = int(action)

        current_bundle = self.last_bundle
        true_label = int(current_bundle.true_label)

        values = current_bundle.values.copy()
        confidences = current_bundle.confidences.copy()

        weights = self._apply_action_strategy(action, values, confidences)
        self.state.weights = weights

        # suppression logic
        if action == 3:
            self.state.suppressed[3] = True
        else:
            self.state.suppressed[:] = False

        effective_weights = weights.copy()

        if self.state.suppressed[3]:
            effective_weights[3] *= 0.25

        effective_weights = np.clip(effective_weights, 1e-6, None)
        effective_weights = effective_weights / np.sum(effective_weights)

        fused_score = float(np.sum(values * effective_weights))

        conflict = compute_conflict(values, effective_weights)
        uncertainty = compute_uncertainty(confidences, conflict)
        consensus = consensus_score(values)
        ensemble = compute_ensemble_assessment(values, confidences)
        agreement = compute_specialist_agreement(values, confidences)

        decision = self._make_decision(fused_score)

        reward = self._compute_reward(
            decision,
            true_label,
            conflict,
            uncertainty,
            self.state.suppressed[3],
            current_bundle.adversarial_mode
        )

        explanation = generate_explanation(
            decision,
            values,
            confidences,
            effective_weights,
            conflict,
            uncertainty
        )

        info = {
            "decision": DECISIONS[decision],
            "decision_id": decision,
            "true_label": DECISIONS[true_label],
            "true_label_id": true_label,
            "correct": bool(decision == true_label),
            "conflict": float(conflict),
            "uncertainty": float(uncertainty),
            "consensus": float(consensus),
            "ensemble_assessment": float(ensemble),
            "specialist_agreement": float(agreement),
            "adversarial_mode": current_bundle.adversarial_mode,
            "explanation": explanation,
        }

        self.current_step += 1
        self.done = self.current_step >= self.max_steps

        if not self.done:
            next_bundle = generate_signal_bundle(
                step_count=self.current_step,
                difficulty=self.difficulty
            )
            self.last_bundle = next_bundle
            next_obs = self._build_observation(next_bundle)
        else:
            next_obs = np.zeros(22, dtype=np.float32)

        return next_obs, float(reward), bool(self.done), False, info

    def _compute_reward(
        self,
        decision,
        true_label,
        conflict,
        uncertainty,
        suppressed_unreliable,
        adversarial_mode
    ):

        if decision == true_label:
            if true_label == 0:
                reward = 0.8
            elif true_label == 1:
                reward = 0.9
            else:
                reward = 1.0
        else:
            if decision == 0 and true_label == 2:
                reward = -1.0
            elif decision == 2 and true_label == 0:
                reward = -0.7
            else:
                reward = -0.4

        if decision == 1 and uncertainty > 0.45:
            reward += 0.15

        if suppressed_unreliable and adversarial_mode != "none":
            reward += 0.10

        reward += cascade_penalty(
            decision,
            true_label,
            conflict,
            uncertainty
        )

        return float(np.clip(reward, -1.0, 1.0))