

"""
Symbolic processing module for Émile framework.
Handles symbolic field classification, regime detection, and adaptive thresholds.
"""
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field

from emile_cogito.kainos.config import CONFIG
from emile_cogito.kainos.universal_module_logging import LoggedModule, logged_method

@dataclass
class RegimeProperties:
    """Properties of a symbolic regime."""
    name: str
    description: str
    stability: float      # 0-1 stability rating
    creativity: float     # 0-1 creativity rating
    coherence: float      # 0-1 coherence rating
    energy: float         # 0-1 energy level
    associated_words: List[str] = field(default_factory=list)

# Global regime properties dictionary
REGIME_PROPERTIES = {
    "stable_coherence": RegimeProperties(
        name="stable_coherence",
        description="A stable state with high internal organization and minimal surplus",
        stability=0.9,
        creativity=0.3,
        coherence=0.9,
        energy=0.4,
        associated_words=["stability", "coherence", "harmony", "balance", "order",
                          "alignment", "equilibrium", "consistency"]
    ),
    "symbolic_turbulence": RegimeProperties(
        name="symbolic_turbulence",
        description="A chaotic state with rapidly changing patterns, moderate surplus, and high variance",
        stability=0.2,
        creativity=0.8,
        coherence=0.3,
        energy=0.7,
        associated_words=["chaos", "turbulence", "fluctuation", "complexity", "instability",
                          "change", "variation", "unpredictability", "disorder"]
    ),
    "flat_rupture": RegimeProperties(
        name="flat_rupture",
        description="A state following rupture where previous structure is lost or flattened",
        stability=0.4,
        creativity=0.2,
        coherence=0.5,
        energy=0.1,
        associated_words=["rupture", "collapse", "reset", "flat", "blank", "neutral",
                         "emptiness", "silence", "aftermath", "disintegration"]
    ),
    "quantum_oscillation": RegimeProperties(
        name="quantum_oscillation",
        description="A rhythmic state with regular oscillation between states",
        stability=0.7,
        creativity=0.6,
        coherence=0.6,
        energy=0.8,
        associated_words=["oscillation", "rhythm", "cycle", "wave", "periodicity",
                         "pattern", "resonance", "alternation", "pulse"]
    )
}

class SymbolicReasoner(LoggedModule):
    """
    Handles symbolic field processing, regime classification and threshold adaptation.

    Implements aspects of Theorems 2, 4, and 6 from QSE theory, particularly concerning
    symbolic curvature interpretation and regime transitions.
    """

    def __init__(self, cfg=CONFIG):
        """
        Initialize the symbolic reasoner.

        Args:
            cfg: Configuration parameters
        """
        super().__init__("symbolic_reasoning")
        self.cfg = cfg

        # Core symbolic fields
        self.psi = None
        self.phi = None
        self.sigma = None

        # Adaptive thresholds
        self.theta_psi = cfg.THETA_PSI
        self.theta_phi = cfg.THETA_PHI

        # Regime tracking
        self.current_regime = "stable_coherence"
        self.regime_history = []
        self.regime_durations = {regime: 0 for regime in REGIME_PROPERTIES.keys()}

        # Metric history
        self.sigma_history = []
        self.sigma_var_history = []
        self.surplus_history = []
        self.oscillation_scores = []

        # Analysis results
        self.last_analysis = {}

    def classify_regime(self, sigma: np.ndarray, surplus: np.ndarray,
                       oscillation_score: float = 0.0) -> Dict[str, Any]:
        """
        Classify the current symbolic regime using fuzzy logic.

        Implements regime detection based on §3 of the QSE paper, identifying
        stable coherence, symbolic turbulence, flat rupture, and quantum oscillation.

        Args:
            sigma: Current symbolic curvature field
            surplus: Current surplus field
            oscillation_score: Score indicating oscillatory behavior (0-1)

        Returns:
            Dictionary with regime classification and membership values
        """
        # Calculate statistical properties
        avg_sigma = float(np.mean(sigma))
        var_sigma = float(np.var(sigma))
        avg_surplus = float(np.mean(surplus))
        var_surplus = float(np.var(surplus))

        # Update history
        self.sigma_history.append(avg_sigma)
        self.sigma_var_history.append(var_sigma)
        self.surplus_history.append(avg_surplus)
        self.oscillation_scores.append(oscillation_score)

        # Keep history bounded
        max_history = 100
        if len(self.sigma_history) > max_history:
            self.sigma_history = self.sigma_history[-max_history:]
            self.sigma_var_history = self.sigma_var_history[-max_history:]
            self.surplus_history = self.surplus_history[-max_history:]
            self.oscillation_scores = self.oscillation_scores[-max_history:]

        # Calculate trend and stability metrics
        trend = 0.0
        stability = 1.0
        if len(self.sigma_history) >= 3:
            # Calculate trend using linear regression
            x = np.arange(len(self.sigma_history))
            trend = np.polyfit(x, self.sigma_history, 1)[0] * len(self.sigma_history)

            # Calculate stability using coefficient of variation
            recent_sigma = np.array(self.sigma_history[-10:])
            if np.mean(np.abs(recent_sigma)) > 0.01:
                stability = 1.0 - min(1.0, np.std(recent_sigma) / (np.mean(np.abs(recent_sigma)) + 0.01))

        # Calculate fuzzy membership values for each regime
        memberships = {}

        # Stable Coherence regime membership
        stable_coherence_thresh = self.cfg.REGIME_THRESHOLDS["stable_coherence"]
        stable_score = 1.0

        # Low sigma mean and variance indicate stability
        if avg_sigma < stable_coherence_thresh.get("mean_min", 0.0):
            stable_score *= 0.5
        if avg_sigma > stable_coherence_thresh.get("mean_max", 0.1):
            dist = (avg_sigma - stable_coherence_thresh.get("mean_max", 0.1)) / 0.1
            stable_score *= max(0.0, 1.0 - dist)
        if var_sigma > stable_coherence_thresh.get("var_max", 0.01):
            dist = (var_sigma - stable_coherence_thresh.get("var_max", 0.01)) / 0.01
            stable_score *= max(0.0, 1.0 - dist)

        # Oscillation reduces stable coherence membership
        stable_score *= max(0.0, 1.0 - oscillation_score)

        memberships["stable_coherence"] = float(stable_score)

        # Symbolic Turbulence regime membership
        turbulence_thresh = self.cfg.REGIME_THRESHOLDS["symbolic_turbulence"]
        turbulence_score = 1.0

        # Moderate sigma mean and high variance indicate turbulence
        if avg_sigma < turbulence_thresh.get("mean_min", 0.1):
            dist = (turbulence_thresh.get("mean_min", 0.1) - avg_sigma) / 0.1
            turbulence_score *= max(0.0, 1.0 - dist)
        if avg_sigma > turbulence_thresh.get("mean_max", 0.4):
            dist = (avg_sigma - turbulence_thresh.get("mean_max", 0.4)) / 0.1
            turbulence_score *= max(0.0, 1.0 - dist)
        if var_sigma < turbulence_thresh.get("var_min", 0.01):
            dist = (turbulence_thresh.get("var_min", 0.01) - var_sigma) / 0.01
            turbulence_score *= max(0.0, 1.0 - dist * 2)

        # High variability and low stability increase turbulence score
        turbulence_score *= min(1.0, var_sigma * 10)
        turbulence_score *= (1.0 - stability) * 0.5 + 0.5

        memberships["symbolic_turbulence"] = float(turbulence_score)

        # Flat Rupture regime membership
        flat_rupture_thresh = self.cfg.REGIME_THRESHOLDS["flat_rupture"]
        rupture_score = 1.0

        # Negative sigma mean indicates rupture (Φ > Ψ)
        if avg_sigma > flat_rupture_thresh.get("mean_max", -0.1):
            dist = (avg_sigma - flat_rupture_thresh.get("mean_max", -0.1)) / 0.1
            rupture_score *= max(0.0, 1.0 - dist)
        if avg_sigma < flat_rupture_thresh.get("mean_min", -0.9):
            dist = (flat_rupture_thresh.get("mean_min", -0.9) - avg_sigma) / 0.1
            rupture_score *= max(0.0, 1.0 - dist)

        # Low variance is characteristic of flat rupture
        if var_sigma > flat_rupture_thresh.get("var_max", 0.05):
            dist = (var_sigma - flat_rupture_thresh.get("var_max", 0.05)) / 0.05
            rupture_score *= max(0.0, 1.0 - dist)

        memberships["flat_rupture"] = float(rupture_score)

        # Quantum Oscillation regime membership
        oscillation_thresh = self.cfg.REGIME_THRESHOLDS["quantum_oscillation"]
        oscillation_regime_score = 1.0

        # Oscillation score directly contributes to regime membership
        oscillation_regime_score *= oscillation_score

        # Moderate sigma values are conducive to oscillation
        if avg_sigma < oscillation_thresh.get("mean_min", 0.1):
            dist = (oscillation_thresh.get("mean_min", 0.1) - avg_sigma) / 0.1
            oscillation_regime_score *= max(0.0, 1.0 - dist)
        if avg_sigma > oscillation_thresh.get("mean_max", 0.3):
            dist = (avg_sigma - oscillation_thresh.get("mean_max", 0.3)) / 0.1
            oscillation_regime_score *= max(0.0, 1.0 - dist)

        memberships["quantum_oscillation"] = float(oscillation_regime_score)

        # Determine winning regime
        if memberships:
            sorted_regimes = sorted(memberships.items(), key=lambda x: x[1], reverse=True)
            winning_regime = sorted_regimes[0][0]
            winning_confidence = sorted_regimes[0][1]

            # Only switch regime if confidence is above threshold
            confidence_threshold = 0.3
            if winning_confidence >= confidence_threshold:
                # Update regime durations
                for regime in self.regime_durations:
                    if regime == winning_regime:
                        self.regime_durations[regime] += 1
                    else:
                        self.regime_durations[regime] = 0

                self.current_regime = winning_regime

        # Record regime history
        self.regime_history.append(self.current_regime)
        if len(self.regime_history) > 1000:
            self.regime_history = self.regime_history[-1000:]

        # Create analysis result
        analysis = {
            "regime": self.current_regime,
            "memberships": memberships,
            "mean_sigma": avg_sigma,
            "variance_sigma": var_sigma,
            "mean_surplus": avg_surplus,
            "variance_surplus": var_surplus,
            "trend": trend,
            "stability": stability,
            "oscillation_score": oscillation_score,
            "properties": REGIME_PROPERTIES[self.current_regime]
        }

        self.last_analysis = analysis
        return analysis

    def update_temporal_regime_context(self, tau_prime: float, subjective_time: float):
        """Update regime context with temporal awareness"""

        if not hasattr(self, 'temporal_regime_history'):
            self.temporal_regime_history = []

        regime_entry = {
            'regime': self.current_regime,
            'tau_prime': tau_prime,
            'subjective_time': subjective_time,
            'empirical_time': time.time(),
            'regime_duration': getattr(self, 'regime_durations', {}).get(self.current_regime, 0)
        }

        self.temporal_regime_history.append(regime_entry)

        # Keep bounded history
        if len(self.temporal_regime_history) > 100:
            self.temporal_regime_history = self.temporal_regime_history[-100:]


    def adjust_thresholds(self, metrics: Dict[str, Any]) -> None:
        """
        Adaptively adjust thresholds based on current metrics.

        This implements a form of symbolic homeostasis, where the system
        self-regulates its sensitivity to maintain functionality.

        Args:
            metrics: Dictionary of metrics from the QSE engine
        """
        # Extract relevant metrics
        coherence = metrics.get('phase_coherence', 0.5)
        entropy = metrics.get('normalized_entropy', 0.5)

        # Adjust theta_psi based on coherence
        # If coherence is low, lower theta_psi to make Psi activation easier
        coherence_factor = 1.0 + 0.1 * (0.5 - coherence)  # 0.9 to 1.1 range
        self.theta_psi = np.clip(self.cfg.THETA_PSI * coherence_factor, 0.1, 0.9)

        # Adjust theta_phi based on entropy
        # If entropy is high, raise theta_phi to make Phi activation harder
        entropy_factor = 1.0 + 0.2 * (entropy - 0.5)  # 0.9 to 1.1 range
        self.theta_phi = np.clip(self.cfg.THETA_PHI * entropy_factor, 0.2, 0.9)

        # Regime-specific adaptations
        if self.current_regime == "flat_rupture" and self.regime_durations[self.current_regime] > 10:
            # Been in flat_rupture too long, make it easier to get out
            self.theta_psi *= 0.95  # Lower Psi threshold

        elif self.current_regime == "symbolic_turbulence" and self.regime_durations[self.current_regime] > 15:
            # Been in turbulence too long, try to stabilize
            self.theta_phi *= 0.97  # Lower Phi threshold

    @logged_method
    def step(self, surplus: np.ndarray,
            metrics: Optional[Dict[str, Any]] = None,
            oscillation_score: float = 0.0) -> Dict[str, Any]:
        """
        Process a single step of symbolic reasoning.

        Args:
            surplus: Current surplus field
            metrics: Optional metrics from QSE engine
            oscillation_score: Score indicating oscillatory behavior

        Returns:
            Dictionary with symbolic analysis results
        """
        # Calculate symbolic fields with adaptive thresholds
        psi = 1.0 / (1.0 + np.exp(-self.cfg.K_PSI * (surplus - self.theta_psi)))
        phi = np.maximum(0.0, self.cfg.K_PHI * (surplus - self.theta_phi))
        sigma = psi - phi

        # Store fields
        self.psi = psi
        self.phi = phi
        self.sigma = sigma

        # Classify current regime
        analysis = self.classify_regime(sigma, surplus, oscillation_score)

        # Adjust thresholds if metrics provided
        if metrics is not None:
            self.adjust_thresholds(metrics)

        # Add logging for regime changes
        if hasattr(self, 'previous_regime') and analysis['regime'] != self.previous_regime:
            self.log_event("REGIME_TRANSITION",
                          f"Regime changed from {self.previous_regime} to {analysis['regime']}",
                          {'old_regime': self.previous_regime, 'new_regime': analysis['regime']})

        self.previous_regime = analysis['regime']

        return analysis

    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of the symbolic reasoner.

        Returns:
            Dictionary with current symbolic state
        """
        return {
            "psi": self.psi.copy() if self.psi is not None else None,
            "phi": self.phi.copy() if self.phi is not None else None,
            "sigma": self.sigma.copy() if self.sigma is not None else None,
            "current_regime": self.current_regime,
            "theta_psi": self.theta_psi,
            "theta_phi": self.theta_phi,
            "regime_durations": dict(self.regime_durations),
            "analysis": dict(self.last_analysis)
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Set the symbolic reasoner state."""
        if "psi" in state and state["psi"] is not None:
            self.psi = state["psi"].copy()
        if "phi" in state and state["phi"] is not None:
            self.phi = state["phi"].copy()
        if "sigma" in state and state["sigma"] is not None:
            self.sigma = state["sigma"].copy()
        if "current_regime" in state:
            self.current_regime = state["current_regime"]
        if "theta_psi" in state:
            self.theta_psi = state["theta_psi"]
        if "theta_phi" in state:
            self.theta_phi = state["theta_phi"]
        if "regime_durations" in state:
            self.regime_durations = dict(state["regime_durations"])

    def get_regime_history(self) -> List[str]:
        """Get the history of symbolic regimes."""
        return list(self.regime_history)

    def get_metrics_history(self) -> Dict[str, List[float]]:
        """Get the history of symbolic metrics."""
        return {
            "sigma_mean": self.sigma_history,
            "sigma_variance": self.sigma_var_history,
            "surplus_mean": self.surplus_history,
            "oscillation_scores": self.oscillation_scores
        }

from emile_cogito.kainos.module_wide_flow_mapper import auto_map_module_flow
auto_map_module_flow(__name__)  # Maps the entire module!
