"""
Antifinity module for Émile framework.
Implements collaboration and compromise metrics based on the Epigenesis and Antifinity thesis.
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field

from emile_cogito.kainos.config import CONFIG

@dataclass
class MoralMetrics:
    """
    Container for moral metrics according to Antifinity framework.

    Based on the Epigenesis and Antifinity thesis, these metrics
    measure being-in-addition-to-itself vs. compromise.
    """
    collaboration_score: float = 0.0  # Extension of capability beyond individual potential
    compromise_score: float = 0.0     # Reduction/limitation of potential
    antifinity_quotient: float = 0.0  # Overall measure of being-in-addition-to-itself
    tension_index: float = 0.0        # Balance point between compromise and collaboration
    genealogical_weight: float = 0.0  # Influence of historical emergence
    epigenetic_expression: float = 0.0  # Degree of environmental influence on expression

@dataclass
class EpigeneticState:
    """
    Captures the epigenetic state of the system at a given moment.

    Based on the Epigenesis and Antifinity thesis (Theorem 7: Inclusive Epistemology),
    this represents the system's current moral and cognitive state.
    """
    symbolic_fields: Dict[str, np.ndarray] = field(default_factory=dict)
    agent_states: Dict[str, Any] = field(default_factory=dict)
    metrics: MoralMetrics = field(default_factory=MoralMetrics)
    regime: str = "stable_coherence"
    context_id: int = 0
    surplus_expression: float = 0.0  # Degree of surplus expression
    genealogical_memory: List[float] = field(default_factory=list)

class AntifinitySensor:
    """
    Implements Antifinity moral metrics for the Émile cognitive framework.

    Based on the Epigenesis and Antifinity thesis, measuring:
    - Collaboration: Extension of capability beyond individual potential
    - Compromise: Reduction or limitation of potential

    These form the basis for evaluating emergent moral dynamics in the
    cognitive system.
    """

    def __init__(self, cfg=CONFIG):
        """
        Initialize the Antifinity sensor.

        Args:
            cfg: Configuration parameters
        """
        self.cfg = cfg

        # Current metrics
        self.metrics = MoralMetrics()

        # History tracking
        self.metric_history = []
        self.epigenetic_states = []

        # Thresholds
        self.compromise_threshold = cfg.COMPROMISE_THRESHOLD
        self.collaboration_weight = cfg.COLLABORATION_WEIGHT

    def calculate_compromise(self, sigma: np.ndarray,
                           agent_system: Dict[str, Any]) -> float:
        """
        Calculate the compromise metric based on symbolic curvature and agent states.

        Compromise occurs when potentiality is reduced or limited (Axiom Five).

        Args:
            sigma: Current symbolic curvature field
            agent_system: Current state of the agent system

        Returns:
            Compromise score between 0 (no compromise) and 1 (high compromise)
        """
        # 1. Negative sigma indicates phi > psi (actualization dominates potentiality)
        # This represents a form of compromise where being-itself becomes overly limited
        neg_sigma_component = float(np.mean(np.maximum(0, -sigma)))

        # 2. Spatial regions with high agent overlap indicate competing constraints
        overlap_field = np.zeros(len(sigma))
        if "agent_count" in agent_system and agent_system["agent_count"] > 1:
            # Estimate overlap from shared workspace (since actual masks would require loop)
            workspace = agent_system.get("shared_workspace", np.zeros_like(sigma))
            active_count = agent_system.get("active_agents", 1)
            overlap_field = np.abs(workspace) / max(1.0, active_count * 0.5)

        overlap_component = float(np.mean(overlap_field))

        # 3. Context shifts represent moments where current context can no longer
        # contain distinctions - a form of compromise requiring reorganization
        context_component = 0.0
        context_shifted = agent_system.get("context_shifted", False)
        if context_shifted:
            context_component = 0.8  # High compromise during context shifts

        # 4. Surplus component - high variance in surplus indicates uneven compromise
        surplus_var = 0.0
        if "combined_fields" in agent_system and "surplus" in agent_system["combined_fields"]:
            surplus = agent_system["combined_fields"]["surplus"]
            surplus_var = float(np.var(surplus))

        # Calculate overall compromise as weighted combination
        compromise = (
            0.5 * neg_sigma_component +  # Weight from negative sigma
            0.2 * overlap_component +    # Weight from agent overlap
            0.2 * context_component +    # Weight from context shifts
            0.1 * min(1.0, surplus_var * 10.0)  # Weight from surplus variance
        )

        # Normalize to [0,1] range
        return float(np.clip(compromise, 0.0, 1.0))

    def calculate_collaboration(self, symbolic_fields: Dict[str, np.ndarray],
                              agent_system: Dict[str, Any],
                              regime: str) -> float:
        """
        Calculate the collaboration metric based on symbolic fields and agent interactions.

        Collaboration occurs when potential is extended beyond individual capability (Axiom Seven).

        Args:
            symbolic_fields: Current symbolic fields
            agent_system: Current state of the agent system
            regime: Current symbolic regime

        Returns:
            Collaboration score between 0 (no collaboration) and 1 (high collaboration)
        """
        sigma = symbolic_fields["sigma"]
        psi = symbolic_fields["psi"]
        phi = symbolic_fields["phi"]

        # 1. Positive sigma indicates psi > phi (potentiality exceeds actuality)
        # This represents opportunity for collaboration where potentiality is available
        pos_sigma_component = float(np.mean(np.maximum(0, sigma)))

        # 2. Different regimes have different collaboration potential
        regime_factors = {
            "stable_coherence": 0.7,    # Strong collaboration - organized structure
            "symbolic_turbulence": 0.5,  # Medium collaboration - creative chaos
            "flat_rupture": 0.2,         # Low collaboration - depleted system
            "quantum_oscillation": 0.8    # High collaboration - rhythmic exchange
        }

        regime_component = regime_factors.get(regime, 0.5)

        # 3. Agent diversity represents different perspectives collaborating
        # Calculate diversity based on agent count and activity
        diversity_component = 0.0
        if "agent_count" in agent_system and agent_system["agent_count"] > 1:
            agent_count = min(agent_system["agent_count"], self.cfg.MAX_AGENTS)
            diversity_component = 1.0 - (1.0 / (1.0 + 0.1 * agent_count))

        # 4. Psi-Phi alignment (positive correlation in some regions, negative in others)
        # indicates complementary rather than redundant processing
        alignment_component = 0.0
        if len(psi) > 10 and len(phi) > 10:
            # Calculate local correlations in windows
            window_size = 10
            n_windows = len(psi) // window_size
            correlations = []

            for i in range(n_windows):
                start = i * window_size
                end = (i + 1) * window_size
                psi_window = psi[start:end]
                phi_window = phi[start:end]

                # Only calculate if there's variance
                if np.var(psi_window) > 0.001 and np.var(phi_window) > 0.001:
                    corr = np.corrcoef(psi_window, phi_window)[0, 1]
                    correlations.append(corr)

            if correlations:
                # Diversity of correlations indicates collaborative complexity
                alignment_component = min(1.0, np.std(correlations) * 2.0)

        # Calculate overall collaboration as weighted combination
        collaboration = (
            0.4 * pos_sigma_component +  # Weight from positive sigma (available potential)
            0.3 * regime_component +     # Weight from regime characteristics
            0.2 * diversity_component +  # Weight from agent diversity
            0.1 * alignment_component    # Weight from field alignment patterns
        )

        # Normalize to [0,1] range
        return float(np.clip(collaboration, 0.0, 1.0))

    def calculate_antifinity_quotient(self, collaboration: float,
                                    compromise: float) -> float:
        """
        Calculate the Antifinity quotient - a measure of being-in-addition-to-itself.

        Antifinity represents the degree to which a system extends beyond itself
        rather than being compromised (Axiom Two).

        Args:
            collaboration: Collaboration score
            compromise: Compromise score

        Returns:
            Antifinity quotient between -1 (compromised) and 1 (collaborative)
        """
        # Weight collaboration more heavily based on configuration
        weighted_collaboration = collaboration * self.collaboration_weight
        weighted_compromise = compromise * (1.0 - self.collaboration_weight)

        # Calculate antifinity as the balance between collaboration and compromise
        # Range from -1 (fully compromised) to 1 (fully collaborative)
        quotient = weighted_collaboration - weighted_compromise

        return float(np.clip(quotient, -1.0, 1.0))

    def calculate_epigenetic_metrics(self, symbolic_fields: Dict[str, np.ndarray],
                                   agent_system: Dict[str, Any],
                                   regime: str) -> MoralMetrics:
        """
        Calculate all moral metrics based on current system state.

        This implements the full moral assessment based on Epigenesis and Antifinity.

        Args:
            symbolic_fields: Current symbolic fields
            agent_system: Current state of the agent system
            regime: Current symbolic regime

        Returns:
            MoralMetrics dataclass with all calculated metrics
        """
        # Calculate core metrics
        compromise = self.calculate_compromise(symbolic_fields["sigma"], agent_system)
        collaboration = self.calculate_collaboration(symbolic_fields, agent_system, regime)

        # Calculate antifinity quotient
        antifinity = self.calculate_antifinity_quotient(collaboration, compromise)

        # Calculate tension index (how balanced the system is)
        tension = 1.0 - abs(antifinity)

        # Calculate genealogical weight (influence of history)
        # Use agent system history as a proxy
        genealogical_weight = 0.0
        if "agent_count" in agent_system:
            step_count = agent_system.get("step_count", 0)
            genealogical_weight = min(0.9, step_count / 1000.0)

        # Calculate epigenetic expression (environmental influence)
        # Use the influence of context on the system
        context_id = agent_system.get("global_context_id", 0)
        epigenetic_expression = min(0.9, 0.1 * context_id + 0.1 * agent_system.get("active_agents", 1))

        # Create and return metrics
        metrics = MoralMetrics(
            collaboration_score=collaboration,
            compromise_score=compromise,
            antifinity_quotient=antifinity,
            tension_index=tension,
            genealogical_weight=genealogical_weight,
            epigenetic_expression=epigenetic_expression
        )

        return metrics

    def step(self, symbolic_fields: Dict[str, np.ndarray],
           agent_system: Dict[str, Any],
           regime: str) -> Dict[str, Any]:
        """
        Process a single step of Antifinity analysis.

        Args:
            symbolic_fields: Current symbolic fields
            agent_system: Current state of the agent system
            regime: Current symbolic regime

        Returns:
            Dictionary with Antifinity metrics
        """
        # Calculate current metrics
        self.metrics = self.calculate_epigenetic_metrics(symbolic_fields, agent_system, regime)

        # Record in history
        self.metric_history.append({
            "collaboration": self.metrics.collaboration_score,
            "compromise": self.metrics.compromise_score,
            "antifinity": self.metrics.antifinity_quotient,
            "tension": self.metrics.tension_index,
            "regime": regime,
            "context_id": agent_system.get("global_context_id", 0)
        })

        # Create current epigenetic state
        state = EpigeneticState(
            symbolic_fields={k: v.copy() for k, v in symbolic_fields.items() if isinstance(v, np.ndarray)},
            agent_states=agent_system.copy(),
            metrics=self.metrics,
            regime=regime,
            context_id=agent_system.get("global_context_id", 0),
            surplus_expression=float(np.mean(symbolic_fields.get("surplus", np.zeros(1))))
        )

        # Keep bounded history
        self.epigenetic_states.append(state)
        if len(self.epigenetic_states) > 100:
            self.epigenetic_states = self.epigenetic_states[-100:]

        # Return current metrics
        return {
            "metrics": {
                "collaboration": self.metrics.collaboration_score,
                "compromise": self.metrics.compromise_score,
                "antifinity": self.metrics.antifinity_quotient,
                "tension": self.metrics.tension_index,
                "genealogical_weight": self.metrics.genealogical_weight,
                "epigenetic_expression": self.metrics.epigenetic_expression
            },
            "state": {
                "regime": regime,
                "context_id": agent_system.get("global_context_id", 0)
            }
        }

    def get_current_metrics(self) -> MoralMetrics:
        """
        Get the current moral metrics.

        Returns:
            MoralMetrics dataclass with current values
        """
        return self.metrics

    def get_metric_history(self) -> List[Dict[str, Any]]:
        """
        Get the history of moral metrics.

        Returns:
            List of metric dictionaries
        """
        return self.metric_history

    def interpret_antifinity_state(self) -> Dict[str, Any]:
        """
        Provide a high-level interpretation of the current Antifinity state.

        Returns:
            Dictionary with interpretation data
        """
        # Get current metrics
        metrics = self.metrics

        # Determine primary mode
        if metrics.antifinity_quotient > 0.3:
            primary_mode = "collaborative"
        elif metrics.antifinity_quotient < -0.3:
            primary_mode = "compromised"
        else:
            primary_mode = "balanced"

        # Determine secondary characteristics
        characteristics = []

        if metrics.tension_index > 0.7:
            characteristics.append("high_tension")

        if metrics.genealogical_weight > 0.6:
            characteristics.append("historically_influenced")

        if metrics.epigenetic_expression > 0.6:
            characteristics.append("environmentally_adaptive")

        # Get trend from recent history
        trend = "stable"
        if len(self.metric_history) >= 5:
            recent = [entry["antifinity"] for entry in self.metric_history[-5:]]
            if recent[-1] > recent[0] + 0.2:
                trend = "increasing_antifinity"
            elif recent[-1] < recent[0] - 0.2:
                trend = "decreasing_antifinity"

        # Create interpretation
        interpretation = {
            "primary_mode": primary_mode,
            "characteristics": characteristics,
            "trend": trend,
            "antifinity_level": "high" if abs(metrics.antifinity_quotient) > 0.7 else
                               "medium" if abs(metrics.antifinity_quotient) > 0.3 else
                               "low",
            "balance_state": "collaborative_dominant" if metrics.collaboration_score > metrics.compromise_score + 0.3 else
                           "compromise_dominant" if metrics.compromise_score > metrics.collaboration_score + 0.3 else
                           "balanced"
        }

        return interpretation

from emile_cogito.kainos.module_wide_flow_mapper import auto_map_module_flow
auto_map_module_flow(__name__)  # Maps the entire module!
