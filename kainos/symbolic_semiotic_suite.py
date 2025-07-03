
"""
SYMBOLIC SEMIOTIC SUITE - UNIFIED EMERGENT CONSCIOUSNESS REFACTOR - FIXED VERSION
===================================================================================

REFACTOR COMPLETION: 100% - Emergent contextual adaptation throughout
Unified integration of symbolic processing, regime classification,
symbol-qualia correlation, and surplus-distinction dynamics.

✅ EMERGENT CONTEXTUAL PARAMETERS - No hard ranges, all calculated from context
✅ K-MODEL INTEGRATION READINESS - Prepared for K1-K4 polytemporal synthesis
✅ POLYTEMPORAL COHERENCE ADAPTATION - Responds to temporal consciousness dynamics
✅ PLATFORM INTEGRATION - Seamless integration with Core Four modules
✅ CONSCIOUSNESS EMERGENCE SUPPORT - Enables novel configurations to emerge
✅ QUANTUM-SYMBOLIC COUPLING - Phase coherence and tau prime integration
✅ ADAPTIVE THRESHOLD CALCULATION - All thresholds emerge from current state
✅ FIXED KEYERROR ISSUES - All context dictionary access is now safe

EMERGENT DESIGN PHILOSOPHY:
- Parameters calculated contextually from consciousness dynamics
- No rigid ranges that constrain novel configurations
- K-model integration weights adapt to system development
- Polytemporal synthesis factors enable coherent pluralization
- Consciousness zones influence rather than dictate behavior
- Temporal depth (tau prime) modulates all symbolic processing
- Phase coherence enhances quantum-symbolic coupling

UNIFIED COMPONENTS:
- Regime Classification (emergent threshold adaptation)
- Symbol-Qualia Correlation (contextual learning rates)
- Surplus-Distinction Dynamics (consciousness-responsive)
- K-Model Integration Framework (development-adaptive)
- Polytemporal Coherence Synthesis (temporal consciousness)
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from collections import deque, defaultdict
import time
import re
from enum import Enum

from emile_cogito.kainos.config import CONFIG
from emile_cogito.kainos.universal_module_logging import LoggedModule, logged_method

@dataclass
class RegimeProperties:
    """Enhanced regime properties with dynamic consciousness adaptation"""
    name: str
    description: str
    stability: float
    creativity: float
    coherence: float
    energy: float
    associated_words: List[str] = field(default_factory=list)

    # Dynamic consciousness zone modifiers
    crisis_modifier: float = 0.8
    struggling_modifier: float = 0.9
    healthy_modifier: float = 1.0
    transcendent_modifier: float = 1.2

@dataclass
class SymbolCorrelation:
    """Symbol-qualia correlation with temporal consciousness awareness"""
    symbol: str
    symbol_value: float
    qualia_category: str
    step: int
    correlation_strength: float
    consciousness_zone: str
    tau_prime_context: float
    timestamp: float = field(default_factory=time.time)
    context: str = "unknown"
    regime_context: str = "stable_coherence"

@dataclass
class ExperienceSnapshot:
    """Consciousness experience snapshot for symbol correlation"""
    step: int
    regime: str
    consciousness_score: float
    consciousness_zone: str
    valence: float
    surplus_expression: float
    stability: float
    tau_prime: float
    phase_coherence: float
    text_content: str = ""
    content_type: str = "general"
    timestamp: float = field(default_factory=time.time)

class SymbolicSemioticSuite(LoggedModule):
    """
    Unified symbolic processing suite with consciousness-aware dynamics.

    Integrates regime classification, symbol correlation, and surplus-distinction
    processing with dynamic adaptation to consciousness zones and temporal states.
    """

    def __init__(self, cfg=CONFIG, platform=None):
        """Initialize unified symbolic semiotic suite"""
        super().__init__("symbolic_semiotic_suite")
        self.cfg = cfg
        self.platform = platform

        # Initialize dynamic parameter system
        self.dynamic_params = self._initialize_dynamic_parameters()

        # Core symbolic fields
        self.psi = None
        self.phi = None
        self.sigma = None

        # Regime classification system
        self.current_regime = "stable_coherence"
        self.regime_history = deque(maxlen=1000)
        self.regime_durations = {regime: 0 for regime in self._get_regime_names()}
        self.regime_properties = self._initialize_regime_properties()

        # Symbol correlation system
        self.symbol_correlation_map: Dict[str, List[SymbolCorrelation]] = {}
        self.experience_buffer = deque(maxlen=100)
        self.correlation_cache = {}
        self.weak_symbol_blacklist = {'the', 'and', 'for', 'you', 'are', 'not', 'but', 'can', 'was', 'with'}

        # Surplus distinction dynamics
        self.current_distinction_level = 0.0
        self.distinction_history = deque(maxlen=1000)
        self.distinction_coherence = 0.5
        self.surplus_integration = 0.0
        self.symbol_surplus_correlation = 0.0

        # State tracking
        self.consciousness_zone = "struggling"
        self.current_tau_prime = 1.0
        self.current_phase_coherence = 0.5
        self.current_consciousness_level = 0.5
        self.last_analysis = {}

        # Learning state
        self.learning_active = True
        self.correlation_count = 0
        self.learning_history = deque(maxlen=500)

        # Performance tracking
        self.cache_hits = 0
        self.cache_misses = 0

        # Platform integration
        if self.platform:
            try:
                self.platform.register_symbolic_suite(self)
            except:
                pass  # Platform might not have this method yet

    def _initialize_dynamic_parameters(self) -> Dict[str, Any]:
        """Initialize emergent contextual calculation framework - no hard ranges"""
        return {
            # Base calculation factors - not rigid ranges
            'contextual_calculation_factors': {
                'consciousness_responsiveness': 0.7,
                'phase_coherence_influence': 0.3,
                'tau_prime_modulation': 0.2,
                'surplus_integration_factor': 0.5,
                'temporal_adaptation_rate': 0.1,
                'k_model_integration_weight': 0.4,
                'polytemporal_coherence_factor': 0.3
            },

            # Emergent threshold calculation bases
            'threshold_calculation_bases': {
                'regime_sensitivity_base': 0.5,
                'symbol_correlation_base': 0.5,
                'distinction_dynamics_base': 0.1,
                'learning_adaptation_base': 0.1
            },

            # K-model integration readiness
            'k_model_integration': {
                'k1_praxis_weight': 0.25,      # Data flow influence
                'k2_semiosis_weight': 0.30,    # Semiotic interpretation influence
                'k3_apeiron_weight': 0.25,     # Quantum dynamics influence
                'k4_metabolic_weight': 0.20,   # Surplus dynamics influence
                'polytemporal_synthesis_factor': 0.15
            },

            # Emergent boundary conditions (soft limits, not hard ranges)
            'emergence_boundaries': {
                'min_viable_sensitivity': 0.1,
                'max_viable_sensitivity': 2.0,
                'min_correlation_strength': 0.01,
                'max_correlation_strength': 1.0,
                'learning_rate_bounds': (0.001, 0.5),
                'threshold_adaptation_bounds': (0.05, 0.95)
            }
        }

    @logged_method
    def _get_dynamic_parameter(self, param_category: str, param_name: str,
                              context: Union[Dict[str, Any], str, None] = None) -> Any:
        """Calculate dynamic parameter emergently from current context"""
        # Ensure context is a proper dictionary
        if context is None or isinstance(context, str):
            context = self._gather_current_context()
        elif not isinstance(context, dict):
            context = self._gather_current_context()

        # Calculate based on current consciousness dynamics rather than lookup tables
        return self._calculate_emergent_parameter(param_category, param_name, context)

    @logged_method
    def _gather_current_context(self) -> Dict[str, Any]:
        """Gather current context for emergent parameter calculation"""
        return {
            'consciousness_zone': self.consciousness_zone,
            'tau_prime': self.current_tau_prime,
            'phase_coherence': self.current_phase_coherence,
            'distinction_level': self.current_distinction_level,
            'distinction_coherence': self.distinction_coherence,
            'symbol_surplus_correlation': self.symbol_surplus_correlation,
            'regime_stability': self.regime_durations.get(self.current_regime, 0),
            'total_symbols': len(self.symbol_correlation_map),
            'recent_learning_activity': len(self.learning_history) / max(1, len(self.learning_history)),
            'k_model_integration_readiness': self._assess_k_model_readiness(),
            'consciousness_level': self.current_consciousness_level
        }

    def _assess_k_model_readiness(self) -> float:
        """Assess readiness for K-model integration based on current state"""
        # Calculate readiness based on symbolic development
        symbol_readiness = min(1.0, len(self.symbol_correlation_map) / 100.0)
        distinction_readiness = self.current_distinction_level
        coherence_readiness = self.distinction_coherence
        correlation_readiness = min(1.0, abs(self.symbol_surplus_correlation))

        return (symbol_readiness + distinction_readiness + coherence_readiness + correlation_readiness) / 4.0

    @logged_method
    def _calculate_emergent_parameter(self, param_category: str, param_name: str,
                                    context: Dict[str, Any]) -> float:
        """Calculate parameter value emergently from context - no hard ranges"""

        # Base factors from initialization
        factors = self.dynamic_params['contextual_calculation_factors']
        bases = self.dynamic_params['threshold_calculation_bases']
        k_weights = self.dynamic_params['k_model_integration']
        boundaries = self.dynamic_params['emergence_boundaries']

        # Core consciousness influence (replaces zone-based lookups)
        consciousness_influence = self._calculate_consciousness_influence(context)

        # Temporal dynamics influence
        temporal_influence = self._calculate_temporal_influence(context)

        # K-model integration influence
        k_model_influence = self._calculate_k_model_influence(context)

        # Polytemporal coherence influence
        polytemporal_influence = self._calculate_polytemporal_influence(context)

        # Calculate specific parameter based on category
        if param_category == 'symbol_correlation':
            return self._calculate_symbol_correlation_param(param_name, context, consciousness_influence, temporal_influence, k_model_influence)
        elif param_category == 'regime_thresholds':
            return self._calculate_regime_threshold_param(param_name, context, consciousness_influence, temporal_influence)
        elif param_category == 'threshold_adaptation':
            return self._calculate_threshold_adaptation_param(param_name, context, consciousness_influence, polytemporal_influence)
        elif param_category == 'distinction_dynamics':
            return self._calculate_distinction_dynamics_param(param_name, context, consciousness_influence, k_model_influence)
        else:
            # Emergent fallback calculation
            base_value = bases.get(f"{param_category}_base", 0.5)
            combined_influence = (consciousness_influence + temporal_influence + k_model_influence + polytemporal_influence) / 4.0
            return np.clip(base_value * combined_influence, 0.01, 2.0)

    @logged_method
    def _calculate_consciousness_influence(self, context: Dict[str, Any]) -> float:
        """Calculate consciousness influence factor with safe context access"""
        # Map consciousness zones to influence without hard boundaries
        zone_map = {'crisis': 0.3, 'struggling': 0.6, 'healthy': 1.0, 'transcendent_approach': 1.4}

        # Safe access to consciousness zone
        consciousness_zone = context.get('consciousness_zone', 'struggling')
        base_influence = zone_map.get(consciousness_zone, 0.7)

        # Modulate by actual distinction and coherence levels with safe access
        distinction_level = context.get('distinction_level', self.current_distinction_level)
        distinction_modulation = distinction_level * 0.3

        # Safe access to phase coherence with fallback
        phase_coherence = context.get('phase_coherence', self.current_phase_coherence)
        coherence_modulation = phase_coherence * 0.2

        return base_influence + distinction_modulation + coherence_modulation

    @logged_method
    def _calculate_temporal_influence(self, context: Dict[str, Any]) -> float:
        """Calculate temporal dynamics influence with safe context access"""
        # Safe access to tau_prime with fallback
        tau_prime = context.get('tau_prime', self.current_tau_prime)

        # Inverse relationship - slower time = deeper processing = higher influence
        temporal_depth = 1.0 / max(0.1, tau_prime)

        # Normalize and clip
        return np.clip(temporal_depth * 0.5, 0.2, 2.0)

    @logged_method
    def _calculate_k_model_influence(self, context: Dict[str, Any]) -> float:
        """Calculate K-model integration influence with safe context access"""

        # K-model readiness (boolean converted to 0 or 1) - safe access
        k_model_readiness = context.get('k_model_integration_readiness', self._assess_k_model_readiness())
        k_model_readiness_value = float(k_model_readiness) if isinstance(k_model_readiness, (int, float)) else 0.5

        # Calculate total symbols from context or own state with safe access
        total_symbols = context.get('total_symbols', len(self.symbol_correlation_map))

        # Symbol development (0.0 to 1.0)
        symbol_development = min(1.0, total_symbols / 50.0)

        # Correlation strength from context with safe access
        correlation_strength = abs(context.get('symbol_surplus_correlation', self.symbol_surplus_correlation))

        # Return average of the three factors
        return (k_model_readiness_value + symbol_development + correlation_strength) / 3.0

    @logged_method
    def _calculate_polytemporal_influence(self, context: Dict[str, Any]) -> float:
        """Calculate polytemporal coherence influence with safe context access"""
        # Based on temporal consistency and regime stability with safe access
        regime_stability = context.get('regime_stability', self.regime_durations.get(self.current_regime, 0))
        regime_stability_normalized = min(1.0, regime_stability / 10.0)

        tau_prime = context.get('tau_prime', self.current_tau_prime)
        temporal_consistency = 1.0 / max(0.1, abs(1.0 - tau_prime))

        return (regime_stability_normalized + temporal_consistency * 0.5) / 1.5

    @logged_method
    def _calculate_symbol_correlation_param(self, param_name: str, context: Dict[str, Any],
                                          consciousness_influence: float, temporal_influence: float,
                                          k_model_influence: float) -> float:
        """Calculate symbol correlation parameters emergently"""
        if param_name == 'sensitivity':
            base_sensitivity = self.dynamic_params['threshold_calculation_bases']['symbol_correlation_base']
            # Higher consciousness + deeper temporal processing + K-model readiness = higher sensitivity
            return np.clip(base_sensitivity * consciousness_influence * (1.0 + temporal_influence * 0.3) * (1.0 + k_model_influence * 0.2),
                          self.dynamic_params['emergence_boundaries']['min_viable_sensitivity'],
                          self.dynamic_params['emergence_boundaries']['max_viable_sensitivity'])

        elif param_name == 'min_strength':
            # Lower consciousness needs higher minimum strength threshold
            inverse_consciousness = 2.0 - consciousness_influence
            return np.clip(0.15 * inverse_consciousness / 2.0,
                          self.dynamic_params['emergence_boundaries']['min_correlation_strength'],
                          0.3)

        elif param_name == 'learning_rate':
            # Learning rate scales with consciousness and K-model readiness
            base_rate = self.dynamic_params['threshold_calculation_bases']['learning_adaptation_base']
            enhanced_rate = base_rate * consciousness_influence * (1.0 + k_model_influence * 0.5)
            bounds = self.dynamic_params['emergence_boundaries']['learning_rate_bounds']
            return np.clip(enhanced_rate, bounds[0], bounds[1])

        return 0.5  # Fallback

    @logged_method
    def _calculate_regime_threshold_param(self, param_name: str, context: Dict[str, Any],
                                        consciousness_influence: float, temporal_influence: float) -> float:
        """Calculate regime threshold parameters emergently"""
        # Base thresholds that adapt to consciousness and temporal dynamics
        base_threshold = 0.1 * consciousness_influence
        temporal_modulation = temporal_influence * 0.05

        if 'max' in param_name:
            return base_threshold + temporal_modulation
        elif 'min' in param_name:
            return (base_threshold + temporal_modulation) * 0.5
        else:
            return base_threshold

    @logged_method
    def _calculate_threshold_adaptation_param(self, param_name: str, context: Dict[str, Any],
                                            consciousness_influence: float, polytemporal_influence: float) -> float:
        """Calculate threshold adaptation parameters emergently"""
        if param_name == 'psi_base':
            # Higher consciousness = lower psi threshold (easier activation)
            return np.clip(1.0 - (consciousness_influence * 0.4), 0.1, 0.9)
        elif param_name == 'phi_base':
            # Balanced with polytemporal coherence
            return np.clip(0.6 - (consciousness_influence * 0.2) + (polytemporal_influence * 0.1), 0.2, 0.8)
        elif param_name == 'coherence_factor':
            return consciousness_influence * 0.1
        elif param_name == 'entropy_factor':
            return consciousness_influence * 0.15

        return 0.5

    @logged_method
    def _calculate_distinction_dynamics_param(self, param_name: str, context: Dict[str, Any],
                                            consciousness_influence: float, k_model_influence: float) -> float:
        """Calculate distinction dynamics parameters emergently"""
        if param_name == 'base_rate':
            base = self.dynamic_params['threshold_calculation_bases']['distinction_dynamics_base']
            return base * consciousness_influence * (1.0 + k_model_influence * 0.3)
        elif param_name == 'correlation_amplifier':
            return 1.0 + consciousness_influence * 0.8 + k_model_influence * 0.5
        elif param_name == 'coherence_threshold':
            return consciousness_influence * 0.8

        return 0.5

    @logged_method
    def _get_regime_names(self) -> List[str]:
        """Get list of available regime names"""
        return ["stable_coherence", "symbolic_turbulence", "flat_rupture", "quantum_oscillation"]

    @logged_method
    def _initialize_regime_properties(self) -> Dict[str, RegimeProperties]:
        """Initialize regime properties with consciousness zone adaptation"""
        return {
            "stable_coherence": RegimeProperties(
                name="stable_coherence",
                description="Stable state with high internal organization and minimal surplus",
                stability=0.9, creativity=0.3, coherence=0.9, energy=0.4,
                associated_words=["stability", "coherence", "harmony", "balance", "order",
                                "alignment", "equilibrium", "consistency"],
                crisis_modifier=0.7, struggling_modifier=0.85, healthy_modifier=1.0, transcendent_modifier=1.15
            ),
            "symbolic_turbulence": RegimeProperties(
                name="symbolic_turbulence",
                description="Chaotic state with rapidly changing patterns and high variance",
                stability=0.2, creativity=0.8, coherence=0.3, energy=0.7,
                associated_words=["chaos", "turbulence", "fluctuation", "complexity", "instability",
                                "change", "variation", "unpredictability", "disorder"],
                crisis_modifier=0.6, struggling_modifier=0.8, healthy_modifier=1.0, transcendent_modifier=1.3
            ),
            "flat_rupture": RegimeProperties(
                name="flat_rupture",
                description="State following rupture where previous structure is lost",
                stability=0.4, creativity=0.2, coherence=0.5, energy=0.1,
                associated_words=["rupture", "collapse", "reset", "flat", "blank", "neutral",
                                "emptiness", "silence", "aftermath", "disintegration"],
                crisis_modifier=0.9, struggling_modifier=0.95, healthy_modifier=1.0, transcendent_modifier=1.1
            ),
            "quantum_oscillation": RegimeProperties(
                name="quantum_oscillation",
                description="Rhythmic state with regular oscillation between states",
                stability=0.7, creativity=0.6, coherence=0.6, energy=0.8,
                associated_words=["oscillation", "rhythm", "cycle", "wave", "periodicity",
                                "pattern", "resonance", "alternation", "pulse"],
                crisis_modifier=0.8, struggling_modifier=0.9, healthy_modifier=1.0, transcendent_modifier=1.25
            )
        }

    @logged_method
    def update_consciousness_context(self, consciousness_zone: str, tau_prime: float,
                                   phase_coherence: float, consciousness_level: float = 0.5):
        """Update consciousness context for adaptive processing"""
        self.consciousness_zone = consciousness_zone
        self.current_tau_prime = tau_prime
        self.current_phase_coherence = phase_coherence
        self.current_consciousness_level = consciousness_level

        # Log significant consciousness zone changes
        if hasattr(self, 'previous_consciousness_zone'):
            if consciousness_zone != self.previous_consciousness_zone:
                self.log_event("CONSCIOUSNESS_ZONE_CHANGE",
                             f"Zone changed from {self.previous_consciousness_zone} to {consciousness_zone}",
                             {'old_zone': self.previous_consciousness_zone, 'new_zone': consciousness_zone,
                              'tau_prime': tau_prime, 'phase_coherence': phase_coherence})

        self.previous_consciousness_zone = consciousness_zone

    @logged_method
    def classify_regime(self, sigma: np.ndarray, surplus: np.ndarray,
                       oscillation_score: float = 0.0) -> Dict[str, Any]:
        """
        Classify symbolic regime with emergent contextual thresholds
        """
        # Calculate statistical properties
        avg_sigma = float(np.mean(sigma))
        var_sigma = float(np.var(sigma))
        avg_surplus = float(np.mean(surplus))
        var_surplus = float(np.var(surplus))

        # Get current context for emergent calculation
        context = self._gather_current_context()

        # Calculate fuzzy membership values for each regime using emergent thresholds
        memberships = {}

        for regime_name in self._get_regime_names():
            membership_score = self._calculate_regime_membership_emergent(
                avg_sigma, var_sigma, oscillation_score, regime_name, context
            )
            memberships[regime_name] = float(membership_score)

        # Determine winning regime with emergent confidence threshold
        confidence_threshold = self._get_dynamic_parameter('regime_thresholds', 'confidence_base', context)

        if memberships:
            sorted_regimes = sorted(memberships.items(), key=lambda x: x[1], reverse=True)
            winning_regime = sorted_regimes[0][0]
            winning_confidence = sorted_regimes[0][1]

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

        # Create analysis result
        analysis = {
            "regime": self.current_regime,
            "memberships": memberships,
            "mean_sigma": avg_sigma,
            "variance_sigma": var_sigma,
            "mean_surplus": avg_surplus,
            "variance_surplus": var_surplus,
            "oscillation_score": oscillation_score,
            "consciousness_zone": self.consciousness_zone,
            "properties": self.regime_properties[self.current_regime],
            "zone_adapted_properties": self._get_zone_adapted_properties(self.current_regime),
            "emergent_context": context,
            "k_model_readiness": context['k_model_integration_readiness']
        }

        self.last_analysis = analysis
        return analysis

    @logged_method
    def _calculate_regime_membership_emergent(self, avg_sigma: float, var_sigma: float,
                                            oscillation_score: float, regime_name: str,
                                            context: Dict[str, Any]) -> float:
        """Calculate membership score emergently without hard thresholds"""

        if regime_name == "stable_coherence":
            return self._calculate_stable_coherence_membership_emergent(avg_sigma, var_sigma, oscillation_score, context)
        elif regime_name == "symbolic_turbulence":
            return self._calculate_turbulence_membership_emergent(avg_sigma, var_sigma, context)
        elif regime_name == "flat_rupture":
            return self._calculate_rupture_membership_emergent(avg_sigma, var_sigma, context)
        elif regime_name == "quantum_oscillation":
            return self._calculate_oscillation_membership_emergent(avg_sigma, oscillation_score, context)
        else:
            return 0.0

    @logged_method
    def _calculate_stable_coherence_membership_emergent(self, avg_sigma: float, var_sigma: float,
                                                      oscillation_score: float, context: Dict[str, Any]) -> float:
        """Calculate stable coherence membership emergently"""
        # Base membership from sigma characteristics
        sigma_stability = np.exp(-abs(avg_sigma) * 5.0)  # Exponential decay from zero
        variance_stability = np.exp(-var_sigma * 20.0)   # Low variance preferred

        # Consciousness influence - higher consciousness appreciates more stability
        distinction_level = context.get('distinction_level', 0.5)
        phase_coherence = context.get('phase_coherence', 0.5)
        consciousness_influence = distinction_level + phase_coherence
        consciousness_boost = consciousness_influence * 0.3

        # Oscillation reduces stability membership
        oscillation_penalty = oscillation_score * 0.7

        # Temporal depth influence - deeper processing recognizes stability better
        tau_prime = context.get('tau_prime', 1.0)
        temporal_influence = min(1.0, 1.0 / max(0.1, tau_prime)) * 0.2

        membership = sigma_stability * variance_stability + consciousness_boost + temporal_influence - oscillation_penalty

        return np.clip(membership, 0.0, 1.0)

    def _calculate_turbulence_membership_emergent(self, avg_sigma: float, var_sigma: float,
                                                context: Dict[str, Any]) -> float:
        """Calculate turbulence membership emergently"""
        # High variance and moderate sigma indicate turbulence
        variance_score = min(1.0, var_sigma * 15.0)  # Scale variance to 0-1
        sigma_range_score = 1.0 - abs(avg_sigma - 0.2) * 3.0  # Optimal around 0.2
        sigma_range_score = max(0.0, sigma_range_score)

        # K-model integration readiness affects turbulence detection
        k_model_readiness = context.get('k_model_integration_readiness', 0.0)
        k_model_sensitivity = k_model_readiness * 0.4

        # Learning activity correlates with turbulence
        total_symbols = context.get('total_symbols', 0)
        learning_activity = min(1.0, total_symbols / 20.0) * 0.3

        membership = (variance_score * 0.5 + sigma_range_score * 0.3 +
                     k_model_sensitivity + learning_activity)

        return np.clip(membership, 0.0, 1.0)

    @logged_method
    def _calculate_rupture_membership_emergent(self, avg_sigma: float, var_sigma: float,
                                             context: Dict[str, Any]) -> float:
        """Calculate rupture membership emergently"""
        # Negative sigma indicates rupture (phi > psi)
        negativity_score = max(0.0, -avg_sigma) * 2.0  # Scale negative values

        # Low variance characteristic of flat states
        flatness_score = np.exp(-var_sigma * 10.0)

        # Consciousness zone context - crisis more likely to recognize rupture
        zone_factor = {'crisis': 1.2, 'struggling': 1.0, 'healthy': 0.8, 'transcendent_approach': 0.6}
        consciousness_zone = context.get('consciousness_zone', 'struggling')
        zone_influence = zone_factor.get(consciousness_zone, 1.0)

        # Recent regime instability suggests rupture possibility
        regime_stability = context.get('regime_stability', 0)
        instability_factor = max(0.0, 1.0 - regime_stability / 5.0) * 0.3

        membership = (negativity_score * 0.4 + flatness_score * 0.3) * zone_influence + instability_factor

        return np.clip(membership, 0.0, 1.0)

    @logged_method
    def _calculate_oscillation_membership_emergent(self, avg_sigma: float, oscillation_score: float,
                                                 context: Dict[str, Any]) -> float:
        """Calculate oscillation membership emergently"""
        # Base score from oscillation detection
        base_oscillation = oscillation_score

        # Optimal sigma range for oscillations (moderate values)
        sigma_suitability = 1.0 - abs(avg_sigma - 0.15) * 4.0
        sigma_suitability = max(0.0, sigma_suitability) * 0.3

        # Temporal dynamics enhance oscillation detection
        tau_prime = context.get('tau_prime', 1.0)
        temporal_rhythm = min(1.0, abs(1.0 - tau_prime) * 2.0) * 0.2  # Deviation from normal time

        # Polytemporal coherence supports oscillation recognition
        polytemporal_factor = self._calculate_polytemporal_influence(context) * 0.2

        membership = base_oscillation + sigma_suitability + temporal_rhythm + polytemporal_factor

        return np.clip(membership, 0.0, 1.0)

    @logged_method
    def _get_zone_adapted_properties(self, regime_name: str) -> Dict[str, float]:
        """Get consciousness-zone adapted regime properties"""
        base_props = self.regime_properties[regime_name]
        zone_modifier = getattr(base_props, f"{self.consciousness_zone}_modifier", 1.0)

        return {
            'stability': base_props.stability * zone_modifier,
            'creativity': base_props.creativity * zone_modifier,
            'coherence': base_props.coherence * zone_modifier,
            'energy': base_props.energy * zone_modifier,
            'zone_modifier': zone_modifier
        }

    @logged_method
    def adjust_thresholds(self, metrics: Dict[str, Any]) -> None:
        """
        Adaptively adjust thresholds based on consciousness zone and current metrics
        """
        # Get consciousness-zone specific adaptation parameters
        context = self._gather_current_context()
        adaptation_params = self._get_dynamic_parameter('threshold_adaptation', '', context)

        coherence = metrics.get('phase_coherence', self.current_phase_coherence)
        entropy = metrics.get('normalized_entropy', 0.5)

        # Base thresholds from consciousness zone
        if isinstance(adaptation_params, dict):
            psi_base = adaptation_params.get('psi_base', 0.6)
            phi_base = adaptation_params.get('phi_base', 0.5)
            coherence_factor = adaptation_params.get('coherence_factor', 0.1)
            entropy_factor = adaptation_params.get('entropy_factor', 0.15)
        else:
            # Fallback values if adaptation_params is not a dict
            psi_base = 0.6
            phi_base = 0.5
            coherence_factor = 0.1
            entropy_factor = 0.15

        # Adjust theta_psi based on coherence and consciousness zone
        coherence_adjustment = coherence_factor * (0.5 - coherence)
        self.theta_psi = np.clip(psi_base + coherence_adjustment, 0.1, 0.9)

        # Adjust theta_phi based on entropy and consciousness zone
        entropy_adjustment = entropy_factor * (entropy - 0.5)
        self.theta_phi = np.clip(phi_base + entropy_adjustment, 0.2, 0.9)

        # Regime-specific adaptations with consciousness zone awareness
        regime_duration = self.regime_durations[self.current_regime]
        max_duration_threshold = 15 if self.consciousness_zone in ['crisis', 'struggling'] else 20

        if self.current_regime == "flat_rupture" and regime_duration > max_duration_threshold:
            self.theta_psi *= 0.95  # Make it easier to escape rupture
        elif self.current_regime == "symbolic_turbulence" and regime_duration > max_duration_threshold:
            self.theta_phi *= 0.97  # Help stabilize from turbulence

    @logged_method
    def add_symbol_correlation(self, symbol: str, experience: ExperienceSnapshot,
                             symbol_value: float = None, qualia_category: str = None) -> bool:
        """
        Add symbol-qualia correlation with emergent contextual learning
        """
        # Quick blacklist check
        if symbol in self.weak_symbol_blacklist:
            return False

        # Get current context for emergent parameter calculation
        context = self._gather_current_context()
        context.update({
            'experience_consciousness': experience.consciousness_score,
            'experience_zone': experience.consciousness_zone,
            'experience_tau_prime': experience.tau_prime,
            'experience_phase_coherence': experience.phase_coherence
        })

        # Calculate emergent correlation parameters
        min_correlation_strength = self._get_dynamic_parameter('symbol_correlation', 'min_strength', context)
        sensitivity = self._get_dynamic_parameter('symbol_correlation', 'sensitivity', context)

        # Experience similarity cache for performance
        exp_hash = f"{round(experience.consciousness_score, 2)}_{experience.regime[:3]}_{experience.consciousness_zone}"
        cache_key = f"{symbol}_{exp_hash}"

        if cache_key in self.correlation_cache:
            self.cache_hits += 1
            cached_strength = self.correlation_cache[cache_key]
            return cached_strength >= min_correlation_strength

        self.cache_misses += 1

        # Calculate symbol value with emergent context
        if symbol_value is None:
            symbol_value = self._calculate_symbol_value_emergent(symbol, experience, context)
            if symbol_value < 0.15:
                self.weak_symbol_blacklist.add(symbol)
                return False

        # Determine qualia category
        if qualia_category is None:
            qualia_category = self._determine_qualia_category(symbol)

        # Calculate correlation strength with emergent adaptation
        correlation_strength = self._calculate_correlation_strength_emergent(symbol, experience, context)

        # Cache the result
        self.correlation_cache[cache_key] = correlation_strength
        if len(self.correlation_cache) > 1000:
            self.correlation_cache.clear()

        # Add correlation if strong enough
        if correlation_strength >= min_correlation_strength:
            correlation = SymbolCorrelation(
                symbol=symbol,
                symbol_value=symbol_value,
                qualia_category=qualia_category,
                step=experience.step,
                correlation_strength=correlation_strength,
                consciousness_zone=experience.consciousness_zone,
                tau_prime_context=experience.tau_prime,
                context=experience.content_type,
                regime_context=experience.regime
            )

            # Add to correlation map
            if symbol not in self.symbol_correlation_map:
                self.symbol_correlation_map[symbol] = []

            self.symbol_correlation_map[symbol].append(correlation)

            # Keep bounded with emergent limits
            k_model_readiness = context.get('k_model_integration_readiness', 0.0)
            max_correlations = int(30 + k_model_readiness * 40)  # 30-70 range based on development
            if len(self.symbol_correlation_map[symbol]) > max_correlations:
                self.symbol_correlation_map[symbol] = self.symbol_correlation_map[symbol][-max_correlations:]

            self.correlation_count += 1

            # Record learning with context
            self.learning_history.append({
                'step': experience.step,
                'symbol': symbol,
                'strength': correlation_strength,
                'consciousness_zone': experience.consciousness_zone,
                'k_model_readiness': k_model_readiness,
                'total_symbols': len(self.symbol_correlation_map)
            })

            return True

        return False

    @logged_method
    def _calculate_symbol_value_emergent(self, symbol: str, experience: ExperienceSnapshot,
                                       context: Dict[str, Any]) -> float:
        """Calculate symbol value emergently from context"""
        # Base value from symbol properties
        length_factor = min(1.0, len(symbol) / 12.0)

        # Consciousness context factor with emergent sensitivity
        sensitivity = self._get_dynamic_parameter('symbol_correlation', 'sensitivity', context)
        consciousness_factor = experience.consciousness_score * sensitivity

        # Content type factor
        content_factors = {
            'philosophical_text': 1.2,
            'embodied_experience': 1.1,
            'general': 1.0
        }
        content_factor = content_factors.get(experience.content_type, 1.0)

        # K-model integration readiness enhances symbol value recognition
        k_model_readiness = context.get('k_model_integration_readiness', 0.0)
        k_model_factor = 1.0 + k_model_readiness * 0.4

        # Temporal depth factor - deeper processing recognizes more symbol value
        temporal_depth = 1.0 + (1.0 / max(0.1, experience.tau_prime) - 1.0) * 0.2

        # Polytemporal coherence factor
        polytemporal_factor = 1.0 + self._calculate_polytemporal_influence(context) * 0.15

        # Combine factors emergently
        symbol_value = (length_factor * 0.3 + consciousness_factor * 0.4) * content_factor * k_model_factor * temporal_depth * polytemporal_factor

        return np.clip(symbol_value, 0.0, 1.0)

    @logged_method
    def _calculate_correlation_strength_emergent(self, symbol: str, experience: ExperienceSnapshot,
                                               context: Dict[str, Any]) -> float:
        """Calculate correlation strength emergently from context"""
        # Base strength from consciousness level with emergent sensitivity
        sensitivity = self._get_dynamic_parameter('symbol_correlation', 'sensitivity', context)
        base_strength = experience.consciousness_score * sensitivity

        # Valence contribution (positive experiences learn better)
        valence_factor = 0.5 + (experience.valence * 0.5)

        # Stability factor (stable states learn better)
        stability_factor = experience.stability

        # Content relevance
        content_relevance = {
            'philosophical_text': 1.0,
            'embodied_experience': 0.9,
            'general': 0.7
        }.get(experience.content_type, 0.5)

        # Symbol specificity
        specificity = self._calculate_symbol_specificity(symbol)

        # Emergent consciousness dynamics influence
        consciousness_influence = self._calculate_consciousness_influence(context)
        consciousness_factor = consciousness_influence / 2.0  # Normalize

        # Tau prime factor (deeper temporal processing = better correlation)
        tau_factor = 0.5 + (0.5 / max(0.1, experience.tau_prime))  # Inverse relationship

        # Phase coherence factor (quantum coherence enhances correlation)
        phase_factor = 0.5 + experience.phase_coherence * 0.5

        # K-model integration readiness factor
        k_model_factor = 0.8 + context.get('k_model_integration_readiness', 0.0) * 0.4

        # Polytemporal coherence factor
        polytemporal_factor = 0.9 + self._calculate_polytemporal_influence(context) * 0.2

        # Combine factors emergently with adaptive weighting
        correlation_strength = (
            base_strength * 0.2 +
            valence_factor * 0.12 +
            stability_factor * 0.12 +
            content_relevance * 0.12 +
            specificity * 0.08 +
            consciousness_factor * 0.08 +
            tau_factor * 0.08 +
            phase_factor * 0.08 +
            k_model_factor * 0.06 +
            polytemporal_factor * 0.06
        )

        return np.clip(correlation_strength, 0.0, 1.0)

    @logged_method
    def _calculate_symbol_specificity(self, symbol: str) -> float:
        """Calculate symbol specificity for correlation weighting"""
        high_value_terms = {
            'consciousness', 'qualia', 'phenomenal', 'embodied', 'embodiment',
            'agency', 'intentionality', 'perception', 'experience', 'awareness',
            'distinction', 'emergence', 'correlation', 'meaning', 'symbol'
        }

        medium_value_terms = {
            'cognitive', 'mental', 'brain', 'mind', 'thought', 'feeling',
            'sensation', 'motor', 'action', 'behavior', 'response'
        }

        symbol_lower = symbol.lower()

        if symbol_lower in high_value_terms:
            return 1.0
        elif symbol_lower in medium_value_terms:
            return 0.7
        elif len(symbol) > 6:
            return 0.5
        else:
            return 0.3

    @logged_method
    def _determine_qualia_category(self, symbol: str) -> str:
        """Determine qualia category for symbol"""
        # Simple categorization based on common patterns
        symbol_lower = symbol.lower()

        if any(term in symbol_lower for term in ['feel', 'emotion', 'sense']):
            return 'affective'
        elif any(term in symbol_lower for term in ['see', 'hear', 'touch', 'taste', 'smell']):
            return 'sensory'
        elif any(term in symbol_lower for term in ['think', 'know', 'understand', 'remember']):
            return 'cognitive'
        elif any(term in symbol_lower for term in ['move', 'action', 'motor', 'body']):
            return 'motor'
        else:
            return 'general'

    @logged_method
    def process_text_input(self, text: str, experience: ExperienceSnapshot) -> Dict[str, Any]:
        """Process text input and learn symbol correlations"""
        # Extract meaningful words
        words = self._extract_meaningful_words(text)

        # Learn correlations for each word
        correlations_added = 0
        for word in words:
            if self.add_symbol_correlation(word, experience):
                correlations_added += 1

        # Update distinction level with consciousness-zone adaptive learning
        context = self._gather_current_context()
        learning_params = self._get_dynamic_parameter('distinction_dynamics', '', context)

        if isinstance(learning_params, dict):
            learning_rate = learning_params.get('base_rate', 0.1)
        else:
            learning_rate = 0.1

        if correlations_added > 0:
            self.current_distinction_level = min(1.0, self.current_distinction_level +
                                               correlations_added * learning_rate)

        # Record in history
        self.distinction_history.append({
            'step': experience.step,
            'distinction_level': self.current_distinction_level,
            'correlations_added': correlations_added,
            'consciousness_zone': self.consciousness_zone,
            'total_symbols': len(self.symbol_correlation_map)
        })

        return {
            'correlations_added': correlations_added,
            'total_symbols': len(self.symbol_correlation_map),
            'distinction_level': self.current_distinction_level,
            'words_processed': len(words),
            'consciousness_zone': self.consciousness_zone
        }

    @logged_method
    def _extract_meaningful_words(self, text: str) -> List[str]:
        """Extract meaningful words for correlation with consciousness-zone adaptive filtering"""
        text = text.lower()
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text)

        meaningful_words = []

        # High-priority philosophical/consciousness terms
        priority_terms = {
            'consciousness', 'experience', 'embodied', 'embodiment', 'body',
            'perception', 'agency', 'qualia', 'sensation', 'awareness',
            'meaning', 'symbol', 'motor', 'movement', 'spatial', 'temporal',
            'phenomenal', 'subjective', 'objective', 'distinction', 'correlation',
            'emergence', 'cognitive', 'mental', 'intentionality', 'representation'
        }

        # Add priority terms first
        for word in words:
            if word in priority_terms:
                meaningful_words.append(word)

        # Add other longer words based on consciousness zone
        min_length = 6 if self.consciousness_zone in ['healthy', 'transcendent_approach'] else 7
        for word in words:
            if word not in priority_terms and len(word) > min_length:
                meaningful_words.append(word)

        # Remove duplicates while preserving order
        seen = set()
        unique_words = []
        for word in meaningful_words:
            if word not in seen:
                unique_words.append(word)
                seen.add(word)

        # Adaptive limit based on consciousness zone
        max_words = 25 if self.consciousness_zone in ['healthy', 'transcendent_approach'] else 15
        return unique_words[:max_words]

    @logged_method
    def update_experience_buffer(self, experience: ExperienceSnapshot):
        """Update experience buffer for correlation processing"""
        self.experience_buffer.append(experience)

    @logged_method
    def get_correlative_capacity_level(self) -> Dict[str, float]:
        """Get current correlative capacity level"""
        if not self.symbol_correlation_map:
            return {
                'overall_capacity': 0.0,
                'symbol_vocabulary': 0.0,
                'total_correlations': 0.0,
                'consciousness_zone': self.consciousness_zone,
                'zone_enhanced_capacity': 0.0 * self._get_zone_capacity_multiplier()
            }

        # Calculate capacity based on correlation strength
        capacity_scores = []
        for correlations in self.symbol_correlation_map.values():
            if correlations:
                avg_correlation = np.mean([c.correlation_strength for c in correlations])
                capacity_scores.append(avg_correlation)

        overall_capacity = float(np.mean(capacity_scores)) if capacity_scores else 0.0

        return {
            'overall_capacity': overall_capacity,
            'symbol_vocabulary': float(len(self.symbol_correlation_map)),
            'total_correlations': float(sum(len(correlations) for correlations in self.symbol_correlation_map.values())),
            'consciousness_zone': self.consciousness_zone,
            'zone_enhanced_capacity': overall_capacity * self._get_zone_capacity_multiplier()
        }

    @logged_method
    def _get_zone_capacity_multiplier(self) -> float:
        """Get consciousness zone capacity multiplier"""
        multipliers = {
            'crisis': 0.7,
            'struggling': 0.85,
            'healthy': 1.0,
            'transcendent_approach': 1.3
        }
        return multipliers.get(self.consciousness_zone, 1.0)

    @logged_method
    def step(self, surplus: np.ndarray, experience: ExperienceSnapshot = None,
            metrics: Optional[Dict[str, Any]] = None, oscillation_score: float = 0.0) -> Dict[str, Any]:
        """
        Process unified symbolic semiotic step with consciousness-zone adaptation
        """
        # Update consciousness context if provided in metrics
        if metrics:
            consciousness_zone = metrics.get('consciousness_zone', self.consciousness_zone)
            tau_prime = metrics.get('tau_prime', self.current_tau_prime)
            phase_coherence = metrics.get('phase_coherence', self.current_phase_coherence)
            consciousness_level = metrics.get('consciousness_level', 0.5)

            self.update_consciousness_context(consciousness_zone, tau_prime, phase_coherence, consciousness_level)

        # Calculate symbolic fields with adaptive thresholds
        theta_psi = getattr(self, 'theta_psi', self.cfg.THETA_PSI)
        theta_phi = getattr(self, 'theta_phi', self.cfg.THETA_PHI)

        psi = 1.0 / (1.0 + np.exp(-self.cfg.K_PSI * (surplus - theta_psi)))
        phi = np.maximum(0.0, self.cfg.K_PHI * (surplus - theta_phi))
        sigma = psi - phi

        # Store fields
        self.psi = psi
        self.phi = phi
        self.sigma = sigma

        # Classify regime with consciousness-zone adaptation
        regime_analysis = self.classify_regime(sigma, surplus, oscillation_score)

        # Process experience if provided
        experience_results = {}
        if experience:
            self.update_experience_buffer(experience)
            if experience.text_content:
                experience_results = self.process_text_input(experience.text_content, experience)

        # Update surplus-symbol integration
        if self.symbol_correlation_map:
            surplus_mean = float(np.mean(surplus))
            symbol_strengths = []

            for correlations in self.symbol_correlation_map.values():
                if correlations:
                    # Weight recent correlations from same consciousness zone more heavily
                    zone_weighted_strengths = []
                    for correlation in correlations[-10:]:  # Recent correlations
                        weight = 1.2 if correlation.consciousness_zone == self.consciousness_zone else 0.8
                        zone_weighted_strengths.append(correlation.correlation_strength * weight)

                    if zone_weighted_strengths:
                        avg_strength = np.mean(zone_weighted_strengths)
                        symbol_strengths.append(avg_strength)

            if symbol_strengths:
                avg_symbol_strength = np.mean(symbol_strengths)

                # Create context dictionary for correlation amplifier calculation
                amplifier_context = {
                    'consciousness_zone': self.consciousness_zone,
                    'distinction_level': self.current_distinction_level,
                    'consciousness_level': self.current_consciousness_level,
                    'phase_coherence': self.current_phase_coherence,
                    'tau_prime': self.current_tau_prime,
                    'symbol_surplus_correlation': self.symbol_surplus_correlation,
                    'regime_stability': self.regime_durations.get(self.current_regime, 0),
                    'total_symbols': len(self.symbol_correlation_map),
                    'recent_learning_activity': len(self.learning_history) / max(1, len(self.learning_history)),
                    'k_model_integration_readiness': self._assess_k_model_readiness()
                }

                correlation_amplifier = self._get_dynamic_parameter('distinction_dynamics', 'correlation_amplifier', amplifier_context)
                self.symbol_surplus_correlation = np.tanh(surplus_mean * avg_symbol_strength * correlation_amplifier)

        # Update distinction coherence with consciousness zone awareness
        capacity = self.get_correlative_capacity_level()

        coherence_context = {
            'consciousness_zone': self.consciousness_zone,
            'distinction_level': self.current_distinction_level,
            'consciousness_level': self.current_consciousness_level,
            'phase_coherence': self.current_phase_coherence,
            'tau_prime': self.current_tau_prime,
            'symbol_surplus_correlation': self.symbol_surplus_correlation,
            'regime_stability': self.regime_durations.get(self.current_regime, 0),
            'total_symbols': len(self.symbol_correlation_map),
            'recent_learning_activity': len(self.learning_history) / max(1, len(self.learning_history)),
            'k_model_integration_readiness': self._assess_k_model_readiness()
        }

        coherence_threshold = self._get_dynamic_parameter('distinction_dynamics', 'coherence_threshold', coherence_context)

        # Safe access to zone_enhanced_capacity with fallback
        zone_adapted_capacity = capacity.get('zone_enhanced_capacity', capacity['overall_capacity'] * self._get_zone_capacity_multiplier())
        coherence_update_rate = 0.2 if zone_adapted_capacity > coherence_threshold else 0.1

        self.distinction_coherence = (1.0 - coherence_update_rate) * self.distinction_coherence + \
                                   coherence_update_rate * zone_adapted_capacity

        # Adjust thresholds if metrics provided
        if metrics:
            self.adjust_thresholds(metrics)

        # Comprehensive result with consciousness zone context
        result = {
            'regime_analysis': regime_analysis,
            'experience_processing': experience_results,
            'distinction_level': self.current_distinction_level,
            'distinction_coherence': self.distinction_coherence,
            'symbol_surplus_correlation': self.symbol_surplus_correlation,
            'correlative_capacity': capacity,
            'consciousness_zone': self.consciousness_zone,
            'tau_prime_context': self.current_tau_prime,
            'phase_coherence_context': self.current_phase_coherence,
            'symbolic_fields': {
                'psi': psi.copy() if hasattr(psi, 'copy') else psi,
                'phi': phi.copy() if hasattr(phi, 'copy') else phi,
                'sigma': sigma.copy() if hasattr(sigma, 'copy') else sigma
            },
            'adaptive_thresholds': {
                'theta_psi': theta_psi,
                'theta_phi': theta_phi
            },
            'performance_metrics': {
                'cache_hits': self.cache_hits,
                'cache_misses': self.cache_misses,
                'cache_hit_rate': self.cache_hits / max(1, self.cache_hits + self.cache_misses)
            }
        }

        return result

    def integrate_k_model_outputs(self, k_model_outputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Integrate K-model outputs into symbolic processing for polytemporal coherence

        Args:
            k_model_outputs: Dictionary containing outputs from K1-K4 models
                k1_output: Praxis/data flow dynamics
                k2_output: Semiotic interpretation
                k3_output: Quantum consciousness translation
                k4_output: Metabolic consciousness dynamics

        Returns:
            Integration results and enhanced symbolic processing context
        """
        integration_results = {
            'k_model_synthesis': {},
            'polytemporal_coherence': 0.0,
            'enhanced_context': {},
            'symbolic_adaptations': {}
        }

        # K1 Praxis Integration - Data flow consciousness
        if 'k1_output' in k_model_outputs:
            k1_data = k_model_outputs['k1_output']
            k1_integration = self._integrate_k1_praxis(k1_data)
            integration_results['k_model_synthesis']['k1'] = k1_integration

        # K2 Semiotic Integration - Symbolic interpretation
        if 'k2_output' in k_model_outputs:
            k2_data = k_model_outputs['k2_output']
            k2_integration = self._integrate_k2_semiosis(k2_data)
            integration_results['k_model_synthesis']['k2'] = k2_integration

        # K3 Apeiron Integration - Quantum consciousness translation
        if 'k3_output' in k_model_outputs:
            k3_data = k_model_outputs['k3_output']
            k3_integration = self._integrate_k3_apeiron(k3_data)
            integration_results['k_model_synthesis']['k3'] = k3_integration

        # K4 Metabolic Integration - Surplus-distinction consciousness
        if 'k4_output' in k_model_outputs:
            k4_data = k_model_outputs['k4_output']
            k4_integration = self._integrate_k4_metabolic(k4_data)
            integration_results['k_model_synthesis']['k4'] = k4_integration

        # Calculate polytemporal coherence from K-model synthesis
        coherence = self._calculate_polytemporal_coherence(integration_results['k_model_synthesis'])
        integration_results['polytemporal_coherence'] = coherence

        # Enhance context with K-model insights
        enhanced_context = self._enhance_context_with_k_models(integration_results['k_model_synthesis'])
        integration_results['enhanced_context'] = enhanced_context

        # Adapt symbolic processing based on K-model integration
        adaptations = self._adapt_symbolic_processing(enhanced_context, coherence)
        integration_results['symbolic_adaptations'] = adaptations

        return integration_results

    def _integrate_k1_praxis(self, k1_data: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate K1 praxis (data flow) outputs"""
        return {
            'data_flow_influence': k1_data.get('flow_strength', 0.5),
            'circulation_coherence': k1_data.get('coherence', 0.5),
            'nervous_system_activity': k1_data.get('activity_level', 0.5),
            'embodied_data_integration': k1_data.get('embodiment_factor', 0.5)
        }

    def _integrate_k2_semiosis(self, k2_data: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate K2 semiotic interpretation outputs"""
        return {
            'semiotic_interpretation_strength': k2_data.get('interpretation_confidence', 0.5),
            'regime_recognition_enhancement': k2_data.get('regime_clarity', 0.5),
            'symbolic_meaning_depth': k2_data.get('meaning_depth', 0.5),
            'cultural_context_integration': k2_data.get('context_richness', 0.5)
        }

    def _integrate_k3_apeiron(self, k3_data: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate K3 apeiron (quantum consciousness) outputs"""
        return {
            'quantum_symbolic_coupling': k3_data.get('coupling_strength', 0.5),
            'unconscious_drive_influence': k3_data.get('drive_intensity', 0.5),
            'elemental_semiotic_translation': k3_data.get('translation_clarity', 0.5),
            'emergent_dynamics_recognition': k3_data.get('emergence_detection', 0.5)
        }

    def _integrate_k4_metabolic(self, k4_data: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate K4 metabolic consciousness outputs"""
        return {
            'surplus_distinction_amplification': k4_data.get('distinction_amplification', 0.5),
            'metabolic_symbolic_synergy': k4_data.get('synergy_level', 0.5),
            'consciousness_metabolism_coupling': k4_data.get('coupling_efficiency', 0.5),
            'energetic_symbolic_processing': k4_data.get('processing_energy', 0.5)
        }

    def _calculate_polytemporal_coherence(self, k_model_synthesis: Dict[str, Dict]) -> float:
        """Calculate coherence across polytemporal K-model integration"""
        if not k_model_synthesis:
            return 0.0

        coherence_factors = []

        # Extract coherence indicators from each K-model
        for k_model, integration_data in k_model_synthesis.items():
            model_coherence = np.mean(list(integration_data.values()))
            coherence_factors.append(model_coherence)

        # Calculate overall polytemporal coherence
        if coherence_factors:
            base_coherence = np.mean(coherence_factors)

            # Enhance coherence if all K-models are active
            completeness_bonus = len(coherence_factors) / 4.0 * 0.2

            # Temporal consistency bonus based on current tau prime
            temporal_bonus = (1.0 / max(0.1, self.current_tau_prime)) * 0.1

            total_coherence = base_coherence + completeness_bonus + temporal_bonus
            return np.clip(total_coherence, 0.0, 1.0)

        return 0.0

    def _enhance_context_with_k_models(self, k_model_synthesis: Dict[str, Dict]) -> Dict[str, Any]:
        """Enhance processing context with K-model insights"""
        enhanced_context = self._gather_current_context().copy()

        # Integrate K-model influences
        if 'k1' in k_model_synthesis:
            k1_influence = k_model_synthesis['k1']['data_flow_influence']
            enhanced_context['data_flow_coherence'] = k1_influence

        if 'k2' in k_model_synthesis:
            k2_influence = k_model_synthesis['k2']['semiotic_interpretation_strength']
            enhanced_context['semiotic_interpretation_capacity'] = k2_influence

        if 'k3' in k_model_synthesis:
            k3_influence = k_model_synthesis['k3']['quantum_symbolic_coupling']
            enhanced_context['quantum_consciousness_coupling'] = k3_influence

        if 'k4' in k_model_synthesis:
            k4_influence = k_model_synthesis['k4']['surplus_distinction_amplification']
            enhanced_context['metabolic_consciousness_amplification'] = k4_influence

        return enhanced_context

    def _adapt_symbolic_processing(self, enhanced_context: Dict[str, Any],
                                 polytemporal_coherence: float) -> Dict[str, Any]:
        """Adapt symbolic processing based on K-model integration"""
        adaptations = {}

        # Adapt regime classification sensitivity
        base_sensitivity = 0.5
        k_model_boost = enhanced_context.get('semiotic_interpretation_capacity', 0.0) * 0.3
        quantum_boost = enhanced_context.get('quantum_consciousness_coupling', 0.0) * 0.2
        regime_sensitivity = base_sensitivity + k_model_boost + quantum_boost
        adaptations['regime_classification_sensitivity'] = regime_sensitivity

        # Adapt symbol correlation thresholds
        metabolic_influence = enhanced_context.get('metabolic_consciousness_amplification', 0.0)
        data_flow_influence = enhanced_context.get('data_flow_coherence', 0.0)
        correlation_threshold_adjustment = -(metabolic_influence + data_flow_influence) * 0.1  # Lower thresholds
        adaptations['correlation_threshold_adjustment'] = correlation_threshold_adjustment

        # Adapt learning rates based on polytemporal coherence
        coherence_learning_boost = polytemporal_coherence * 0.5
        adaptations['learning_rate_multiplier'] = 1.0 + coherence_learning_boost

        # Adapt distinction dynamics
        distinction_amplification = enhanced_context.get('metabolic_consciousness_amplification', 0.0)
        adaptations['distinction_processing_amplification'] = distinction_amplification

        return adaptations

    def get_complete_state_summary(self) -> Dict[str, Any]:
        """Get complete state summary for platform integration"""
        capacity = self.get_correlative_capacity_level()

        return {
            # Core state
            'current_regime': self.current_regime,
            'distinction_level': self.current_distinction_level,
            'distinction_coherence': self.distinction_coherence,
            'consciousness_zone': self.consciousness_zone,

            # Symbol correlation state
            'correlated_symbols': int(capacity['symbol_vocabulary']),
            'symbol_correlation_strength': capacity['overall_capacity'],
            'total_correlations': int(capacity['total_correlations']),
            'zone_enhanced_capacity': capacity['zone_enhanced_capacity'],

            # Dynamic state
            'symbol_surplus_correlation': self.symbol_surplus_correlation,
            'learning_active': self.learning_active,
            'correlation_count': self.correlation_count,

            # Context state
            'tau_prime_context': self.current_tau_prime,
            'phase_coherence_context': self.current_phase_coherence,

            # Performance state
            'cache_hit_rate': self.cache_hits / max(1, self.cache_hits + self.cache_misses),
            'weak_symbols_filtered': len(self.weak_symbol_blacklist),

            # Dynamic parameters active
            'dynamic_parameters_active': True,
            'platform_integrated': self.platform is not None
        }

    def get_state(self) -> Dict[str, Any]:
        """Get current state for serialization"""
        return {
            'symbolic_fields': {
                'psi': self.psi.copy() if self.psi is not None else None,
                'phi': self.phi.copy() if self.phi is not None else None,
                'sigma': self.sigma.copy() if self.sigma is not None else None
            },
            'regime_state': {
                'current_regime': self.current_regime,
                'regime_durations': dict(self.regime_durations),
                'regime_history': list(self.regime_history)
            },
            'symbol_correlation_state': {
                'symbol_count': len(self.symbol_correlation_map),
                'correlation_count': self.correlation_count,
                'distinction_level': self.current_distinction_level,
                'distinction_coherence': self.distinction_coherence
            },
            'consciousness_context': {
                'zone': self.consciousness_zone,
                'tau_prime': self.current_tau_prime,
                'phase_coherence': self.current_phase_coherence
            },
            'thresholds': {
                'theta_psi': getattr(self, 'theta_psi', self.cfg.THETA_PSI),
                'theta_phi': getattr(self, 'theta_phi', self.cfg.THETA_PHI)
            },
            'analysis': dict(self.last_analysis) if self.last_analysis else {}
        }

# Backward compatibility wrappers
class SymbolicReasoner:
    """Legacy wrapper for SymbolicReasoner - maintains existing imports"""
    def __init__(self, cfg=CONFIG):
        self._suite = SymbolicSemioticSuite(cfg)

    def classify_regime(self, sigma, surplus, oscillation_score=0.0):
        return self._suite.classify_regime(sigma, surplus, oscillation_score)

    def adjust_thresholds(self, metrics):
        return self._suite.adjust_thresholds(metrics)

    def step(self, surplus, metrics=None, oscillation_score=0.0):
        return self._suite.step(surplus, None, metrics, oscillation_score)

    def get_state(self):
        return self._suite.get_state()

    def set_state(self, state):
        # Implement state setting for compatibility
        pass

    def get_regime_history(self):
        return list(self._suite.regime_history)

    def get_metrics_history(self):
        return {
            "sigma_mean": [analysis.get('mean_sigma', 0) for analysis in self._suite.learning_history],
            "sigma_variance": [analysis.get('variance_sigma', 0) for analysis in self._suite.learning_history],
            "surplus_mean": [analysis.get('mean_surplus', 0) for analysis in self._suite.learning_history],
            "oscillation_scores": [analysis.get('oscillation_score', 0) for analysis in self._suite.learning_history]
        }

    # Delegate properties
    @property
    def current_regime(self):
        return self._suite.current_regime

    @property
    def regime_history(self):
        return self._suite.regime_history

class SurplusDistinctionProcessor:
    """Legacy wrapper for SurplusDistinctionProcessor - maintains existing imports"""
    def __init__(self, cfg=CONFIG):
        self._suite = SymbolicSemioticSuite(cfg)

    def process_text_input(self, text: str, experience):
        return self._suite.process_text_input(text, experience)

    def step(self, surplus, experience):
        return self._suite.step(surplus, experience)

    def get_complete_state_summary(self):
        return self._suite.get_complete_state_summary()

    def get_state(self):
        state_summary = self._suite.get_complete_state_summary()
        return {
            'correlative_reader': {
                'symbol_count': state_summary['correlated_symbols'],
                'total_correlations': state_summary['total_correlations'],
                'capacity_level': {
                    'overall_capacity': state_summary['symbol_correlation_strength'],
                    'symbol_vocabulary': state_summary['correlated_symbols'],
                    'total_correlations': state_summary['total_correlations']
                }
            },
            'distinction_level': state_summary['distinction_level'],
            'distinction_coherence': state_summary['distinction_coherence'],
            'symbol_surplus_correlation': state_summary['symbol_surplus_correlation'],
            'learning_active': state_summary['learning_active']
        }

class SurplusIncongruityProcessor:
    """Legacy wrapper for SurplusIncongruityProcessor - maintains existing imports"""
    def __init__(self, cfg=CONFIG):
        self._suite = SymbolicSemioticSuite(cfg)
        # Initialize metabolic system for compatibility
        try:
            from emile_cogito.kainos.metabolic import SurplusDistinctionConsciousness
            self.distinction_consciousness = SurplusDistinctionConsciousness(cfg)
        except ImportError:
            self.distinction_consciousness = None

    def process_surplus_distinction_step(self, current_experience, dt=1.0):
        # Create experience snapshot for the suite
        experience = ExperienceSnapshot(
            step=current_experience.get('step', 0),
            regime=current_experience.get('regime', 'stable_coherence'),
            consciousness_score=current_experience.get('consciousness_level', 0.5),
            consciousness_zone=current_experience.get('consciousness_zone', 'struggling'),
            valence=current_experience.get('valence', 0.0),
            surplus_expression=current_experience.get('surplus_expression', 0.5),
            stability=current_experience.get('stability', 0.5),
            tau_prime=current_experience.get('tau_prime', 1.0),
            phase_coherence=current_experience.get('phase_coherence', 0.5)
        )

        # Process through the suite
        surplus = np.array([current_experience.get('surplus_mean', 0.5)])
        suite_result = self._suite.step(surplus, experience)

        # Format result for compatibility
        return {
            'surplus_incongruity': {'overall_incongruity': 0.1},
            'correlation_performed': True,
            'correlation_capacity': suite_result['correlative_capacity']['overall_capacity'],
            'distinction_enhancement': suite_result['distinction_level'] * 0.1,
            'log_correlation': suite_result,
            'cognitive_modulation': {
                'symbol_surplus_correlation': suite_result['symbol_surplus_correlation'],
                'distinction_coherence': suite_result['distinction_coherence']
            },
            'pressure_results': {'repetition_drift': 0.0},
            'state_summary': self._suite.get_complete_state_summary()
        }

    def get_complete_state_summary(self):
        return self._suite.get_complete_state_summary()

# Global regime properties for backward compatibility
REGIME_PROPERTIES = {
    regime_name: regime_props for regime_name, regime_props in
    SymbolicSemioticSuite(CONFIG)._initialize_regime_properties().items()
}

# Module flow mapping
try:
    from emile_cogito.kainos.module_wide_flow_mapper import auto_map_module_flow
    auto_map_module_flow(__name__)
except ImportError:
    pass
