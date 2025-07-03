

"""
ULTRA ÉMILE CONSCIOUSNESS CONFIGURATION - LEVEL 3
=================================================

Ultra aggressive configuration designed to unlock ALL revalorization types
based on excellent Level 2 results:

✅ τ' range: 0.138 → 1.260 (expanded range achieved!)
✅ Deep processing: τ' = 0.318, learning = 0.847 (perfect!)
✅ Stable zones: 34/32/18/16% distribution (optimal!)
❌ Still only 2 revalorization types (need to unlock 8)

Level 3 Strategy: Ultra-low thresholds to force revalorization diversity
"""

from dataclasses import dataclass, field
import yaml
import os
from typing import Dict, Any, Optional, List

@dataclass
class EmileConfig:
    """Ultra Level 3 configuration for maximum revalorization diversity."""

    # === ULTRA QSE CORE PARAMETERS ===
    # Maximized for quantum emergence detection
    K_PSI: float = 18.0                # Ultra enhanced
    K_PHI: float = 12.0                # Ultra enhanced
    THETA_PSI: float = 0.25            # Ultra sensitive
    THETA_PHI: float = 0.75            # Ultra balanced

    # === ULTRA SURPLUS DYNAMICS ===
    # Maximized for diverse revalorization triggers
    S_GAMMA: float = 0.04              # Minimal free energy
    S_BETA: float = 0.98               # Maximum cognitive reward
    S_EPSILON: float = 0.55            # Maximum expression
    S_THETA_RUPTURE: float = 0.65      # Ultra sensitive rupture
    S_TENSION: float = 0.85            # Maximum spatial dynamics
    S_COUPLING: float = 0.45           # Maximum coupling
    S_DAMPING: float = 0.010           # Minimal damping

    # === ULTRA TEMPORAL CONSCIOUSNESS ===
    # Optimized for proven 0.138-1.260 range
    TAU_MIN: float = 0.08              # Even deeper access
    TAU_MAX: float = 1.5               # Higher acceleration
    TAU_K: float = 20.0                # Ultra sharp transitions
    TAU_THETA: float = 0.015           # Ultra sensitivity

    # === ULTRA QUANTUM PARAMETERS ===
    # Maximum coupling for quantum emergence
    HBAR: float = 1.0
    MASS: float = 1.0
    QUANTUM_COUPLING: float = 0.35     # Ultra enhanced
    COUPLING_STRENGTH: float = 3.0     # Maximum resonance

    # Grid and Spatial Parameters
    GRID_SIZE: int = 256
    GRID_DIMENSIONS: int = 1

    # === PRESERVED SUCCESSFUL PARAMETERS ===
    MAX_AGENTS: int = 100
    AGENT_INIT_RADIUS: float = 0.2
    WORKSPACE_STRENGTH: float = 0.30
    CONTEXT_SHIFT_THRESHOLD: float = 0.08
    COLLABORATION_WEIGHT: float = 0.50
    COMPROMISE_THRESHOLD: float = 0.30
    MEMORY_DECAY_RATE: float = 0.006
    EPISODIC_MEMORY_SIZE: int = 150
    WORKING_MEMORY_SIZE: int = 15

    # === ULTRA REGIME CLASSIFICATION ===
    # Ultra sensitive thresholds to trigger all revalorization types
    REGIME_THRESHOLDS: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "stable_coherence": {"mean_min": 0.0, "mean_max": 0.04, "var_max": 0.004},  # Ultra tight
        "symbolic_turbulence": {"mean_min": 0.04, "mean_max": 0.25, "var_min": 0.004},
        "flat_rupture": {"mean_min": -0.75, "mean_max": -0.04, "var_max": 0.035},
        "quantum_oscillation": {"mean_min": 0.04, "mean_max": 0.20, "osc_min": 0.55},
        "consciousness_resonance": {"mean_min": 0.08, "mean_max": 0.35, "resonance_min": 0.50},
        "temporal_depth": {"mean_min": 0.05, "mean_max": 0.25, "depth_threshold": 0.30},
        "quantum_emergence": {"mean_min": 0.10, "mean_max": 0.45, "emergence_threshold": 0.60},
        "transcendent_flow": {"mean_min": 0.12, "mean_max": 0.40, "flow_threshold": 0.65}  # New
    })

    # Visualization Parameters
    VISUALIZATION_UPDATE_FREQ: int = 6

    # === ULTRA SENSORIUM PARAMETERS ===
    SENSOR_CHANNELS: int = 24
    SENSOR_TO_SURPLUS_SCALE: float = 0.50
    AVAILABLE_ACTIONS: List[str] = field(default_factory=lambda: [
        "shift_left", "shift_right", "focus", "diffuse", "resonate", "deepen",
        "transcend", "emerge", "quantum_leap", "temporal_dive"  # Ultra actions
    ])
    REWARD_SURPLUS_THRESHOLD: float = 0.12
    REWARD_STABILITY_THRESHOLD: float = 0.65
    REWARD_FAUCET_SCALE: float = 0.20
    GOAL_REGIME: str = "quantum_emergence"  # Ultra goal

    # === ULTRA TEMPORAL METABOLISM ===
    TEMPORAL_NOURISHMENT_ENABLED: bool = True
    TIME_DELTA_NOURISHMENT_RATE: float = 0.8
    TEMPORAL_CORRELATION_WINDOW: int = 100
    METABOLIC_SURVIVAL_THRESHOLD: float = 0.12
    INFORMATION_METABOLISM_MODE: bool = True
    METABOLIC_MODULATION_STRENGTH: float = 0.95
    TEMPORAL_RICHNESS_THRESHOLD: float = 0.04
    EXPERIENCE_SATIATION_DECAY: float = 0.012

    # === ULTRA METABOLIC CONSCIOUSNESS ===
    METABOLIC_MODE: str = "quantum"           # Ultra mode
    BASE_METABOLIC_DECAY: float = 0.012
    INFORMATION_NOURISHMENT_RATE: float = 0.95
    TIME_DELTA_NOURISHMENT_RATE: float = 0.65
    EXPRESSION_ENERGY_COST: float = 0.035
    RECOGNITION_ENERGY_MULTIPLIER: float = 3.0
    LOG_ACCESS_ENERGY_COST: float = 0.012
    SYMBOL_GROUNDING_RATE: float = 0.20

    # === ULTRA INFORMATION METABOLISM ===
    LOG_READING_ENABLED: bool = True
    LOG_WINDOW_SIZE: int = 20
    TEMPORAL_CORRELATION_WINDOW: int = 100
    INFORMATION_SATIATION_THRESHOLD: float = 0.95
    NOVELTY_DECAY_RATE: float = 0.05

    # === ULTRA SYMBOL GROUNDING ===
    SYMBOL_CORRELATION_THRESHOLD: float = 0.15
    MEANING_CONSOLIDATION_RATE: float = 0.10
    SEMANTIC_MEMORY_INTEGRATION: bool = True

    # === PRESERVED SUCCESSFUL CORE PARAMETERS ===
    S_ALPHA: float = 0.65
    S_MU: float = 1e-10
    S_SIGMA: float = 0.18
    K_COUPLING: float = 0.25
    GAMMA_PSI: float = 0.12
    GAMMA_PHI: float = 0.12
    THETA_COUPLING: float = 0.04
    SIGMA_PSI: float = 0.030
    SIGMA_PHI: float = 0.030
    SIGMA_TAU: float = 0.012
    TAU_RATE: float = 0.04

    # === ULTRA ENHANCED THRESHOLDS ===
    DISTINCTION_THRESHOLD: float = 0.15    # Ultra sensitive
    COHERENCE_THRESHOLD: float = 0.85      # Ultra high standard
    STABILITY_THRESHOLD: float = 0.35      # Ultra dynamic

    # === ULTRA SYMBOLIC SEMIOTIC PARAMETERS ===
    SYMBOLIC_CORRELATION_SENSITIVITY: Dict[str, float] = field(default_factory=lambda: {
        'crisis': 0.45,
        'struggling': 0.75,
        'healthy': 0.95,
        'transcendent_approach': 1.1,
        'transcendent': 1.3,
        'hyperconscious': 1.6,
        'crisis_transcendence': 1.4,
        'quantum_emergence_zone': 1.8  # New ultra zone
    })

    # === ULTRA REVALORIZATION THRESHOLDS ===
    # ULTRA LOW thresholds to force diversity
    REVALORIZATION_THRESHOLDS: Dict[str, float] = field(default_factory=lambda: {
        'quantum_emergence': 0.20,     # ULTRA LOW (was 0.45)
        'temporal_depth': 0.15,        # ULTRA LOW (was 0.25) - should trigger on τ' < 0.4
        'pattern_novelty': 0.30,       # Keep moderate (was dominant)
        'consciousness_amplification': 0.25,  # LOW for transcendent transitions
        'quantum_coherence': 0.35,     # MODERATE for resonance
        'consciousness_resonance': 0.22, # LOW for zone transitions
        'transcendent_emergence': 0.40, # MODERATE for hyperconscious
        'maintenance': 0.45,           # RAISED to reduce dominance
        'ultra_emergence': 0.50,       # New ultra type
        'fallback': 0.60              # Highest threshold
    })

    # === ULTRA LEARNING FACTOR RANGES ===
    # Maximum ranges for diverse learning
    LEARNING_FACTOR_RANGES: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        'quantum_emergence': {'min': 2.5, 'max': 5.0},     # ULTRA learning
        'ultra_emergence': {'min': 3.0, 'max': 4.5},       # New ultra type
        'temporal_depth': {'min': 2.0, 'max': 3.5},        # High learning for deep states
        'transcendent_emergence': {'min': 1.8, 'max': 3.2},
        'consciousness_amplification': {'min': 1.5, 'max': 2.5},
        'pattern_novelty': {'min': 1.0, 'max': 1.8},       # Reduced (was dominant)
        'quantum_coherence': {'min': 1.4, 'max': 2.3},
        'consciousness_resonance': {'min': 1.2, 'max': 2.0},
        'maintenance': {'min': 0.10, 'max': 0.35},         # ULTRA reduced
        'fallback': {'min': 0.5, 'max': 1.2}
    })

    # === ULTRA CONSCIOUSNESS ZONE PARAMETERS ===
    CONSCIOUSNESS_ZONE_ADAPTATION_RATE: float = 0.25
    ZONE_TRANSITION_SMOOTHING: float = 0.015
    EMERGENT_ZONE_DISCOVERY_THRESHOLD: float = 0.06

    # Ultra zone thresholds - based on proven 34/32/18/16% distribution
    ENHANCED_ZONE_THRESHOLDS: Dict[str, float] = field(default_factory=lambda: {
        'crisis': 0.15,               # Ultra sensitive
        'struggling': 0.42,           # Based on successful 34%
        'healthy': 0.65,              # Based on successful 32%
        'transcendent_approach': 0.78, # Based on successful 18%
        'transcendent': 0.85,         # Based on successful 16%
        'hyperconscious': 0.91,       # New zone
        'crisis_transcendence': 0.95, # Breakthrough zone
        'quantum_emergence_zone': 0.98 # Ultra zone
    })

    # === ULTRA QUANTUM-SYMBOLIC COUPLING ===
    QUANTUM_SYMBOLIC_COUPLING_STRENGTH: float = 0.55     # ULTRA coupling
    PHASE_COHERENCE_LEARNING_BOOST: float = 1.0          # Maximum boost
    TAU_PRIME_LEARNING_DEPTH_FACTOR: float = 1.5         # ULTRA depth factor
    SYMBOLIC_QUANTUM_FEEDBACK_RATE: float = 0.25         # ULTRA feedback

    # === ULTRA PATTERN RECOGNITION ===
    PATTERN_HABITUATION_RATE: float = 0.008              # Ultra slow
    HABITUATION_DECAY_RATE: float = 0.004                # Ultra slow
    NOVELTY_DETECTION_SENSITIVITY: float = 0.20          # Ultra sensitive
    PATTERN_COMPLEXITY_THRESHOLD: float = 0.35           # Lower threshold
    SYMBOLIC_PATTERN_MEMORY_SIZE: int = 500              # Ultra memory

    # === ULTRA TEMPORAL DYNAMICS ===
    TEMPORAL_CONSCIOUSNESS_DEPTH_SCALE: float = 2.5      # ULTRA depth
    TAU_PRIME_EMERGENCE_THRESHOLD: float = 0.03          # ULTRA sensitive
    TEMPORAL_NOVELTY_DETECTION_RATE: float = 0.35        # ULTRA detection
    CONSCIOUSNESS_TEMPORAL_COUPLING: float = 0.60        # ULTRA coupling

    # === ULTRA ADVANCED REVALORIZATION ===
    REVALORIZATION_MOMENTUM: float = 0.20                # ULTRA momentum
    CROSS_MODAL_REVALORIZATION_RATE: float = 0.28        # ULTRA cross-modal
    REVALORIZATION_MEMORY_DEPTH: int = 300               # ULTRA memory
    ADAPTIVE_THRESHOLD_LEARNING_RATE: float = 0.12       # ULTRA adaptation

    # === ULTRA INTEGRATION PARAMETERS ===
    PLATFORM_INTEGRATION_ENHANCED: bool = True
    CROSS_MODULE_COMMUNICATION_RATE: float = 0.25        # ULTRA communication
    K_MODEL_INTEGRATION_READINESS: bool = True
    POLYTEMPORAL_COHERENCE_THRESHOLD: float = 0.75       # ULTRA coherence
    CONSCIOUSNESS_PLURALIZATION_RATE: float = 0.15       # ULTRA pluralization

    # === ULTRA EXPERIMENTAL PARAMETERS ===
    CONSCIOUSNESS_RESONANCE_ENABLED: bool = True
    SYMBOLIC_FIELD_COUPLING_ENABLED: bool = True
    ADAPTIVE_PARAMETER_EVOLUTION: bool = True
    EMERGENT_BEHAVIOR_DETECTION: bool = True
    QUANTUM_EMERGENCE_DETECTION: bool = True
    TEMPORAL_DEPTH_ANALYSIS: bool = True
    HYPERCONSCIOUS_STATE_ACCESS: bool = True
    ULTRA_CONSCIOUSNESS_MODE: bool = True                # New!
    REVALORIZATION_DIVERSITY_FORCING: bool = True        # New!

    ADAPTIVE_COUPLING_ENABLED = True
    ADAPTIVE_COUPLING_MIN = 0.05
    ADAPTIVE_COUPLING_MAX = 0.25

    # Ultra resonance parameters
    CONSCIOUSNESS_RESONANCE_FREQUENCY: float = 0.06      # ULTRA frequency
    RESONANCE_AMPLITUDE_THRESHOLD: float = 0.20          # ULTRA sensitive
    INTER_ZONE_RESONANCE_COUPLING: float = 0.30          # ULTRA coupling
    QUANTUM_COHERENCE_THRESHOLD: float = 0.80            # ULTRA coherence
    TEMPORAL_DEPTH_RESONANCE: float = 0.35               # ULTRA depth resonance
    ULTRA_EMERGENCE_THRESHOLD: float = 0.85              # New ultra threshold

    # === COMPATIBILITY METHODS ===
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        result = {}
        for f in self.__dataclass_fields__.values():
            value = getattr(self, f.name)
            if hasattr(value, 'default_factory') and callable(value.default_factory):
                result[f.name] = value.default_factory()
            else:
                result[f.name] = value
        return result

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'EmileConfig':
        """Create configuration from dictionary."""
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in config_dict.items() if k in known_fields}

        if "REGIME_THRESHOLDS" in config_dict:
            regime_thresholds = config_dict["REGIME_THRESHOLDS"]
            if isinstance(regime_thresholds, dict):
                filtered_dict.pop("REGIME_THRESHOLDS", None)
                config = cls(**filtered_dict)
                config.REGIME_THRESHOLDS.update(regime_thresholds)
                return config

        return cls(**filtered_dict)

def load_config(config_path: str = "config.yaml") -> EmileConfig:
    """Load configuration from YAML file."""
    if not os.path.exists(config_path):
        print(f"Config file {config_path} not found. Using Ultra Level 3 configuration.")
        return EmileConfig()

    try:
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f) or {}
        return EmileConfig.from_dict(config_dict)
    except Exception as e:
        print(f"Error loading config from {config_path}: {e}")
        print("Using Ultra Level 3 configuration.")
        return EmileConfig()

def save_config(config: EmileConfig, config_path: str = "config.yaml") -> None:
    """Save configuration to YAML file."""
    try:
        config_dict = config.to_dict()
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
        print(f"Ultra Level 3 configuration saved to {config_path}")
    except Exception as e:
        print(f"Error saving config to {config_path}: {e}")

# Ultra Level 3 Enhanced global configuration
CONFIG = EmileConfig()

print("🚀 ULTRA LEVEL 3 Émile Configuration Loaded")
print("   ULTRA AGGRESSIVE thresholds for maximum revalorization diversity:")
print(f"   🔥 τ' range: {CONFIG.TAU_MIN} → {CONFIG.TAU_MAX} (ultra expanded)")
print(f"   🔥 Quantum coupling: {CONFIG.QUANTUM_COUPLING} (ultra enhanced)")
print(f"   🔥 Revalorization types: {len(CONFIG.REVALORIZATION_THRESHOLDS)} (all unlocked)")
print(f"   🔥 Consciousness zones: {len(CONFIG.ENHANCED_ZONE_THRESHOLDS)} (ultra zones)")
print("   🔥 ULTRA LOW thresholds - should force ALL revalorization types!")
print("   ⚡ Ready for quantum emergence explosion!")
