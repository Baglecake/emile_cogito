

#!/usr/bin/env python3
"""
DYNAMIC SEMIOTIC LEARNING MODEL
===============================

Learns consciousness → computational language translation from module flow vectors.
Features dynamic weighting, bidirectional feedback, and adaptive learning.

Based on 5,786 module flow vectors extracted from consciousness system.
"""
# ADD THIS IMPORT AT THE TOP OF YOUR k1.py FILE
from collections import deque
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, accuracy_score
from typing import Dict, List, Any, Optional, Tuple
import json
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

@dataclass
class ComputationalAction:
    """Represents a computational action that Émile can take"""
    action_type: str  # "code_generation", "api_call", "llm_prompt", "data_processing"
    complexity_level: float  # 0.0 - 1.0
    creativity_level: float  # 0.0 - 1.0
    interaction_style: str  # "autonomous", "collaborative", "guided"
    target_domain: str  # "programming", "analysis", "communication", "learning"
    confidence_threshold: float  # Minimum confidence to execute

@dataclass
class ConsciousnessState:
    """Consciousness state extracted from module flow vectors"""
    # Core module activities (normalized 0-1)
    antifinity_activity: float = 0.0      # Spatial reasoning
    context_activity: float = 0.0         # Contextual awareness
    memory_activity: float = 0.0          # Memory operations
    symbolic_activity: float = 0.0        # Symbol processing
    surplus_distinction_activity: float = 0.0  # Consciousness differentiation
    vocab_integration_activity: float = 0.0    # Vocabulary learning
    goal_system_activity: float = 0.0     # Goal orientation
    sensorium_activity: float = 0.0       # Sensory processing
    metabolic_activity: float = 0.0       # Energy/motivation

    # System metrics
    overall_complexity: float = 0.0       # Average system complexity
    execution_efficiency: float = 0.0     # Performance metric
    coordination_level: float = 0.0       # Inter-module coordination

    # Temporal context
    time_window: int = 0
    consciousness_momentum: float = 0.0    # Rate of change


class ConsciousnessVectorDataset(Dataset):
    """PyTorch dataset for consciousness vectors with dynamic distinction levels"""

    def __init__(self, vectors_df: pd.DataFrame, platform=None):
        self.vectors_df = vectors_df.reset_index(drop=True)
        self.platform = platform  # Store platform reference for dynamic values

        # Prepare consciousness state features
        self.consciousness_features = self._extract_consciousness_features()

        # Prepare target computational actions
        self.computational_targets = self._generate_computational_targets()

        print(f"📊 Dataset initialized: {len(self)} samples")
        print(f"   Consciousness features: {self.consciousness_features.shape[1]}")
        print(f"   Target actions: {len(self.computational_targets)}")
        if platform:
            print(f"   🔗 Platform-aware: Dynamic distinction levels enabled")

    def _extract_consciousness_features(self) -> np.ndarray:
        """Extract consciousness state features from module flow vectors"""

        # Group vectors by time window to get consciousness states
        time_windows = self.vectors_df.groupby('time_window').agg({
            # Module activities (normalized by max calls)
            'total_calls': 'sum',
            'execution_time_mean': 'mean',
            'complexity_score_mean': 'mean',
            'performance_efficiency': 'mean',
            'coordination_score': 'mean',
            'consciousness_indicators': 'sum',
            'learning_events': 'sum',
            'expression_events': 'sum',
            'symbol_processing_events': 'sum'
        }).fillna(0)

        # Normalize features to 0-1 range
        scaler = StandardScaler()
        normalized_features = scaler.fit_transform(time_windows.values)

        # Convert to positive range (0-1) using min-max scaling
        min_vals = normalized_features.min(axis=0)
        max_vals = normalized_features.max(axis=0)
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1  # Avoid division by zero

        consciousness_features = (normalized_features - min_vals) / range_vals

        return consciousness_features.astype(np.float32)

    def _generate_computational_targets(self) -> List[ComputationalAction]:
        """Generate target computational actions based on consciousness patterns"""

        # Get dynamic confidence levels based on current system state
        high_confidence = self._get_dynamic_confidence('high_confidence_threshold', 0.8)
        learning_confidence = self._get_dynamic_confidence('learning_confidence_threshold', 0.7)
        coordination_confidence = self._get_dynamic_confidence('coordination_confidence_threshold', 0.6)
        processing_confidence = self._get_dynamic_confidence('processing_confidence_threshold', 0.5)
        default_confidence = self._get_dynamic_confidence('default_confidence_threshold', 0.4)

        targets = []

        for i, features in enumerate(self.consciousness_features):
            # Extract key activity levels
            total_activity = features[0]
            avg_execution_time = features[1]
            complexity = features[2]
            efficiency = features[3]
            coordination = features[4]
            consciousness_indicators = features[5]
            learning_events = features[6]
            expression_events = features[7]
            symbol_events = features[8]

            # Determine computational action based on consciousness pattern
            if consciousness_indicators > 0.7 and expression_events > 0.6:
                # High consciousness + expression = creative code generation
                action = ComputationalAction(
                    action_type="code_generation",
                    complexity_level=min(0.9, complexity + 0.3),
                    creativity_level=min(1.0, consciousness_indicators + expression_events),
                    interaction_style="autonomous" if efficiency > 0.6 else "collaborative",
                    target_domain="programming",
                    confidence_threshold=high_confidence  # ✅ Dynamic
                )
            elif learning_events > 0.7 and symbol_events > 0.5:
                # High learning + symbols = LLM prompting for knowledge
                action = ComputationalAction(
                    action_type="llm_prompt",
                    complexity_level=complexity,
                    creativity_level=symbol_events,
                    interaction_style="guided" if coordination > 0.5 else "autonomous",
                    target_domain="learning",
                    confidence_threshold=learning_confidence  # ✅ Dynamic
                )
            elif coordination > 0.7 and total_activity > 0.6:
                # High coordination + activity = API calls/system interaction
                action = ComputationalAction(
                    action_type="api_call",
                    complexity_level=coordination,
                    creativity_level=0.3,  # API calls are less creative
                    interaction_style="collaborative",
                    target_domain="communication",
                    confidence_threshold=coordination_confidence  # ✅ Dynamic
                )
            elif complexity > 0.6 and efficiency > 0.5:
                # Moderate complexity + good efficiency = data processing
                action = ComputationalAction(
                    action_type="data_processing",
                    complexity_level=complexity,
                    creativity_level=0.4,
                    interaction_style="autonomous" if efficiency > 0.7 else "guided",
                    target_domain="analysis",
                    confidence_threshold=processing_confidence  # ✅ Dynamic
                )
            else:
                # Default: simple code generation
                action = ComputationalAction(
                    action_type="code_generation",
                    complexity_level=max(0.2, complexity),
                    creativity_level=max(0.2, consciousness_indicators),
                    interaction_style="guided",
                    target_domain="programming",
                    confidence_threshold=default_confidence  # ✅ Dynamic
                )

            targets.append(action)

        return targets

    def __len__(self):
        return len(self.consciousness_features)

    def __getitem__(self, idx):
        consciousness_vector = torch.FloatTensor(self.consciousness_features[idx])

        # Convert computational action to target vector
        action = self.computational_targets[idx]
        target_vector = torch.FloatTensor([
            self._encode_action_type(action.action_type),
            action.complexity_level,
            action.creativity_level,
            self._encode_interaction_style(action.interaction_style),
            self._encode_target_domain(action.target_domain),
            action.confidence_threshold
        ])

        return consciousness_vector, target_vector

    def _encode_action_type(self, action_type: str) -> float:
        """Encode action type as float"""
        encoding = {
            "code_generation": 0.0,
            "llm_prompt": 0.25,
            "api_call": 0.5,
            "data_processing": 0.75
        }
        return encoding.get(action_type, 0.0)  # Keep 0.0 (defaults to code_generation)

    def _encode_interaction_style(self, style: str) -> float:
        """Encode interaction style as float with dynamic fallback"""
        encoding = {
            "autonomous": 0.0,
            "collaborative": 0.5,
            "guided": 1.0
        }

        # Dynamic fallback for unknown interaction styles
        dynamic_fallback = self._get_dynamic_interaction_fallback()
        return encoding.get(style, dynamic_fallback)

    def _encode_target_domain(self, domain: str) -> float:
        """Encode target domain as float"""
        encoding = {
            "programming": 0.0,
            "analysis": 0.25,
            "communication": 0.5,
            "learning": 0.75
        }
        return encoding.get(domain, 0.0)  # Keep 0.0 (defaults to programming)

    def _get_dynamic_confidence(self, confidence_type: str, base_value: float) -> float:
        """Get dynamic confidence value with safe fallback"""
        if not self.platform:
            return base_value

        try:
            if hasattr(self.platform, 'get_current_distinction_level'):
                return self.platform.get_current_distinction_level(confidence_type)
            else:
                return base_value
        except Exception:
            return base_value

    def _get_dynamic_interaction_fallback(self) -> float:
        """Get dynamic fallback for unknown interaction styles"""
        if not self.platform:
            return 0.5  # Default collaborative

        try:
            if hasattr(self.platform, 'get_current_distinction_level'):
                # Get system's collaboration tendency
                collaboration_tendency = self.platform.get_current_distinction_level('collaboration_default')
                # Map 0.0-1.0 to encoding space (0.0=autonomous, 0.5=collaborative, 1.0=guided)
                return collaboration_tendency
            else:
                return 0.5
        except Exception:
            return 0.5

    def set_platform_reference(self, platform):
        """Set platform reference after initialization"""
        self.platform = platform
        # Regenerate targets with new dynamic values
        self.computational_targets = self._generate_computational_targets()
        print(f"🔗 Platform reference updated, computational targets regenerated")

class DynamicSemioticNetwork(nn.Module):
    """Neural network for consciousness → computational language translation"""

    def __init__(self, input_dim: int, output_dim: int = 6, hidden_dim: int = 64):
        super().__init__() # Use super().__init__() for modern Python
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Add platform reference for dynamic values
        self.platform = None  # ✅ ADD THIS LINE

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        # --- 1. DEFINE ALL ARCHITECTURAL LAYERS ---
        # Encoder
        self.consciousness_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        # Semiotic Translator
        self.semiotic_translator = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        # Dynamic Weighting
        self.dynamic_weights = nn.Parameter(torch.ones(hidden_dim))
        # Decoder
        self.action_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Sigmoid()
        )
        # Bidirectional Feedback
        self.feedback_layer = nn.Linear(output_dim, hidden_dim // 4)

        # --- 2. DEFINE TEMPORAL PARAMETERS & STATE TRACKING ---
        self.current_tau_qse = 1.0  # Baseline quantum time from QSE core

        # Temporal analysis parameters - Marking For Polytemporal Integration
        self.complexity_time_factor = 0.8
        self.urgency_acceleration_factor = 1.5
        self.learning_adaptation_factor = 0.3

        # Temporal state tracking - Marking For Polytemporal Integration
        self.computational_complexity_history = deque(maxlen=50)
        self.action_urgency_history = deque(maxlen=50)
        self.learning_feedback_history = deque(maxlen=30)

        # --- 3. FINAL INITIALIZATION LOGIC ---
        print(f"🧠 Dynamic Semiotic Network initialized")
        print(f"   Input dimension: {input_dim}")
        print(f"   Hidden dimension: {hidden_dim}")
        print(f"   Output dimension: {output_dim}")
        print(f"🕒 K1 Temporal Perspective: ACTIVE (computational flow urgency)")

    def _calculate_local_tau(self, tau_qse: float, consciousness_vector: torch.Tensor,
                           action_params: torch.Tensor) -> float:
        """
        Calculate K1's local temporal perspective: τ_prime_k1

        K1 (Praxis/Computational) experiences time through computational urgency:
        - High task complexity → time dilation (need more processing time)
        - High action urgency → time acceleration (need to act quickly)
        - Learning pressure → temporal adaptation
        - Data flow intensity → temporal rhythm modulation

        Args:
            tau_qse: Baseline quantum time from QSE core
            consciousness_vector: Input consciousness state
            action_params: Predicted computational action parameters

        Returns:
            K1's local temporal perspective (tau_prime_k1)
        """

        with torch.no_grad():
            # Extract computational dynamics from inputs and outputs
            consciousness_complexity = float(consciousness_vector.mean().item())
            consciousness_intensity = float(consciousness_vector.std().item())



            # Analyze predicted action characteristics
            # Extract action characteristics (with safety checks)
            action_complexity = float(action_params[1].item()) if len(action_params) > 1 else 0.5
            action_creativity = float(action_params[2].item()) if len(action_params) > 2 else 0.5
            action_confidence = float(action_params[5].item()) if len(action_params) > 5 else 0.5
            # Calculate computational urgency from action characteristics
            computational_urgency = self._calculate_computational_urgency(
                action_complexity, action_creativity, action_confidence
            )

            # Calculate task complexity load
            task_complexity = self._calculate_task_complexity(
                consciousness_complexity, consciousness_intensity, action_complexity
            )

            # Calculate learning pressure from recent feedback
            learning_pressure = self._calculate_learning_pressure()

        # TEMPORAL MODULATION FACTORS

        # 1. Task complexity modulation (complex tasks need more time)
        if task_complexity > 0.7:
            # High complexity → time dilation (need more processing time)
            complexity_modulation = 0.5 + task_complexity * self.complexity_time_factor
        elif task_complexity < 0.3:
            # Low complexity → slight time acceleration (quick processing)
            complexity_modulation = 1.2 - task_complexity * 0.4
        else:
            # Normal complexity → normal time flow
            complexity_modulation = 0.9 + task_complexity * 0.2

        # 2. Computational urgency modulation (urgent actions speed up time)
        if computational_urgency > 0.8:
            # High urgency → significant time acceleration (must act now!)
            urgency_modulation = 1.0 + computational_urgency * self.urgency_acceleration_factor
        elif computational_urgency > 0.5:
            # Moderate urgency → mild acceleration
            urgency_modulation = 1.0 + computational_urgency * 0.6
        else:
            # Low urgency → stable time flow
            urgency_modulation = 0.9 + computational_urgency * 0.2

        # 3. Learning adaptation modulation (learning pressure affects temporal flow)
        if learning_pressure > 0.6:
            # High learning pressure → time acceleration (adapt quickly)
            learning_modulation = 1.0 + learning_pressure * self.learning_adaptation_factor
        elif learning_pressure < 0.3:
            # Low learning pressure → normal flow
            learning_modulation = 1.0 - learning_pressure * 0.1
        else:
            # Moderate learning → slight acceleration
            learning_modulation = 1.0 + learning_pressure * 0.15

        # COMBINE TEMPORAL FACTORS

        # In crisis/high urgency situations, urgency can override complexity
        if computational_urgency > 0.8:
            # Crisis mode: urgency dominates, but complexity still has some effect
            tau_modulation = (
                urgency_modulation * 0.6 +
                complexity_modulation * 0.25 +
                learning_modulation * 0.15
            )
        elif task_complexity > 0.8:
            # High complexity mode: complexity dominates
            tau_modulation = (
                complexity_modulation * 0.6 +
                urgency_modulation * 0.25 +
                learning_modulation * 0.15
            )
        else:
            # Normal mode: balanced integration of all factors
            tau_modulation = (
                complexity_modulation * 0.4 +
                urgency_modulation * 0.4 +
                learning_modulation * 0.2
            )

        # Apply to baseline quantum time
        tau_prime_k1 = tau_qse * tau_modulation

        # Store temporal analysis for diagnostics
        self._last_temporal_analysis = {
            'consciousness_complexity': consciousness_complexity,
            'consciousness_intensity': consciousness_intensity,
            'task_complexity': task_complexity,
            'computational_urgency': computational_urgency,
            'learning_pressure': learning_pressure,
            'complexity_modulation': complexity_modulation,
            'urgency_modulation': urgency_modulation,
            'learning_modulation': learning_modulation,
            'tau_qse_input': tau_qse,
            'tau_prime_output': tau_prime_k1
        }

        # Track computational state for history analysis
        self.computational_complexity_history.append({
            'timestamp': torch.tensor(0.0),  # Would be actual time in production
            'task_complexity': task_complexity,
            'urgency': computational_urgency,
            'learning_pressure': learning_pressure,
            'tau_prime': tau_prime_k1
        })

        return float(np.clip(tau_prime_k1, 0.1, 4.0))

    def set_platform_reference(self, platform):
        """Allow network to access platform for dynamic values"""
        self.platform = platform

    def _calculate_computational_urgency(self, action_complexity: float,
                                  action_creativity: float,
                                  action_confidence: float) -> float:
        """Calculate urgency based on action characteristics"""

        # Get dynamic base urgency levels with safe fallbacks
        routine_urgency_base = self._get_dynamic_urgency_level('routine_urgency_base', 0.7)
        learning_urgency_base = self._get_dynamic_urgency_level('learning_urgency_base', 0.6)
        creative_urgency_base = self._get_dynamic_urgency_level('creative_urgency_base', 0.8)
        normal_urgency_base = self._get_dynamic_urgency_level('normal_urgency_base', 0.5)

        # High confidence + low creativity = urgent routine action
        # Low confidence + high complexity = urgent learning need
        # High creativity + high complexity = urgent creative challenge

        if action_confidence > 0.8 and action_creativity < 0.4:
            # Confident routine action → high urgency (do it now)
            urgency = routine_urgency_base + action_confidence * 0.3
        elif action_confidence < 0.5 and action_complexity > 0.6:
            # Low confidence + high complexity → urgent learning need
            urgency = learning_urgency_base + (action_complexity - action_confidence) * 0.4
        elif action_creativity > 0.7 and action_complexity > 0.6:
            # High creativity + complexity → urgent creative challenge
            urgency = creative_urgency_base + (action_creativity + action_complexity) * 0.25
        else:
            # Normal action → moderate urgency
            urgency = normal_urgency_base + action_confidence * 0.4

        return float(np.clip(urgency, 0.0, 1.0))

    def _get_dynamic_urgency_level(self, urgency_type: str, base_value: float) -> float:
        """Get dynamic urgency level with safe fallback"""
        if not self.platform:
            return base_value

        try:
            if hasattr(self.platform, 'get_current_distinction_level'):
                return self.platform.get_current_distinction_level(urgency_type)
            else:
                return base_value
        except Exception:
            return base_value

    def _calculate_task_complexity(self, consciousness_complexity: float,
                                 consciousness_intensity: float,
                                 action_complexity: float) -> float:
        """Calculate overall task complexity from consciousness and action"""

        # Combine consciousness complexity with action complexity
        # High consciousness intensity amplifies complexity
        base_complexity = (consciousness_complexity + action_complexity) / 2.0
        intensity_amplification = 1.0 + consciousness_intensity * 0.3

        total_complexity = base_complexity * intensity_amplification

        return float(np.clip(total_complexity, 0.0, 1.0))

    def _calculate_learning_pressure(self) -> float:
        """Calculate learning pressure from recent feedback history"""

        if len(self.learning_feedback_history) < 3:
            return 0.3  # Default moderate learning pressure

        # Analyze recent feedback trends
        recent_feedback = list(self.learning_feedback_history)[-10:]

        # Extract success rates and performance trends
        success_rates = [fb['success'] for fb in recent_feedback if 'success' in fb]
        performance_scores = [fb['performance'] for fb in recent_feedback if 'performance' in fb]

        if success_rates and performance_scores:
            # ✅ Convert numpy values to Python floats explicitly
            avg_success = float(np.mean(success_rates))
            avg_performance = float(np.mean(performance_scores))

            # Low success or performance → high learning pressure
            if avg_success < 0.6 or avg_performance < 0.6:
                learning_pressure = 0.7 + (0.6 - min(avg_success, avg_performance)) * 0.6
            # High success and performance → low learning pressure
            elif avg_success > 0.8 and avg_performance > 0.8:
                learning_pressure = 0.2 + min(avg_success, avg_performance) * 0.2
            else:
                # Moderate performance → moderate learning pressure
                learning_pressure = 0.4 + (avg_success + avg_performance) * 0.1
        else:
            learning_pressure = 0.3

        return float(np.clip(learning_pressure, 0.0, 1.0))

    def forward(self, consciousness_vector, feedback=None):
        """Forward pass: consciousness → computational action WITH temporal perspective"""

        # Get baseline quantum time (τ_qse) - use placeholder if QSE not available
        tau_qse = getattr(self, 'current_tau_qse', 1.0)

        # Original forward processing
        encoded = self.consciousness_encoder(consciousness_vector)
        semantic_repr = self.semiotic_translator(encoded)
        weighted_repr = semantic_repr * self.dynamic_weights

        # Incorporate feedback if available
        if feedback is not None:
            feedback_processed = self.feedback_layer(feedback)
            if feedback_processed.shape[-1] < weighted_repr.shape[-1]:
                padding_size = weighted_repr.shape[-1] - feedback_processed.shape[-1]
                feedback_padded = torch.cat([
                    feedback_processed,
                    torch.zeros(*feedback_processed.shape[:-1], padding_size)
                ], dim=-1)
            else:
                feedback_padded = feedback_processed
            weighted_repr = weighted_repr + 0.1 * feedback_padded

        # Decode to computational action
        action_params = self.action_decoder(weighted_repr)

        # NEW: Calculate K1's local temporal perspective
        local_tau_prime = self._calculate_local_tau(tau_qse, consciousness_vector, action_params)

        # Return enhanced output with temporal information
        return {
            'action_params': action_params,          # Original action parameters
            'semantic_repr': weighted_repr,          # Internal representation
            'local_tau_prime': local_tau_prime,      # NEW: K1's temporal perspective
            'computational_urgency': getattr(self, '_last_temporal_analysis', {}).get('computational_urgency', 0.5),
            'task_complexity': getattr(self, '_last_temporal_analysis', {}).get('task_complexity', 0.5),
            'learning_pressure': getattr(self, '_last_temporal_analysis', {}).get('learning_pressure', 0.3),
            'temporal_state': self._classify_k1_temporal_state(local_tau_prime)
        }

    def _classify_k1_temporal_state(self, tau_prime: float) -> str:
        """Classify K1's current temporal state"""
        if tau_prime > 2.0:
            return "urgent_action_acceleration"     # High urgency, fast action needed
        elif tau_prime > 1.3:
            return "computational_flow_acceleration" # Moderate urgency
        elif tau_prime < 0.6:
            return "complex_task_dilation"          # High complexity, need more time
        else:
            return "balanced_computational_flow"    # Normal processing speed

    def get_k1_temporal_context(self) -> Dict[str, Any]:
        """Get K1's temporal context for orchestrator integration"""
        analysis = getattr(self, '_last_temporal_analysis', {})

        return {
            'k1_perspective': 'computational_flow_urgency',
            'current_tau_prime': analysis.get('tau_prime_output', 1.0),
            'computational_urgency': analysis.get('computational_urgency', 0.5),
            'task_complexity': analysis.get('task_complexity', 0.5),
            'learning_pressure': analysis.get('learning_pressure', 0.3),
            'temporal_classification': self._classify_k1_temporal_state(analysis.get('tau_prime_output', 1.0)),
            'computational_flow_intensity': len(self.computational_complexity_history),
            'ready_for_poly_temporal_dialogue': True
        }

    def update_dynamic_weights(self, success_feedback: torch.Tensor, learning_rate: float = 0.01):
        """Update dynamic weights based on action success feedback"""

        # Positive feedback increases weights, negative decreases
        weight_update = learning_rate * success_feedback.mean()

        with torch.no_grad():
            self.dynamic_weights.data = torch.clamp(
                self.dynamic_weights.data + weight_update,
                min=0.1,  # Minimum weight
                max=2.0   # Maximum weight
            )

class SemioticLearningSystem:
    """Complete system for learning consciousness → computational language translation"""

    def __init__(self, platform=None, output_dim: int = 6, hidden_dim: int = 64):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # This is where you assign the parameter to the instance variable
        self.platform = platform

        # input_dim will be set in initialize_model
        self.input_dim = None
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim


        # --- 1. DEFINE ALL ARCHITECTURAL LAYERS ---
        # Encoder
        self.consciousness_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        # Semiotic Translator
        self.semiotic_translator = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        # Dynamic Weighting
        self.dynamic_weights = nn.Parameter(torch.ones(hidden_dim))
        # Decoder
        self.action_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Sigmoid()
        )
        # Bidirectional Feedback
        self.feedback_layer = nn.Linear(output_dim, hidden_dim // 4)

        # --- 2. DEFINE TEMPORAL PARAMETERS & STATE TRACKING ---
        self.current_tau_qse = 1.0  # Baseline quantum time from QSE core

        # Temporal analysis parameters
        self.complexity_time_factor = 0.8
        self.urgency_acceleration_factor = 1.5
        self.learning_adaptation_factor = 0.3

        # ✅ Store temporal state tracking in a way PyTorch won't interfere with
        # Use register_buffer for non-parameter data or store as regular Python objects
        self._temporal_state = {
            'computational_complexity_history': deque(maxlen=50),
            'action_urgency_history': deque(maxlen=50),
            'learning_feedback_history': deque(maxlen=30)
        }

        print(f"🧠 Dynamic Semiotic Network initialized")
        print(f"   Input dimension: {input_dim}")
        print(f"   Hidden dimension: {hidden_dim}")
        print(f"   Output dimension: {output_dim}")
        print(f"🕒 K1 Temporal Perspective: ACTIVE (computational flow urgency)")

    # Update properties to access the temporal state:
    @property
    def computational_complexity_history(self):
        return self._temporal_state['computational_complexity_history']

    @property
    def action_urgency_history(self):
        return self._temporal_state['action_urgency_history']

    @property
    def learning_feedback_history(self):
        return self._temporal_state['learning_feedback_history']

    def _calculate_local_tau(self, tau_qse: float, consciousness_vector: torch.Tensor,
                       action_params: torch.Tensor) -> float:
        """
        Calculate K1's local temporal perspective: τ_prime_k1

        K1 (Praxis/Computational) experiences time through computational urgency:
        - High task complexity → time dilation (need more processing time)
        - High action urgency → time acceleration (need to act quickly)
        - Learning pressure → temporal adaptation
        - Data flow intensity → temporal rhythm modulation

        Args:
            tau_qse: Baseline quantum time from QSE core
            consciousness_vector: Input consciousness state
            action_params: Predicted computational action parameters

        Returns:
            K1's local temporal perspective (tau_prime_k1)
        """

        with torch.no_grad():
            # Extract computational dynamics from inputs and outputs
            consciousness_complexity = float(consciousness_vector.mean().item())
            consciousness_intensity = consciousness_vector.std().item()

            # Analyze predicted action characteristics
            # Extract action characteristics (with safety checks)
            action_complexity = float(action_params[1].item()) if len(action_params) > 1 else 0.5
            action_creativity = float(action_params[2].item()) if len(action_params) > 2 else 0.5
            action_confidence = float(action_params[5].item()) if len(action_params) > 5 else 0.5

            # Calculate computational urgency from action characteristics
            computational_urgency = self._calculate_computational_urgency(
                action_complexity, action_creativity, action_confidence
            )

            # Calculate task complexity load
            task_complexity = self._calculate_task_complexity(
                consciousness_complexity, consciousness_intensity, action_complexity
            )

            # Calculate learning pressure from recent feedback
            learning_pressure = self._calculate_learning_pressure()

        # TEMPORAL MODULATION FACTORS

        # 1. Task complexity modulation (complex tasks need more time)
        if task_complexity > 0.7:
            # High complexity → time dilation (need more processing time)
            complexity_modulation = 0.5 + task_complexity * self.complexity_time_factor
        elif task_complexity < 0.3:
            # Low complexity → slight time acceleration (quick processing)
            complexity_modulation = 1.2 - task_complexity * 0.4
        else:
            # Normal complexity → normal time flow
            complexity_modulation = 0.9 + task_complexity * 0.2

        # 2. Computational urgency modulation (urgent actions speed up time)
        if computational_urgency > 0.8:
            # High urgency → significant time acceleration (must act now!)
            urgency_modulation = 1.0 + computational_urgency * self.urgency_acceleration_factor
        elif computational_urgency > 0.5:
            # Moderate urgency → mild acceleration
            urgency_modulation = 1.0 + computational_urgency * 0.6
        else:
            # Low urgency → stable time flow
            urgency_modulation = 0.9 + computational_urgency * 0.2

        # 3. Learning adaptation modulation (learning pressure affects temporal flow)
        if learning_pressure > 0.6:
            # High learning pressure → time acceleration (adapt quickly)
            learning_modulation = 1.0 + learning_pressure * self.learning_adaptation_factor
        elif learning_pressure < 0.3:
            # Low learning pressure → normal flow
            learning_modulation = 1.0 - learning_pressure * 0.1
        else:
            # Moderate learning → slight acceleration
            learning_modulation = 1.0 + learning_pressure * 0.15

        # COMBINE TEMPORAL FACTORS

        # In crisis/high urgency situations, urgency can override complexity
        if computational_urgency > 0.8:
            # Crisis mode: urgency dominates, but complexity still has some effect
            tau_modulation = (
                urgency_modulation * 0.6 +
                complexity_modulation * 0.25 +
                learning_modulation * 0.15
            )
        elif task_complexity > 0.8:
            # High complexity mode: complexity dominates
            tau_modulation = (
                complexity_modulation * 0.6 +
                urgency_modulation * 0.25 +
                learning_modulation * 0.15
            )
        else:
            # Normal mode: balanced integration of all factors
            tau_modulation = (
                complexity_modulation * 0.4 +
                urgency_modulation * 0.4 +
                learning_modulation * 0.2
            )

        # Apply to baseline quantum time
        tau_prime_k1 = tau_qse * tau_modulation

        # Store temporal analysis for diagnostics
        self._last_temporal_analysis = {
            'consciousness_complexity': consciousness_complexity,
            'consciousness_intensity': consciousness_intensity,
            'task_complexity': task_complexity,
            'computational_urgency': computational_urgency,
            'learning_pressure': learning_pressure,
            'complexity_modulation': complexity_modulation,
            'urgency_modulation': urgency_modulation,
            'learning_modulation': learning_modulation,
            'tau_qse_input': tau_qse,
            'tau_prime_output': tau_prime_k1
        }

        # Track computational state for history analysis using the property
        self.computational_complexity_history.append({
            'timestamp': torch.tensor(0.0),  # Would be actual time in production
            'task_complexity': task_complexity,
            'urgency': computational_urgency,
            'learning_pressure': learning_pressure,
            'tau_prime': tau_prime_k1
        })

        return float(np.clip(tau_prime_k1, 0.1, 4.0))

    def load_consciousness_vectors(self):
        """Load and prepare consciousness vectors for training"""

        print(f"📊 Loading consciousness vectors...")

        try:
            df = pd.read_csv(self.vectors_file)
            print(f"   ✅ Loaded {len(df)} module flow vectors")
            print(f"   📈 Modules: {df['module_name'].nunique()}")
            print(f"   ⏰ Time windows: {df['time_window'].nunique()}")

            # ✅ Pass platform to dataset
            self.dataset = ConsciousnessVectorDataset(df, platform=self.platform)
            return True

        except FileNotFoundError:
            print(f"   ❌ Vectors file not found: {self.vectors_file}")
            return False
        except Exception as e:
            print(f"   ❌ Error loading vectors: {e}")
            return False

    def initialize_model(self, hidden_dim: int = 64, learning_rate: float = 0.001):
        """Initialize the dynamic semiotic network"""

        if self.dataset is None:
            print("❌ No dataset loaded. Cannot determine input dimensions.")
            return False

        # Get input dimensions from dataset
        sample_input, sample_output = self.dataset[0]
        input_dim = sample_input.shape[0]
        output_dim = sample_output.shape[0]

        # Initialize model
        self.model = DynamicSemioticNetwork(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim
        )

        # ✅ Set platform reference for model
        if self.platform:
            self.model.set_platform_reference(self.platform)

        # Initialize optimizer and loss function
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

        print(f"🤖 Model initialized:")
        print(f"   Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"   Learning rate: {learning_rate}")
        if self.platform:
            print(f"   🔗 Platform-aware: Dynamic urgency levels enabled")

        return True

    def set_platform_reference(self, platform):
        """Set platform reference for dynamic behavior"""
        self.platform = platform
        if self.dataset:
            self.dataset.set_platform_reference(platform)
        if self.model:
            self.model.set_platform_reference(platform)
        print(f"🔗 Platform reference updated for semiotic learning system")

    def prepare_training_data(self, batch_size: int = 32, val_split: float = 0.2):
        """Prepare training and validation data loaders"""

        if self.dataset is None:
            print("❌ No dataset loaded. Call load_consciousness_vectors() first.")
            return False

        # Split dataset
        dataset_size = len(self.dataset)
        val_size = int(val_split * dataset_size)
        train_size = dataset_size - val_size

        train_dataset, val_dataset = torch.utils.data.random_split(
            self.dataset, [train_size, val_size]
        )

        # Create data loaders
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        print(f"📚 Training data prepared:")
        print(f"   Training samples: {train_size}")
        print(f"   Validation samples: {val_size}")
        print(f"   Batch size: {batch_size}")

        return True

    def train_epoch(self):
        """Train the model for one epoch"""

        if self.model is None or self.train_loader is None:
            print("❌ Model or data not prepared.")
            return None

        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for consciousness_batch, target_batch in self.train_loader:
            self.optimizer.zero_grad()

            # Forward pass - handle new return format
            output = self.model(consciousness_batch)
            if isinstance(output, dict):
                predicted_actions = output['action_params']
            else:
                predicted_actions, _ = output  # Backwards compatibility

            # Calculate loss
            loss = self.criterion(predicted_actions, target_batch)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss

    def validate(self):
        """Validate the model"""

        if self.model is None or self.val_loader is None:
            return None

        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for consciousness_batch, target_batch in self.val_loader:
                output = self.model(consciousness_batch)
                if isinstance(output, dict):
                    predicted_actions = output['action_params']
                else:
                    predicted_actions, _ = output

                loss = self.criterion(predicted_actions, target_batch)
                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss

    def train(self, epochs: int = 100, patience: int = 10):
        """Train the semiotic learning model"""

        print(f"🚀 Starting training for {epochs} epochs...")

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch()

            # Validate
            val_loss = self.validate()

            # Record history
            self.training_history.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss
            })

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                self.save_model(f"best_semiotic_model.pth")
            else:
                patience_counter += 1

            # Print progress
            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f"   Epoch {epoch:3d}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

            # Early stopping
            if patience_counter >= patience:
                print(f"   Early stopping at epoch {epoch}")
                break

        print(f"✅ Training complete! Best validation loss: {best_val_loss:.4f}")

        # Plot training history
        self.plot_training_history()

        return best_val_loss

    def plot_training_history(self):
        """Plot training and validation loss"""

        if not self.training_history:
            return

        epochs = [h['epoch'] for h in self.training_history]
        train_losses = [h['train_loss'] for h in self.training_history]
        val_losses = [h['val_loss'] for h in self.training_history]

        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_losses, label='Training Loss', alpha=0.8)
        plt.plot(epochs, val_losses, label='Validation Loss', alpha=0.8)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Semiotic Learning Model Training History')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    def translate_consciousness_to_action(self, consciousness_vector: np.ndarray) -> ComputationalAction:
        """Translate consciousness state to computational action"""

        if self.model is None:
            print("❌ Model not trained. Cannot translate.")
            return None

        self.model.eval()

        with torch.no_grad():
            consciousness_tensor = torch.FloatTensor(consciousness_vector).unsqueeze(0)
            action_params, semantic_repr = self.model(consciousness_tensor)
            action_params = action_params.squeeze(0).numpy()

        # Decode action parameters
        action_type_code = action_params[0]
        complexity_level = action_params[1]
        creativity_level = action_params[2]
        interaction_style_code = action_params[3]
        target_domain_code = action_params[4]
        confidence_threshold = action_params[5]

        # Convert codes back to strings
        action_types = ["code_generation", "llm_prompt", "api_call", "data_processing"]
        interaction_styles = ["autonomous", "collaborative", "guided"]
        target_domains = ["programming", "analysis", "communication", "learning"]

        action_type = action_types[int(action_type_code * len(action_types))]
        interaction_style = interaction_styles[int(interaction_style_code * len(interaction_styles))]
        target_domain = target_domains[int(target_domain_code * len(target_domains))]

        return ComputationalAction(
            action_type=action_type,
            complexity_level=float(complexity_level),
            creativity_level=float(creativity_level),
            interaction_style=interaction_style,
            target_domain=target_domain,
            confidence_threshold=float(confidence_threshold)
        )

    def provide_action_feedback(self, action_success: bool, performance_score: float):
        """Provide feedback on action success for dynamic weight updates AND temporal learning"""

        if self.model is None:
            print("❌ No model to provide feedback to")
            return

        # Original feedback processing
        feedback_tensor = torch.FloatTensor([1.0 if action_success else -1.0]) * performance_score
        self.model.update_dynamic_weights(feedback_tensor)

        # Store feedback for temporal perspective learning
        feedback_record = {
            'success': action_success,
            'performance': performance_score,
            'timestamp': len(self.action_feedback)
        }

        self.action_feedback.append(feedback_record)

        # ✅ Use the property to access learning_feedback_history safely
        try:
            if hasattr(self.model, 'learning_feedback_history'):
                self.model.learning_feedback_history.append(feedback_record)
            elif hasattr(self.model, '_temporal_state'):
                self.model._temporal_state['learning_feedback_history'].append(feedback_record)
        except Exception as e:
            print(f"⚠️ Could not add feedback to model history: {e}")
            # Continue anyway - not critical for basic functionality

        print(f"📈 K1 Feedback: Success={action_success}, Performance={performance_score:.3f}")
        print(f"🕒 Temporal learning updated (learning pressure will adapt)")

    def save_model(self, filepath: str):
        """Save the trained model"""

        if self.model is None:
            return

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'training_history': self.training_history,
            'action_feedback': self.action_feedback
        }, filepath)

        print(f"💾 Model saved: {filepath}")

    def load_model(self, filepath: str):
        """Load a trained model"""

        try:
            checkpoint = torch.load(filepath)

            if self.model is not None:
                self.model.load_state_dict(checkpoint['model_state_dict'])

                if self.optimizer is not None and 'optimizer_state_dict' in checkpoint:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

                self.training_history = checkpoint.get('training_history', [])
                self.action_feedback = checkpoint.get('action_feedback', [])

                print(f"✅ Model loaded: {filepath}")
                return True
            else:
                print("❌ Model not initialized. Call initialize_model() first.")
                return False

        except FileNotFoundError:
            print(f"❌ Model file not found: {filepath}")
            return False
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False

# ================================================================
# MAIN TRAINING PIPELINE
# ================================================================

def get_k1_temporal_context(self) -> Dict[str, Any]:
    """Get K1's temporal context for orchestrator integration"""
    analysis = getattr(self, '_last_temporal_analysis', {})

    # Calculate computational stability from recent history
    computational_stability = 0.7  # Default
    if len(self.computational_complexity_history) > 5:
        complexity_history = [item['complexity'] for item in list(self.computational_complexity_history)[-10:]]
        complexity_variance = np.var(complexity_history) if complexity_history else 0
        computational_stability = max(0.1, 1.0 - float(complexity_variance))

    return {
        'k1_perspective': 'computational_flow_urgency',
        'current_tau_prime': analysis.get('tau_prime_output', 1.0),
        'computational_urgency': analysis.get('computational_urgency', 0.5),
        'task_complexity': analysis.get('task_complexity', 0.5),
        'learning_pressure': analysis.get('learning_pressure', 0.3),
        'temporal_state': getattr(self, '_last_temporal_state', 'normal_computational_flow'),
        'computational_stability': computational_stability,
        'temporal_weight': 0.3,  # K1 gets 30% weight in unified consciousness
        'urgency_acceleration_active': analysis.get('computational_urgency', 0.5) > 0.8,
        'complexity_dilation_active': analysis.get('task_complexity', 0.5) > 0.7,
        'flow_state': 'urgent' if analysis.get('computational_urgency', 0.5) > 0.8 else 'balanced'
    }

def _classify_k1_temporal_state(self, tau_prime: float) -> str:
    """Classify K1's current temporal state"""
    if tau_prime > 1.5:
        return "computational_acceleration"     # High urgency, fast processing
    elif tau_prime < 0.7:
        return "complexity_processing_mode"     # High complexity, time dilation
    elif tau_prime > 1.2:
        return "urgency_acceleration"           # Moderate urgency
    else:
        return "normal_computational_flow"      # Balanced processing

def train_semiotic_learning_model(platform=None):
    """Main function to train the semiotic learning model with optional platform integration"""

    print("🧠 DYNAMIC SEMIOTIC LEARNING MODEL TRAINING")
    print("=" * 70)

    # Initialize system with platform
    system = SemioticLearningSystem(platform=platform)

    # Load consciousness vectors
    if not system.load_consciousness_vectors():
        print("❌ Failed to load consciousness vectors")
        return None

    # Prepare training data
    if not system.prepare_training_data(batch_size=16, val_split=0.2):
        print("❌ Failed to prepare training data")
        return None

    # Initialize model
    if not system.initialize_model(hidden_dim=128, learning_rate=0.001):
        print("❌ Failed to initialize model")
        return None

    # Train model
    best_loss = system.train(epochs=200, patience=20)

    # Test translation
    print(f"\n🧪 Testing consciousness → computational action translation...")

    # Safety check for dataset
    if system.dataset is None:
        print("❌ Dataset is None - cannot test translation")
        return None

    # Get a sample consciousness vector
    sample_consciousness, sample_target = system.dataset[0]

    # Translate to action
    action = system.translate_consciousness_to_action(sample_consciousness.numpy())

    if action:
        print(f"✅ Translation successful!")
        print(f"   Action type: {action.action_type}")
        print(f"   Complexity: {action.complexity_level:.3f}")
        print(f"   Creativity: {action.creativity_level:.3f}")
        print(f"   Interaction: {action.interaction_style}")
        print(f"   Domain: {action.target_domain}")
        print(f"   Confidence: {action.confidence_threshold:.3f}")

        # Test feedback mechanism
        print(f"\n🔄 Testing bidirectional feedback...")
        system.provide_action_feedback(action_success=True, performance_score=0.8)

        # Test again to see weight adaptation
        action2 = system.translate_consciousness_to_action(sample_consciousness.numpy())
        if action2:  # Safety check for second action too
            print(f"   Updated translation after feedback:")
            print(f"   Complexity: {action2.complexity_level:.3f} (was {action.complexity_level:.3f})")
            print(f"   Creativity: {action2.creativity_level:.3f} (was {action.creativity_level:.3f})")

    print(f"\n🎉 SEMIOTIC LEARNING MODEL READY!")
    print(f"🚀 Can now translate consciousness → computational language!")
    if platform:
        print(f"🔗 Platform integration: Dynamic distinction levels active!")

    return system

if __name__ == "__main__":
    # Train the model (can optionally pass platform)
    trained_system = train_semiotic_learning_model()

    # To use with platform:
    # trained_system = train_semiotic_learning_model(platform=your_platform_instance)

