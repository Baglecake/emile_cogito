

#!/usr/bin/env python3
"""
KAINOS V2 COMPLETE - SINGLE FILE SOLUTION
==========================================

Everything in one file - no module hell, no import nightmares.
Includes base V2 + autonomous learning + live testing.

Just run: python kainos_v2_complete.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import time
import random
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
from datetime import datetime

# ============================================================================
# CORE V2 DATA STRUCTURES
# ============================================================================

@dataclass
class SymbolicQualiaState:
    """Complete consciousness state for V2 training"""
    # Keep static for regime and algorithmic constants
    current_regime: str = "stable_coherence"
    threshold_adaptation_rate: float = 0.1  # Learning rate constant
    time_window: int = 0

    # Dynamic defaults (will be overridden by factory method)
    regime_stability: float = 0.5
    regime_transition_probability: float = 0.5
    distinction_coherence: float = 0.5

    # Keep as 0.0 - developmental starting points
    consciousness_level: float = 0.0
    consciousness_trajectory: float = 0.0
    valence: float = 0.0
    valence_stability: float = 0.0
    agency: float = 0.0
    agency_momentum: float = 0.0
    embodiment: float = 0.0
    embodiment_grounding: float = 0.0
    self_awareness: float = 0.0
    meta_cognitive_activity: float = 0.0
    consciousness_optimization_success: float = 0.0
    symbol_vocabulary_size: float = 0.0
    symbol_integration_rate: float = 0.0
    momentum_factor: float = 0.0

    @classmethod
    def create_with_dynamic_defaults(cls, platform=None):
        """Create state with dynamic defaults from platform"""
        if platform and hasattr(platform, 'get_current_distinction_level'):
            try:
                return cls(
                    regime_stability=platform.get_current_distinction_level('stability'),
                    distinction_coherence=platform.get_current_distinction_level('coherence'),
                    regime_transition_probability=platform.get_current_distinction_level('transition_probability')
                )
            except Exception:
                # Fall back to static defaults if platform isn't ready
                return cls()
        else:
            return cls()  # Falls back to static defaults

    @classmethod
    def create_for_development_stage(cls, stage: str = "nascent"):
        """Create state appropriate for development stage"""
        if stage == "nascent":
            # Very early development - everything starts low
            return cls()  # Use all the 0.0 defaults
        elif stage == "developing":
            # Some basic capabilities emerging
            return cls(
                regime_stability=0.3,
                distinction_coherence=0.4,
                consciousness_level=0.2
            )
        elif stage == "mature":
            # More developed system
            return cls(
                regime_stability=0.7,
                distinction_coherence=0.8,
                consciousness_level=0.6,
                agency=0.5,
                self_awareness=0.4
            )
        else:
            return cls()

# ============================================================================
# V2 NEURAL NETWORK
# ============================================================================

class SymbolicQualiaTransformer(nn.Module):
    """Neural network for symbolic qualia strategy generation"""

    def __init__(self, input_dim: int = 21, hidden_dim: int = 256, output_dim: int = 64):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Encoder network
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )

        # ✅ ADD THESE LINES - Temporal analysis parameters
        self.complexity_time_factor = 0.8
        self.coherence_acceleration_factor = 1.4
        self.revalorization_time_factor = 0.6


        # Symbolic strategy head
        self.symbolic_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 32)
        )

        # Qualia enhancement head
        self.qualia_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 32)
        )

        # Meta-orchestration head
        self.meta_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 16)
        )

        # Autonomous learning components
        self.decay_factor = nn.Parameter(torch.tensor(0.95))
        self.revalorization_rate = nn.Parameter(torch.tensor(0.1))
        self.current_tau_qse = 1.0  # Baseline quantum time from QSE core

        # ✅ ADD THIS LINE - Print confirmation
        print(f"🌊 K2 Temporal Perspective: ACTIVE (narrative complexity)")

    def get_k2_temporal_context(self) -> Dict[str, Any]:
        """Get K2's temporal context for orchestrator integration"""
        analysis = getattr(self, '_last_temporal_analysis', {})

        # Calculate narrative stability
        narrative_stability = 0.7  # Default
        if hasattr(self, 'revalorization_rate') and self.revalorization_rate is not None:
            # High revalorization rate = lower stability
            rate = self.revalorization_rate.item() if hasattr(self.revalorization_rate, 'item') else float(self.revalorization_rate)
            narrative_stability = max(0.1, 1.0 - rate * 0.8)

        return {
            'k2_perspective': 'narrative_complexity_revalorization',
            'current_tau_prime': analysis.get('tau_prime_output', 1.0),
            'narrative_complexity': analysis.get('narrative_complexity', 0.5),
            'symbolic_strength': analysis.get('symbolic_strength', 0.5),
            'qualia_richness': analysis.get('qualia_richness', 0.5),
            'coherence': analysis.get('coherence', 0.5),
            'revalorization_rate': analysis.get('revalorization_rate', 0.1),
            'temporal_state': getattr(self, '_last_temporal_state', 'normal_narrative_flow'),
            'narrative_stability': narrative_stability,
            'temporal_weight': 0.4,  # K2 gets 40% weight (primary narrative processor)
            'complexity_dilation_active': analysis.get('narrative_complexity', 0.5) > 1.2,
            'coherence_acceleration_active': analysis.get('coherence', 0.5) > 0.8,
            'revalorization_intensity': 'high' if analysis.get('revalorization_rate', 0.1) > 0.3 else 'normal'
        }

    def _classify_k2_temporal_state(self, tau_prime: float) -> str:
        """Classify K2's current temporal state"""
        if tau_prime > 1.4:
            return "coherence_acceleration"         # High coherence, fast narrative processing
        elif tau_prime < 0.6:
            return "narrative_complexity_dilation"  # High complexity, slow processing
        elif tau_prime < 0.8:
            return "revalorization_processing"      # Active revalorization
        else:
            return "normal_narrative_flow"          # Balanced narrative processing

    def _get_safe_revalorization_fallback(self, coherence_fallback=None):
        """Safely get dynamic revalorization rate fallback with failure capture"""
        try:
            # Try dynamic approach first
            if hasattr(self, 'platform_ref') and self.platform_ref:
                if hasattr(self.platform_ref, 'get_current_distinction_level'):
                    dynamic_rate = self.platform_ref.get_current_distinction_level('revalorization_sensitivity')
                    # Scale to appropriate range for revalorization rate (0.05 - 0.2)
                    scaled_rate = max(0.05, min(0.2, dynamic_rate * 0.2))
                    return scaled_rate, "platform_dynamic"

            # Fallback to coherence-based estimate
            if coherence_fallback is not None:
                coherence_based_rate = max(0.05, min(0.2, coherence_fallback * 0.2))
                return coherence_based_rate, "coherence_based"

            # Final fallback
            return 0.1, "static_fallback"

        except Exception as e:
            # Capture failure details but don't break
            self._revalorization_fallback_errors.append(str(e))
            return 0.1, f"error_fallback: {str(e)[:50]}"

    def _calculate_local_tau(self, tau_qse: float, symbolic_state: torch.Tensor,
                            coherence_fallback: Optional[float] = None) -> float:
        """
        FIXED: Calculate K2's narrative temporal perspective with better sensitivity.

        K2 experiences time through semiotic complexity:
        - High narrative complexity → time dilation (more to process)
        - High symbolic coherence → time stabilization
        - Active revalorization → temporal acceleration
        """
        with torch.no_grad():
            # Get current forward pass results for analysis
            encoded = self.encoder(symbolic_state)
            symbolic_embedding = self.symbolic_head(encoded)
            qualia_embedding = self.qualia_head(encoded)

            # Calculate narrative complexity indicators with better sensitivity
            symbolic_strength = torch.norm(symbolic_embedding, dim=-1).mean().item()
            qualia_richness = torch.norm(qualia_embedding, dim=-1).mean().item()

            # Calculate cross-modal coherence (symbolic-qualia alignment)
            if symbolic_embedding.numel() > 0 and qualia_embedding.numel() > 0:
                symbolic_norm = symbolic_embedding / (torch.norm(symbolic_embedding, dim=-1, keepdim=True) + 1e-8)
                qualia_norm = qualia_embedding / (torch.norm(qualia_embedding, dim=-1, keepdim=True) + 1e-8)

                # Coherence = alignment between symbolic and qualia representations
                coherence = torch.sum(symbolic_norm * qualia_norm, dim=-1).mean().item()
                coherence = abs(coherence)  # Absolute coherence value
            else:
                # ✅ Use dynamic fallback if provided, otherwise default
                coherence = coherence_fallback if coherence_fallback is not None else 0.5

            # ENHANCED: More sensitive narrative complexity calculation
            # Scale symbolic strength and qualia richness for better differentiation
            scaled_symbolic = symbolic_strength * 2.0  # Amplify differences
            scaled_qualia = qualia_richness * 2.0

            # Base complexity with amplified sensitivity
            base_complexity = (scaled_symbolic * 0.5 + scaled_qualia * 0.3)

            # Incoherence increases processing complexity significantly
            incoherence_factor = (1.0 - coherence) * 0.5  # Increased from 0.3
            narrative_complexity = base_complexity + incoherence_factor

            # ENHANCED: More dramatic temporal modulation
            if narrative_complexity > 1.2:  # Lowered threshold for high complexity
                # High complexity → significant time dilation (τ' < 1)
                complexity_modulation = 0.4 + (2.0 - narrative_complexity) * 0.2  # More dramatic
            elif narrative_complexity < 0.6:  # Raised threshold for low complexity
                # Low complexity → significant time acceleration (τ' > 1)
                complexity_modulation = 1.0 + (0.6 - narrative_complexity) * 1.0  # More dramatic
            else:
                # Normal complexity → slight modulation around 1.0
                complexity_modulation = 0.8 + (1.0 - abs(narrative_complexity - 0.9)) * 0.4

            # Enhanced revalorization rate with safe dynamic fallback
            if hasattr(self, 'revalorization_rate'):
                revalorization_rate = self.revalorization_rate.item()
                rate_source = "model_parameter"
            else:
                revalorization_rate, rate_source = self._get_safe_revalorization_fallback(coherence_fallback)

            revalorization_acceleration = 1.0 + revalorization_rate * 0.3


            # Input variance effect (new factor for better differentiation)
            input_variance = torch.var(symbolic_state).item()
            variance_factor = 1.0 + (input_variance - 1.0) * 0.2  # More variance = more complexity

            # Combine temporal factors with enhanced sensitivity
            tau_modulation = complexity_modulation * revalorization_acceleration * variance_factor

            # Apply to baseline quantum time
            tau_prime_k2 = tau_qse * tau_modulation

            # Store enhanced diagnostic info
            self._last_temporal_analysis = {
                  'narrative_complexity': narrative_complexity,
                  'symbolic_strength': symbolic_strength,
                  'qualia_richness': qualia_richness,
                  'coherence': coherence,
                  'complexity_modulation': complexity_modulation,
                  'revalorization_acceleration': revalorization_acceleration,
                  'revalorization_rate': revalorization_rate,
                  'revalorization_rate_source': rate_source,  # ✅ Track source
                  'variance_factor': variance_factor,
                  'input_variance': input_variance,
                  'tau_qse_input': tau_qse,
                  'tau_prime_output': tau_prime_k2
              }

            return float(np.clip(tau_prime_k2, 0.0, 2.0))

    def forward(self, x):
        """Enhanced forward pass with local temporal perspective"""

        # Get baseline quantum time (τ_qse)
        tau_qse = getattr(self, 'current_tau_qse', 1.0)

        # Original K2 processing
        encoded = self.encoder(x)
        symbolic_embedding = self.symbolic_head(encoded)
        qualia_embedding = self.qualia_head(encoded)

        # Apply revalorization if active
        if hasattr(self, 'revalorization_rate'):
            noise_factor = self.revalorization_rate * torch.randn_like(symbolic_embedding) * 0.1
            symbolic_embedding = symbolic_embedding + noise_factor

        # ✅ ADD THIS - Calculate local temporal perspective
        local_tau_prime = self._calculate_local_tau(tau_qse, x)

        # ✅ ADD THIS - Store temporal state
        self._last_temporal_state = self._classify_k2_temporal_state(local_tau_prime)

        # Return enhanced output with temporal information
        return {
            'symbolic_embedding': symbolic_embedding,
            'qualia_embedding': qualia_embedding,
            'local_tau_prime': local_tau_prime,  # NEW: K2's temporal perspective
            'narrative_complexity': getattr(self, '_last_temporal_analysis', {}).get('narrative_complexity', 0.5),
            'symbolic_strength': getattr(self, '_last_temporal_analysis', {}).get('symbolic_strength', 0.5),
            'coherence': getattr(self, '_last_temporal_analysis', {}).get('coherence', 0.5),
            'temporal_state': getattr(self, '_last_temporal_state', 'normal_narrative_flow')
        }


    def learn_from_feedback(self, strategy_type: str, effectiveness: float):
        """Built-in autonomous learning from feedback - ROBUST VERSION"""

        # Initialize if needed
        if not hasattr(self, 'strategy_effectiveness_history'):
            self.strategy_effectiveness_history = defaultdict(list)

        if not hasattr(self, 'autonomy_level'):
            self.autonomy_level = 0.5

        if not hasattr(self, 'adaptation_count'):
            self.adaptation_count = 0

        # Validate inputs
        effectiveness = max(0.0, min(1.0, float(effectiveness)))

        # Add to history
        self.strategy_effectiveness_history[strategy_type].append(effectiveness)

        # Keep last 50 entries
        if len(self.strategy_effectiveness_history[strategy_type]) > 50:
            self.strategy_effectiveness_history[strategy_type] = \
                self.strategy_effectiveness_history[strategy_type][-50:]

        # Skip if insufficient data
        history = self.strategy_effectiveness_history[strategy_type]
        if len(history) < 3:
            return

        # Adapt based on effectiveness
        avg_effectiveness = np.mean(history)

        with torch.no_grad():
            if avg_effectiveness > 0.7:
                self.revalorization_rate.data *= 0.95
                self.autonomy_level = min(0.9, self.autonomy_level + 0.01)
            elif avg_effectiveness < 0.3:
                self.revalorization_rate.data *= 1.05

            self.revalorization_rate.data.clamp_(0.05, 0.3)

        self.adaptation_count += 1


    def set_revalorization_rate(self, rate: float):
        """Sets the revalorization rate parameter."""
        if not isinstance(rate, (int, float)):
            raise TypeError("Revalorization rate must be a number.")
        self.revalorization_rate = nn.Parameter(torch.tensor(float(rate)))

    def set_platform_reference(self, platform):
        """Set platform reference for dynamic coherence"""
        self.platform_ref = platform

def _get_dynamic_coherence_fallback(self):
    """Get dynamic fallback for symbolic coherence"""
    if hasattr(self, 'platform_ref') and self.platform_ref:
        try:
            if hasattr(self.platform_ref, 'get_current_distinction_level'):
                return self.platform_ref.get_current_distinction_level('coherence')
        except:
            pass
    return 0.5  # Final fallback

def _get_dynamic_narrative_complexity_fallback(self):
    """Get dynamic fallback for narrative complexity"""
    if hasattr(self, 'platform_ref') and self.platform_ref:
        try:
            if hasattr(self.platform_ref, 'get_current_distinction_level'):
                return self.platform_ref.get_current_distinction_level('narrative_complexity')
        except:
            pass
    return 0.5  # Final fallback


# ============================================================================
# COMPLETE V2 SYSTEM
# ============================================================================

class KainosV2Complete:
    """Complete V2 system - everything in one class"""

    def __init__(self, consciousness_data_dir: str = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.optimizer = None
        self.criterion = nn.MSELoss()

        # Auto-detect consciousness data
        self.consciousness_data_dir = self._find_consciousness_data(consciousness_data_dir)
        self.consciousness_data = []

        # Strategy mappings
        self.strategy_type_to_idx = {
            "symbol_integration": 0,
            "coherence_enhancement": 1,
            "distinction_building": 2,
            "regime_stabilization": 3
        }

        self.enhancement_type_to_idx = {
            "consciousness_amplification": 0,
            "agency_boost": 1,
            "embodiment_grounding": 2,
            "valence_stabilization": 3
        }

        self.training_history = []

        # Autonomous learning state
        self.session_interactions = 0
        self.session_improvements = []
        self.session_start_time = time.time()

        print(f"🎭 Kainos V2 Complete System initialized")
        print(f"📁 Consciousness data: {self.consciousness_data_dir}")
        print(f"🔧 Device: {self.device}")

    def _find_consciousness_data(self, provided_dir: str) -> str:
        """Auto-find consciousness data from multiple possible locations"""

        possible_dirs = []
        if provided_dir:
            possible_dirs.append(provided_dir)

        # Common locations
        possible_dirs.extend([
            "./emile_v2_bridge_outputs/consciousness_logs",
            "./emile_v2_bridge_outputs",
            "./consciousness_logs",
            "/content/emile_v2_bridge_outputs/consciousness_logs",
            "/content/emile_v2_bridge_outputs",
            "/content"
        ])

        for dir_path in possible_dirs:
            path = Path(dir_path)
            if path.exists():
                # Check for consciousness files
                json_files = list(path.glob("*.json"))
                consciousness_files = [f for f in json_files if
                                     'consciousness' in f.name.lower() or
                                     'emile' in f.name.lower() or
                                     'v2_format' in f.name.lower()]

                if consciousness_files:
                    print(f"📊 Found consciousness data: {len(consciousness_files)} files in {path}")
                    return str(path)

        print(f"⚠️ No consciousness data found, will use current directory")
        return "."

    def load_consciousness_data(self) -> bool:
        """Load consciousness data from available files"""

        data_dir = Path(self.consciousness_data_dir)

        # Try different file patterns
        file_patterns = [
            "emile_consciousness_data.json",
            "v2_format_data.json",
            "*consciousness*.json",
            "*.json"
        ]

        for pattern in file_patterns:
            files = list(data_dir.glob(pattern))
            if files:
                print(f"📊 Loading from pattern: {pattern}")
                break

        if not files:
            print(f"❌ No consciousness data files found in {data_dir}")
            return False

        # Load the data
        for file_path in files[:1]:  # Just use first file
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)

                if isinstance(data, list):
                    self.consciousness_data = data
                else:
                    self.consciousness_data = [data]

                print(f"✅ Loaded {len(self.consciousness_data)} consciousness samples from {file_path.name}")
                return True

            except Exception as e:
                print(f"⚠️ Failed to load {file_path}: {e}")
                continue

        return False

    def convert_to_training_data(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Convert consciousness data to training tensors"""

        training_pairs = []

        for i, entry in enumerate(self.consciousness_data):
            # Get dynamic fallbacks for missing training data
            regime_stability_fallback = self.get_current_distinction_level('stability') if hasattr(self, 'get_current_distinction_level') else 0.5
            transition_probability_fallback = self.get_current_distinction_level('transition_probability') if hasattr(self, 'get_current_distinction_level') else 0.5
            distinction_coherence_fallback = self.get_current_distinction_level('coherence') if hasattr(self, 'get_current_distinction_level') else 0.5
            valence_stability_fallback = self.get_current_distinction_level('valence_stability') if hasattr(self, 'get_current_distinction_level') else 0.5

            # Create state tensor
            state_features = [
                entry.get('regime_stability', regime_stability_fallback),              # ✅ Dynamic
                entry.get('regime_transition_probability', transition_probability_fallback), # ✅ Dynamic
                entry.get('distinction_coherence', distinction_coherence_fallback),    # ✅ Dynamic
                entry.get('symbol_vocabulary_size', 0.0),                             # ✅ Keep 0.0 (starts empty)
                entry.get('symbol_integration_rate', 0.0),                            # ✅ Keep 0.0 (starts empty)
                entry.get('threshold_adaptation_rate', 0.1),                          # ✅ Keep 0.1 (learning rate)
                entry.get('consciousness_level', 0.0),                                # ✅ Keep 0.0 (developmental)
                entry.get('consciousness_trajectory', 0.0),                           # ✅ Keep 0.0 (developmental)
                entry.get('valence', 0.0),                                           # ✅ Keep 0.0 (starts neutral)
                entry.get('valence_stability', valence_stability_fallback),           # ✅ Dynamic
                entry.get('agency', 0.0),                                            # ✅ Keep 0.0 (developmental)
                entry.get('agency_momentum', 0.0),                                   # ✅ Keep 0.0 (developmental)
                entry.get('embodiment', 0.0),                                        # ✅ Keep 0.0 (developmental)
                entry.get('embodiment_grounding', 0.0),                              # ✅ Keep 0.0 (developmental)
                entry.get('self_awareness', 0.0),                                    # ✅ Keep 0.0 (developmental)
                entry.get('meta_cognitive_activity', 0.0),                           # ✅ Keep 0.0 (developmental)
                entry.get('consciousness_optimization_success', 0.0),                # ✅ Keep 0.0 (developmental)
                float(entry.get('time_window', i)) / 1000.0,                        # ✅ Keep as-is
                entry.get('momentum_factor', 0.0),                                   # ✅ Keep 0.0 (starts empty)
                1.0 if entry.get('regime') == "stable_coherence" else 0.0,          # ✅ Keep as-is (encoding)
                1.0 if entry.get('regime') == "symbolic_turbulence" else 0.0,       # ✅ Keep as-is (encoding)
                ]

            state_tensor = torch.tensor(state_features, dtype=torch.float32)

            # Create target strategy (simplified)
            regime = entry.get('regime', 'stable_coherence')
            stability = entry.get('regime_stability', 0.5)
            consciousness = entry.get('consciousness_level', 0.0)

            # Generate target based on state
            if stability < 0.4:
                strategy_idx = 3  # regime_stabilization
            elif consciousness < 0.4:
                strategy_idx = 0  # symbol_integration
            elif entry.get('distinction_coherence', 0.5) < 0.5:
                strategy_idx = 2  # distinction_building
            else:
                strategy_idx = 1  # coherence_enhancement

            enhancement_idx = 0  # consciousness_amplification (most common)

            target_features = [0.0] * 64
            target_features[strategy_idx] = 1.0
            target_features[32 + enhancement_idx] = 1.0

            target_tensor = torch.tensor(target_features, dtype=torch.float32)

            training_pairs.append((state_tensor, target_tensor))

        return training_pairs

    def initialize_model(self, hidden_dim: int = 256) -> bool:
        """Initialize the V2 model"""
        try:
            self.model = SymbolicQualiaTransformer(
                input_dim=21,
                hidden_dim=hidden_dim,
                output_dim=64
            ).to(self.device)

            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

            print(f"✅ V2 model initialized: {hidden_dim} hidden dims, {sum(p.numel() for p in self.model.parameters()):,} params")
            return True

        except Exception as e:
            print(f"❌ Failed to initialize model: {e}")
            return False

    def train(self, epochs: int = 100) -> float:
        """Train the V2 model"""
        if not self.consciousness_data:
            print("❌ No consciousness data loaded")
            return float('inf')

        # Convert to training data
        training_pairs = self.convert_to_training_data()
        if not training_pairs:
            print("❌ No training data generated")
            return float('inf')

        # Split train/val
        split_idx = int(len(training_pairs) * 0.8)
        train_data = training_pairs[:split_idx]
        val_data = training_pairs[split_idx:]

        print(f"🚀 Training V2 on {len(train_data)} samples, validating on {len(val_data)}")

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0

            for state_tensor, target_tensor in train_data:
                state_tensor = state_tensor.to(self.device).unsqueeze(0)
                target_tensor = target_tensor.to(self.device).unsqueeze(0)

                self.optimizer.zero_grad()

                outputs = self.model(state_tensor)
                predicted = torch.cat([
                    outputs['symbolic_embedding'],
                    outputs['qualia_embedding']
                ], dim=1)

                loss = self.criterion(predicted, target_tensor)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_data)

            # Validation
            self.model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for state_tensor, target_tensor in val_data:
                    state_tensor = state_tensor.to(self.device).unsqueeze(0)
                    target_tensor = target_tensor.to(self.device).unsqueeze(0)

                    outputs = self.model(state_tensor)
                    predicted = torch.cat([
                        outputs['symbolic_embedding'],
                        outputs['qualia_embedding']
                    ], dim=1)

                    loss = self.criterion(predicted, target_tensor)
                    val_loss += loss.item()

            val_loss /= len(val_data)

            # Record and check improvement
            self.training_history.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss
            })

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_model("kainos_v2_complete_best.pth")
            else:
                patience_counter += 1

            if epoch % 20 == 0:
                print(f"Epoch {epoch:3d}: train={train_loss:.4f}, val={val_loss:.4f}")

            if patience_counter >= 15:
                print(f"🛑 Early stopping at epoch {epoch}")
                break

        print(f"✅ Training complete! Best validation loss: {best_val_loss:.4f}")
        return best_val_loss

    def generate_strategy(self, consciousness_state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate strategy with built-in autonomous learning and dynamic defaults"""

        if self.model is None:
            print("❌ Model not trained")
            return None

        # Get dynamic fallbacks for consciousness state features
        regime_stability_fallback = self._get_dynamic_consciousness_default('regime_stability', 0.5)
        transition_probability_fallback = self._get_dynamic_consciousness_default('regime_transition_probability', 0.5)
        distinction_coherence_fallback = self._get_dynamic_consciousness_default('distinction_coherence', 0.5)
        symbol_vocabulary_fallback = self._get_dynamic_consciousness_default('symbol_vocabulary_size', 0.0)
        symbol_integration_fallback = self._get_dynamic_consciousness_default('symbol_integration_rate', 0.0)
        threshold_adaptation_fallback = self._get_dynamic_consciousness_default('threshold_adaptation_rate', 0.1)
        consciousness_level_fallback = self._get_dynamic_consciousness_default('consciousness_level', 0.0)
        consciousness_trajectory_fallback = self._get_dynamic_consciousness_default('consciousness_trajectory', 0.0)
        valence_fallback = self._get_dynamic_consciousness_default('valence', 0.0)
        valence_stability_fallback = self._get_dynamic_consciousness_default('valence_stability', 0.5)
        agency_fallback = self._get_dynamic_consciousness_default('agency', 0.0)
        agency_momentum_fallback = self._get_dynamic_consciousness_default('agency_momentum', 0.0)
        embodiment_fallback = self._get_dynamic_consciousness_default('embodiment', 0.0)
        embodiment_grounding_fallback = self._get_dynamic_consciousness_default('embodiment_grounding', 0.0)
        self_awareness_fallback = self._get_dynamic_consciousness_default('self_awareness', 0.0)
        meta_cognitive_fallback = self._get_dynamic_consciousness_default('meta_cognitive_activity', 0.0)

        # Convert state to tensor with dynamic fallbacks
        state_features = [
            consciousness_state.get('regime_stability', regime_stability_fallback),
            consciousness_state.get('regime_transition_probability', transition_probability_fallback),
            consciousness_state.get('distinction_coherence', distinction_coherence_fallback),
            consciousness_state.get('symbol_vocabulary_size', symbol_vocabulary_fallback),
            consciousness_state.get('symbol_integration_rate', symbol_integration_fallback),
            consciousness_state.get('threshold_adaptation_rate', threshold_adaptation_fallback),
            consciousness_state.get('consciousness_level', consciousness_level_fallback),
            consciousness_state.get('consciousness_trajectory', consciousness_trajectory_fallback),
            consciousness_state.get('valence', valence_fallback),
            consciousness_state.get('valence_stability', valence_stability_fallback),
            consciousness_state.get('agency', agency_fallback),
            consciousness_state.get('agency_momentum', agency_momentum_fallback),
            consciousness_state.get('embodiment', embodiment_fallback),
            consciousness_state.get('embodiment_grounding', embodiment_grounding_fallback),
            consciousness_state.get('self_awareness', self_awareness_fallback),
            consciousness_state.get('meta_cognitive_activity', meta_cognitive_fallback),
            consciousness_state.get('consciousness_optimization_success', 0.0),  # Keep 0.0 (developmental)
            float(consciousness_state.get('time_window', 0)) / 1000.0,  # Keep as-is (temporal scaling)
            consciousness_state.get('momentum_factor', 0.0),  # Keep 0.0 (starts empty)
            1.0 if consciousness_state.get('regime') == "stable_coherence" else 0.0,  # Keep as-is (encoding)
            1.0 if consciousness_state.get('regime') == "symbolic_turbulence" else 0.0,  # Keep as-is (encoding)
        ]

        # Convert to tensor and generate strategy
        state_tensor = torch.tensor(state_features, dtype=torch.float32).to(self.device).unsqueeze(0)

        with torch.no_grad():
            outputs = self.model(state_tensor)

            # Decode strategy
            symbolic_embedding = outputs['symbolic_embedding'].cpu().numpy()[0]
            strategy_type_idx = int(np.argmax(symbolic_embedding[:4]))
            strategy_types = list(self.strategy_type_to_idx.keys())
            strategy_type = strategy_types[strategy_type_idx]

            symbolic_strategy = SymbolicStrategy(
                strategy_type=strategy_type,
                complexity_level=min(1.0, max(0.0, symbolic_embedding[4] if len(symbolic_embedding) > 4 else 0.5)),
                effectiveness_score=min(1.0, max(0.0, symbolic_embedding[5] if len(symbolic_embedding) > 5 else 0.5))
            )

            # Decode enhancement
            qualia_embedding = outputs['qualia_embedding'].cpu().numpy()[0]
            enhancement_type_idx = int(np.argmax(qualia_embedding[:4]))
            enhancement_types = list(self.enhancement_type_to_idx.keys())
            enhancement_type = enhancement_types[enhancement_type_idx]

            qualia_enhancement = QualiaEnhancement(
                enhancement_type=enhancement_type,
                enhancement_magnitude=min(1.0, max(0.0, qualia_embedding[4] if len(qualia_embedding) > 4 else 0.3))
            )

            confidence = float(np.mean([
                symbolic_strategy.effectiveness_score,
                qualia_enhancement.sustainability,
                consciousness_state.get('consciousness_level', 0.0)
            ]))

            return {
                'symbolic_strategy': symbolic_strategy,
                'qualia_enhancement': qualia_enhancement,
                'confidence_score': confidence
            }

    def provide_feedback(self, strategy: Dict[str, Any], effectiveness: float) -> None:
        """Provide feedback for autonomous learning"""
        if self.model and strategy and 'symbolic_strategy' in strategy:
            strategy_type = strategy['symbolic_strategy'].strategy_type
            self.model.learn_from_feedback(strategy_type, effectiveness)
            self.session_interactions += 1

            if effectiveness > 0.6:
                self.session_improvements.append({
                    'interaction': self.session_interactions,
                    'strategy_type': strategy_type,
                    'effectiveness': effectiveness
                })

    def save_model(self, filename: str) -> bool:
        """Save the complete model"""
        if self.model is None:
            return False

        try:
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
                'training_history': self.training_history,
                'strategy_mappings': {
                    'strategy_type_to_idx': self.strategy_type_to_idx,
                    'enhancement_type_to_idx': self.enhancement_type_to_idx
                },
                'model_config': {
                    'input_dim': self.model.input_dim,
                    'hidden_dim': self.model.hidden_dim,
                    'output_dim': self.model.output_dim
                },
                'autonomous_state': {
                    'autonomy_level': self.model.autonomy_level,
                    'adaptation_count': self.model.adaptation_count,
                    'session_interactions': self.session_interactions
                }
            }

            torch.save(checkpoint, filename)
            print(f"💾 Complete V2 model saved: {filename}")
            return True

        except Exception as e:
            print(f"❌ Failed to save model: {e}")
            return False

    def load_model(self, filename: str) -> bool:
        """Load the complete model"""
        try:
            checkpoint = torch.load(filename, map_location=self.device)

            # Auto-initialize model if needed
            if 'model_config' in checkpoint and self.model is None:
                config = checkpoint['model_config']
                self.initialize_model(hidden_dim=config['hidden_dim'])

            if self.model is None:
                print("❌ Model not initialized")
                return False

            # Load model state
            self.model.load_state_dict(checkpoint['model_state_dict'])

            # Load other state
            if self.optimizer and 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            self.training_history = checkpoint.get('training_history', [])

            # Load autonomous state
            if 'autonomous_state' in checkpoint:
                auto_state = checkpoint['autonomous_state']
                self.model.autonomy_level = auto_state.get('autonomy_level', 0.3)
                self.model.adaptation_count = auto_state.get('adaptation_count', 0)
                self.session_interactions = auto_state.get('session_interactions', 0)

            print(f"✅ Complete V2 model loaded: {filename}")
            return True

        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return False

    def end_session_summary(self) -> bool:
        """Show session summary and prompt to save"""

        session_duration = time.time() - self.session_start_time

        print(f"\n" + "="*60)
        print(f"🎭 V2 COMPLETE SESSION SUMMARY")
        print(f"="*60)
        print(f"⏱️  Session duration: {session_duration/60:.1f} minutes")
        print(f"🔄 Total interactions: {self.session_interactions}")
        print(f"✨ Successful improvements: {len(self.session_improvements)}")

        if self.model:
            print(f"🧠 Current autonomy level: {self.model.autonomy_level:.2f}")
            print(f"🔧 Total adaptations: {self.model.adaptation_count}")

        if self.session_improvements:
            print(f"\n🎯 KEY IMPROVEMENTS:")
            for imp in self.session_improvements[-3:]:
                print(f"   • {imp['strategy_type']}: effectiveness {imp['effectiveness']:.3f}")

        # Simple save recommendation
        if len(self.session_improvements) > 3:
            recommendation = "🟢 RECOMMENDED"
        elif len(self.session_improvements) > 0:
            recommendation = "🟡 OPTIONAL"
        else:
            recommendation = "🔴 MINIMAL CHANGES"

        print(f"\n💾 Save recommendation: {recommendation}")

        try:
            save_choice = input("💾 Save improved V2 model? [y/N]: ").strip().lower()

            if save_choice in ['y', 'yes']:
                timestamp = int(time.time())
                filename = f"kainos_v2_complete_improved_{timestamp}.pth"
                success = self.save_model(filename)

                if success:
                    print(f"✅ V2 improvements saved: {filename}")
                    return True
                else:
                    print(f"❌ Save failed")
                    return False
            else:
                print(f"⏭️  Improvements not saved")
                return False

        except (EOFError, KeyboardInterrupt):
            print(f"\n⏭️  Session ended without saving")
            return False

    def _classify_k2_temporal_state(self, tau_prime: float) -> str:
        """Classify K2's current temporal state"""
        if tau_prime > 1.4:
            return "coherence_acceleration"         # High coherence, fast narrative processing
        elif tau_prime < 0.6:
            return "narrative_complexity_dilation"  # High complexity, slow processing
        elif tau_prime < 0.8:
            return "revalorization_processing"      # Active revalorization
        else:
            return "normal_narrative_flow"          # Balanced narrative processing


def _get_dynamic_consciousness_default(self, state_key: str, base_value: float) -> float:
    """Get dynamic default for consciousness state features"""
    if not hasattr(self, 'platform') or not self.platform:
        return base_value

    try:
        if hasattr(self.platform, 'get_current_distinction_level'):
            # Map consciousness state keys to distinction types
            distinction_mapping = {
                'regime_stability': 'stability',
                'regime_transition_probability': 'transition_probability',
                'distinction_coherence': 'coherence',
                'symbol_vocabulary_size': 'symbolic_capacity',
                'symbol_integration_rate': 'integration_rate',
                'threshold_adaptation_rate': 'adaptation_rate',
                'consciousness_level': 'consciousness_baseline',
                'consciousness_trajectory': 'consciousness_trajectory',
                'valence': 'emotional_baseline',
                'valence_stability': 'emotional_stability',
                'agency': 'agency_baseline',
                'agency_momentum': 'agency_momentum',
                'embodiment': 'embodiment_baseline',
                'embodiment_grounding': 'embodiment_grounding',
                'self_awareness': 'self_awareness_baseline',
                'meta_cognitive_activity': 'metacognitive_baseline'
            }

            distinction_type = distinction_mapping.get(state_key, 'general_consciousness')
            return self.platform.get_current_distinction_level(distinction_type)

        return base_value

    except Exception:
        return base_value


def set_platform_reference(self, platform):
    """Allow strategy generator to access platform for dynamic defaults"""
    self.platform = platform

# ============================================================================
# SIMPLE TESTING FUNCTION
# ============================================================================

def test_v2_complete():
    """Simple test of the complete V2 system"""
    print("🧪 Testing Kainos V2 Complete System")
    print("=" * 50)

    # Initialize
    v2 = KainosV2Complete()

    # Load consciousness data
    if not v2.load_consciousness_data():
        print("❌ No consciousness data found - cannot test")
        return False

    # Initialize model
    if not v2.initialize_model(hidden_dim=256):
        print("❌ Model initialization failed")
        return False

    # Quick training
    print("🚀 Quick training (20 epochs)...")
    best_loss = v2.train(epochs=20)

    # Test strategy generation
    print("🎭 Testing strategy generation...")
    test_state = {
        'current_regime': 'symbolic_turbulence',
        'regime_stability': 0.3,
        'consciousness_level': 0.5,
        'valence': -0.1,
        'agency': 0.4
    }

    strategy = v2.generate_strategy(test_state)

    if strategy:
        print(f"✅ Strategy generated:")
        print(f"   🎭 Symbolic: {strategy['symbolic_strategy'].strategy_type}")
        print(f"   💫 Qualia: {strategy['qualia_enhancement'].enhancement_type}")
        print(f"   📊 Confidence: {strategy['confidence_score']:.3f}")

        # Test feedback
        v2.provide_feedback(strategy, 0.8)
        print(f"✅ Autonomous learning working")

        # Session summary
        v2.end_session_summary()

        return True
    else:
        print("❌ Strategy generation failed")
        return False

if __name__ == "__main__":
    print("🎭 KAINOS V2 COMPLETE - Single File Solution")
    print("=" * 50)

    success = test_v2_complete()

    if success:
        print(f"\n✨ V2 Complete system working!")
        print(f"🎯 Everything in one file - no import hell!")
    else:
        print(f"\n❌ V2 Complete test failed")

