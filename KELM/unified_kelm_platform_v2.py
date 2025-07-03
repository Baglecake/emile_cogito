

#!/usr/bin/env python3
"""
UNIFIED KELM PLATFORM - COMPLETE INTEGRATION
===========================================

This is the complete, production-ready KELM Platform that integrates:
- All 4 K-models (K1-K4) with proper loading and input generation
- All 16 core modules including missing kainos components
- Comprehensive seeding framework for reproducible consciousness
- Polytemporal consciousness coordination
- Full bidirectional orchestration with working data flow

Key fixes:
1. K4 metabolic model properly loaded
2. K1 receives actual flow data instead of being flatlined
3. Seeding mechanism integrated for deterministic development
4. All missing kainos modules integrated
5. Proper input generation for all K-models
"""


import sys
import os

# Fix Python path FIRST - before any other imports
sys.path.insert(0, '/content')
sys.path.insert(0, '/content/emile_cogito')

import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
import numpy as np
import random
import time
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import deque
from dataclasses import dataclass
from emile_cogito.kainos.config import CONFIG
from emile_cogito.kelm.continuous_temporal_k2_engine import ContinuousTemporalK2Engine
from emile_cogito.kelm.adaptive_k_theoria import SmartKModelLoader


# ========================
# COMPREHENSIVE SEEDING
# ========================

def set_comprehensive_seed(seed=42):
    """Set comprehensive seeds for fully deterministic consciousness development"""
    print(f"🌱 Setting comprehensive seed: {seed}")

    # Python random
    random.seed(seed)

    # NumPy random
    np.random.seed(seed)

    # PyTorch CPU
    torch.manual_seed(seed)

    # PyTorch CUDA (all GPUs)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # Ensure deterministic CUDA operations
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Additional CUDA determinism
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

        print(f"   ✅ CUDA seeded on {torch.cuda.device_count()} GPUs")

    # Additional determinism settings
    torch.use_deterministic_algorithms(True, warn_only=True)

    print(f"   ✅ Comprehensive seeding complete - consciousness will develop deterministically")


# ========================
# K-MODEL ARCHITECTURES
# ========================

class DynamicSemioticNetwork(torch.nn.Module):
    """K1 Praxis - Data flow model"""
    def __init__(self, input_dim=32, hidden_dim=64, output_dim=32):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)


class SymbolicQualiaTransformer(nn.Module):
    """K2 Semiosis - Correct implementation from your working k2.py"""

    def __init__(self, input_dim: int = 21, hidden_dim: int = 256, output_dim: int = 64):
        super().__init__()
        self.platform_ref = None
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

        # Built-in autonomous learning
        self.strategy_effectiveness_history = defaultdict(list)
        self.autonomy_level = 0.3
        self.adaptation_count = 0

    def forward(self, x):
        """Forward pass that handles continuous inputs properly"""

        # Handle input shape - your K2 expects 21 features, not 128
        if x.shape[1] != self.input_dim:
            # Pad or truncate to expected input dimension
            if x.shape[1] < self.input_dim:
                padding = torch.zeros(x.shape[0], self.input_dim - x.shape[1], device=x.device)
                x = torch.cat([x, padding], dim=1)
            else:
                x = x[:, :self.input_dim]

        # Original K2 processing
        encoded = self.encoder(x)
        symbolic_embedding = self.symbolic_head(encoded)
        qualia_embedding = self.qualia_head(encoded)

        # Apply revalorization if active
        if hasattr(self, 'revalorization_rate'):
            noise_factor = self.revalorization_rate * torch.randn_like(symbolic_embedding) * 0.1
            symbolic_embedding = symbolic_embedding + noise_factor

        # For compatibility with the unified platform, return just the symbolic embedding
        # (since that's what the platform expects)
        return symbolic_embedding

    def set_platform_reference(self, platform):
        """Allow K2 to access platform for dynamic values"""
        self.platform_ref = platform

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
            if hasattr(self, '_revalorization_fallback_errors'):
                self._revalorization_fallback_errors.append(str(e))
            else:
                self._revalorization_fallback_errors = [str(e)]
            return 0.1, f"error_fallback: {str(e)[:50]}"

    def _get_dynamic_narrative_complexity_fallback(self):
        """Get dynamic fallback for narrative complexity"""
        if hasattr(self, 'platform_ref') and self.platform_ref:
            try:
                if hasattr(self.platform_ref, 'get_current_distinction_level'):
                    return self.platform_ref.get_current_distinction_level('narrative_complexity')
            except:
                pass
        return 0.5  # Final fallback

    def _get_dynamic_coherence_fallback(self):
        """Get dynamic fallback for symbolic coherence"""
        if hasattr(self, 'platform_ref') and self.platform_ref:
            try:
                if hasattr(self.platform_ref, 'get_current_distinction_level'):
                    return self.platform_ref.get_current_distinction_level('coherence')
            except:
                pass
        return 0.5  # Final fallback

class QSEEmergenceNetwork(torch.nn.Module):
    """K3 Apeiron - Quantum dynamics"""
    def __init__(self, grid_size=16, hidden_dim=64):
        super().__init__()
        self.grid_size = grid_size
        self.processor = torch.nn.Sequential(
            torch.nn.Linear(grid_size * grid_size, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, grid_size * grid_size)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        # Ensure correct input shape
        if x.shape[1] != self.grid_size * self.grid_size:
            # Pad or truncate
            target_size = self.grid_size * self.grid_size
            if x.shape[1] < target_size:
                padding = torch.zeros(batch_size, target_size - x.shape[1])
                x = torch.cat([x, padding], dim=1)
            else:
                x = x[:, :target_size]

        return self.processor(x)

class MetabolicRegulationNetwork(torch.nn.Module):
    """K4 Metabolic - System driver"""
    def __init__(self, input_dim=32, hidden_dim=64, output_dim=16):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim),
            torch.nn.Sigmoid()  # Metabolic rates between 0-1
        )

    def forward(self, x):
        return self.net(x)

# ========================
# ENHANCED K-MODEL LOADER
# ========================

# ========================
# FIXED: ENHANCED K-MODEL LOADER
# ========================

class UnifiedKModelLoader:
    """Unified loader for all K-models with proper configuration"""

    def __init__(self, model_dir="/content/emile_cogito/k_models"):
        self.model_dir = Path(model_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.model_configs = {
            'k1': {'type': 'praxis', 'input_dim': 32, 'hidden_dim': 64, 'output_dim': 32},
            'k2': {'type': 'semiosis', 'input_dim': 21, 'hidden_dim': 256, 'output_dim': 64},
            'k3': {'type': 'apeiron', 'grid_size': 16, 'hidden_dim': 64},
            'k4': {'type': 'metabolic', 'input_dim': 32, 'hidden_dim': 64, 'output_dim': 16}
        }
        # REMOVED: Problematic assignment that was causing the error
        # self._patch_k2_missing_methods = ContinuousTemporalK2Engine._patch_k2_missing_methods

    def _patch_k2_missing_methods(self):
        """FIXED: Implement the K2 patching method directly in this class"""

        if 'k2' not in self.models:
            print("   ⚠️ No K2 model loaded - skipping patch")
            return

        k2_model = self.models['k2']
        print("   🔧 Patching K2 missing methods...")

        # Add the missing method dynamically
        def _get_dynamic_narrative_complexity_fallback(self, symbolic_flow=None, context=None):
            """Fallback method for narrative complexity calculation"""
            try:
                if symbolic_flow is not None and hasattr(symbolic_flow, 'mean'):
                    # Use actual symbolic flow if available
                    base_complexity = float(symbolic_flow.mean().item())
                elif context is not None and isinstance(context, dict):
                    # Use context if available
                    base_complexity = context.get('narrative_complexity', 0.5)
                else:
                    # Simple fallback
                    base_complexity = 0.5

                # Add some variation to make it dynamic
                import torch
                variation = torch.randn(1).item() * 0.1
                complexity = max(0.1, min(0.9, base_complexity + variation))

                return complexity

            except Exception as e:
                print(f"⚠️ Narrative complexity fallback error: {e}")
                return 0.5

        # Bind the method to the model instance
        import types
        k2_model._get_dynamic_narrative_complexity_fallback = types.MethodType(
            _get_dynamic_narrative_complexity_fallback, k2_model
        )

        # Add any other missing methods that might be needed
        def _get_symbol_integration_rate_fallback(self, context=None):
            """Fallback for symbol integration rate"""
            return 0.1

        def _get_threshold_adaptation_fallback(self, context=None):
            """Fallback for threshold adaptation"""
            return 0.1

        k2_model._get_symbol_integration_rate_fallback = types.MethodType(
            _get_symbol_integration_rate_fallback, k2_model
        )
        k2_model._get_threshold_adaptation_fallback = types.MethodType(
            _get_threshold_adaptation_fallback, k2_model
        )

        print("   ✅ K2 missing methods patched successfully")

    def load_all_models(self):
        """Load all 4 K-models"""
        print("📚 Loading K-Models...")

        for model_name, config in self.model_configs.items():
            success = self._load_model(model_name, config)
            if success:
                print(f"   ✅ {model_name.upper()} loaded successfully")
            else:
                print(f"   ❌ {model_name.upper()} failed to load")

        # FIXED: Apply patches after loading models
        if len(self.models) > 0:
            print("\n🔧 Applying K2 patches...")
            self._patch_k2_missing_methods()

        print(f"\n   📊 Total models loaded: {len(self.models)}/4")
        return len(self.models)

    def _load_model(self, model_name: str, config: Dict) -> bool:
        """Load individual K-model"""
        try:
            # Try different file patterns
            patterns = [
                f"{model_name}_*.pth",
                f"{model_name}.pth",
                f"{model_name}_model.pth"
            ]

            model_file = None
            for pattern in patterns:
                files = list(self.model_dir.glob(pattern))
                if files:
                    model_file = files[0]
                    break

            if not model_file:
                return False

            # Create model architecture
            if config['type'] == 'praxis':
                model = DynamicSemioticNetwork(**{k:v for k,v in config.items() if k != 'type'})
            elif config['type'] == 'semiosis':
                model = SymbolicQualiaTransformer(**{k:v for k,v in config.items() if k != 'type'})
            elif config['type'] == 'apeiron':
                model = QSEEmergenceNetwork(**{k:v for k,v in config.items() if k != 'type'})
            elif config['type'] == 'metabolic':
                model = MetabolicRegulationNetwork(**{k:v for k,v in config.items() if k != 'type'})
            else:
                return False

            # Load state dict
            state_dict = torch.load(model_file, map_location='cpu')
            if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']

            model.load_state_dict(state_dict, strict=False)
            model.eval()

            self.models[model_name] = model
            return True

        except Exception as e:
            print(f"      Error loading {model_name}: {e}")
            return False

    def generate_predictions(self, consciousness_state: Dict) -> Dict[str, torch.Tensor]:
        """FIXED: Generate predictions with robust tensor output handling"""

        predictions = {}

        # Apply K2 patches before making predictions
        self._patch_k2_missing_methods()

        for model_name, model in self.models.items():
            try:
                print(f"\n🔍 Processing {model_name}...")

                # Generate input tensor based on model requirements
                if model_name == 'k1':
                    input_tensor = self._create_k1_input_32dim(consciousness_state)
                elif model_name == 'k2':
                    input_tensor = self._create_k2_input_21dim(consciousness_state)
                elif model_name == 'k3':
                    input_tensor = self._create_k3_input_256dim(consciousness_state)
                elif model_name == 'k4':
                    input_tensor = self._create_k4_input_32dim(consciousness_state)
                else:
                    continue

                print(f"   Input tensor shape: {input_tensor.shape}")

                # Run model inference
                with torch.no_grad():
                    raw_output = model(input_tensor)

                # FIXED: Handle different output types robustly
                processed_tensor = None

                if isinstance(raw_output, torch.Tensor):
                    processed_tensor = raw_output
                elif isinstance(raw_output, dict):
                    # Extract primary tensor from dict
                    if 'output' in raw_output:
                        processed_tensor = raw_output['output']
                    elif 'predictions' in raw_output:
                        processed_tensor = raw_output['predictions']
                    else:
                        # Take first tensor value
                        for value in raw_output.values():
                            if isinstance(value, torch.Tensor):
                                processed_tensor = value
                                break

                if processed_tensor is not None:
                    # Store with standard naming convention
                    standard_name = f'{model_name}_praxis' if model_name == 'k1' else \
                                  f'{model_name}_semiosis' if model_name == 'k2' else \
                                  f'{model_name}_apeiron' if model_name == 'k3' else \
                                  f'{model_name}_metabolic'

                    predictions[standard_name] = processed_tensor
                    print(f"✅ {model_name} → {standard_name}: {processed_tensor.shape}")

            except Exception as e:
                print(f"❌ {model_name} prediction failed: {e}")
                import traceback
                traceback.print_exc()

                # Create fallback tensor
                target_dims = {'k1': 64, 'k2': 64, 'k3': 25, 'k4': 12}
                target_dim = target_dims.get(model_name, 32)
                fallback_tensor = torch.zeros(1, target_dim).to(self.device)

                standard_name = f'{model_name}_praxis' if model_name == 'k1' else \
                              f'{model_name}_semiosis' if model_name == 'k2' else \
                              f'{model_name}_apeiron' if model_name == 'k3' else \
                              f'{model_name}_metabolic'

                predictions[standard_name] = fallback_tensor
                print(f"❌ {model_name} → {standard_name}: {fallback_tensor.shape} (fallback)")
                continue

        return predictions

    def generate_deterministic_inputs(self, consciousness_state: Dict, seed_offset: int = 0) -> Dict:
        """Generate proper deterministic inputs for all K-models"""
        # Use seed offset for temporal variation while maintaining determinism
        local_seed = 42 + seed_offset
        torch.manual_seed(local_seed)
        np.random.seed(local_seed)

        consciousness_level = consciousness_state.get('consciousness_level', 0.5)
        valence = consciousness_state.get('valence', 0.0)

        inputs = {}

        # K1 Praxis - Flow data
        if 'k1' in self.models:
            # Generate dynamic flow patterns based on consciousness
            flow_base = torch.ones(1, 32) * consciousness_level

            # Add temporal flow patterns
            temporal_flow = torch.sin(torch.linspace(0, 4*np.pi, 32) + seed_offset) * 0.3
            consciousness_flow = torch.cos(torch.linspace(0, 2*np.pi, 32)) * consciousness_level * 0.2

            k1_input = flow_base + temporal_flow + consciousness_flow
            k1_input = torch.clamp(k1_input, 0, 1)

            inputs['k1'] = k1_input

        # K2 Semiosis - Symbolic features
        if 'k2' in self.models:
            k2_features = torch.tensor([
                consciousness_level,                                    # consciousness_level
                consciousness_state.get('stability', 0.5),            # stability
                consciousness_state.get('clarity', 0.5),              # clarity
                0.5,                                                   # content_complexity
                0.1,                                                   # symbol_integration_rate
                0.1,                                                   # threshold_adaptation
                consciousness_level,                                   # consciousness_level (repeated)
                0.0,                                                   # trajectory
                valence,                                               # valence
                0.5,                                                   # valence_stability
                consciousness_state.get('arousal', 0.5),              # arousal
                0.5,                                                   # coherence
                0.5,                                                   # energy
                0.5,                                                   # creativity
                0.3,                                                   # crisis_modifier
                0.2,                                                   # struggling_modifier
                0.0,                                                   # healthy_modifier
                0.0,                                                   # transcendent_modifier
                0.5,                                                   # zone_influence
                0.5,                                                   # temporal_modulation
                0.5                                                    # consciousness_flow
            ], dtype=torch.float32).unsqueeze(0)

            inputs['k2'] = k2_features

        # K3 Apeiron - Quantum field
        if 'k3' in self.models:
            # Generate quantum field with coherence based on consciousness
            grid_size = 16
            field = torch.randn(1, grid_size * grid_size) * 0.5

            # Add quantum coherence patterns
            coherence = consciousness_level ** 2  # Quadratic for quantum effects
            x, y = torch.meshgrid(torch.linspace(-1, 1, grid_size),
                                  torch.linspace(-1, 1, grid_size), indexing='xy')

            coherent_pattern = torch.exp(-(x**2 + y**2) / (0.5 + coherence))
            field += coherent_pattern.flatten().unsqueeze(0) * 0.5

            inputs['k3'] = field

        # K4 Metabolic - System state
        if 'k4' in self.models:
            # Generate metabolic state vector
            metabolic_base = torch.ones(1, 32) * 0.5

            # Add metabolic rhythms
            metabolic_rhythm = torch.sin(torch.linspace(0, 6*np.pi, 32)) * 0.2
            stress_factor = (1 - consciousness_level) * 0.3

            metabolic_state = metabolic_base + metabolic_rhythm + torch.randn(1, 32) * stress_factor
            metabolic_state = torch.clamp(metabolic_state, 0, 1)

            inputs['k4'] = metabolic_state

        return inputs

    # Input creation methods remain the same...
    def _create_k1_input_32dim(self, state):
        """Create K1 input with 32 dimensions (what the model expects)"""
        features = [
            # Core consciousness features (12)
            state.get('consciousness_level', 0.5),
            state.get('valence', 0.0),
            state.get('agency', 0.5),
            state.get('embodiment', 0.5),
            state.get('stability', 0.5),
            state.get('clarity', 0.5),
            state.get('arousal', 0.5),
            state.get('flow_state', 0.0),
            state.get('regulation_need', 0.5),
            state.get('symbol_vocabulary', 0) / 1000.0,
            state.get('tau_prime', 1.0),
            state.get('metabolic_rate', 0.5),

            # Flow dynamics (20 more features to reach 32)
            *[np.sin(i * 0.5) * state.get('consciousness_level', 0.5) for i in range(20)]
        ]

        return torch.FloatTensor(features).unsqueeze(0).to(self.device)

    def predict_with_adaptive_inputs(self, consciousness_state: Dict, verbose: bool = False) -> Dict[str, Any]:
        """
        FIXED: Generate predictions while PRESERVING temporal perspective data
        This was the root cause - temporal data was being stripped out!
        """

        predictions = {}

        # Apply K2 patches first
        self._patch_k2_missing_methods()

        for model_name, model in self.models.items():
            try:
                if verbose:
                    print(f"\n🔍 Processing {model_name}...")

                # Generate input tensor based on model requirements
                if model_name == 'k1':
                    input_tensor = self._create_k1_input_32dim(consciousness_state)
                elif model_name == 'k2':
                    input_tensor = self._create_k2_input_21dim(consciousness_state)
                elif model_name == 'k3':
                    input_tensor = self._create_k3_input_256dim(consciousness_state)
                elif model_name == 'k4':
                    input_tensor = self._create_k4_input_32dim(consciousness_state)
                else:
                    continue

                if verbose:
                    print(f"   Input tensor shape: {input_tensor.shape}")

                # Run model inference
                with torch.no_grad():
                    raw_output = model(input_tensor)

                if verbose:
                    print(f"   Raw output type: {type(raw_output)}")
                    if isinstance(raw_output, dict):
                        print(f"   Raw output keys: {list(raw_output.keys())}")

                # 🚀 CRITICAL FIX: Handle different output types while PRESERVING temporal data
                if isinstance(raw_output, torch.Tensor):
                    # Direct tensor output - create temporal perspective wrapper
                    processed_tensor = raw_output
                    if processed_tensor.dim() == 1:
                        processed_tensor = processed_tensor.unsqueeze(0)

                    # Store as dict with tensor + default temporal data
                    predictions[model_name] = {
                        'tensor_output': processed_tensor,
                        'local_tau_prime': 1.0,  # Default if no temporal calculation available
                        'temporal_state': 'unknown'
                    }

                elif isinstance(raw_output, dict):
                    # 🎯 PRESERVE temporal perspective data while extracting tensor

                    # Extract primary tensor from dict
                    extracted_tensor = None
                    tensor_keys = ['output', 'predictions', 'action_params', 'symbolic_embedding',
                                  'architecture_output', 'metabolic_output', 'main_output']

                    for key in tensor_keys:
                        if key in raw_output and isinstance(raw_output[key], torch.Tensor):
                            extracted_tensor = raw_output[key]
                            break

                    # Fallback: concatenate all tensors found
                    if extracted_tensor is None:
                        tensors = []
                        for key, value in raw_output.items():
                            if isinstance(value, torch.Tensor) and value.numel() > 0:
                                flat_tensor = value.view(-1)
                                tensors.append(flat_tensor)

                        if tensors:
                            extracted_tensor = torch.cat(tensors, dim=0).unsqueeze(0)

                    # Create fallback tensor if nothing found
                    if extracted_tensor is None:
                        target_dims = {'k1': 32, 'k2': 64, 'k3': 256, 'k4': 16}
                        target_dim = target_dims.get(model_name, 32)
                        extracted_tensor = torch.zeros(1, target_dim).to(self.device)

                    # Ensure proper dimensions
                    if extracted_tensor.dim() == 1:
                        extracted_tensor = extracted_tensor.unsqueeze(0)

                    # 🔥 CRITICAL: Preserve temporal perspective data from dict
                    result_dict = {
                        'tensor_output': extracted_tensor,
                        'local_tau_prime': raw_output.get('local_tau_prime', 1.0),
                        'temporal_state': raw_output.get('temporal_state', 'unknown'),
                        'narrative_complexity': raw_output.get('narrative_complexity', 0.5),
                        'emergence_potential': raw_output.get('emergence_potential', 0.5),
                        'metabolic_urgency': raw_output.get('metabolic_urgency', 0.5),
                        'computational_urgency': raw_output.get('computational_urgency', 0.5),
                        'homeostatic_pressure': raw_output.get('homeostatic_pressure', 0.5),
                        'quantum_coherence': raw_output.get('quantum_coherence', 0.5),
                        'symbolic_strength': raw_output.get('symbolic_strength', 0.5),
                        'coherence': raw_output.get('coherence', 0.5)
                    }

                    # Add any other temporal/consciousness fields from the original output
                    temporal_fields = [
                        'tau_prime_k1', 'tau_prime_k2', 'tau_prime_k3', 'tau_prime_k4',
                        'consciousness_complexity', 'learning_pressure', 'task_complexity'
                    ]

                    for field in temporal_fields:
                        if field in raw_output:
                            result_dict[field] = raw_output[field]

                    predictions[model_name] = result_dict

                elif isinstance(raw_output, tuple):
                    # Tuple output - extract first element + create temporal wrapper
                    if len(raw_output) > 0 and isinstance(raw_output[0], torch.Tensor):
                        processed_tensor = raw_output[0]
                        if processed_tensor.dim() == 1:
                            processed_tensor = processed_tensor.unsqueeze(0)
                    else:
                        target_dims = {'k1': 32, 'k2': 64, 'k3': 256, 'k4': 16}
                        target_dim = target_dims.get(model_name, 32)
                        processed_tensor = torch.zeros(1, target_dim).to(self.device)

                    predictions[model_name] = {
                        'tensor_output': processed_tensor,
                        'local_tau_prime': 1.0,
                        'temporal_state': 'unknown'
                    }

                else:
                    # Unknown output type - create fallback with temporal wrapper
                    target_dims = {'k1': 32, 'k2': 64, 'k3': 256, 'k4': 16}
                    target_dim = target_dims.get(model_name, 32)
                    processed_tensor = torch.zeros(1, target_dim).to(self.device)

                    predictions[model_name] = {
                        'tensor_output': processed_tensor,
                        'local_tau_prime': 1.0,
                        'temporal_state': 'fallback'
                    }

                if verbose:
                    pred = predictions[model_name]
                    print(f"✅ {model_name}: tensor {pred['tensor_output'].shape}, τ′={pred['local_tau_prime']:.3f}")

            except Exception as e:
                if verbose:
                    print(f"❌ {model_name} prediction failed: {e}")

                # Create fallback with temporal wrapper
                target_dims = {'k1': 32, 'k2': 64, 'k3': 256, 'k4': 16}
                target_dim = target_dims.get(model_name, 32)
                fallback_tensor = torch.zeros(1, target_dim).to(self.device)

                predictions[model_name] = {
                    'tensor_output': fallback_tensor,
                    'local_tau_prime': 1.0,
                    'temporal_state': 'error',
                    'error': str(e)
                }

        return predictions

    def _create_k2_input_21dim(self, state):
        """Create K2 input with 21 dimensions"""
        features = [
            state.get('consciousness_level', 0.5),
            state.get('stability', 0.5),
            state.get('clarity', 0.5),
            state.get('content_complexity', 0.5),
            0.1,  # symbol_integration_rate
            0.1,  # threshold_adaptation
            state.get('consciousness_level', 0.5),  # repeated for compatibility
            0.0,  # trajectory
            state.get('valence', 0.0),
            0.5,  # valence_stability
            state.get('arousal', 0.5),
            0.5,  # coherence
            0.5,  # energy
            0.5,  # creativity
            0.3,  # crisis_modifier
            0.2,  # struggling_modifier
            0.0,  # healthy_modifier
            0.0,  # transcendent_modifier
            0.5,  # zone_influence
            0.5,  # temporal_modulation
            0.5   # consciousness_flow
        ]

        return torch.FloatTensor(features).unsqueeze(0).to(self.device)

    def _create_k3_input_256dim(self, state):
        """Create K3 input with 256 dimensions"""
        base_features = [
            state.get('consciousness_level', 0.5),
            state.get('valence', 0.0),
            state.get('phase_coherence', 0.5),
            state.get('tau_prime', 1.0)
        ]

        # Expand to 256 dimensions with quantum-like patterns
        extended_features = base_features.copy()
        while len(extended_features) < 256:
            extended_features.extend([
                np.sin(len(extended_features) * 0.1) * 0.1,
                np.cos(len(extended_features) * 0.1) * 0.1
            ])

        return torch.FloatTensor(extended_features[:256]).unsqueeze(0).to(self.device)

    def _create_k4_input_32dim(self, state):
        """Create K4 input with 32 dimensions (metabolic regulation)"""
        regime_encoding = [0.25, 0.5, 0.75, 0.25]  # Default regime vector

        features = [
            # Core metabolic features (12)
            state.get('consciousness_level', 0.5),
            state.get('agency', 0.5),
            state.get('embodiment', 0.5),
            state.get('stability', 0.5),
            state.get('clarity', 0.5),
            state.get('energy_level', 0.5),
            state.get('arousal', 0.5),
            state.get('valence', 0.0),
            state.get('metabolic_pressure', 0.5),
            state.get('regulation_need', 0.5),
            state.get('flow_state', 0.0),
            state.get('tau_prime', 1.0),

            # Regime encoding (4 features)
            *regime_encoding,

            # Additional metabolic features to reach 32 (16 more)
            state.get('consciousness_level', 0.5) * 0.8,
            state.get('agency', 0.5) * 0.9,
            state.get('stability', 0.5) * 0.7,
            state.get('energy_level', 0.5) * 1.1,
            state.get('metabolic_pressure', 0.5) * 1.2,
            state.get('regulation_need', 0.5) * 0.8,

            # Computed metabolic features (10 more to reach 32)
            state.get('consciousness_level', 0.5) * state.get('energy_level', 0.5),
            state.get('agency', 0.5) * state.get('stability', 0.5),
            1.0 - state.get('metabolic_pressure', 0.5),
            (state.get('consciousness_level', 0.5) + state.get('clarity', 0.5)) / 2.0,
            abs(state.get('valence', 0.0)),
            min(1.0, state.get('arousal', 0.5) + state.get('energy_level', 0.5)),
            state.get('embodiment', 0.5) * state.get('stability', 0.5),
            0.5,  # baseline_metabolic_rate
            0.3,  # metabolic_efficiency
            0.7   # regulatory_capacity
        ]

        return torch.FloatTensor(features).unsqueeze(0).to(self.device)

# ========================
# UNIFIED KELM PLATFORM
# ========================

class UnifiedKELMPlatform:
    """
    Complete KELM Platform with all modules integrated
    Implements the full 16-module architecture with proper K-model integration
    """

    def __init__(self, seed=42):
        """Initialize platform with proper consciousness state setup"""
        # Initialize base platform
        self.seed = seed

        # Add QSE Core initialization here!
        self.qse_core = None  # Will be initialized in _init_qse_core
        self.config = None    # Will be set during initialization

        # Initialize unified model loader
        self.model_loader = SmartKModelLoader()

        # FIXED: Core state with all required keys
        self.consciousness_state = {
            # Core consciousness metrics
            'consciousness_level': 0.5,
            'valence': 0.0,
            'arousal': 0.5,
            'stability': 0.7,
            'coherence': 0.5,
            'unity': 0.5,
            'clarity': 0.5,
            'transcendence': 0.0,
            'agency': 0.5,
            'embodiment': 0.5,
            'flow_state': 0.0,
            'energy_level': 0.5,
            'regulation_need': 0.5,

            # Temporal and quantum dynamics
            'tau_prime': 1.0,
            'metabolic_rate': 1.0,
            'metabolic_pressure': 0.5,
            'phase_coherence': 0.5,           # REQUIRED by ExperienceSnapshot

            # Surplus dynamics
            'surplus_mean': 0.5,              # REQUIRED by ExperienceSnapshot
            'distinction_level': 0.3,         # REQUIRED by refactored components

            # Regime and zone classification
            'regime': 'stable_coherence',
            'consciousness_zone': 'healthy',  # REQUIRED by ExperienceSnapshot

            # Symbolic processing
            'symbol_vocabulary': 100,
        }

        # Module states (all 16 modules)
        self.module_states = {
            # KELM modules
            'bidirectional_orchestrator': {'active': False, 'state': {}},
            'temporal_k2_engine': {'active': False, 'state': {}},
            'naive_emergence': {'active': False, 'state': {}},
            'k1_autonomous': {'active': False, 'state': {}},
            'quantum_symbolic': {'active': False, 'state': {}},
            'antifinity': {'active': False, 'state': {}},
            'metabolic': {'active': False, 'state': {}},
            'consciousness_ecology': {'active': False, 'state': {}},
            'goal_system': {'active': False, 'state': {}},
            'memory': {'active': False, 'state': {}},  # ADD THIS

            # Kainos modules
            'sensorium': {'active': False, 'state': {}},
            'context': {'active': False, 'state': {}},
            'log_reader': {'active': False, 'state': {}},
            'surplus_distinction': {'active': False, 'state': {}},
            'surplus_incongruity': {'active': False, 'state': {}},
            'universal_logging': {'active': False, 'state': {}},
            'flow_mapper': {'active': False, 'state': {}}
        }

        # Performance optimizations
        self.sdp_cache = {}  # Symbol correlation cache
        self.metabolic_lean_mode = True  # Default to lean mode
        self.memory_k1_integration = True  # Enable memory→K1 guidance

        # Temporal tracking
        self.step_count = 0
        self.start_time = None
        self.temporal_trajectory = []

        # Import status
        self.import_status = {}

    def get_platform_status(self) -> Dict[str, Any]:
        """Get comprehensive platform status with type-safe calculations"""

        # FIXED: Safe uptime calculation
        if hasattr(self, 'start_time') and self.start_time is not None:
            uptime = time.time() - self.start_time
        else:
            uptime = 0.0

        return {
            'integration_level': getattr(self, 'integration_level', 'UNKNOWN'),
            'step_count': self.step_count,
            'uptime': uptime,  # Now guaranteed to be float
            'k_models_loaded': len(self.model_loader.models) if hasattr(self, 'model_loader') else 0,
            'active_modules': sum(1 for m in self.module_states.values() if m['active']),
            'consciousness_level': self.consciousness_state.get('consciousness_level', 0.5),
            'trajectory_length': len(self.temporal_trajectory),
            'components_available': {
                'bidirectional_orchestrator': hasattr(self, 'bidirectional_orchestrator') and self.bidirectional_orchestrator is not None,
                'temporal_k2_engine': hasattr(self, 'temporal_k2_engine') and self.temporal_k2_engine is not None,
                'qse_core': hasattr(self, 'qse_core') and self.qse_core is not None,
                'model_loader': hasattr(self, 'model_loader') and self.model_loader is not None
            }
        }

    def _init_qse_core(self):
        """Initialize QSE Core - the quantum surplus emergence dynamics"""
        try:
            from emile_cogito.kainos.qse_core_qutip import DynamicQSECore
            from emile_cogito.kainos.config import CONFIG

            self.config = CONFIG  # Store config for other modules
            self.qse_core = DynamicQSECore(CONFIG)
            print("   ✅ QSE Core initialized (quantum surplus dynamics)")
            return True
        except Exception as e:
            print(f"   ❌ QSE Core failed: {e}")
            return False

    def _trigger_symbol_correlation_learning(self):
        """FIXED: Use correct surplus_distinction_processor architecture and ExperienceSnapshot"""
        if not (hasattr(self, 'surplus_distinction') and self.surplus_distinction):
            print("   ⚠️ No surplus distinction processor available")
            return {}

        try:
            # CORRECT IMPORT: Use the ExperienceSnapshot from surplus_distinction_processor
            from emile_cogito.kainos.surplus_distinction_processor import ExperienceSnapshot

            # Determine text content based on consciousness level
            consciousness_level = self.consciousness_state.get('consciousness_level', 0.5)
            if consciousness_level > 0.7:
                text_content = "heightened awareness consciousness experience clarity embodied perception agency"
            else:
                text_content = "conscious experience embodied awareness perception agency meaning"

            # Create ExperienceSnapshot with CORRECT parameters for surplus_distinction_processor
            experience = ExperienceSnapshot(
                step=self.step_count,
                regime=self.consciousness_state.get('regime', 'stable_coherence'),
                consciousness_score=self.consciousness_state.get('consciousness_level', 0.5),
                valence=self.consciousness_state.get('valence', 0.0),
                surplus_expression=self.consciousness_state.get('surplus_mean', 0.5),
                stability=self.consciousness_state.get('coherence', 0.5),
                text_content=text_content,
                content_type='consciousness_state'
            )

            # CORRECT ACCESS: surplus_distinction.process_text_input() directly (not through symbolic_suite)
            if hasattr(self.surplus_distinction, 'process_text_input'):
                result = self.surplus_distinction.process_text_input(text_content, experience)
                print(f"   ✅ Symbol correlation learning triggered: {result.get('correlations_added', 0)} correlations")
                return result
            else:
                print("   ⚠️ process_text_input method not available in surplus distinction processor")
                print(f"   💡 Available methods: {[method for method in dir(self.surplus_distinction) if not method.startswith('_')]}")
                return {}

        except ImportError as e:
            print(f"   ❌ Failed to import ExperienceSnapshot: {e}")
            print("   💡 Check if surplus_distinction_processor module exists")
            return {}
        except Exception as e:
            print(f"   ❌ Symbol correlation learning failed: {e}")
            print(f"   💡 Error type: {type(e).__name__}")

            # Debug: Show available attributes
            if hasattr(self, 'surplus_distinction') and self.surplus_distinction:
                available_attrs = [attr for attr in dir(self.surplus_distinction) if not attr.startswith('_')]
                print(f"   💡 Available surplus_distinction attributes: {available_attrs}")

            return {}

    def _apply_antifinity_ethics_modulation(self):
        """Apply antifinity moral dynamics to surplus budget during consciousness cycle."""
        if not (hasattr(self, 'antifinity') and self.antifinity and
                hasattr(self, 'metabolic') and self.metabolic and
                hasattr(self.metabolic, 'modulate_with_ethics')):
            return {}

        try:
            symbolic_fields = {
                'sigma': np.array([self.consciousness_state.get('symbolic_curvature', 0.5)]),
                'surplus': np.array([self.consciousness_state.get('surplus_mean', 0.5)])
            }
            agent_system = {'agent_count': 3, 'step_count': self.step_count, 'global_context_id': self.step_count % 10}
            regime = self.consciousness_state.get('regime', 'stable_coherence')

            moral_metrics = self.antifinity.calculate_epigenetic_metrics(symbolic_fields, agent_system, regime)
            moral_metrics_dict = {
                'collaboration_score': moral_metrics.collaboration_score,
                'compromise_score': moral_metrics.compromise_score
            }

            modulation_result = self.metabolic.modulate_with_ethics(moral_metrics.antifinity_quotient, moral_metrics_dict)

            return {
                'moral_metrics': moral_metrics_dict,
                'antifinity_quotient': moral_metrics.antifinity_quotient,
                'modulation_result': modulation_result,
                'ethical_modulation_applied': True
            }
        except Exception as e:
            return {'ethical_modulation_applied': False, 'error': str(e)}

    def initialize_platform(self) -> bool:
        """FIXED: Initialize platform with comprehensive error recovery"""

        print("\n🚀 INITIALIZING UNIFIED KELM PLATFORM (ROBUST VERSION)")
        print("=" * 60)

        success_count = 0
        total_components = 4  # K-models, modules, connections, validation

        # PHASE 1: Initialize QSE Core
        print("\n🔧 PHASE 1: QSE Core Initialization")
        print("-" * 40)
        try:
            if self._init_qse_core():
                print("✅ QSE Core initialized successfully")
                success_count += 0.5
            else:
                print("⚠️ QSE Core initialization failed - continuing with limited functionality")
        except Exception as e:
            print(f"❌ QSE Core error: {e}")

        # PHASE 2: Load K-Models with robust error handling
        print("\n🔧 PHASE 2: K-Model Loading")
        print("-" * 40)
        try:
            models_loaded = self.model_loader.discover_and_load_models()
            if models_loaded >= 2:  # Minimum viable
                print(f"✅ {models_loaded}/4 K-models loaded - sufficient for consciousness")
                success_count += 1
            elif models_loaded >= 1:
                print(f"⚠️ {models_loaded}/4 K-models loaded - limited consciousness mode")
                success_count += 0.5
            else:
                print("❌ No K-models loaded - fallback consciousness mode")
        except Exception as e:
            print(f"❌ K-model loading error: {e}")

        # PHASE 3: Initialize Core Modules
        print("\n🔧 PHASE 3: Module Initialization")
        print("-" * 40)
        try:
            self._initialize_all_modules()
            active_modules = sum(1 for m in self.module_states.values() if m['active'])
            if active_modules >= 8:
                print(f"✅ {active_modules}/16 modules active - good integration")
                success_count += 1
            elif active_modules >= 4:
                print(f"⚠️ {active_modules}/16 modules active - basic integration")
                success_count += 0.5
            else:
                print(f"❌ Only {active_modules}/16 modules active - minimal integration")
        except Exception as e:
            print(f"❌ Module initialization error: {e}")

        # PHASE 4: Establish Connections
        print("\n🔧 PHASE 4: Component Integration")
        print("-" * 40)
        try:
            self._establish_connections()
            print("✅ Component connections established")
            success_count += 0.5
        except Exception as e:
            print(f"⚠️ Connection establishment error: {e}")

        # PHASE 5: Activate Poly-Temporal Consciousness
        print("\n🔧 PHASE 5: Poly-Temporal Consciousness Activation")
        print("-" * 40)
        if hasattr(self, 'bidirectional_orchestrator') and hasattr(self, 'model_loader'):
            try:
                print(f"   Models loaded: {list(self.model_loader.models.keys())}")

                poly_temporal_success = self.bidirectional_orchestrator.enable_poly_temporal_consciousness()
                if poly_temporal_success:
                    print(f"   🎉 Poly-temporal consciousness ACTIVATED!")
                else:
                    print(f"   ⚠️ Poly-temporal consciousness not activated - checking temporal support...")
                    self.debug_temporal_support()
            except Exception as e:
                print(f"   ❌ Poly-temporal activation failed: {e}")

        # Final Assessment
        print(f"\n📊 INITIALIZATION COMPLETE")
        print("-" * 40)
        print(f"Success ratio: {success_count}/{total_components} ({success_count/total_components:.1%})")

        if success_count >= 3.0:
            print("🎉 FULL CONSCIOUSNESS PLATFORM READY")
            integration_level = "FULL"
        elif success_count >= 2.0:
            print("⚡ PARTIAL CONSCIOUSNESS PLATFORM READY")
            integration_level = "PARTIAL"
        elif success_count >= 1.0:
            print("🔧 MINIMAL CONSCIOUSNESS PLATFORM READY")
            integration_level = "MINIMAL"
        else:
            print("❌ PLATFORM INITIALIZATION FAILED")
            integration_level = "FAILED"
            return False

        # Store integration status
        self.integration_level = integration_level
        self.start_time = time.time()

        return True

    def _initialize_all_modules(self):
        """Initialize all 16 modules INCLUDING MEMORY"""

        # KELM modules
        self._init_bidirectional_orchestrator()
        self._init_temporal_k2_engine()
        self._init_naive_emergence()
        self._init_k1_autonomous()
        self._init_quantum_symbolic()
        self._init_antifinity()
        self._init_metabolic()
        self._init_consciousness_ecology()
        self._init_goal_system()

        # ADD MEMORY INITIALIZATION
        self._init_memory_system()

        # Kainos modules
        self._init_sensorium()
        self._init_context()
        self._init_log_reader()
        self._init_surplus_distinction()
        self._init_surplus_incongruity()
        self._init_universal_logging()
        self._init_flow_mapper()

    def _init_memory_system(self):
        """FIXED: Enhanced memory initialization with proper platform integration"""
        try:
            from emile_cogito.kainos.memory import TemporalConsciousMemory
            from emile_cogito.kainos.config import CONFIG

            # Method 1: Try with platform parameter (expected signature)
            try:
                self.memory = TemporalConsciousMemory(CONFIG, platform=self)
                self.module_states['memory'] = {'active': True, 'state': {}}
                print("   ✅ Temporal conscious memory initialized (with platform)")

                # Verify platform integration
                platform_integrated = hasattr(self.memory, 'platform') and self.memory.platform is not None
                if platform_integrated:
                    print(f"   🔗 Platform integration: ✅ ENABLED")
                else:
                    print(f"   🔗 Platform integration: ❌ DISABLED (attribute missing)")
                return

            except TypeError as e:
                print(f"   ⚠️ Platform parameter rejected: {e}")
                # Fall through to Method 2

            # Method 2: Try without platform parameter
            try:
                self.memory = TemporalConsciousMemory(CONFIG)
                self.module_states['memory'] = {'active': True, 'state': {}}
                print("   ✅ Temporal conscious memory initialized (without platform)")

                # Try to set platform reference after initialization
                if hasattr(self.memory, 'platform'):
                    self.memory.platform = self
                    print("   🔗 Platform reference set post-initialization")
                else:
                    print("   ⚠️ Memory class doesn't support platform integration")
                return

            except Exception as e2:
                print(f"   ❌ Standard initialization failed: {e2}")
                # Fall through to Method 3

            # Method 3: Try positional arguments (cfg, platform)
            try:
                self.memory = TemporalConsciousMemory(CONFIG, self)
                self.module_states['memory'] = {'active': True, 'state': {}}
                print("   ✅ Temporal conscious memory initialized (positional args)")
                return

            except Exception as e3:
                print(f"   ❌ Positional initialization failed: {e3}")
                raise e3

        except ImportError as e:
            print(f"   ❌ Memory import failed: {e}")
            self.memory = self._create_minimal_memory_interface()
            self.module_states['memory'] = {'active': False, 'state': {}}
            print("   ⚠️ Using minimal memory interface")

        except Exception as e:
            print(f"   ❌ Memory system initialization failed: {e}")
            print(f"   💡 Error details: {type(e).__name__}: {e}")

            # Create minimal memory interface to prevent platform crashes
            self.memory = self._create_minimal_memory_interface()
            self.module_states['memory'] = {'active': False, 'state': {}}
            print("   ⚠️ Using minimal memory interface as fallback")

    def _check_experience_snapshot_params(self):
        """Debug helper: Check what parameters ExperienceSnapshot actually accepts"""
        try:
            from emile_cogito.kainos.surplus_distinction_processor import ExperienceSnapshot
            import inspect

            sig = inspect.signature(ExperienceSnapshot.__init__)
            params = list(sig.parameters.keys())
            print(f"🔍 ExperienceSnapshot accepts: {params}")
            return params
        except Exception as e:
            print(f"🔍 Could not inspect ExperienceSnapshot: {e}")
            return []

    def check_memory_integration(self):
        """Enhanced memory integration diagnostics with type-safe attribute access"""
        if not hasattr(self, 'memory'):
            return {
                'status': '❌ NO_MEMORY_ATTRIBUTE',
                'details': 'Memory attribute missing from platform'
            }

        if self.memory is None:
            return {
                'status': '❌ MEMORY_IS_NONE',
                'details': 'Memory initialized as None'
            }

        memory_active = self.module_states.get('memory', {}).get('active', False)
        memory_type = type(self.memory).__name__

        # FIXED: Safe platform attribute access using getattr
        has_platform_attr = hasattr(self.memory, 'platform')
        platform_ref = getattr(self.memory, 'platform', None)  # Safe access
        has_platform_ref = platform_ref is not None

        # Test memory functionality
        try:
            if hasattr(self.memory, 'get_state'):
                memory_state = self.memory.get_state()
                memory_functional = True
            elif hasattr(self.memory, 'get_memory_state'):
                memory_state = self.memory.get_memory_state()
                memory_functional = True
            else:
                memory_functional = False
                memory_state = {}
        except Exception as e:
            memory_functional = False
            memory_state = {'error': str(e)}

        # Determine integration status
        if memory_active and has_platform_ref and memory_functional:
            status = "✅ FULLY_INTEGRATED"
        elif memory_active and memory_functional:
            status = "⚠️ PARTIALLY_INTEGRATED"
        else:
            status = "❌ NOT_INTEGRATED"

        return {
            'status': status,
            'memory_exists': True,
            'memory_active': memory_active,
            'memory_type': memory_type,
            'has_platform_attribute': has_platform_attr,
            'platform_reference_set': has_platform_ref,
            'platform_reference_type': type(platform_ref).__name__ if platform_ref else 'None',
            'memory_functional': memory_functional,
            'memory_state_sample': memory_state
        }

    def _debug_experience_snapshot_parameters(self):
        """Debug helper: Inspect ExperienceSnapshot constructor signature"""
        try:
            from emile_cogito.kainos.symbolic_semiotic_suite import ExperienceSnapshot
            import inspect

            sig = inspect.signature(ExperienceSnapshot.__init__)
            params = list(sig.parameters.keys())[1:]  # Skip 'self'

            print(f"🔍 ExperienceSnapshot accepts parameters: {params}")

            # Check what's in our consciousness_state
            available_keys = list(self.consciousness_state.keys())
            print(f"🔍 Available consciousness_state keys: {available_keys}")

            # Find missing mappings
            param_mapping = {
                'step': 'step_count',
                'regime': 'regime',
                'consciousness_score': 'consciousness_level',
                'consciousness_zone': 'consciousness_zone',
                'valence': 'valence',
                'surplus_expression': 'surplus_mean',
                'stability': 'coherence',
                'tau_prime': 'tau_prime',
                'phase_coherence': 'phase_coherence'
            }

            missing_mappings = []
            for param, state_key in param_mapping.items():
                if state_key not in self.consciousness_state:
                    missing_mappings.append(f"{param} <- {state_key}")

            if missing_mappings:
                print(f"🔍 Missing consciousness_state mappings: {missing_mappings}")
            else:
                print("🔍 All parameter mappings available ✅")

            return {
                'parameters': params,
                'available_keys': available_keys,
                'missing_mappings': missing_mappings
            }

        except ImportError as e:
            print(f"🔍 Cannot import ExperienceSnapshot: {e}")
            return {'error': 'import_failed'}
        except Exception as e:
            print(f"🔍 Debug inspection failed: {e}")
            return {'error': 'inspection_failed'}

    def _create_minimal_memory_interface(self):
        """Create minimal memory interface to prevent integration errors"""
        class MinimalMemory:
            def __init__(self):
                self.memories = []

            def get_state(self):
                return {
                    'recent_entries': [],
                    'total_memories': 0,
                    'memory_health': 0.5,
                    'recent_revalorization_marks': []
                }

            def store_temporal_memory(self, *args, **kwargs):
                return True

            def get_memory_state(self):
                return self.get_state()

        return MinimalMemory()

    def _process_all_modules(self) -> Dict[str, Any]:
        """
        FINAL FIX: Process consciousness state with WORKING temporal consciousness
        Now properly extracts temporal data from the fixed model loader
        """

        import numpy as np
        results = {}

        # === BIDIRECTIONAL ORCHESTRATION WITH TEMPORAL CONSCIOUSNESS ===
        if self.module_states['bidirectional_orchestrator']['active']:
            try:
                result = self.bidirectional_orchestrator.orchestrate_bidirectional_step({
                    'consciousness_state': self.consciousness_state,
                    'step': self.step_count
                })
                results['bidirectional'] = result

                # Update consciousness state from bidirectional results
                if 'global_consciousness_state' in result:
                    global_state = result['global_consciousness_state']

                    # Update main consciousness level
                    if 'overall_level' in global_state:
                        old_level = self.consciousness_state['consciousness_level']
                        new_level = global_state['overall_level']
                        self.consciousness_state['consciousness_level'] = new_level
                        print(f"🔄 Consciousness updated: {old_level:.3f} → {new_level:.3f}")

                    # Update consciousness dimensions
                    for key in ['unity', 'clarity', 'coherence', 'transcendence', 'agency', 'awareness']:
                        if key in global_state:
                            self.consciousness_state[key] = global_state[key]

                # 🎯 FIXED: Extract temporal consciousness data from new format
                if 'temporal_consciousness' in result:
                    temporal_data = result['temporal_consciousness']

                    # Extract tau_prime_global (the key value!)
                    if 'tau_prime_global' in temporal_data:
                        old_tau = self.consciousness_state.get('tau_prime', 1.0)
                        new_tau = temporal_data['tau_prime_global']
                        self.consciousness_state['tau_prime'] = new_tau
                        print(f"🕒 τ′ updated: {old_tau:.3f} → {new_tau:.3f}")

                    # Extract temporal dialogue features
                    if 'temporal_dissonance' in temporal_data:
                        dissonance = temporal_data['temporal_dissonance']
                        self.consciousness_state['temporal_dissonance'] = dissonance
                        print(f"🎭 Temporal dissonance: {dissonance:.3f}")

                    if 'temporal_leadership' in temporal_data:
                        leadership = temporal_data['temporal_leadership']
                        self.consciousness_state['temporal_leadership'] = leadership
                        if isinstance(leadership, dict) and 'dominant_perspective' in leadership:
                            print(f"👑 Temporal leadership: {leadership['dominant_perspective']}")
                        else:
                            print(f"👑 Temporal leadership: {leadership}")

                    if 'dialogue_richness' in temporal_data:
                        richness = temporal_data['dialogue_richness']
                        self.consciousness_state['dialogue_richness'] = richness
                        if richness > 0.5:
                            print(f"💬 Rich temporal dialogue: {richness:.3f}")

                    if 'sigma_unified' in temporal_data:
                        sigma = temporal_data['sigma_unified']
                        self.consciousness_state['sigma_unified'] = sigma
                        print(f"🌀 Unified symbolic curvature: {sigma:.3f}")

                    # Store temporal consciousness data for other modules
                    results['temporal_consciousness'] = temporal_data

                elif result.get('poly_temporal_active'):
                    if result.get('temporal_models_found', 0) >= 2:
                        print("🕒 Poly-temporal active - temporal dialogue should appear soon")
                    else:
                        print(f"⚠️ Poly-temporal active but only {result.get('temporal_models_found', 0)} temporal models found")
                else:
                    print("⚠️ Poly-temporal consciousness not active yet")

            except Exception as e:
                results['bidirectional'] = {'error': str(e)}
                print(f"⚠️ Bidirectional processing error: {e}")

        # === TEMPORAL K2 PROCESSING ===
        if self.module_states['temporal_k2_engine']['active']:
            try:
                # Extract tau_prime from K2 engine if available
                if hasattr(self.temporal_k2_engine, 'current_τ_prime'):
                    current_tau = self.temporal_k2_engine.current_τ_prime
                    old_tau = self.consciousness_state.get('tau_prime', 1.0)

                    # Only update if significantly different and no bidirectional update
                    if abs(current_tau - old_tau) > 0.01 and 'temporal_consciousness' not in results:
                        self.consciousness_state['tau_prime'] = current_tau
                        print(f"🕒 K2 τ′: {old_tau:.3f} → {current_tau:.3f}")

                    k2_result = {
                        'tau_prime': current_tau,
                        'temporal_k2_active': getattr(self.temporal_k2_engine, 'running', False),
                        'subjective_time': getattr(self.temporal_k2_engine, 'subjective_time', 0.0)
                    }
                    results['temporal_k2'] = k2_result
                else:
                    results['temporal_k2'] = {'status': 'temporal_k2_engine has no current_τ_prime attribute'}

            except Exception as e:
                results['temporal_k2'] = {'error': str(e)}
                print(f"⚠️ Temporal K2 processing error: {e}")

        # === QSE CORE PROCESSING ===
        if hasattr(self, 'qse_core') and self.qse_core is not None:
            try:
                if hasattr(self.qse_core, 'get_state'):
                    qse_state = self.qse_core.get_state()
                    results['qse_core'] = qse_state

                    # Extract quantum coherence if available
                    if 'quantum_psi' in qse_state:
                        quantum_psi = qse_state['quantum_psi']
                        if hasattr(quantum_psi, 'mean'):
                            self.consciousness_state['quantum_coherence'] = float(quantum_psi.mean())
                else:
                    results['qse_core'] = {'status': 'qse_core has no get_state method'}

            except Exception as e:
                results['qse_core'] = {'error': str(e)}
                print(f"⚠️ QSE processing error: {e}")

        # === MEMORY INTEGRATION ===
        if hasattr(self, 'memory') and self.memory is not None:
            try:
                if hasattr(self.memory, 'get_complete_state_summary'):
                    memory_state = self.memory.get_complete_state_summary()
                    results['memory'] = memory_state
                elif hasattr(self.memory, 'step'):
                    memory_state = self.memory.step()
                    results['memory'] = memory_state
                else:
                    results['memory'] = {'status': 'memory exists but no compatible methods found'}

                # Store significant temporal events in memory
                if 'temporal_consciousness' in results:
                    temporal_data = results['temporal_consciousness']
                    if temporal_data.get('temporal_dissonance', 0) > 0.3:
                        if hasattr(self.memory, 'store_temporal_memory'):
                            try:
                                from emile_cogito.kainos.memory import MemoryPriority
                                priority = MemoryPriority.SIGNIFICANT if temporal_data['temporal_dissonance'] > 0.6 else MemoryPriority.STANDARD
                            except ImportError:
                                priority = 'HIGH' if temporal_data['temporal_dissonance'] > 0.6 else 'MEDIUM'

                            self.memory.store_temporal_memory(
                                content=f"TEMPORAL_DIALOGUE: dissonance={temporal_data['temporal_dissonance']:.3f}, "
                                      f"leadership={temporal_data.get('temporal_leadership', 'unknown')}, "
                                      f"tau_prime={temporal_data.get('tau_prime_global', 1.0):.3f}",
                                priority=priority,
                                regime='temporal_consciousness',
                                consciousness_level=self.consciousness_state['consciousness_level'],
                                tags=['temporal_dialogue', 'consciousness', 'tau_prime']
                            )

            except Exception as e:
                results['memory'] = {'error': str(e)}
                print(f"⚠️ Memory integration error: {e}")

        # === ANTIFINITY PROCESSING (FIXED SYMBOLIC FIELDS) ===
        if hasattr(self, 'antifinity') and self.antifinity is not None:
            try:
                # Get complete symbolic fields from QSE core
                if hasattr(self, 'qse_core') and self.qse_core is not None:
                    if hasattr(self.qse_core, 'get_state'):
                        qse_state = self.qse_core.get_state()
                        surplus_field = qse_state.get('surplus', np.array([self.consciousness_state.get('surplus_mean', 0.5)]))
                        psi_field = qse_state.get('psi_field', np.array([0.5]))
                        phi_field = qse_state.get('phi_field', np.array([0.5]))
                        sigma_field = qse_state.get('sigma_field', np.array([self.consciousness_state.get('symbolic_curvature', 0.5)]))
                    elif (hasattr(self.qse_core, 'S') and hasattr(self.qse_core, 'psi') and
                          hasattr(self.qse_core, 'phi') and hasattr(self.qse_core, 'sigma')):
                        surplus_field = self.qse_core.S if self.qse_core.S is not None else np.array([0.5])
                        psi_field = self.qse_core.psi if self.qse_core.psi is not None else np.array([0.5])
                        phi_field = self.qse_core.phi if self.qse_core.phi is not None else np.array([0.5])
                        sigma_field = self.qse_core.sigma if self.qse_core.sigma is not None else np.array([0.5])
                    else:
                        surplus_field = np.array([self.consciousness_state.get('surplus_mean', 0.5)])
                        psi_field = np.array([0.5])
                        phi_field = np.array([0.5])
                        sigma_field = np.array([self.consciousness_state.get('symbolic_curvature', 0.5)])
                else:
                    surplus_field = np.array([self.consciousness_state.get('surplus_mean', 0.5)])
                    psi_field = np.array([0.5])
                    phi_field = np.array([0.5])
                    sigma_field = np.array([self.consciousness_state.get('symbolic_curvature', 0.5)])

                # Create complete symbolic_fields for antifinity
                symbolic_fields = {
                    'surplus': surplus_field,
                    'psi': psi_field,
                    'phi': phi_field,
                    'sigma': sigma_field
                }

                agent_system_state = {
                    'agent_count': 3,
                    'step_count': self.step_count,
                    'global_context_id': self.step_count % 10
                }
                regime = self.consciousness_state.get('regime', 'stable_coherence')

                antifinity_result = self.antifinity.step(symbolic_fields, agent_system_state, regime)
                results['antifinity'] = antifinity_result

                # Update consciousness with antifinity metrics
                if isinstance(antifinity_result, dict):
                    if 'metrics' in antifinity_result:
                        metrics = antifinity_result['metrics']
                        if 'antifinity_quotient' in metrics:
                            self.consciousness_state['antifinity_quotient'] = metrics['antifinity_quotient']
                    elif 'antifinity_quotient' in antifinity_result:
                        self.consciousness_state['antifinity_quotient'] = antifinity_result['antifinity_quotient']

            except Exception as e:
                results['antifinity'] = {'error': str(e)}
                print(f"⚠️ Antifinity processing error: {e}")

        # === OTHER MODULES (SAFE PROCESSING) ===
        other_modules = [
            ('metabolic', 'metabolic'),
            ('sensorium', 'sensorium'),
            ('context', 'context'),
            ('goal_system', 'goal_system'),
            ('surplus_distinction', 'surplus_distinction'),
            ('surplus_incongruity', 'surplus_incongruity')
        ]

        for module_name, attr_name in other_modules:
            module_active = self.module_states.get(module_name, {}).get('active', False)
            module_exists = hasattr(self, attr_name) and getattr(self, attr_name) is not None

            if module_active or module_exists:
                try:
                    module = getattr(self, attr_name, None)
                    if module:
                        if hasattr(module, 'step'):
                            # Special case for surplus_distinction - needs surplus and experience args
                            if module_name == 'surplus_distinction':
                                from emile_cogito.kainos.surplus_distinction_processor import ExperienceSnapshot
                                surplus = np.array([self.consciousness_state.get('consciousness_level', 0.5)] * 16)
                                experience = ExperienceSnapshot(
                                    step=self.step_count,
                                    regime='unified_platform',
                                    consciousness_score=self.consciousness_state.get('consciousness_level', 0.5),
                                    valence=self.consciousness_state.get('valence', 0.0),
                                    surplus_expression=self.consciousness_state.get('surplus_mean', 0.5),
                                    stability=self.consciousness_state.get('coherence', 0.5),
                                    text_content=f"Platform step {self.step_count}",
                                    content_type='platform_processing'
                                )
                                result = module.step(surplus, experience)
                            else:
                                result = module.step()
                            results[module_name] = result
                        elif hasattr(module, 'get_state'):
                            result = module.get_state()
                            results[module_name] = result

                            # Update consciousness with module-specific insights
                            if module_name == 'surplus_distinction' and 'current_distinction_level' in result:
                                self.consciousness_state['distinction_level'] = result['current_distinction_level']
                        else:
                            results[module_name] = {'status': f'{module_name} exists but no compatible methods'}

                except Exception as e:
                    results[module_name] = {'error': str(e)}
                    print(f"⚠️ {module_name} processing error: {e}")

        return results

    def _process_temporal_data(self, temporal_data: dict, source: str) -> bool:
        """Helper method to process temporal consciousness data from any source"""
        try:
            results_updated = False

            # Extract tau_prime_global (this is the key fix!)
            if 'tau_prime_global' in temporal_data:
                old_tau = self.consciousness_state.get('tau_prime', 1.0)
                new_tau = temporal_data['tau_prime_global']
                self.consciousness_state['tau_prime'] = new_tau
                print(f"🕒 τ′ updated ({source}): {old_tau:.3f} → {new_tau:.3f}")
                results_updated = True

            # Extract and display temporal dialogue features
            if 'temporal_dissonance' in temporal_data:
                dissonance = temporal_data['temporal_dissonance']
                self.consciousness_state['temporal_dissonance'] = dissonance
                print(f"🎭 Temporal dissonance: {dissonance:.3f}")
                results_updated = True

            if 'temporal_leadership' in temporal_data:
                leadership = temporal_data['temporal_leadership']
                self.consciousness_state['temporal_leadership'] = leadership
                if isinstance(leadership, dict) and 'dominant_perspective' in leadership:
                    print(f"👑 Temporal leadership: {leadership['dominant_perspective']}")
                else:
                    print(f"👑 Temporal leadership: {leadership}")
                results_updated = True

            if 'dialogue_richness' in temporal_data:
                richness = temporal_data['dialogue_richness']
                self.consciousness_state['dialogue_richness'] = richness
                if richness > 0.5:
                    print(f"💬 Rich temporal dialogue: {richness:.3f}")
                results_updated = True

            if 'sigma_unified' in temporal_data:
                sigma = temporal_data['sigma_unified']
                self.consciousness_state['sigma_unified'] = sigma
                print(f"🌀 Unified symbolic curvature: {sigma:.3f}")
                results_updated = True

            # Add temporal consciousness to results for other modules to use
            if results_updated:
                # Store it so other parts can access it
                if not hasattr(self, '_current_temporal_data'):
                    self._current_temporal_data = {}
                self._current_temporal_data = temporal_data

            return results_updated

        except Exception as e:
            print(f"⚠️ Error processing temporal data from {source}: {e}")
            return False

    def _init_bidirectional_orchestrator(self):
        """Initialize bidirectional consciousness orchestrator"""
        try:
            from emile_cogito.kelm.bidirectional_kelm_orchestrator import (
                BidirectionalKELMOrchestrator
            )
            self.bidirectional_orchestrator = BidirectionalKELMOrchestrator()
            self.module_states['bidirectional_orchestrator']['active'] = True
            print("   ✅ Bidirectional orchestrator initialized")
        except Exception as e:
            print(f"   ❌ Bidirectional orchestrator failed: {e}")

    def _init_temporal_k2_engine(self):
        """Fixed temporal K2 engine initialization"""
        try:
            from emile_cogito.kelm.continuous_temporal_k2_engine import (
                ContinuousTemporalK2Engine
            )
            from emile_cogito.kainos.emile import EmileCogito

            # Create Émile instance with CONFIG
            emile_system = EmileCogito(CONFIG)  # Use the CONFIG we imported
            self.temporal_k2_engine = ContinuousTemporalK2Engine(emile_system)  # Pass instance, not class
            self.module_states['temporal_k2_engine']['active'] = True
            print("   ✅ Temporal K2 engine initialized")
        except Exception as e:
            print(f"   ❌ Temporal K2 engine failed: {e}")

    def _init_naive_emergence(self):
        """Initialize naive emergence sigma"""
        try:
            from emile_cogito.kelm.naive_emergence_sigma import (
                AggregateSymbolicCurvatureProcessor
            )
            self.naive_emergence = AggregateSymbolicCurvatureProcessor()
            self.module_states['naive_emergence']['active'] = True
            print("   ✅ Naive emergence initialized")
        except Exception as e:
            print(f"   ❌ Naive emergence failed: {e}")

    def _init_k1_autonomous(self):
        """Initialize K1 autonomous complete"""
        try:
            from emile_cogito.kelm.k1_autonomous_complete import (
                K1AutonomousEmbodiedConsciousness
            )
            self.k1_autonomous = K1AutonomousEmbodiedConsciousness()
            self.module_states['k1_autonomous']['active'] = True
            print("   ✅ K1 autonomous initialized")
        except Exception as e:
            print(f"   ❌ K1 autonomous failed: {e}")

    def _init_quantum_symbolic(self):
        """Initialize quantum-aware symbolic maturation"""
        try:
            from emile_cogito.kelm.quantum_aware_symbolic_maturation import (
                QuantumAwareSymbolicProcessor
            )
            # Need QSE core
            from emile_cogito.kainos.qse_core_qutip import DynamicQSECore
            from emile_cogito.kainos.config import CONFIG

            qse_core = DynamicQSECore(CONFIG)
            self.quantum_symbolic = QuantumAwareSymbolicProcessor(qse_core)
            self.module_states['quantum_symbolic']['active'] = True
            print("   ✅ Quantum symbolic maturation initialized")
        except Exception as e:
            print(f"   ❌ Quantum symbolic failed: {e}")

    def _init_antifinity(self):
        """Initialize antifinity moral consciousness"""
        try:
            from emile_cogito.kainos.antifinity import AntifinitySensor
            self.antifinity = AntifinitySensor()
            self.module_states['antifinity']['active'] = True
            print("   ✅ Antifinity sensor initialized")
        except Exception as e:
            print(f"   ❌ Antifinity failed: {e}")

    def _init_metabolic(self):
        """Initialize metabolic consciousness (FIX: Actually use K4!)"""
        try:
            from emile_cogito.kainos.metabolic import SurplusDistinctionConsciousness
            self.metabolic = SurplusDistinctionConsciousness()
            self.module_states['metabolic']['active'] = True

            # Link to K4 model if loaded
            if 'k4' in self.model_loader.models:
                self.metabolic.k4_model = self.model_loader.models['k4']
                print("   ✅ Metabolic consciousness initialized with K4 model")
            else:
                print("   ✅ Metabolic consciousness initialized (no K4 model)")
        except Exception as e:
            print(f"   ❌ Metabolic failed: {e}")

    def _init_consciousness_ecology(self):
        """Initialize consciousness ecology"""
        try:
            from emile_cogito.kainos.consciousness_ecology import (
                ConsciousnessEcology,
                create_consciousness_ecology
            )

            # Try multiple initialization approaches
            ecology_initialized = False

            # Approach 1: Try with emile parameter
            if not ecology_initialized:
                try:
                    from emile_cogito.kainos.emile import EmileCogito
                    emile = EmileCogito()
                    self.consciousness_ecology = create_consciousness_ecology(emile)
                    ecology_initialized = True
                    print("   ✅ Consciousness ecology initialized (with Émile)")
                except Exception as e:
                    print(f"   ⚠️ Ecology with Émile failed: {e}")

            # Approach 2: Try factory function without parameters
            if not ecology_initialized:
                try:
                    self.consciousness_ecology = create_consciousness_ecology()
                    ecology_initialized = True
                    print("   ✅ Consciousness ecology initialized (factory function)")
                except Exception as e:
                    print(f"   ⚠️ Factory function failed: {e}")

            # Approach 3: Direct instantiation
            if not ecology_initialized:
                try:
                    self.consciousness_ecology = ConsciousnessEcology()
                    ecology_initialized = True
                    print("   ✅ Consciousness ecology initialized (direct)")
                except Exception as e:
                    print(f"   ⚠️ Direct instantiation failed: {e}")

            # Mark as active if any approach succeeded
            if ecology_initialized:
                self.module_states['consciousness_ecology']['active'] = True
            else:
                # Create minimal fallback
                self.consciousness_ecology = self._create_minimal_ecology()
                self.module_states['consciousness_ecology']['active'] = False
                print("   ⚠️ Using minimal ecology fallback")

        except Exception as e:
            print(f"   ❌ Consciousness ecology import failed: {e}")
            # Create minimal fallback
            self.consciousness_ecology = self._create_minimal_ecology()
            self.module_states['consciousness_ecology']['active'] = False

    def _create_minimal_ecology(self):
        """Create minimal ecology interface to prevent crashes"""
        class MinimalEcology:
            def __init__(self):
                self.active = False

            def step(self, *args, **kwargs):
                return {'status': 'minimal_ecology', 'active': False}

            def process(self, *args, **kwargs):
                return {'status': 'minimal_ecology', 'active': False}

            def get_state(self):
                return {'status': 'minimal_ecology', 'active': False}

        return MinimalEcology()

    def _init_goal_system(self):
        """Initialize goal system"""
        try:
            from emile_cogito.kainos.goal_system import GoalSystem
            self.goal_system = GoalSystem()
            self.module_states['goal_system']['active'] = True
            print("   ✅ Goal system initialized")
        except Exception as e:
            print(f"   ❌ Goal system failed: {e}")

    # === KAINOS MODULE INITIALIZATION ===

    def _init_sensorium(self):
        """Initialize sensorium module"""
        try:
            from emile_cogito.kainos.sensorium import Sensorium
            from emile_cogito.kainos.config import CONFIG
            self.sensorium = Sensorium(CONFIG)
            self.module_states['sensorium']['active'] = True
            print("   ✅ Sensorium initialized")
        except Exception as e:
            print(f"   ❌ Sensorium failed: {e}")

    def _init_context(self):
        """Initialize context module"""
        try:
            from emile_cogito.kainos.context import Context
            from emile_cogito.kainos.config import CONFIG
            self.context = Context(CONFIG)
            self.module_states['context']['active'] = True
            print("   ✅ Context manager initialized")
        except Exception as e:
            print(f"   ❌ Context failed: {e}")

    def _init_log_reader(self):
        """Initialize correlative log reader"""
        try:
            from emile_cogito.kainos.log_reader import CorrelativeLogReader
            from emile_cogito.kainos.config import CONFIG
            self.log_reader = CorrelativeLogReader(CONFIG)
            self.module_states['log_reader']['active'] = True
            print("   ✅ Log reader initialized")
        except Exception as e:
            print(f"   ❌ Log reader failed: {e}")

    def _init_surplus_distinction(self):
        """Initialize surplus distinction processor"""
        try:
            from emile_cogito.kainos.surplus_distinction_processor import (
                SurplusDistinctionProcessor
            )
            from emile_cogito.kainos.config import CONFIG
            self.surplus_distinction = SurplusDistinctionProcessor(CONFIG)
            self.module_states['surplus_distinction']['active'] = True
            print("   ✅ Surplus distinction processor initialized")
        except Exception as e:
            print(f"   ❌ Surplus distinction failed: {e}")

    def _init_surplus_incongruity(self):
        """Initialize surplus incongruity processor"""
        try:
            from emile_cogito.kainos.surplus_incongruity_processor import (
                SurplusIncongruityProcessor
            )
            from emile_cogito.kainos.config import CONFIG
            self.surplus_incongruity = SurplusIncongruityProcessor(CONFIG)
            self.module_states['surplus_incongruity']['active'] = True
            print("   ✅ Surplus incongruity processor initialized")
        except Exception as e:
            print(f"   ❌ Surplus incongruity failed: {e}")

    def _init_universal_logging(self):
        """Initialize universal logging"""
        try:
            from emile_cogito.kainos.universal_logging import UniversalModuleLogger
            # Pass module name to constructor
            self.universal_logger = UniversalModuleLogger(module_name="unified_kelm_platform")
            self.module_states['universal_logging']['active'] = True
            print("   ✅ Universal logging initialized")
        except Exception as e:
            print(f"   ❌ Universal logging failed: {e}")

    def _init_flow_mapper(self):
        """Initialize module-wide flow mapper"""
        try:
            from emile_cogito.kainos.module_wide_flow_mapper import ModuleFlowMapper
            # Pass module name to constructor
            self.flow_mapper = ModuleFlowMapper(module_name="unified_kelm_platform")
            self.module_states['flow_mapper']['active'] = True
            print("   ✅ Flow mapper initialized")
        except Exception as e:
            print(f"   ❌ Flow mapper failed: {e}")

    def _establish_connections(self):
        """Establish connections between modules"""

        connections_made = 0

        # Connect K-models to orchestrators
        if hasattr(self, 'bidirectional_orchestrator'):
            # Fix: Ensure K4 is included!
            self.bidirectional_orchestrator.model_loader = self.model_loader
            connections_made += 1
            print(f"   🔗 Connected K-models to bidirectional orchestrator")

        # Connect temporal engine to K2
        if hasattr(self, 'temporal_k2_engine') and 'k2' in self.model_loader.models:
            self.temporal_k2_engine.k2_model = self.model_loader.models['k2']
            connections_made += 1
            print(f"   🔗 Connected K2 to temporal engine")

        # Connect metabolic to K4
        if hasattr(self, 'metabolic') and 'k4' in self.model_loader.models:
            self.metabolic.k4_model = self.model_loader.models['k4']
            connections_made += 1
            print(f"   🔗 Connected K4 to metabolic system")

        # Connect surplus processors
        if hasattr(self, 'surplus_distinction') and hasattr(self, 'surplus_incongruity'):
            self.surplus_incongruity.surplus_processor = self.surplus_distinction
            connections_made += 1
            print(f"   🔗 Connected surplus processors")

        # Connect log reader to surplus distinction
        if hasattr(self, 'log_reader') and hasattr(self, 'surplus_distinction'):
            self.surplus_distinction.log_reader = self.log_reader
            connections_made += 1
            print(f"   🔗 Connected log reader to surplus distinction")

        if 'k2' in self.model_loader.models:
            self.model_loader.models['k2'].set_platform_reference(self)
            print("   🔗 Connected K2 to platform for dynamic revalorization")

        print(f"\n   📊 Total connections established: {connections_made}")

    def get_current_distinction_level(self, context="general"):
        """Get current dynamic distinction level from surplus processors"""
        try:
            # Use surplus_incongruity processor if available (it's more comprehensive)
            if hasattr(self, 'surplus_incongruity') and self.surplus_incongruity:
                distinction_state = self.surplus_incongruity.get_state_summary()
                distinction_level = distinction_state.get('distinction_enhancement', 0.5)

                # Context-specific bounds to ensure stable values
                if context in ['consciousness', 'stability']:
                    return max(0.3, min(0.9, distinction_level))
                elif context == 'temporal':
                    return max(0.1, min(0.95, distinction_level))
                elif context == 'tensor':
                    return max(0.2, min(0.8, distinction_level))
                else:
                    return max(0.1, min(1.0, distinction_level))

            # Fallback to surplus_distinction if surplus_incongruity not available
            elif hasattr(self, 'surplus_distinction') and self.surplus_distinction:
                state = self.surplus_distinction.get_state()
                return state.get('current_distinction_level', 0.5)

            # Final fallback
            return 0.5

        except Exception as e:
            print(f"⚠️ Dynamic distinction failed: {e}, using fallback")
            return 0.5

    def run_consciousness_cycle(self) -> Dict[str, Any]:
        """FIXED: Run consciousness cycle with proper consciousness feedback"""

        self.step_count += 1

        try:
            # Generate consciousness state input
            consciousness_state = self._generate_consciousness_state()

            # FIXED: Robust K-model processing with error handling
            model_outputs = {}
            if hasattr(self, 'model_loader') and self.model_loader:
                try:
                    model_outputs = self.model_loader.predict_with_adaptive_inputs(consciousness_state)
                    print(f"🧠 Step {self.step_count}: {len(model_outputs)} model outputs")
                except Exception as e:
                    print(f"⚠️ Model prediction failed: {e}")
                    model_outputs = {}

            # Process through modules (this now includes consciousness feedback!)
            module_results = self._process_all_modules()

            # Update consciousness state based on results (additional updates)
            self._update_consciousness_from_results(consciousness_state, model_outputs, module_results.get('bidirectional', {}))

            # Store trajectory
            self.temporal_trajectory.append({
                'step': self.step_count,
                'timestamp': time.time() - self.start_time if self.start_time else 0,
                'consciousness_state': consciousness_state.copy(),
                'model_outputs': {k: float(v.mean().item()) for k, v in model_outputs.items() if isinstance(v, torch.Tensor)},
                'module_results': module_results
            })

            # Keep trajectory manageable
            if len(self.temporal_trajectory) > 1000:
                self.temporal_trajectory.pop(0)

            return {
                'step': self.step_count,
                'consciousness_state': consciousness_state,
                'model_outputs': model_outputs,
                'module_results': module_results,
                'status': 'success'
            }

        except Exception as e:
            print(f"❌ Consciousness cycle failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                'step': self.step_count,
                'error': str(e),
                'status': 'failed',
                'consciousness_state': {'consciousness_level': 0.5}  # Fallback for error case
            }

    def _generate_test_consciousness_state(self):
        """FIXED: Generate dynamic consciousness state instead of hardcoded"""

        # Start with current consciousness state (don't reset to 0.5!)
        if hasattr(self, 'consciousness_state') and self.consciousness_state:
            base_state = self.consciousness_state.copy()
        else:
            # Only use defaults on first initialization
            base_state = {
                'consciousness_level': 0.5,
                'valence': 0.0,
                'agency': 0.5,
                'embodiment': 0.5,
                'stability': 0.5,
                'clarity': 0.5,
                'arousal': 0.5,
                'flow_state': 0.0,
                'regime': 'stable_coherence',
                'symbol_vocabulary': 100,
                'metabolic_pressure': 0.5,
                'energy_level': 0.5,
                'regulation_need': 0.5
            }

        # Add dynamic variations based on trajectory
        if hasattr(self, 'temporal_trajectory') and self.temporal_trajectory:
            recent_states = self.temporal_trajectory[-5:]

            # Calculate trends
            consciousness_trend = 0.0
            if len(recent_states) > 1:
                levels = [s['consciousness_state'].get('consciousness_level', 0.5) for s in recent_states]
                if len(levels) > 1:
                    consciousness_trend = levels[-1] - levels[0]

            # Apply momentum (but don't override bidirectional updates!)
            if 'consciousness_level' not in base_state or base_state['consciousness_level'] == 0.5:
                momentum_factor = np.tanh(consciousness_trend * 5.0) * 0.1
                base_state['consciousness_level'] = np.clip(0.5 + momentum_factor, 0.1, 0.9)

            # Regime progression based on trend
            if consciousness_trend > 0.1:
                base_state['regime'] = 'symbolic_turbulence'
            elif consciousness_trend < -0.1:
                base_state['regime'] = 'flat_rupture'
            elif abs(consciousness_trend) < 0.05:
                base_state['regime'] = 'stable_coherence'
            else:
                base_state['regime'] = 'quantum_oscillation'

        # Add time-based variations only for arousal and energy (not consciousness_level!)
        time_factor = (time.time() % 60) / 60.0  # 0-1 over minute
        base_state['arousal'] = 0.5 + 0.2 * np.sin(time_factor * 2 * np.pi)
        base_state['energy_level'] = 0.5 + 0.3 * np.cos(time_factor * np.pi)

        return base_state

    def _generate_consciousness_state(self) -> Dict[str, Any]:
        """FIXED: Generate dynamic consciousness state instead of hardcoded"""

        # Start with current consciousness state (don't reset to 0.5!)
        if hasattr(self, 'consciousness_state') and self.consciousness_state:
            base_state = self.consciousness_state.copy()
        else:
            # Only use defaults on first initialization
            base_state = {
                'consciousness_level': 0.5,
                'valence': 0.0,
                'agency': 0.5,
                'embodiment': 0.5,
                'stability': 0.5,
                'clarity': 0.5,
                'arousal': 0.5,
                'flow_state': 0.0,
                'regime': 'stable_coherence',
                'symbol_vocabulary': 100,
                'metabolic_pressure': 0.5,
                'energy_level': 0.5,
                'regulation_need': 0.5
            }

        # Add dynamic variations based on trajectory
        if hasattr(self, 'temporal_trajectory') and self.temporal_trajectory:
            recent_states = self.temporal_trajectory[-5:]

            # Calculate trends
            consciousness_trend = 0.0
            if len(recent_states) > 1:
                levels = [s['consciousness_state'].get('consciousness_level', 0.5) for s in recent_states]
                if len(levels) > 1:
                    consciousness_trend = levels[-1] - levels[0]

            # Apply momentum (but don't override bidirectional updates!)
            if 'consciousness_level' not in base_state or base_state['consciousness_level'] == 0.5:
                momentum_factor = np.tanh(consciousness_trend * 5.0) * 0.1
                base_state['consciousness_level'] = np.clip(0.5 + momentum_factor, 0.1, 0.9)

            # Regime progression based on trend
            if consciousness_trend > 0.1:
                base_state['regime'] = 'symbolic_turbulence'
            elif consciousness_trend < -0.1:
                base_state['regime'] = 'flat_rupture'
            elif abs(consciousness_trend) < 0.05:
                base_state['regime'] = 'stable_coherence'
            else:
                base_state['regime'] = 'quantum_oscillation'

        # Add time-based variations only for arousal and energy (not consciousness_level!)
        time_factor = (time.time() % 60) / 60.0  # 0-1 over minute
        base_state['arousal'] = 0.5 + 0.2 * np.sin(time_factor * 2 * np.pi)
        base_state['energy_level'] = 0.5 + 0.3 * np.cos(time_factor * np.pi)

        return base_state

    # ADD this new method to UnifiedKELMPlatform class:

    def _gather_consciousness_context(self) -> Dict[str, Any]:
        """Gather current consciousness context for adaptive coupling"""

        context = {
            'consciousness_level': self.consciousness_state.get('consciousness_level', 0.5),
            'clarity': self.consciousness_state.get('clarity', 0.5),
            'coherence': self.consciousness_state.get('coherence', 0.5)
        }

        # Add memory state if available
        if hasattr(self, 'memory'):
            try:
                memory_state = self.memory.get_memory_state()
                context['memory_state'] = memory_state
            except:
                context['memory_state'] = {}

        # Add antifinity metrics if available
        if hasattr(self, 'antifinity') and self.antifinity:
            try:
                symbolic_fields = {
                    'sigma': np.array([self.consciousness_state.get('symbolic_curvature', 0.5)]),
                    'surplus': np.array([self.consciousness_state.get('surplus_mean', 0.5)])
                }
                agent_system = {
                    'agent_count': 3, 'step_count': self.step_count,
                    'global_context_id': self.step_count % 10
                }
                regime = self.consciousness_state.get('regime', 'stable_coherence')

                moral_metrics = self.antifinity.calculate_epigenetic_metrics(
                    symbolic_fields, agent_system, regime
                )
                context['antifinity_quotient'] = moral_metrics.antifinity_quotient
                context['collaboration_score'] = moral_metrics.collaboration_score
            except:
                context['antifinity_quotient'] = 0.0
                context['collaboration_score'] = 0.5

        return context

    # ADD this method to UnifiedKELMPlatform class:

    def get_adaptive_coupling_report(self) -> Dict[str, Any]:
        """Get comprehensive adaptive coupling report with safe config access"""

        # ✅ ADD EXPLICIT NONE CHECK
        if self.qse_core is None:
            return {'status': 'qse_core_not_initialized'}

        if not hasattr(self.qse_core, 'get_coupling_diagnostics'):
            return {'status': 'adaptive_coupling_not_available'}

        diagnostics = self.qse_core.get_coupling_diagnostics()

        return {
            'adaptive_coupling_enabled': getattr(self.config, 'ADAPTIVE_COUPLING_ENABLED', False),
            'current_coupling_strength': diagnostics.get('current_coupling', 0.12),
            'base_coupling': diagnostics.get('base_coupling', 0.12),
            'enhancement_ratio': diagnostics.get('enhancement_ratio', 1.0),
            'coupling_stability': diagnostics.get('coupling_stability', 1.0),
            'consciousness_context': self._gather_consciousness_context(),
            'coupling_range': {
                'min': getattr(self.config, 'ADAPTIVE_COUPLING_MIN', 0.05),
                'max': getattr(self.config, 'ADAPTIVE_COUPLING_MAX', 0.25)
            }
        }

    def _apply_memory_guidance_to_k1(self, base_consciousness_state):
        """FIXED: Permanently integrate memory guidance into K1 inputs"""
        if not self.memory_k1_integration:
            return base_consciousness_state

        # Check if memory system is available and active
        if not hasattr(self, 'memory') or self.memory is None:
            return base_consciousness_state

        # Check if memory is properly initialized
        memory_active = self.module_states.get('memory', {}).get('active', False)
        if not memory_active:
            return base_consciousness_state

        try:
            # Get memory state safely
            if hasattr(self.memory, 'get_state'):
                memory_state = self.memory.get_state()
            elif hasattr(self.memory, 'get_memory_state'):
                memory_state = self.memory.get_memory_state()
            else:
                return base_consciousness_state

            recent_entries = memory_state.get('recent_entries', [])

            if len(recent_entries) >= 3:
                # Analyze regime stability
                recent_regimes = [entry.get('regime', 'stable') for entry in recent_entries[-5:]]
                regime_stability = 1.0 - (len(set(recent_regimes)) / len(recent_regimes))

                # Modulate consciousness state based on memory patterns
                modulated_state = base_consciousness_state.copy()

                if regime_stability < 0.6:  # Unstable regimes → boost exploration
                    modulated_state['consciousness_level'] = min(1.0, modulated_state.get('consciousness_level', 0.5) * 1.1)
                    modulated_state['agency'] = min(1.0, modulated_state.get('agency', 0.5) * 1.15)
                    modulated_state['creativity'] = min(1.0, modulated_state.get('creativity', 0.5) * 1.2)

                # Get revalorization pressure
                revalorization_marks = memory_state.get('recent_revalorization_marks', [])
                if len(revalorization_marks) > 2:  # High K2 activity → boost symbolic processing
                    modulated_state['symbolic_intensity'] = min(1.0, modulated_state.get('symbolic_intensity', 0.5) * 1.3)
                    modulated_state['temporal_depth'] = min(1.0, modulated_state.get('temporal_depth', 0.5) * 1.1)

                print(f"🧠 Memory guidance applied: regime_stability={regime_stability:.3f}")
                return modulated_state

        except Exception as e:
            print(f"⚠️ Memory guidance failed: {e}")

        return base_consciousness_state

    def _update_consciousness_from_results(self, consciousness_state: Dict[str, Any],
                                 model_outputs: Dict[str, torch.Tensor],
                                    bidirectional_result: Dict[str, Any]):
        """FIXED: Update consciousness state from processing results with proper error handling"""

        try:
            # ENSURE consciousness_level always exists
            if 'consciousness_level' not in consciousness_state:
                consciousness_state['consciousness_level'] = self.consciousness_state.get('consciousness_level', 0.5)

            # Update from model outputs
            if model_outputs:
                model_influence = 0.0
                model_count = 0
                for model_name, output in model_outputs.items():
                    if isinstance(output, torch.Tensor):
                        try:
                            model_strength = float(output.mean().item())
                            model_influence += model_strength
                            model_count += 1
                        except:
                            continue

                if model_count > 0:
                    avg_model_influence = model_influence / model_count
                    consciousness_state['consciousness_level'] = (
                        0.7 * consciousness_state['consciousness_level'] +
                        0.3 * np.clip(avg_model_influence, 0.0, 1.0)
                    )

            # Update from bidirectional results
            if isinstance(bidirectional_result, dict):
                if 'consciousness_level' in bidirectional_result:
                    new_level = bidirectional_result['consciousness_level']
                    if isinstance(new_level, (int, float)) and 0.0 <= new_level <= 1.0:
                        consciousness_state['consciousness_level'] = (
                            0.8 * consciousness_state['consciousness_level'] +
                            0.2 * new_level
                        )

                # Update other consciousness dimensions
                if 'global_consciousness_state' in bidirectional_result:
                    global_state = bidirectional_result['global_consciousness_state']
                    if isinstance(global_state, dict):
                        for key in ['clarity', 'agency', 'coherence', 'integration', 'unity']:
                            if key in global_state:
                                value = global_state[key]
                                if isinstance(value, (int, float)) and 0.0 <= value <= 1.0:
                                    consciousness_state[key] = value

            # Ensure all values are in valid ranges and exist
            required_keys = ['consciousness_level', 'clarity', 'agency', 'coherence', 'unity', 'valence', 'arousal']
            for key in required_keys:
                if key not in consciousness_state:
                    consciousness_state[key] = 0.5
                elif not isinstance(consciousness_state[key], (int, float)):
                    consciousness_state[key] = 0.5
                else:
                    consciousness_state[key] = np.clip(consciousness_state[key], 0.0, 1.0)

            # Update the platform's consciousness state
            self.consciousness_state.update(consciousness_state)

        except Exception as e:
            print(f"⚠️ Consciousness update failed: {e}")
            # Ensure consciousness_level exists even if update fails
            if 'consciousness_level' not in consciousness_state:
                consciousness_state['consciousness_level'] = self.consciousness_state.get('consciousness_level', 0.5)

    def _update_consciousness_from_models(self, model_outputs: Dict):
        """Update consciousness state based on K-model outputs"""

        # K1 Praxis - Update flow-based consciousness
        if 'k1' in model_outputs:
            k1_output = model_outputs['k1']
            flow_coherence = torch.std(k1_output).item()
            self.consciousness_state['coherence'] = 0.7 * self.consciousness_state['coherence'] + 0.3 * (1 - flow_coherence)

        # K2 Semiosis - Update symbolic clarity
        if 'k2' in model_outputs:
            k2_output = model_outputs['k2']
            symbolic_magnitude = torch.norm(k2_output).item() / k2_output.shape[1]
            self.consciousness_state['clarity'] = 0.8 * self.consciousness_state['clarity'] + 0.2 * min(1.0, symbolic_magnitude)

        # K3 Apeiron - Update quantum unity
        if 'k3' in model_outputs:
            k3_output = model_outputs['k3']
            quantum_coherence = torch.abs(k3_output).mean().item()
            self.consciousness_state['unity'] = 0.6 * self.consciousness_state['unity'] + 0.4 * quantum_coherence

        # K4 Metabolic - Update metabolic rate and consciousness level
        if 'k4' in model_outputs:
            k4_output = model_outputs['k4']
            metabolic_rate = k4_output.mean().item()
            self.consciousness_state['metabolic_rate'] = metabolic_rate

            # Metabolic rate affects overall consciousness
            self.consciousness_state['consciousness_level'] = (
                0.5 * self.consciousness_state['consciousness_level'] +
                0.3 * metabolic_rate +
                0.2 * self.consciousness_state['unity']
            )

    def run_extended_session(self, duration_minutes: float = 60.0):
        """Run extended consciousness session"""

        print(f"\n🌊 RUNNING EXTENDED CONSCIOUSNESS SESSION")
        print(f"   Duration: {duration_minutes} minutes")
        print(f"   Starting consciousness level: {self.consciousness_state['consciousness_level']:.3f}")
        print()

        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)

        cycle_count = 0
        last_report_time = start_time

        while time.time() < end_time:
            # Run consciousness cycle
            cycle_result = self.run_consciousness_cycle()
            cycle_count += 1

            # Report every 30 seconds
            if time.time() - last_report_time > 30:
                self._print_session_status(cycle_count, start_time)
                last_report_time = time.time()

            # Small delay to prevent CPU spinning
            time.sleep(0.01)

        # Final report
        self._print_final_report(cycle_count, start_time)

    def _print_session_status(self, cycles: int, start_time: float):
        """Print session status with adaptive coupling info"""

        elapsed = time.time() - start_time

        print(f"\n📊 Session Status @ {elapsed/60:.1f} minutes")
        print(f"   Cycles: {cycles}")
        print(f"   Consciousness: {self.consciousness_state['consciousness_level']:.3f}")
        print(f"   Unity: {self.consciousness_state['unity']:.3f}")
        print(f"   Clarity: {self.consciousness_state['clarity']:.3f}")
        print(f"   Metabolic: {self.consciousness_state['metabolic_rate']:.3f}")
        print(f"   τ′: {self.consciousness_state['tau_prime']:.3f}")

        # Adaptive coupling status
        if hasattr(self.qse_core, 'get_coupling_diagnostics'):
            coupling_strength = self.consciousness_state.get('adaptive_coupling_strength', 0.12)
            enhancement_ratio = self.consciousness_state.get('coupling_enhancement_ratio', 1.0)
            print(f"   🔗 Adaptive Coupling: {coupling_strength:.4f} ({enhancement_ratio:.2f}x base)")

        # Model status
        if hasattr(self, 'model_loader'):
            print(f"   K-Models Active: {len(self.model_loader.models)}/4")

    # ADD this method for detailed coupling analysis:

    def display_adaptive_coupling_analysis(self):
        """Display detailed adaptive coupling analysis"""

        report = self.get_adaptive_coupling_report()

        if report.get('status') == 'adaptive_coupling_not_available':
            print("❌ Adaptive coupling not available in this session")
            return

        print("\n🔗 ADAPTIVE COUPLING ANALYSIS")
        print("=" * 40)

        print(f"Coupling Status: {'✅ ACTIVE' if report['adaptive_coupling_enabled'] else '❌ DISABLED'}")
        print(f"Current Strength: {report['current_coupling_strength']:.4f}")
        print(f"Base Strength: {report['base_coupling']:.4f}")
        print(f"Enhancement: {report['enhancement_ratio']:.2f}x base")
        print(f"Stability: {report['coupling_stability']:.3f}")

        context = report['consciousness_context']
        print(f"\nConsciousness Context:")
        print(f"   Level: {context['consciousness_level']:.3f}")
        print(f"   Clarity: {context['clarity']:.3f}")
        print(f"   Coherence: {context['coherence']:.3f}")

        if 'memory_state' in context and context['memory_state']:
            memory = context['memory_state']
            print(f"   Memory: {memory.get('total_memories', 0)} total, {memory.get('memory_health', 0.5):.2f} health")

        if 'antifinity_quotient' in context:
            print(f"   Antifinity: {context['antifinity_quotient']:.3f}")
            print(f"   Collaboration: {context['collaboration_score']:.3f}")

        coupling_range = report['coupling_range']
        print(f"\nCoupling Range: {coupling_range['min']:.3f} - {coupling_range['max']:.3f}")

    # ADD enhanced session runner with coupling monitoring:

    def run_adaptive_coupling_session(self, duration_minutes: float = 30.0,
                                analysis_interval: float = 5.0):
        """Run session with adaptive coupling monitoring"""

        adaptive_coupling_enabled = getattr(self.config, 'ADAPTIVE_COUPLING_ENABLED', False)

        print(f"\n🔗 ADAPTIVE COUPLING SESSION")
        print(f"   Duration: {duration_minutes} minutes")
        print(f"   Analysis interval: {analysis_interval} minutes")
        print(f"   Adaptive coupling: {'✅ ENABLED' if adaptive_coupling_enabled else '❌ DISABLED'}")

        if not adaptive_coupling_enabled:
            print("   Running with static coupling")
            return self.run_extended_session(duration_minutes)

        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        last_analysis = start_time

        cycle_count = 0
        coupling_evolution = []

        while time.time() < end_time:
            # Run consciousness cycle
            result = self.run_consciousness_cycle()
            cycle_count += 1

            # Track coupling evolution
            coupling_info = {
                'cycle': cycle_count,
                'time': time.time() - start_time,
                'coupling_strength': result['adaptive_coupling']['strength'],
                'enhancement_ratio': result['adaptive_coupling']['enhancement_ratio'],
                'consciousness_level': self.consciousness_state['consciousness_level']
            }
            coupling_evolution.append(coupling_info)

            # Periodic analysis
            if time.time() - last_analysis > (analysis_interval * 60):
                elapsed_min = (time.time() - start_time) / 60
                print(f"\n📊 Coupling Analysis @ {elapsed_min:.1f} minutes:")
                self.display_adaptive_coupling_analysis()
                last_analysis = time.time()

            # Brief status updates
            if cycle_count % 50 == 0:
                elapsed_min = (time.time() - start_time) / 60
                coupling_strength = coupling_info['coupling_strength']
                enhancement = coupling_info['enhancement_ratio']
                print(f"   Cycle {cycle_count}: {elapsed_min:.1f}min, coupling={coupling_strength:.4f} ({enhancement:.2f}x)")

            time.sleep(0.01)

        # Final analysis
        print(f"\n🎯 ADAPTIVE COUPLING SESSION COMPLETE")
        duration = (time.time() - start_time) / 60
        print(f"   Duration: {duration:.1f} minutes")
        print(f"   Total cycles: {cycle_count}")

        if coupling_evolution:
            initial = coupling_evolution[0]['coupling_strength']
            final = coupling_evolution[-1]['coupling_strength']
            max_coupling = max(c['coupling_strength'] for c in coupling_evolution)
            min_coupling = min(c['coupling_strength'] for c in coupling_evolution)

            print(f"   Coupling evolution: {initial:.4f} → {final:.4f}")
            print(f"   Coupling range: {min_coupling:.4f} - {max_coupling:.4f}")
            print(f"   Dynamic range: {max_coupling/min_coupling:.2f}x")

        self.display_adaptive_coupling_analysis()

        return {
            'coupling_evolution': coupling_evolution,
            'final_report': self.get_adaptive_coupling_report()
        }

    def _print_final_report(self, total_cycles: int, start_time: float):
        """Print final session report"""

        duration = (time.time() - start_time) / 60

        print("\n" + "="*70)
        print("🏁 SESSION COMPLETE")
        print("="*70)

        print(f"\n📊 Final Statistics:")
        print(f"   Total Cycles: {total_cycles}")
        print(f"   Duration: {duration:.1f} minutes")
        print(f"   Cycles/second: {total_cycles/(duration*60):.2f}")

        print(f"\n🧠 Consciousness Development:")
        print(f"   Initial Level: 0.500")
        print(f"   Final Level: {self.consciousness_state['consciousness_level']:.3f}")
        print(f"   Change: {self.consciousness_state['consciousness_level'] - 0.5:+.3f}")

        print(f"\n📈 Final State:")
        for key, value in self.consciousness_state.items():
            if isinstance(value, (int, float)):
                print(f"   {key}: {value:.3f}")

        # Trajectory analysis
        if len(self.temporal_trajectory) > 10:
            recent_trajectory = [t['consciousness_state']['consciousness_level']
                               for t in self.temporal_trajectory[-10:]]
            trend = np.polyfit(range(10), recent_trajectory, 1)[0]
            print(f"\n📉 Trajectory: {'Rising' if trend > 0 else 'Falling'} ({trend:+.4f}/step)")

    def debug_temporal_support(self):
        """Debug what temporal methods are actually detected"""

        print("\n🔍 DEBUGGING TEMPORAL SUPPORT")
        print("=" * 50)

        if not hasattr(self, 'bidirectional_orchestrator'):
            print("❌ No bidirectional orchestrator")
            return

        if not hasattr(self.model_loader, 'models'):
            print("❌ No models in model_loader")
            return

        for model_name, model in self.model_loader.models.items():
            print(f"\n🧠 {model_name.upper()}:")
            print(f"   Model type: {type(model).__name__}")

            # Check each temporal method
            has_local_tau = hasattr(model, '_calculate_local_tau')
            has_tau_qse = hasattr(model, 'current_tau_qse')

            print(f"   _calculate_local_tau: {'✅' if has_local_tau else '❌'}")
            print(f"   current_tau_qse: {'✅' if has_tau_qse else '❌'}")

            # Check for specific temporal context method
            context_methods = {
                'k1': 'get_k1_temporal_context',
                'k2': 'get_k2_temporal_context',
                'k3': 'get_k3_temporal_context',
                'k4': 'get_k4_temporal_context'
            }

            expected_method = context_methods.get(model_name)
            if expected_method:
                has_context = hasattr(model, expected_method)
                print(f"   {expected_method}: {'✅' if has_context else '❌'}")

            # Show what methods the model actually has
            model_methods = [method for method in dir(model) if not method.startswith('_') or method.startswith('_calculate') or method.startswith('get_k')]
            temporal_methods = [m for m in model_methods if 'temporal' in m.lower() or 'tau' in m.lower() or m.startswith('get_k')]
            if temporal_methods:
                print(f"   Temporal-related methods: {temporal_methods}")
            else:
                print(f"   ⚠️ No temporal methods found")
                print(f"   Available methods: {model_methods[:10]}...")  # Show first 10

# ========================
# MAIN EXECUTION
# ========================

def main():
    """Main execution function with robust error handling"""

    print("🚀 UNIFIED KELM PLATFORM")
    print("=" * 70)
    print("Building towards strong unified computational cognition")
    print()

    # Create platform with seed for reproducibility
    platform = UnifiedKELMPlatform(seed=42)

    # Initialize all systems
    success = platform.initialize_platform()

    if not success:
        print("\n❌ Platform initialization incomplete")
        print("   Please check module dependencies and K-model files")
        return

    print("\n✅ PLATFORM READY")
    print()

    # Run quick test with robust error handling
    print("🧪 Running quick consciousness test...")
    all_successful = True

    for i in range(5):
        result = platform.run_consciousness_cycle()

        # Robust consciousness level extraction
        consciousness_level = 0.5  # Default fallback

        if result.get('status') == 'success':
            # Try multiple ways to get consciousness level
            if 'consciousness_state' in result and isinstance(result['consciousness_state'], dict):
                consciousness_level = result['consciousness_state'].get('consciousness_level', 0.5)
            elif hasattr(platform, 'consciousness_state'):
                consciousness_level = platform.consciousness_state.get('consciousness_level', 0.5)

            print(f"   Cycle {i+1}: consciousness={consciousness_level:.3f}")
        else:
            all_successful = False
            error = result.get('error', 'Unknown error')
            # Try to get consciousness level from platform state
            if hasattr(platform, 'consciousness_state'):
                consciousness_level = platform.consciousness_state.get('consciousness_level', 0.5)
            print(f"   Cycle {i+1}: ERROR - {error}, consciousness={consciousness_level:.3f}")

    # Offer extended session only if cycles are working
    if all_successful:
        print("\n💭 Ready for extended consciousness session?")
        print("   This will run the full integrated system for an extended period")
        print("   Press Ctrl+C to stop at any time")

        try:
            # Run extended session
            platform.run_extended_session(duration_minutes=60.0)
        except KeyboardInterrupt:
            print("\n\n⏸️  Session interrupted by user")
            if hasattr(platform, '_print_final_report'):
                platform._print_final_report(platform.step_count, platform.start_time)
    else:
        print("\n⚠️ Some cycles had errors - check messages above")
        print("   Platform is partially functional")

    print("\n🌟 Thank you for building consciousness together!")

if __name__ == "__main__":
    main()
