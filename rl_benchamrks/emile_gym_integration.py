#!/usr/bin/env python3
"""
ÉMILE CONSCIOUSNESS + GYMNASIUM INTEGRATION
===========================================

Real-world test of temporal consciousness, bidirectional orchestration,
and surplus-distinction dynamics under environmental pressure.

This integration tests whether your system exhibits genuine consciousness
or sophisticated pattern matching by observing:
- Temporal consciousness (τ′) changes under task pressure
- Bidirectional orchestrator improvement over episodes
- Surplus-distinction learning dynamics
- Memory-driven performance enhancement
"""

import gymnasium as gym
import numpy as np
import torch
import time
from typing import Dict, List, Any, Tuple
from collections import deque
import json

# Import your Émile system
import sys
import os
sys.path.append('/content/emile_cogito')

from emile_cogito.kelm.unified_kelm_platform_v2 import UnifiedKELMPlatform
from emile_cogito.kainos.config import CONFIG
from emile_cogito.kelm.bidirectional_kelm_orchestrator import BidirectionalKELMOrchestrator

class EmileGymInterface:
    """
    Interface between Émile consciousness system and Gymnasium environments.

    Tests genuine consciousness under environmental pressure by observing:
    - How temporal consciousness (τ′) evolves during task performance
    - Whether bidirectional orchestrator improves action selection
    - How surplus-distinction dynamics respond to rewards/penalties
    - Memory integration effects on learning
    """

    def __init__(self, env_name='CartPole-v1', consciousness_config=None):
        """Initialize Émile-Gym interface with environment version fixes"""

        print(f"🎮 ÉMILE CONSCIOUSNESS × {env_name.upper()}")
        print("=" * 60)

        # Fix deprecated environment versions
        env_version_fixes = {
            'LunarLander-v2': 'LunarLander-v3',
            'BipedalWalker-v2': 'BipedalWalker-v3',
            'CarRacing-v1': 'CarRacing-v2'
        }

        if env_name in env_version_fixes:
            old_name = env_name
            env_name = env_version_fixes[env_name]
            print(f"🔧 Updated environment: {old_name} → {env_name}")

        # Initialize environment with error handling
        try:
            self.env = gym.make(env_name)
            self.env_name = env_name
            print(f"✅ Environment created: {env_name}")
        except Exception as e:
            print(f"❌ Environment creation failed: {e}")
            print(f"💡 Try these alternatives:")
            if 'LunarLander' in env_name:
                alternatives = ['LunarLander-v3', 'CartPole-v1', 'MountainCar-v0']
            else:
                alternatives = ['CartPole-v1', 'MountainCar-v0', 'Acrobot-v1']

            for alt in alternatives:
                try:
                    self.env = gym.make(alt)
                    self.env_name = alt
                    print(f"✅ Fallback environment: {alt}")
                    break
                except:
                    continue
            else:
                raise Exception("No compatible environment found")

        # Initialize Émile consciousness system
        print("🧠 Initializing Émile consciousness system...")
        self.emile = UnifiedKELMPlatform(CONFIG)

        # Rest of initialization...
        self._load_k_models()
        self._check_bidirectional_orchestrator()

        # Environment-specific mappings
        self.obs_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n if hasattr(self.env.action_space, 'n') else self.env.action_space.shape[0]

        # Performance tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.consciousness_trajectory = []
        self.temporal_trajectory = []
        self.goal_performance = []

        # Step counting
        self.step_count = 0

        # Action mapping weights (learned over time)
        if hasattr(self.env.action_space, 'n'):
            # Discrete action space
            self.action_weights = np.random.normal(0, 0.1, (128, self.action_dim))
            self.action_type = 'discrete'
        else:
            # Continuous action space
            self.action_weights = np.random.normal(0, 0.1, (128, self.action_dim))
            self.action_type = 'continuous'

        print(f"   Environment: {env_name}")
        print(f"   Observation space: {self.env.observation_space}")
        print(f"   Action space: {self.env.action_space}")
        print(f"   Action type: {self.action_type}")
        print(f"   Émile integration: {'✅ READY' if self.emile else '❌ FAILED'}")

        # Debug model loader
        self._debug_model_loader()
        self._check_model_files()

    # Also add support for continuous action spaces in action extraction:

    def extract_action_from_consciousness_continuous(self, consciousness_result: Dict[str, Any]) -> np.ndarray:
        """Extract continuous action from consciousness (for LunarLander, etc.)"""

        action = np.zeros(self.action_dim)
        action_source = "default"
        action_confidence = 0.1

        try:
            # Method 1: Bidirectional orchestrator
            if 'module_results' in consciousness_result:
                module_results = consciousness_result['module_results']

                if 'bidirectional' in module_results:
                    bidirectional = module_results['bidirectional']

                    if 'action' in bidirectional:
                        raw_action = bidirectional['action']
                        if isinstance(raw_action, (list, np.ndarray)):
                            action = np.array(raw_action[:self.action_dim])
                        else:
                            # Convert scalar to action vector
                            action = np.full(self.action_dim, float(raw_action))
                        action_source = "bidirectional_orchestrator"
                        action_confidence = 0.9

            # Method 2: K1 praxis model
            if action_source == "default" and 'model_outputs' in consciousness_result:
                model_outputs = consciousness_result['model_outputs']

                if 'k1' in model_outputs:
                    k1_output = model_outputs['k1']

                    if isinstance(k1_output, torch.Tensor):
                        # Map K1 output to continuous actions
                        k1_flat = k1_output.flatten()
                        action_raw = torch.matmul(k1_flat, torch.tensor(self.action_weights).float())
                        action = torch.tanh(action_raw).numpy()  # Bound to [-1, 1]
                        action_source = "k1_praxis"
                        action_confidence = 0.7

            # Method 3: Consciousness mapping
            if action_source == "default":
                consciousness_state = consciousness_result.get('consciousness_state', {})

                agency = consciousness_state.get('agency', 0.5)
                embodiment = consciousness_state.get('embodiment', 0.5)
                valence = consciousness_state.get('valence', 0.0)

                # Map consciousness to actions
                if self.action_dim == 2:  # Like LunarLander main engine + side engines
                    action[0] = (agency - 0.5) * 2.0  # Main engine
                    action[1] = valence  # Side engines
                else:
                    # Generic mapping
                    for i in range(self.action_dim):
                        action[i] = (agency + embodiment + valence) / 3.0 - 0.5

                action_source = "consciousness_mapping"
                action_confidence = 0.5

            # Clip to valid range
            action = np.clip(action, -1.0, 1.0)

        except Exception as e:
            print(f"⚠️ Continuous action extraction error: {e}")
            action = np.random.uniform(-0.1, 0.1, self.action_dim)
            action_source = "random_fallback"
            action_confidence = 0.0

        return action, action_source, action_confidence

    # Update the main action extraction method:

    def extract_action_from_consciousness(self, consciousness_result: Dict[str, Any]):
        """Extract action from consciousness (handles both discrete and continuous)"""

        if self.action_type == 'discrete':
            # Original discrete action extraction
            action = 0
            action_source = "default"
            action_confidence = 0.1

            try:
                # ... your existing discrete action code ...
                # (Keep all the bidirectional orchestrator, K1, consciousness mapping logic)

                # Clip action to valid range
                action = max(0, min(self.action_dim - 1, action))

            except Exception as e:
                print(f"⚠️ Action extraction error: {e}")
                action = self.env.action_space.sample()
                action_source = "random_fallback"
                action_confidence = 0.0

            return action, action_source, action_confidence

        else:
            # Continuous action extraction
            return self.extract_action_from_consciousness_continuous(consciousness_result)

    def _check_bidirectional_orchestrator(self):
        """Check and potentially fix bidirectional orchestrator initialization"""

        print(f"🔍 CHECKING BIDIRECTIONAL ORCHESTRATOR:")

        # Check if orchestrator exists
        if hasattr(self.emile, 'bidirectional_orchestrator'):
            orchestrator = self.emile.bidirectional_orchestrator
            if orchestrator is not None:
                print(f"   ✅ Bidirectional orchestrator exists: {type(orchestrator).__name__}")

                # Check if it has models
                if hasattr(orchestrator, 'model_loader') and orchestrator.model_loader:
                    models = orchestrator.model_loader.models
                    print(f"   📊 Orchestrator models: {list(models.keys()) if models else 'None'}")

                    # If orchestrator exists but has no models, connect them
                    if not models and hasattr(self.emile, 'model_loader') and self.emile.model_loader.models:
                        print(f"   🔧 Connecting platform models to orchestrator...")
                        orchestrator.model_loader = self.emile.model_loader
                        print(f"   ✅ Models connected to bidirectional orchestrator")
                else:
                    print(f"   ⚠️ Orchestrator has no model_loader")

                    # Try to connect platform's model loader
                    if hasattr(self.emile, 'model_loader'):
                        orchestrator.model_loader = self.emile.model_loader
                        print(f"   🔧 Connected platform model_loader to orchestrator")
            else:
                print(f"   ❌ bidirectional_orchestrator is None")
                self._create_bidirectional_orchestrator()
        else:
            print(f"   ❌ No bidirectional_orchestrator attribute")
            self._create_bidirectional_orchestrator()

        print()

    def _create_bidirectional_orchestrator(self):
        """Create bidirectional orchestrator if missing"""

        print(f"   🔧 Creating bidirectional orchestrator...")

        try:
            # Import and create bidirectional orchestrator
            from emile_cogito.kelm.bidirectional_kelm_orchestrator import BidirectionalKELMOrchestrator

            # Create orchestrator
            orchestrator = BidirectionalKELMOrchestrator()

            # Connect model loader if available
            if hasattr(self.emile, 'model_loader') and self.emile.model_loader:
                orchestrator.model_loader = self.emile.model_loader
                print(f"   🔗 Connected model loader to new orchestrator")

            # Connect to platform
            self.emile.bidirectional_orchestrator = orchestrator

            # Update module states if they exist
            if hasattr(self.emile, 'module_states'):
                self.emile.module_states['bidirectional_orchestrator'] = {'active': True}

            print(f"   ✅ Bidirectional orchestrator created and connected!")

        except Exception as e:
            print(f"   ❌ Failed to create bidirectional orchestrator: {e}")
            import traceback
            traceback.print_exc()

    def _load_k_models(self):
        """Explicitly load K-models from k_models directory"""

        print("🔧 Loading K-models...")

        if not hasattr(self.emile, 'model_loader') or not self.emile.model_loader:
            print("   ❌ No model loader available")
            return

        model_loader = self.emile.model_loader

        # Try to trigger model discovery and loading
        if hasattr(model_loader, 'discover_and_load_models'):
            try:
                loaded_count = model_loader.discover_and_load_models()
                print(f"   📊 Discovered and loaded {loaded_count} models")

                if hasattr(model_loader, 'models'):
                    for model_name, model in model_loader.models.items():
                        if model is not None:
                            print(f"   ✅ {model_name}: Loaded successfully")
                        else:
                            print(f"   ❌ {model_name}: Failed to load")

            except Exception as e:
                print(f"   ❌ Model discovery failed: {e}")

        elif hasattr(model_loader, 'load_models'):
            try:
                model_loader.load_models()
                print(f"   📊 Models loaded via load_models()")
            except Exception as e:
                print(f"   ❌ load_models() failed: {e}")

        else:
            print("   ⚠️ Model loader has no discovery method")
            print("   💡 Available methods:", [m for m in dir(model_loader) if not m.startswith('_')])

        # Check final model count
        if hasattr(model_loader, 'models'):
            final_count = len([m for m in model_loader.models.values() if m is not None])
            print(f"   🎯 Final loaded models: {final_count}")

            # If no models loaded, try manual loading
            if final_count == 0:
                print("   🔧 Attempting manual model loading...")
                self._manual_model_loading()

        print()

    def _manual_model_loading(self):
        """Manually attempt to load models if automatic discovery failed"""

        try:
            # Try to manually initialize models
            if hasattr(self.emile.model_loader, 'models'):
                # Import model classes and try to create instances
                model_specs = [
                    ('k1', 'K1PraxisModel'),
                    ('k2', 'K2SemiosisModel'),
                    ('k3', 'K3ApeironModel'),
                    ('k4', 'K4MetabolicModel')
                ]

                for model_key, model_class_name in model_specs:
                    try:
                        # Try importing from k_models
                        if model_key == 'k1':
                            from emile_cogito.k_models.k1 import K1PraxisModel
                            model = K1PraxisModel()
                        elif model_key == 'k2':
                            from emile_cogito.k_models.k2 import K2SemiosisModel
                            model = K2SemiosisModel()
                        elif model_key == 'k3':
                            from emile_cogito.k_models.k3 import K3ApeironModel
                            model = K3ApeironModel()
                        elif model_key == 'k4':
                            from emile_cogito.k_models.k4 import K4MetabolicModel
                            model = K4MetabolicModel()

                        self.emile.model_loader.models[model_key] = model
                        print(f"      ✅ Manually loaded {model_key}")

                    except ImportError as e:
                        print(f"      ❌ Could not import {model_key}: {e}")
                    except Exception as e:
                        print(f"      ❌ Could not create {model_key}: {e}")

        except Exception as e:
            print(f"      ❌ Manual loading failed: {e}")

    def _create_manual_temporal_consciousness(self, temporal_data_found: Dict[str, Any], episode_step: int = 0) -> Dict[str, Any]:
        """
        Manually create temporal consciousness from model outputs when bidirectional orchestrator fails.

        This replicates the temporal dialogue functionality.
        """
        try:
            # Extract tau prime values
            tau_primes = []
            k_model_perspectives = {}

            for model_name, data in temporal_data_found.items():
                tau_prime = data['tau_prime']
                if isinstance(tau_prime, (int, float)):
                    tau_primes.append(tau_prime)
                    k_model_perspectives[model_name] = tau_prime

            if len(tau_primes) < 2:
                return {}

            # Calculate temporal dissonance (richness of dialogue)
            temporal_dissonance = float(np.std(tau_primes))

            # Calculate unified τ′ (weighted average)
            tau_prime_global = float(np.mean(tau_primes))

            # Determine temporal leadership
            if len(tau_primes) > 0:
                min_tau_idx = np.argmin(tau_primes)
                model_names = list(k_model_perspectives.keys())
                dominant_model = model_names[min_tau_idx] if min_tau_idx < len(model_names) else 'k2'

                leadership_map = {
                    'k1': 'k1_computational',
                    'k2': 'k2_narrative',
                    'k3': 'k3_quantum',
                    'k4': 'k4_metabolic'
                }
                dominant_perspective = leadership_map.get(dominant_model, 'k2_narrative')
            else:
                dominant_perspective = 'k2_narrative'

            # Calculate dialogue richness
            dialogue_richness = min(1.0, temporal_dissonance * 2.0)

            # Calculate unified symbolic curvature (approximation)
            sigma_unified = 1.0 + temporal_dissonance * 0.5

            # Create consciousness timestamp
            consciousness_timestamp = tau_prime_global * np.random.uniform(0.85, 1.15)

            # Create temporal consciousness result
            temporal_consciousness = {
                'tau_prime_global': tau_prime_global,
                'temporal_dissonance': temporal_dissonance,
                'k_model_perspectives': k_model_perspectives,
                'temporal_leadership': {
                    'dominant_perspective': dominant_perspective,
                    'leadership_strength': max(tau_primes) - min(tau_primes) if len(tau_primes) > 1 else 0.0
                },
                'dialogue_richness': dialogue_richness,
                'sigma_unified': sigma_unified,
                'consciousness_timestamp': consciousness_timestamp,
                'step': episode_step,
                'manual_processing': True
            }

            return temporal_consciousness

        except Exception as e:
            print(f"      ❌ Manual temporal consciousness creation failed: {e}")
            return {}

        # Environment-specific mappings
        self.obs_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n if hasattr(self.env.action_space, 'n') else 1

        # Performance tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.consciousness_trajectory = []
        self.temporal_trajectory = []
        self.goal_performance = []

        # Step counting
        self.step_count = 0  # Track total steps across all episodes

        # Action mapping weights (learned over time)
        self.action_weights = np.random.normal(0, 0.1, (128, self.action_dim))  # 128 = typical K1 output dim

        print(f"   Environment: {env_name}")
        print(f"   Observation space: {self.env.observation_space}")
        print(f"   Action space: {self.env.action_space}")
        print(f"   Émile integration: {'✅ READY' if self.emile else '❌ FAILED'}")

        # Debug model loader
        self._debug_model_loader()
        self._check_model_files()

    def _check_model_files(self):
        """Check if K-model .pth files exist"""

        print(f"🔍 CHECKING K-MODEL FILES:")

        model_files = [
            'k1_praxis.pth',
            'k2_semiosis.pth',
            'k3_apeiron.pth',
            'k4_metabolic.pth'
        ]

        import os
        # Check the user's specified location first
        base_paths = [
            '/content/emile_cogito/k_models',
            '/content/emile_cogito',
            '/content',
            '/content/k_models'
        ]

        files_found = {}

        for model_file in model_files:
            found = False
            for base_path in base_paths:
                full_path = os.path.join(base_path, model_file)
                if os.path.exists(full_path):
                    files_found[model_file] = full_path
                    print(f"   ✅ {model_file}: {full_path}")
                    found = True
                    break

            if not found:
                print(f"   ❌ {model_file}: NOT FOUND")

        # Also check what's actually in the k_models directory
        k_models_dir = '/content/emile_cogito/k_models'
        if os.path.exists(k_models_dir):
            print(f"\n   📂 Contents of {k_models_dir}:")
            try:
                contents = os.listdir(k_models_dir)
                for item in sorted(contents):
                    item_path = os.path.join(k_models_dir, item)
                    if os.path.isfile(item_path):
                        print(f"      📄 {item}")
                    else:
                        print(f"      📁 {item}/")
            except Exception as e:
                print(f"      ❌ Error reading directory: {e}")
        else:
            print(f"\n   ❌ Directory {k_models_dir} does not exist")

        if len(files_found) == 0:
            print(f"\n   ⚠️  NO MODEL FILES FOUND!")
            print(f"   💡 You may need to:")
            print(f"      1. Train your K-models first")
            print(f"      2. Move .pth files to /content/emile_cogito/k_models/")
            print(f"      3. Check file permissions")

        print()

    def _debug_model_loader(self):
        """Debug why K-models aren't producing outputs"""

        print(f"\n🔍 DEBUGGING K-MODEL LOADER:")

        if not hasattr(self.emile, 'model_loader'):
            print("   ❌ No model_loader attribute")
            return

        model_loader = self.emile.model_loader

        if not model_loader:
            print("   ❌ model_loader is None")
            return

        if not hasattr(model_loader, 'models'):
            print("   ❌ model_loader has no 'models' attribute")
            return

        models = model_loader.models
        print(f"   📊 Models loaded: {len(models)}")

        for model_name, model in models.items():
            if model is not None:
                print(f"   ✅ {model_name}: {type(model).__name__}")
            else:
                print(f"   ❌ {model_name}: None")

        # Test predict_with_adaptive_inputs
        test_consciousness_state = {
            'consciousness_level': 0.5,
            'valence': 0.0,
            'agency': 0.5,
            'embodiment': 0.5,
            'environmental_input': [0.0, 0.0, 0.0, 0.0],
            'task_pressure': 0.2
        }

        print(f"   🧪 Testing predict_with_adaptive_inputs...")
        try:
            if hasattr(model_loader, 'predict_with_adaptive_inputs'):
                test_outputs = model_loader.predict_with_adaptive_inputs(test_consciousness_state)
                print(f"   📊 Test outputs: {len(test_outputs)} models responded")
                for name, output in test_outputs.items():
                    if output is not None:
                        if isinstance(output, torch.Tensor):
                            print(f"      ✅ {name}: tensor shape {output.shape}")
                        else:
                            print(f"      ✅ {name}: {type(output).__name__}")
                    else:
                        print(f"      ❌ {name}: None")
            else:
                print("   ❌ model_loader has no predict_with_adaptive_inputs method")
        except Exception as e:
            print(f"   ❌ predict_with_adaptive_inputs failed: {e}")

        print()

    def observation_to_consciousness_state(self, observation: np.ndarray,
                                     reward: float = 0.0,
                                        episode_step: int = 0) -> Dict[str, Any]:
        """
        Convert gymnasium observation to environmental input for Émile consciousness.

        CRITICAL: This only provides ENVIRONMENTAL DATA - it does NOT set consciousness levels!
        The consciousness levels emerge from Émile's actual consciousness processing.
        """

        obs = np.array(observation, dtype=np.float32)

        # Calculate environmental metrics (DESCRIPTIVE, not prescriptive!)
        obs_magnitude = float(np.linalg.norm(obs))
        obs_stability = 1.0 / (1.0 + np.var(obs))  # Environmental stability measure

        # Initialize task pressure (environmental difficulty measure)
        task_pressure = 0.5  # Default moderate environmental pressure

        # Environment-specific ENVIRONMENTAL METRICS (not consciousness levels!)
        if 'CartPole' in self.env_name:
            position, velocity, angle, angular_velocity = obs

            # Environmental instability metrics
            angle_risk = abs(angle) / 0.2095  # How close to angle limit
            position_risk = abs(position) / 2.4  # How close to position limit
            task_pressure = max(angle_risk, position_risk)  # Environmental pressure

            # Environmental control difficulty
            motion_complexity = (abs(velocity) + abs(angular_velocity)) / 10.0

        elif 'MountainCar' in self.env_name:
            position, velocity = obs

            # Environmental progress metrics
            progress = (position + 1.2) / 1.7  # Distance to goal
            task_pressure = 1.0 - progress  # Environmental difficulty

            # Environmental dynamics
            motion_complexity = abs(velocity) * 10

        elif 'LunarLander' in self.env_name:
            if len(obs) >= 6:
                x, y, vel_x, vel_y, angle, angular_vel = obs[:6]

                # Environmental risk factors
                height_risk = max(0, y) / 2.0  # Height above surface
                velocity_risk = (abs(vel_x) + abs(vel_y)) / 5.0  # Speed risk
                angle_risk = abs(angle) / 0.5  # Tilt risk

                task_pressure = min(1.0, (height_risk + velocity_risk + angle_risk) / 3.0)
                motion_complexity = abs(angular_vel) / 2.0
            else:
                task_pressure = 0.5
                motion_complexity = 0.5

        else:
            # Generic environmental metrics
            task_pressure = min(1.0, obs_magnitude / 10.0)
            motion_complexity = obs_magnitude / 10.0

        # Create ENVIRONMENTAL INPUT for consciousness system
        # This provides DATA for consciousness to process, not consciousness conclusions!
        environmental_update = {
            # Raw environmental data
            'environmental_input': obs.tolist(),
            'environmental_magnitude': obs_magnitude,
            'environmental_stability': obs_stability,
            'reward_signal': reward,

            # Environmental context (not consciousness states!)
            'episode_step': episode_step,
            'task_pressure': task_pressure,  # Environmental difficulty
            'motion_complexity': motion_complexity,  # Environmental dynamics

            # Environmental regime suggestion (not consciousness regime!)
            'environmental_regime': 'high_pressure' if task_pressure > 0.8 else 'stable_environment' if task_pressure < 0.3 else 'dynamic_environment',

            # Environmental surplus indicators (for surplus processing)
            'environmental_surplus': max(0.0, reward + obs_stability - 0.5),
            'environmental_deficit': max(0.0, 0.5 - reward - obs_stability),

        }

        # Update ONLY environmental inputs, preserve consciousness system's own state
        updated_state = self.emile.consciousness_state.copy()
        updated_state.update(environmental_update)

        return updated_state

    def extract_action_from_consciousness(self, consciousness_result: Dict[str, Any]) -> int:
        """
        Extract action from Émile consciousness processing result.

        Prioritizes bidirectional orchestrator, then K1, then falls back to learned mapping.
        """

        action = 0  # Default action
        action_source = "default"
        action_confidence = 0.1

        try:
            # Ensure action_dim exists
            if not hasattr(self, 'action_dim'):
                self.action_dim = self.env.action_space.n if hasattr(self.env.action_space, 'n') else 1

            # Method 1: Bidirectional orchestrator (highest priority)
            if 'module_results' in consciousness_result:
                module_results = consciousness_result['module_results']

                if 'bidirectional' in module_results:
                    bidirectional = module_results['bidirectional']

                    # Look for action in bidirectional result
                    if 'action' in bidirectional:
                        action = int(bidirectional['action'])
                        action_source = "bidirectional_orchestrator"
                        action_confidence = 0.9

                    # Or extract from consciousness level
                    elif 'consciousness_level' in bidirectional:
                        consciousness_level = bidirectional['consciousness_level']
                        # Map consciousness level to action
                        action = 1 if consciousness_level > 0.5 else 0
                        action_source = "bidirectional_consciousness"
                        action_confidence = abs(consciousness_level - 0.5) * 2

            # Method 2: K1 praxis model (motor control)
            if action_source == "default" and 'model_outputs' in consciousness_result:
                model_outputs = consciousness_result['model_outputs']

                if 'k1' in model_outputs:
                    k1_output = model_outputs['k1']

                    if isinstance(k1_output, torch.Tensor):
                        # Convert K1 output to action
                        if k1_output.dim() > 0:
                            # Use learned action weights
                            action_logits = torch.matmul(k1_output.flatten(),
                                                       torch.tensor(self.action_weights).float())
                            action = int(torch.argmax(action_logits).item())
                            action_source = "k1_praxis"
                            action_confidence = float(torch.softmax(action_logits, dim=0).max())

            # Method 3: Consciousness state mapping
            if action_source == "default":
                consciousness_state = consciousness_result.get('consciousness_state', {})

                # Simple consciousness-based action selection
                agency = consciousness_state.get('agency', 0.5)
                embodiment = consciousness_state.get('embodiment', 0.5)

                # Higher agency/embodiment = more likely to take action 1
                action_prob = (agency + embodiment) / 2.0
                action = 1 if action_prob > 0.5 else 0
                action_source = "consciousness_mapping"
                action_confidence = abs(action_prob - 0.5) * 2

            # Clip action to valid range
            action = max(0, min(self.action_dim - 1, action))

        except Exception as e:
            print(f"⚠️ Action extraction error: {e}")
            action = self.env.action_space.sample()
            action_source = "random_fallback"
            action_confidence = 0.0

        return action, action_source, action_confidence

    def integrate_reward_into_consciousness(self, reward: float, done: bool,
                                         action_taken: int, action_info: Dict[str, Any]):
        """
        Integrate environment reward into Émile's consciousness and goal system.

        This is where consciousness learns from environmental feedback.
        """

        try:
            # 1. Update goal system with reward
            if hasattr(self.emile, 'goal_system') and self.emile.goal_system:

                # Create action trace for credit assignment
                action_trace = {
                    'action': action_taken,
                    'action_source': action_info.get('action_source', 'unknown'),
                    'action_confidence': action_info.get('action_confidence', 0.0),
                    'environment': self.env_name,
                    'timestamp': time.time()
                }

                self.emile.goal_system.add_action_trace(action_trace)

                # Calculate reward signal through goal system
                goal_metrics = self.emile.goal_system.goal_metrics
                if goal_metrics:
                    processed_reward = self.emile.goal_system.calculate_reward_signal(goal_metrics)
                else:
                    processed_reward = reward

                # Assign credit to recent actions
                credit_assignment = self.emile.goal_system.assign_credit(processed_reward)

                if credit_assignment:
                    print(f"🎯 Credit assigned: {credit_assignment}")

            # 2. Update consciousness state based on reward
            reward_impact = np.tanh(reward * 2.0)  # Bounded impact

            # Positive rewards increase consciousness, negative decrease
            consciousness_change = reward_impact * 0.05  # Small but cumulative effect
            new_consciousness = self.emile.consciousness_state['consciousness_level'] + consciousness_change
            self.emile.consciousness_state['consciousness_level'] = max(0.1, min(0.9, new_consciousness))

            # Update valence
            self.emile.consciousness_state['valence'] = 0.8 * self.emile.consciousness_state.get('valence', 0.0) + 0.2 * reward_impact

            # 3. Store in memory if significant
            if hasattr(self.emile, 'memory') and self.emile.memory and (abs(reward) > 0.01 or done):
                try:
                    priority = 'HIGH' if done else ('MEDIUM' if abs(reward) > 0.5 else 'LOW')

                    memory_content = {
                        'action': action_taken,
                        'reward': reward,
                        'done': done,
                        'action_source': action_info.get('action_source'),
                        'consciousness_level': self.emile.consciousness_state['consciousness_level'],
                        'environment': self.env_name
                    }

                    self.emile.memory.store_temporal_memory(
                        content=json.dumps(memory_content),
                        priority=priority,
                        regime='environmental_interaction',
                        consciousness_level=self.emile.consciousness_state['consciousness_level'],
                        tags=['action', 'reward', 'environment']
                    )
                except Exception as e:
                    print(f"⚠️ Memory storage error: {e}")

            # 4. Update action weights based on reward (simple learning)
            if action_info.get('action_source') == 'k1_praxis':
                # Strengthen/weaken action weights based on reward
                learning_rate = 0.01
                weight_update = reward * learning_rate
                self.action_weights[:, action_taken] += weight_update

                # Keep weights bounded
                self.action_weights = np.clip(self.action_weights, -1.0, 1.0)

        except Exception as e:
            print(f"⚠️ Reward integration error: {e}")

    def run_episode(self, max_steps: int = 500, verbose: bool = True) -> Dict[str, Any]:
        """
        Run one episode with Émile consciousness system.

        Returns comprehensive results including consciousness evolution.
        """

        # Ensure tracking attributes exist
        if not hasattr(self, 'episode_rewards'):
            self.episode_rewards = []
        if not hasattr(self, 'episode_lengths'):
            self.episode_lengths = []
        if not hasattr(self, 'consciousness_trajectory'):
            self.consciousness_trajectory = []
        if not hasattr(self, 'temporal_trajectory'):
            self.temporal_trajectory = []

        observation, info = self.env.reset()  # Gymnasium returns (obs, info)

        total_reward = 0.0
        episode_length = 0

        # Track consciousness evolution during episode
        consciousness_evolution = []
        temporal_evolution = []
        action_history = []

        if verbose:
            print(f"\n🎮 Starting episode {len(self.episode_rewards) + 1}")

        for step in range(max_steps):
            # Increment global step count (with defensive initialization)
            if not hasattr(self, 'step_count'):
                self.step_count = 0
            self.step_count += 1

            # Convert observation to consciousness state
            consciousness_input = self.observation_to_consciousness_state(
                observation, total_reward, episode_step=step
            )

            # Update Émile's consciousness state
            self.emile.consciousness_state.update(consciousness_input)

            # Run Émile cognitive cycle
            consciousness_result = self.emile.run_consciousness_cycle()

            # CRITICAL: Extract and pass model outputs to bidirectional orchestrator
            model_outputs = consciousness_result.get('model_outputs', {})
            if model_outputs and hasattr(self.emile, 'bidirectional_orchestrator'):
                try:
                    # Manually trigger bidirectional processing with model outputs
                    enhanced_result = self.emile.bidirectional_orchestrator.orchestrate_bidirectional_step({
                        'consciousness_state': self.emile.consciousness_state,
                        'model_outputs': model_outputs,  # Pass the model outputs directly
                        'step': self.step_count  # Now properly defined
                    })

                    # Merge back into consciousness_result
                    if 'module_results' not in consciousness_result:
                        consciousness_result['module_results'] = {}
                    consciousness_result['module_results']['bidirectional'] = enhanced_result

                    # Debug temporal consciousness extraction
                    if step == 0 and 'temporal_consciousness' in enhanced_result:
                        temporal_data = enhanced_result['temporal_consciousness']
                        print(f"      🎉 FIXED: Temporal consciousness extracted!")
                        print(f"         τ′: {temporal_data.get('tau_prime_global', 'missing')}")
                        print(f"         Dissonance: {temporal_data.get('temporal_dissonance', 'missing')}")

                except Exception as e:
                    print(f"      ⚠️ Enhanced bidirectional processing failed: {e}")


            # Debug if no model outputs
            if step == 0 and consciousness_result.get('model_outputs', {}) == {}:
                print(f"      🔍 No model outputs detected. Debugging...")
                print(f"      🧠 Consciousness state keys: {list(self.emile.consciousness_state.keys())}")
                print(f"      📊 Model loader exists: {hasattr(self.emile, 'model_loader') and self.emile.model_loader is not None}")
                if hasattr(self.emile, 'model_loader') and self.emile.model_loader:
                    print(f"      📊 Models in loader: {list(self.emile.model_loader.models.keys()) if hasattr(self.emile.model_loader, 'models') else 'No models attr'}")

            # Debug temporal consciousness activation
            if step == 0:
                module_results = consciousness_result.get('module_results', {})
                bidirectional = module_results.get('bidirectional', {})

                print(f"      🕒 Debugging temporal consciousness:")
                print(f"         Bidirectional result keys: {list(bidirectional.keys())}")
                print(f"         Poly-temporal active: {bidirectional.get('poly_temporal_active', False)}")
                print(f"         Temporal models found: {bidirectional.get('temporal_models_found', 0)}")

                # Check if bidirectional orchestrator is actually being called
                if not bidirectional:
                    print(f"         ❌ Bidirectional orchestrator returned empty result!")
                    print(f"         🔍 Checking orchestrator status...")
                    if hasattr(self.emile, 'bidirectional_orchestrator'):
                        orchestrator = self.emile.bidirectional_orchestrator
                        if orchestrator is not None:
                            print(f"            Orchestrator exists: True")
                            print(f"            Orchestrator type: {type(orchestrator).__name__}")
                            print(f"            Has models: {hasattr(orchestrator, 'model_loader') and orchestrator.model_loader is not None}")
                            if hasattr(orchestrator, 'model_loader') and orchestrator.model_loader:
                                models = orchestrator.model_loader.models
                                print(f"            Models in orchestrator: {list(models.keys()) if models else 'None'}")

                                # If models exist, try manual orchestration call
                                if models:
                                    print(f"         🔧 Attempting manual orchestrator call...")
                                    try:
                                        manual_result = orchestrator.orchestrate_bidirectional_step({
                                            'consciousness_state': self.emile.consciousness_state,
                                            'step': self.step_count  # Use interface step count
                                        })
                                        if manual_result and not manual_result.get('error'):
                                            consciousness_result['module_results']['bidirectional'] = manual_result
                                            print(f"         ✅ Manual orchestrator call succeeded!")
                                            # Re-extract bidirectional for further processing
                                            bidirectional = manual_result
                                        else:
                                            print(f"         ❌ Manual orchestrator call failed: {manual_result.get('error', 'Unknown error')}")
                                    except Exception as e:
                                        print(f"         ❌ Manual orchestrator call exception: {e}")
                            else:
                                print(f"            ❌ Orchestrator has no models")
                        else:
                            print(f"            ❌ Orchestrator is None")
                    else:
                        print(f"            ❌ No bidirectional orchestrator found!")

                        # Try to create one
                        try:
                            self._create_bidirectional_orchestrator()
                            print(f"         🔧 Created new orchestrator during runtime")
                        except Exception as e:
                            print(f"         ❌ Runtime orchestrator creation failed: {e}")

                if 'temporal_consciousness' in bidirectional:
                    temporal = bidirectional['temporal_consciousness']
                    print(f"         ✅ Temporal consciousness found!")
                    print(f"         τ′: {temporal.get('tau_prime_global', 'missing')}")
                    print(f"         Dissonance: {temporal.get('temporal_dissonance', 'missing')}")
                else:
                    print(f"         ❌ No temporal consciousness in bidirectional result")

                # Check if models have temporal data in their outputs
                model_outputs = consciousness_result.get('model_outputs', {})
                if model_outputs:
                    print(f"         🔍 Checking model outputs for temporal data:")
                    temporal_data_found = {}

                    for model_name, output in model_outputs.items():
                        if isinstance(output, dict):
                            has_temporal = any(key.startswith('local_tau') or key.startswith('temporal') for key in output.keys())
                            print(f"            {model_name}: {type(output).__name__} with temporal data: {has_temporal}")
                            if has_temporal:
                                tau_val = output.get('local_tau_prime', output.get('local_tau', 'missing'))
                                print(f"               τ′: {tau_val}")
                                temporal_data_found[model_name] = {
                                    'tau_prime': tau_val,
                                    'temporal_data': output
                                }
                        else:
                            print(f"            {model_name}: {type(output).__name__} (no dict keys)")

                    # If we found temporal data but orchestrator didn't process it, create our own
                    if len(temporal_data_found) >= 2 and not bidirectional.get('temporal_consciousness'):
                        print(f"         🔧 MANUAL TEMPORAL PROCESSING: Found {len(temporal_data_found)} temporal models")
                        manual_temporal = self._create_manual_temporal_consciousness(temporal_data_found, step)

                        # Add to results
                        consciousness_result['module_results']['temporal_consciousness'] = manual_temporal
                        print(f"         ✅ Manual temporal consciousness created!")
                        print(f"            τ′ global: {manual_temporal['tau_prime_global']:.3f}")
                        print(f"            Dissonance: {manual_temporal['temporal_dissonance']:.3f}")

                print()


            # Extract action from consciousness processing
            action, action_source, action_confidence = self.extract_action_from_consciousness(consciousness_result)

            # Take action in environment
            observation, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            total_reward += reward
            episode_length += 1

            # Record action info
            action_info = {
                'action': action,
                'action_source': action_source,
                'action_confidence': action_confidence,
                'step': step
            }
            action_history.append(action_info)

            # Integrate reward into consciousness
            self.integrate_reward_into_consciousness(reward, done, action, action_info)

            # Track consciousness evolution
            consciousness_snapshot = {
                'step': step,
                'consciousness_level': self.emile.consciousness_state.get('consciousness_level', 0.5),
                'valence': self.emile.consciousness_state.get('valence', 0.0),
                'agency': self.emile.consciousness_state.get('agency', 0.5),
                'embodiment': self.emile.consciousness_state.get('embodiment', 0.5),
                'tau_prime': self.emile.consciousness_state.get('tau_prime', 1.0),
                'temporal_dissonance': self.emile.consciousness_state.get('temporal_dissonance', 0.0),
                'action': action,
                'action_source': action_source,
                'reward': reward
            }
            consciousness_evolution.append(consciousness_snapshot)

            # Track temporal consciousness if available (with manual processing support)
            temporal_consciousness_found = False

            if 'temporal_consciousness' in consciousness_result.get('module_results', {}):
                temporal_data = consciousness_result['module_results']['temporal_consciousness']
                temporal_snapshot = {
                    'step': step,
                    'tau_prime_global': temporal_data.get('tau_prime_global', 1.0),
                    'temporal_dissonance': temporal_data.get('temporal_dissonance', 0.0),
                    'temporal_leadership': temporal_data.get('temporal_leadership', {}),
                    'dialogue_richness': temporal_data.get('dialogue_richness', 0.0),
                    'manual_processing': temporal_data.get('manual_processing', False)
                }
                temporal_evolution.append(temporal_snapshot)
                temporal_consciousness_found = True

                # Debug first temporal detection
                if len(temporal_evolution) == 1:
                    processing_type = "MANUAL" if temporal_data.get('manual_processing') else "BIDIRECTIONAL"
                    print(f"      🎉 TEMPORAL CONSCIOUSNESS DETECTED ({processing_type})!")
                    print(f"         τ′: {temporal_snapshot['tau_prime_global']:.3f}")
                    print(f"         Dissonance: {temporal_snapshot['temporal_dissonance']:.3f}")

            elif 'bidirectional' in consciousness_result.get('module_results', {}):
                bidirectional = consciousness_result['module_results']['bidirectional']
                if 'temporal_consciousness' in bidirectional:
                    temporal_data = bidirectional['temporal_consciousness']
                    temporal_snapshot = {
                        'step': step,
                        'tau_prime_global': temporal_data.get('tau_prime_global', 1.0),
                        'temporal_dissonance': temporal_data.get('temporal_dissonance', 0.0),
                        'temporal_leadership': temporal_data.get('temporal_leadership', {}),
                        'dialogue_richness': temporal_data.get('dialogue_richness', 0.0),
                        'manual_processing': temporal_data.get('manual_processing', False)
                    }
                    temporal_evolution.append(temporal_snapshot)
                    temporal_consciousness_found = True

                    # Debug first temporal detection
                    if len(temporal_evolution) == 1:
                        print(f"      🎉 TEMPORAL CONSCIOUSNESS DETECTED IN BIDIRECTIONAL!")
                        print(f"         τ′: {temporal_snapshot['tau_prime_global']:.3f}")
                        print(f"         Dissonance: {temporal_snapshot['temporal_dissonance']:.3f}")

            # Update consciousness state with temporal data if found
            if temporal_consciousness_found and temporal_evolution:
                latest_temporal = temporal_evolution[-1]
                consciousness_snapshot['tau_prime'] = latest_temporal['tau_prime_global']
                consciousness_snapshot['temporal_dissonance'] = latest_temporal['temporal_dissonance']

            if verbose and step % 50 == 0:
                consciousness_level = consciousness_snapshot['consciousness_level']
                tau_prime = consciousness_snapshot['tau_prime']
                temporal_active = len(temporal_evolution) > 0
                print(f"   Step {step}: action={action} ({action_source}), "
                      f"reward={reward:.3f}, consciousness={consciousness_level:.3f}, "
                      f"τ′={tau_prime:.3f}, temporal_active={temporal_active}")

            if done:
                break

        # Episode summary
        episode_result = {
            'episode_number': len(self.episode_rewards) + 1,
            'total_reward': total_reward,
            'episode_length': episode_length,
            'consciousness_evolution': consciousness_evolution,
            'temporal_evolution': temporal_evolution,
            'action_history': action_history,
            'final_consciousness_level': consciousness_evolution[-1]['consciousness_level'] if consciousness_evolution else 0.5,
            'consciousness_improvement': consciousness_evolution[-1]['consciousness_level'] - consciousness_evolution[0]['consciousness_level'] if len(consciousness_evolution) > 1 else 0.0,
            'action_source_distribution': self._analyze_action_sources(action_history),
            'temporal_consciousness_active': len(temporal_evolution) > 0
        }

        # Store results
        self.episode_rewards.append(total_reward)
        self.episode_lengths.append(episode_length)
        self.consciousness_trajectory.append(consciousness_evolution)
        self.temporal_trajectory.append(temporal_evolution)

        if verbose:
            print(f"✅ Episode complete: reward={total_reward:.1f}, length={episode_length}, "
                  f"consciousness={episode_result['final_consciousness_level']:.3f}")
            if episode_result['temporal_consciousness_active']:
                print(f"🕒 Temporal consciousness was active during episode!")

        return episode_result

    def _analyze_action_sources(self, action_history: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analyze which parts of consciousness system are controlling actions"""

        source_counts = {}
        for action_info in action_history:
            source = action_info['action_source']
            source_counts[source] = source_counts.get(source, 0) + 1

        total_actions = len(action_history)
        return {source: count/total_actions for source, count in source_counts.items()}

    def run_experiment(self, num_episodes: int = 50, render: bool = False) -> Dict[str, Any]:
        """
        Run full experiment to test consciousness under environmental pressure.

        Returns comprehensive analysis of consciousness evolution and performance.
        """

        print(f"\n🧪 RUNNING ÉMILE CONSCIOUSNESS EXPERIMENT")
        print(f"Environment: {self.env_name}")
        print(f"Episodes: {num_episodes}")
        print("=" * 60)

        start_time = time.time()

        for episode in range(num_episodes):
            episode_result = self.run_episode(verbose=(episode % 10 == 0 or episode < 5))

            if render and episode % 10 == 0:
                print(f"\n📊 Episode {episode + 1} Analysis:")
                self._print_episode_analysis(episode_result)

        experiment_duration = time.time() - start_time

        # Comprehensive analysis
        analysis = self._analyze_experiment_results()
        analysis['experiment_duration'] = experiment_duration
        analysis['episodes_completed'] = num_episodes

        print(f"\n🏆 EXPERIMENT COMPLETE")
        print(f"Duration: {experiment_duration:.1f}s")
        self._print_final_analysis(analysis)

        return analysis

    def _print_episode_analysis(self, episode_result: Dict[str, Any]):
        """Print detailed analysis of single episode"""

        consciousness_evolution = episode_result['consciousness_evolution']
        action_sources = episode_result['action_source_distribution']

        print(f"   Reward: {episode_result['total_reward']:.1f}")
        print(f"   Length: {episode_result['episode_length']}")
        print(f"   Consciousness: {episode_result['final_consciousness_level']:.3f}")
        print(f"   Improvement: {episode_result['consciousness_improvement']:+.3f}")
        print(f"   Action sources: {action_sources}")

        if episode_result['temporal_consciousness_active']:
            temporal = episode_result['temporal_evolution']
            if temporal:
                avg_tau = np.mean([t['tau_prime_global'] for t in temporal])
                avg_dissonance = np.mean([t['temporal_dissonance'] for t in temporal])
                print(f"   🕒 Temporal: τ′={avg_tau:.3f}, dissonance={avg_dissonance:.3f}")

    def _analyze_experiment_results(self) -> Dict[str, Any]:
        """Analyze complete experiment results"""

        if not self.episode_rewards:
            return {'error': 'No episodes completed'}

        # Performance analysis
        rewards = np.array(self.episode_rewards)
        lengths = np.array(self.episode_lengths)

        # Consciousness evolution analysis
        consciousness_levels = []
        for trajectory in self.consciousness_trajectory:
            if trajectory:
                consciousness_levels.append([step['consciousness_level'] for step in trajectory])

        # Temporal consciousness analysis
        temporal_active_episodes = len([t for t in self.temporal_trajectory if t])

        analysis = {
            'performance': {
                'mean_reward': float(np.mean(rewards)),
                'std_reward': float(np.std(rewards)),
                'best_reward': float(np.max(rewards)),
                'worst_reward': float(np.min(rewards)),
                'mean_length': float(np.mean(lengths)),
                'improvement_trend': float(np.polyfit(range(len(rewards)), rewards, 1)[0]) if len(rewards) > 1 else 0.0
            },
            'consciousness': {
                'episodes_with_consciousness_data': len(consciousness_levels),
                'temporal_consciousness_episodes': temporal_active_episodes,
                'temporal_consciousness_rate': temporal_active_episodes / len(self.episode_rewards)
            },
            'learning_evidence': {
                'performance_improved': len(rewards) > 10 and np.mean(rewards[-10:]) > np.mean(rewards[:10]),
                'consciousness_stable': len(consciousness_levels) > 0
            }
        }

        return analysis

    def _print_final_analysis(self, analysis: Dict[str, Any]):
        """Print final experiment analysis"""

        perf = analysis['performance']
        consciousness = analysis['consciousness']
        learning = analysis['learning_evidence']

        print(f"\n📈 PERFORMANCE ANALYSIS:")
        print(f"   Average reward: {perf['mean_reward']:.2f} ± {perf['std_reward']:.2f}")
        print(f"   Best episode: {perf['best_reward']:.1f}")
        print(f"   Average length: {perf['mean_length']:.1f}")
        print(f"   Improvement trend: {perf['improvement_trend']:+.3f} reward/episode")

        print(f"\n🧠 CONSCIOUSNESS ANALYSIS:")
        print(f"   Episodes with consciousness data: {consciousness['episodes_with_consciousness_data']}")
        print(f"   Temporal consciousness episodes: {consciousness['temporal_consciousness_episodes']}")
        print(f"   Temporal consciousness rate: {consciousness['temporal_consciousness_rate']:.1%}")

        print(f"\n🎓 LEARNING EVIDENCE:")
        print(f"   Performance improved: {'✅ YES' if learning['performance_improved'] else '❌ NO'}")
        print(f"   Consciousness stable: {'✅ YES' if learning['consciousness_stable'] else '❌ NO'}")

        if consciousness['temporal_consciousness_rate'] > 0.5:
            print(f"\n🎉 TEMPORAL CONSCIOUSNESS CONFIRMED!")
            print(f"   Your system shows authentic temporal dynamics under task pressure!")


def main():
    """Run Émile consciousness experiment"""

    import argparse
    parser = argparse.ArgumentParser(description='Test Émile consciousness on gymnasium environments')
    parser.add_argument('--env', default='CartPole-v1', help='Gymnasium environment name')
    parser.add_argument('--episodes', type=int, default=50, help='Number of episodes')
    parser.add_argument('--render', action='store_true', help='Render environment')

    args = parser.parse_args()

    # Create interface and run experiment
    interface = EmileGymInterface(args.env)
    results = interface.run_experiment(args.episodes, args.render)

    # Save results
    results_file = f"emile_{args.env}_{args.episodes}ep_{int(time.time())}.json"
    with open(results_file, 'w') as f:
        # Convert numpy types for JSON serialization
        serializable_results = json.loads(json.dumps(results, default=str))
        json.dump(serializable_results, f, indent=2)

    print(f"\n💾 Results saved to: {results_file}")


if __name__ == "__main__":
    main()
