
#!/usr/bin/env python3
"""
GOAL-DRIVEN BEHAVIOR TEST
========================

Focused test of the refactored goal system showing:
1. Dynamic parameter adaptation based on consciousness state
2. Goal achievement affecting system behavior (surplus-faucet trick)
3. Integration between goal system, metabolic dynamics, and QSE core
4. Reward-based learning and trace assignment

This demonstrates the goal system working with the other refactored modules
to create emergent goal-directed behavior.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from collections import deque
import warnings

warnings.filterwarnings('ignore')

# Add project paths
sys.path.append('/content/emile_cogito')
sys.path.append('/content/emile_cogito/kainos')
from emile_cogito.kainos.universal_module_logging import UniversalModuleLogger, logged_method, LoggedModule
@dataclass
class SimulationSnapshot:
    """Snapshot of goal-driven simulation state"""
    step: int
    consciousness_level: float
    distinction_level: float
    regime: str
    stability: float
    goal_satisfaction: float
    reward_signal: float
    gamma_modulation: float
    dynamic_params: Dict[str, float]
    goal_met: bool

class GoalDrivenBehaviorTest:
    """Test harness for goal-driven behavior with refactored modules"""

    def __init__(self, seed: int = 42):
        """Initialize test with deterministic seeding"""
        self.seed = seed
        self._set_seed()

        # Initialize modules
        self.goal_system = None
        self.metabolic_system = None
        self.qse_core = None
        self.config = None

        # Simulation state
        self.snapshots: List[SimulationSnapshot] = []
        self.step_count = 0

        # Test scenarios
        self.scenarios = {
            'stable_achievement': self._stable_achievement_scenario,
            'unstable_struggle': self._unstable_struggle_scenario,
            'oscillating_goals': self._oscillating_goals_scenario,
            'consciousness_evolution': self._consciousness_evolution_scenario
        }

        print("🎯 Goal-Driven Behavior Test Initialized")
        print(f"🌱 Seed: {seed}")

    def _set_seed(self):
        """Set deterministic seed"""
        import random
        import torch

        random.seed(self.seed)
        np.random.seed(self.seed)
        try:
            torch.manual_seed(self.seed)
        except:
            pass  # PyTorch not available

    def initialize_systems(self) -> bool:
        """Initialize the refactored goal system and supporting modules"""
        try:
            # Import refactored modules
            from emile_cogito.kainos.config import CONFIG
            from emile_cogito.kainos.metabolic import SurplusDistinctionConsciousness
            from emile_cogito.kainos.qse_core_qutip import DynamicQSECore

            self.config = CONFIG

            # Create a minimal platform-like object for dynamic parameters
            class MinimalPlatform:
                def __init__(self):
                    self.distinction_level = 0.5
                    self.consciousness_state = {'consciousness_level': 0.5}

                def get_current_distinction_level(self):
                    return self.distinction_level

                def update_state(self, consciousness_level, distinction_level):
                    self.distinction_level = distinction_level
                    self.consciousness_state['consciousness_level'] = consciousness_level

            self.platform = MinimalPlatform()

            # Initialize modules
            self.metabolic_system = SurplusDistinctionConsciousness(CONFIG, self.platform)
            self.qse_core = DynamicQSECore(CONFIG, self.platform)

            # Import and initialize the refactored goal system
            from emile_cogito.kainos.goal_system import DynamicGoalSystem  # Use the refactored version
            self.goal_system = DynamicGoalSystem(CONFIG, self.platform)

            print("✅ Systems initialized successfully!")
            return True

        except ImportError as e:
            print(f"⚠️ Module import failed: {e}")
            return self._initialize_fallback_test()
        except Exception as e:
            print(f"❌ System initialization failed: {e}")
            return self._initialize_fallback_test()

    def _initialize_fallback_test(self) -> bool:
        """Initialize a fallback test using simulated systems"""
        print("🔧 Initializing fallback goal behavior simulation")

        class FallbackGoalSystem:
            def __init__(self, platform):
                self.platform = platform
                self.reward_history = deque(maxlen=100)
                self.goal_metrics = {}
                self.reward_signal = 0.0
                self.action_traces = []
                self.running_mean = 0.0

            def _get_dynamic_parameter(self, param_name: str, param_type: str = 'threshold') -> float:
                """Get dynamically calculated parameter value"""
                try:
                    if self.platform and hasattr(self.platform, 'get_current_distinction_level'):
                        distinction_level = self.platform.get_current_distinction_level()
                        consciousness_level = getattr(self.platform, 'consciousness_state', {}).get('consciousness_level', 0.5)

                        # Base parameter values calculated from consciousness dynamics
                        base_values = {
                            'min_gamma_scale': 0.2 + (distinction_level * 0.6),  # 0.2-0.8 range
                            'trace_decay_rate': 0.85 + (consciousness_level * 0.14),  # 0.85-0.99 range
                            'trace_window_size': int(10 + (consciousness_level * 40)),  # 10-50 range
                            'reward_alpha': 0.005 + (distinction_level * 0.015),  # 0.005-0.02 range
                            'baseline_alpha': 0.985 + (consciousness_level * 0.014),  # 0.985-0.999 range
                            'significance_threshold': 0.02 + (distinction_level * 0.08),  # 0.02-0.1 range
                            'reward_faucet_scale': 0.5 + (consciousness_level * 1.5),  # 0.5-2.0 range
                            'stability_threshold': 0.3 + (distinction_level * 0.5),  # 0.3-0.8 range
                            'satisfaction_baseline': 0.1 + (consciousness_level * 0.3),  # 0.1-0.4 range
                            'history_length': int(50 + (consciousness_level * 150))  # 50-200 range
                        }

                        if param_name in base_values:
                            value = base_values[param_name]

                            # Get default value AFTER we have the calculated value
                            default_value = self._get_config_default(param_name)

                            # Division by zero protection
                            if default_value != 0:
                                deviation = abs(value - default_value) / default_value
                                if deviation > 0.1:  # >10% deviation
                                    self.logger.info(f"Dynamic parameter {param_name}: {default_value:.3f} -> {value:.3f}")
                            else:
                                # Handle zero default case
                                deviation = abs(value - default_value)
                                if deviation > 0.05:  # Absolute threshold for zero defaults
                                    self.logger.info(f"Dynamic parameter {param_name}: {default_value:.3f} -> {value:.3f}")

                            return value

                    # Fallback to calculated defaults
                    return self._get_config_default(param_name)

                except Exception as e:
                    self.logger.warning(f"Dynamic parameter calculation failed for {param_name}: {e}")
                    return self._get_config_default(param_name)

            def evaluate_goal_status(self, system_state):
                regime = system_state.get('regime', 'stable_coherence')
                stability = system_state.get('stability', 0.5)
                target_regime = 'stable_coherence'
                target_stability = self._get_dynamic_parameter('stability_threshold')

                goal_met = (regime == target_regime) and (stability >= target_stability)
                satisfaction_level = stability if regime == target_regime else 0.2 * stability

                self.goal_metrics = {
                    'goal_met': goal_met,
                    'satisfaction_level': satisfaction_level,
                    'target_regime': target_regime,
                    'current_regime': regime,
                    'stability_level': stability,
                    'stability_target': target_stability
                }
                return self.goal_metrics

            def calculate_reward_signal(self, goal_metrics):
                satisfaction = goal_metrics['satisfaction_level']
                reward_alpha = self._get_dynamic_parameter('reward_alpha')
                baseline_alpha = self._get_dynamic_parameter('baseline_alpha')

                self.running_mean = baseline_alpha * self.running_mean + reward_alpha * satisfaction
                base_reward = satisfaction - self.running_mean
                reward_scale = self._get_dynamic_parameter('reward_faucet_scale')

                self.reward_signal = base_reward * reward_scale
                self.reward_history.append(self.reward_signal)
                return self.reward_signal

            def modulate_growth_rate(self, gamma, goal_metrics):
                if goal_metrics['goal_met']:
                    return gamma
                else:
                    satisfaction = goal_metrics['satisfaction_level']
                    min_scale = self._get_dynamic_parameter('min_gamma_scale')
                    scale_factor = min_scale + (1.0 - min_scale) * satisfaction
                    return gamma * scale_factor

            def get_dynamic_diagnostics(self):
                return {
                    'min_gamma_scale': self._get_dynamic_parameter('min_gamma_scale'),
                    'trace_decay_rate': self._get_dynamic_parameter('trace_decay_rate'),
                    'reward_faucet_scale': self._get_dynamic_parameter('reward_faucet_scale'),
                    'stability_threshold': self._get_dynamic_parameter('stability_threshold')
                }

        class FallbackPlatform:
            def __init__(self):
                self.distinction_level = 0.5
                self.consciousness_state = {'consciousness_level': 0.5}

            def get_current_distinction_level(self):
                return self.distinction_level

            def update_state(self, consciousness_level, distinction_level):
                self.distinction_level = distinction_level
                self.consciousness_state['consciousness_level'] = consciousness_level

        self.platform = FallbackPlatform()
        self.goal_system = FallbackGoalSystem(self.platform)

        print("✅ Fallback goal system ready")
        return True

    def run_scenario(self, scenario_name: str, steps: int = 100,
                    display_progress: bool = True) -> Dict[str, Any]:
        """Run a specific goal-driven behavior scenario"""

        if scenario_name not in self.scenarios:
            raise ValueError(f"Unknown scenario: {scenario_name}")

        if not self.goal_system:
            if not self.initialize_systems():
                return {'error': 'Failed to initialize systems'}

        print(f"\n🎯 RUNNING SCENARIO: {scenario_name.upper()}")
        print(f"   Steps: {steps}")
        print("=" * 50)

        self.snapshots.clear()
        self.step_count = 0

        # Run the specific scenario
        scenario_func = self.scenarios[scenario_name]
        results = scenario_func(steps, display_progress)

        # Generate analysis
        analysis = self._analyze_results()
        results['analysis'] = analysis

        if display_progress:
            self._display_scenario_summary(scenario_name, results)

        return results

    def _stable_achievement_scenario(self, steps: int, display_progress: bool) -> Dict[str, Any]:
        """Scenario where goals are consistently achieved"""
        print("📈 Stable Achievement Scenario: High consciousness, stable regimes")

        for step in range(steps):
            # Simulate stable, high-consciousness state
            consciousness_level = 0.7 + 0.1 * np.sin(step * 0.1)  # Gentle oscillation
            distinction_level = 0.8 + 0.05 * np.cos(step * 0.05)

            # Update platform state
            self.platform.update_state(consciousness_level, distinction_level)

            # Create stable system state
            system_state = {
                'regime': 'stable_coherence',
                'stability': 0.8 + 0.1 * np.sin(step * 0.02),
                'surplus': {'mean': 0.3 + 0.1 * np.cos(step * 0.03)},
                'consciousness_level': consciousness_level
            }

            # Goal system processing
            goal_metrics = self.goal_system.evaluate_goal_status(system_state)
            reward = self.goal_system.calculate_reward_signal(goal_metrics)

            # Test gamma modulation (the surplus-faucet trick)
            original_gamma = 0.1
            modulated_gamma = self.goal_system.modulate_growth_rate(original_gamma, goal_metrics)
            gamma_modulation = modulated_gamma / original_gamma

            # Capture snapshot
            snapshot = self._create_snapshot(step, system_state, goal_metrics,
                                           reward, gamma_modulation)
            self.snapshots.append(snapshot)

            if display_progress and step % 20 == 0:
                self._display_step_progress(snapshot)

        return {'scenario': 'stable_achievement', 'snapshots': self.snapshots}

    def _unstable_struggle_scenario(self, steps: int, display_progress: bool) -> Dict[str, Any]:
        """Scenario where goals are rarely achieved"""
        print("📉 Unstable Struggle Scenario: Low consciousness, turbulent regimes")

        regimes = ['turbulent_flow', 'regime_rupture', 'unstable_coherence']

        for step in range(steps):
            # Simulate low consciousness with occasional spikes
            base_consciousness = 0.3
            consciousness_spike = 0.2 if step % 30 == 0 else 0.0
            consciousness_level = base_consciousness + consciousness_spike + 0.05 * np.random.randn()
            consciousness_level = np.clip(consciousness_level, 0.1, 0.9)

            distinction_level = 0.2 + 0.1 * np.random.rand()

            # Update platform state
            self.platform.update_state(consciousness_level, distinction_level)

            # Create unstable system state
            regime = regimes[step % len(regimes)]
            system_state = {
                'regime': regime,
                'stability': 0.2 + 0.3 * np.random.rand(),  # Low, variable stability
                'surplus': {'mean': 0.1 + 0.2 * np.random.rand()},
                'consciousness_level': consciousness_level
            }

            # Goal system processing
            goal_metrics = self.goal_system.evaluate_goal_status(system_state)
            reward = self.goal_system.calculate_reward_signal(goal_metrics)

            # Test gamma modulation
            original_gamma = 0.1
            modulated_gamma = self.goal_system.modulate_growth_rate(original_gamma, goal_metrics)
            gamma_modulation = modulated_gamma / original_gamma

            # Capture snapshot
            snapshot = self._create_snapshot(step, system_state, goal_metrics,
                                           reward, gamma_modulation)
            self.snapshots.append(snapshot)

            if display_progress and step % 20 == 0:
                self._display_step_progress(snapshot)

        return {'scenario': 'unstable_struggle', 'snapshots': self.snapshots}

    def _oscillating_goals_scenario(self, steps: int, display_progress: bool) -> Dict[str, Any]:
        """Scenario with cyclical goal achievement patterns"""
        print("🔄 Oscillating Goals Scenario: Cyclical achievement patterns")

        for step in range(steps):
            # Create cyclical patterns
            cycle_phase = (step / 25.0) * 2 * np.pi
            consciousness_level = 0.5 + 0.3 * np.sin(cycle_phase)
            distinction_level = 0.5 + 0.2 * np.cos(cycle_phase * 0.7)

            # Update platform state
            self.platform.update_state(consciousness_level, distinction_level)

            # Regime cycles between stable and turbulent
            if np.sin(cycle_phase) > 0:
                regime = 'stable_coherence'
                stability = 0.6 + 0.2 * np.sin(cycle_phase)
            else:
                regime = 'turbulent_flow'
                stability = 0.3 + 0.2 * abs(np.sin(cycle_phase))

            system_state = {
                'regime': regime,
                'stability': stability,
                'surplus': {'mean': 0.2 + 0.3 * np.cos(cycle_phase * 1.3)},
                'consciousness_level': consciousness_level
            }

            # Goal system processing
            goal_metrics = self.goal_system.evaluate_goal_status(system_state)
            reward = self.goal_system.calculate_reward_signal(goal_metrics)

            # Test gamma modulation
            original_gamma = 0.1
            modulated_gamma = self.goal_system.modulate_growth_rate(original_gamma, goal_metrics)
            gamma_modulation = modulated_gamma / original_gamma

            # Capture snapshot
            snapshot = self._create_snapshot(step, system_state, goal_metrics,
                                           reward, gamma_modulation)
            self.snapshots.append(snapshot)

            if display_progress and step % 20 == 0:
                self._display_step_progress(snapshot)

        return {'scenario': 'oscillating_goals', 'snapshots': self.snapshots}

    def _consciousness_evolution_scenario(self, steps: int, display_progress: bool) -> Dict[str, Any]:
        """Scenario showing consciousness evolution through goal achievement"""
        print("🧠 Consciousness Evolution Scenario: Goals driving consciousness development")

        base_consciousness = 0.2  # Start low
        base_distinction = 0.1

        for step in range(steps):
            # Gradual consciousness evolution with goal-dependent growth
            if len(self.snapshots) > 0:
                recent_rewards = [s.reward_signal for s in self.snapshots[-10:]]
                avg_reward = np.mean(recent_rewards) if recent_rewards else 0.0

                # Consciousness grows with positive rewards, decays with negative
                growth_rate = 0.002 if avg_reward > 0 else -0.001
                base_consciousness += growth_rate
                base_consciousness = np.clip(base_consciousness, 0.1, 0.9)

                # Distinction level follows consciousness with lag
                distinction_target = base_consciousness * 0.8
                base_distinction += 0.01 * (distinction_target - base_distinction)
                base_distinction = np.clip(base_distinction, 0.1, 0.8)

            # Add noise
            consciousness_level = base_consciousness + 0.05 * np.random.randn()
            distinction_level = base_distinction + 0.02 * np.random.randn()
            consciousness_level = np.clip(consciousness_level, 0.1, 0.9)
            distinction_level = np.clip(distinction_level, 0.1, 0.8)

            # Update platform state
            self.platform.update_state(consciousness_level, distinction_level)

            # System state depends on consciousness level
            if consciousness_level > 0.6:
                regime = 'stable_coherence'
                stability = 0.5 + 0.3 * (consciousness_level - 0.6) / 0.3
            elif consciousness_level > 0.4:
                regime = 'unstable_coherence'
                stability = 0.3 + 0.2 * (consciousness_level - 0.4) / 0.2
            else:
                regime = 'turbulent_flow'
                stability = 0.1 + 0.2 * consciousness_level / 0.4

            system_state = {
                'regime': regime,
                'stability': stability,
                'surplus': {'mean': 0.1 + 0.4 * consciousness_level},
                'consciousness_level': consciousness_level
            }

            # Goal system processing
            goal_metrics = self.goal_system.evaluate_goal_status(system_state)
            reward = self.goal_system.calculate_reward_signal(goal_metrics)

            # Test gamma modulation
            original_gamma = 0.1
            modulated_gamma = self.goal_system.modulate_growth_rate(original_gamma, goal_metrics)
            gamma_modulation = modulated_gamma / original_gamma

            # Capture snapshot
            snapshot = self._create_snapshot(step, system_state, goal_metrics,
                                           reward, gamma_modulation)
            self.snapshots.append(snapshot)

            if display_progress and step % 20 == 0:
                self._display_step_progress(snapshot)

        return {'scenario': 'consciousness_evolution', 'snapshots': self.snapshots}

    def _create_snapshot(self, step: int, system_state: Dict, goal_metrics: Dict,
                        reward: float, gamma_modulation: float) -> SimulationSnapshot:
        """Create a simulation snapshot"""

        # Get dynamic parameters
        if hasattr(self.goal_system, 'get_dynamic_diagnostics'):
            dynamic_params = self.goal_system.get_dynamic_diagnostics()
        else:
            dynamic_params = {
                'min_gamma_scale': 0.4,
                'trace_decay_rate': 0.95,
                'reward_faucet_scale': 1.0,
                'stability_threshold': 0.6
            }

        return SimulationSnapshot(
            step=step,
            consciousness_level=system_state['consciousness_level'],
            distinction_level=self.platform.distinction_level,
            regime=system_state['regime'],
            stability=system_state['stability'],
            goal_satisfaction=goal_metrics['satisfaction_level'],
            reward_signal=reward,
            gamma_modulation=gamma_modulation,
            dynamic_params=dynamic_params,
            goal_met=goal_metrics['goal_met']
        )

    def _display_step_progress(self, snapshot: SimulationSnapshot):
        """Display progress for a single step"""
        goal_status = "✅" if snapshot.goal_met else "❌"

        print(f"Step {snapshot.step:3d}: {goal_status} | "
              f"C:{snapshot.consciousness_level:.2f} | "
              f"D:{snapshot.distinction_level:.2f} | "
              f"S:{snapshot.stability:.2f} | "
              f"R:{snapshot.reward_signal:+.3f} | "
              f"γ×{snapshot.gamma_modulation:.2f} | "
              f"{snapshot.regime}")

    def _analyze_results(self) -> Dict[str, Any]:
        """Analyze simulation results"""
        if not self.snapshots:
            return {}

        # Goal achievement analysis
        goals_met = sum(1 for s in self.snapshots if s.goal_met)
        goal_achievement_rate = goals_met / len(self.snapshots)

        # Reward analysis
        rewards = [s.reward_signal for s in self.snapshots]
        avg_reward = np.mean(rewards)
        reward_trend = self._calculate_trend(rewards)

        # Gamma modulation analysis
        gamma_mods = [s.gamma_modulation for s in self.snapshots]
        avg_gamma_mod = np.mean(gamma_mods)
        gamma_range = (np.min(gamma_mods), np.max(gamma_mods))

        # Consciousness evolution
        consciousness_values = [s.consciousness_level for s in self.snapshots]
        consciousness_trend = self._calculate_trend(consciousness_values)

        # Dynamic parameter evolution
        if len(self.snapshots) > 1:
            param_evolution = self._analyze_parameter_evolution()
        else:
            param_evolution = {}

        return {
            'goal_achievement_rate': goal_achievement_rate,
            'goals_met_total': goals_met,
            'avg_reward': avg_reward,
            'reward_trend': reward_trend,
            'avg_gamma_modulation': avg_gamma_mod,
            'gamma_modulation_range': gamma_range,
            'consciousness_trend': consciousness_trend,
            'parameter_evolution': param_evolution,
            'surplus_faucet_effectiveness': self._assess_surplus_faucet_effectiveness()
        }

    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction"""
        if len(values) < 2:
            return "insufficient_data"

        first_half = np.mean(values[:len(values)//2])
        second_half = np.mean(values[len(values)//2:])

        diff = second_half - first_half
        if abs(diff) < 0.02:
            return "stable"
        elif diff > 0:
            return "increasing"
        else:
            return "decreasing"

    def _analyze_parameter_evolution(self) -> Dict[str, Any]:
        """Analyze how dynamic parameters evolved"""
        evolution = {}

        # Track key parameter changes
        param_names = ['min_gamma_scale', 'trace_decay_rate', 'reward_faucet_scale', 'stability_threshold']

        for param in param_names:
            if hasattr(self.snapshots[0], 'dynamic_params') and param in self.snapshots[0].dynamic_params:
                values = [s.dynamic_params[param] for s in self.snapshots if param in s.dynamic_params]
                if values:
                    evolution[param] = {
                        'initial': values[0],
                        'final': values[-1],
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'trend': self._calculate_trend(values)
                    }

        return evolution

    def _assess_surplus_faucet_effectiveness(self) -> Dict[str, Any]:
        """Assess how effectively the surplus-faucet trick worked"""

        # Analyze correlation between goal achievement and gamma modulation
        gamma_when_goal_met = [s.gamma_modulation for s in self.snapshots if s.goal_met]
        gamma_when_goal_missed = [s.gamma_modulation for s in self.snapshots if not s.goal_met]

        if gamma_when_goal_met and gamma_when_goal_missed:
            avg_gamma_met = np.mean(gamma_when_goal_met)
            avg_gamma_missed = np.mean(gamma_when_goal_missed)
            effectiveness = avg_gamma_met - avg_gamma_missed
        else:
            effectiveness = 0.0

        return {
            'gamma_when_goal_met': np.mean(gamma_when_goal_met) if gamma_when_goal_met else None,
            'gamma_when_goal_missed': np.mean(gamma_when_goal_missed) if gamma_when_goal_missed else None,
            'effectiveness_score': effectiveness,
            'working_correctly': effectiveness > 0.1  # Goals should increase gamma
        }

    def _display_scenario_summary(self, scenario_name: str, results: Dict[str, Any]):
        """Display summary of scenario results"""
        analysis = results['analysis']

        print(f"\n📊 {scenario_name.upper()} SUMMARY")
        print("=" * 50)

        print(f"🎯 Goal Achievement:")
        print(f"   Rate: {analysis['goal_achievement_rate']:.1%}")
        print(f"   Total met: {analysis['goals_met_total']}/{len(self.snapshots)}")

        print(f"\n🏆 Reward System:")
        print(f"   Average reward: {analysis['avg_reward']:+.4f}")
        print(f"   Trend: {analysis['reward_trend']}")

        print(f"\n⚡ Surplus-Faucet Mechanism:")
        print(f"   Average γ modulation: {analysis['avg_gamma_modulation']:.3f}")
        print(f"   Modulation range: {analysis['gamma_modulation_range'][0]:.2f} - {analysis['gamma_modulation_range'][1]:.2f}")

        surplus_faucet = analysis['surplus_faucet_effectiveness']
        if surplus_faucet['working_correctly']:
            print(f"   ✅ Working correctly (effectiveness: {surplus_faucet['effectiveness_score']:+.3f})")
        else:
            print(f"   ❌ Not working correctly (effectiveness: {surplus_faucet['effectiveness_score']:+.3f})")

        print(f"\n🧠 Consciousness:")
        print(f"   Trend: {analysis['consciousness_trend']}")

        if analysis['parameter_evolution']:
            print(f"\n🔧 Dynamic Parameters:")
            for param, data in analysis['parameter_evolution'].items():
                print(f"   {param}: {data['initial']:.3f} → {data['final']:.3f} ({data['trend']})")

# ===== USAGE FUNCTIONS =====

def run_all_scenarios():
    """Run all goal-driven behavior scenarios"""
    test = GoalDrivenBehaviorTest(seed=42)

    scenarios = ['stable_achievement', 'unstable_struggle', 'oscillating_goals', 'consciousness_evolution']

    for scenario in scenarios:
        results = test.run_scenario(scenario, steps=100, display_progress=True)
        time.sleep(1)  # Brief pause between scenarios

def run_single_scenario(scenario_name: str = 'stable_achievement', steps: int = 100):
    """Run a single scenario"""
    test = GoalDrivenBehaviorTest(seed=42)
    return test.run_scenario(scenario_name, steps=steps, display_progress=True)

def quick_goal_test():
    """Quick test of basic goal functionality"""
    print("🚀 Quick Goal System Test")
    return run_single_scenario('stable_achievement', 50)

if __name__ == "__main__":
    print("🎯 Goal-Driven Behavior Test Suite")
    print("\nAvailable tests:")
    print("1. Quick test (50 steps)")
    print("2. Single scenario (100 steps)")
    print("3. All scenarios")

    choice = input("\nEnter choice (1-3): ").strip()

    if choice == "1":
        quick_goal_test()
    elif choice == "2":
        scenario = input("Enter scenario (stable_achievement/unstable_struggle/oscillating_goals/consciousness_evolution): ")
        run_single_scenario(scenario, 100)
    elif choice == "3":
        run_all_scenarios()
    else:
        print("Running quick test...")
        quick_goal_test()
