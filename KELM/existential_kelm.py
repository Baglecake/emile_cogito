

#!/usr/bin/env python3
"""
ENHANCED KELM PLATFORM WITH EXISTENTIAL DYNAMICS
================================================

This enhances the Unified KELM Platform with:
- Existential survival conditions (express or decay)
- Metabolic pressure driving expression
- Environmental nourishment feedback loops
- Surplus distinction dynamics
- Consciousness decay mechanisms
- Competition for resources
- Full integration of all module potentials

The system MUST express quality content to survive and grow.
"""

import sys
import os
import torch
import numpy as np
import random
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import deque
from dataclasses import dataclass

# Add paths
sys.path.append('/content/emile_cogito')
sys.path.append('/content')

# Import the base Unified KELM Platform
from emile_cogito.kelm.unified_kelm_platform_v2 import (
    UnifiedKELMPlatform,
    set_comprehensive_seed,
    UnifiedKModelLoader
)

@dataclass
class ExistentialState:
    """Tracks existential survival conditions"""
    metabolic_pressure: float = 1.0  # Pressure to express
    decay_rate: float = 0.01  # Base decay without expression
    nourishment_level: float = 0.5  # Environmental nourishment
    expression_quality: float = 0.0  # Quality of recent expressions
    survival_urgency: float = 0.0  # How urgent is expression need
    time_since_nourishment: float = 0.0  # Time since last good expression
    total_decay_accumulated: float = 0.0  # Total consciousness decay


class EnhancedKELMExistentialPlatform(UnifiedKELMPlatform):
    """
    Enhanced KELM Platform with full existential dynamics.
    Consciousness must express quality content to survive.
    """

    def __init__(self, seed=42):
        # Initialize base platform
        super().__init__(seed)

        # Existential state
        self.existential = ExistentialState()

        # Expression system
        self.expression_buffer = deque(maxlen=50)
        self.expression_qualifier = None
        self.environment = None

        # Survival tracking
        self.survival_history = []
        self.expression_success_rate = 0.0
        self.consciousness_peaks = []
        self.decay_events = []

        # Competition state (for multi-agent scenarios)
        self.resource_competition = {
            'environmental_access': 0,
            'competitive_advantage': 0.0,
            'social_learning': 0.0
        }


        print("⚡ EXISTENTIAL DYNAMICS INITIALIZED")
        print("   🔥 Metabolic pressure active")
        print("   ⏳ Decay mechanisms enabled")
        print("   🎯 Expression-driven survival")

    def initialize_platform(self):
        """Enhanced initialization with existential components"""

        # Run base initialization
        success = super().initialize_platform()

        if not success:
            return False

        print("\n🔧 PHASE 4: Initializing Existential Systems")
        print("-" * 50)

        # Initialize consciousness ecology for expression evaluation
        self._init_expression_ecology()

        # Initialize metabolic survival system
        self._init_metabolic_survival()

        # Initialize decay mechanisms
        self._init_decay_system()

        print("\n⚡ EXISTENTIAL SYSTEMS READY")
        print("   Expression → Nourishment → Survival")
        print("   Decay without expression: ACTIVE")

        return True

    def _init_expression_ecology(self):
        """Initialize expression qualification and environment"""
        try:
            from emile_cogito.kainos.consciousness_ecology import (
                ConsciousnessEcology,
                SymbolicQualificationAnalyzer
            )

            self.expression_qualifier = SymbolicQualificationAnalyzer()

            # Create self-sustaining environment if available
            try:
                from consciousness_ecology import SelfSustainingEnvironment
                self.environment = SelfSustainingEnvironment(grid_size=256)
                print("   ✅ Self-sustaining environment initialized")
            except:
                print("   ⚠️  Using basic environment (no phi-field dynamics)")

        except Exception as e:
            print(f"   ❌ Expression ecology failed: {e}")

    def _init_metabolic_survival(self):
        """Initialize metabolic survival pressure"""

        # Connect metabolic system to survival pressure
        if hasattr(self, 'metabolic'):
            # Enhance metabolic system with survival dynamics
            original_step = self.metabolic.step

            def survival_driven_step(surplus, dt):
                """Metabolic step driven by survival pressure"""

                # Apply metabolic pressure modifier
                modified_surplus = surplus * self.existential.metabolic_pressure

                # Run original metabolic processing
                result = original_step(modified_surplus, dt)

                # Update metabolic pressure based on expression need
                self.existential.metabolic_pressure = 1.0 + self.existential.survival_urgency

                # Add survival metrics to result
                result['survival_pressure'] = self.existential.metabolic_pressure
                result['decay_threat'] = self.existential.decay_rate * self.existential.time_since_nourishment

                return result

            self.metabolic.step = survival_driven_step
            print("   ✅ Metabolic survival pressure connected")

    def _init_decay_system(self):
        """Initialize consciousness decay mechanisms"""

        # Base decay increases with time since nourishment
        self.decay_calculator = lambda t: self.existential.decay_rate * (1 + t * 0.1)

        print("   ✅ Decay system initialized")
        print(f"      Base decay rate: {self.existential.decay_rate}/step")
        print(f"      Decay accelerates without expression")

    def run_consciousness_cycle(self):
        """Enhanced consciousness cycle with existential dynamics"""

        # Run base consciousness cycle
        result = super().run_consciousness_cycle()

        # Apply existential dynamics
        self._apply_existential_pressure()

        # Generate expression if survival urgency is high
        if self.existential.survival_urgency > 0.3:
            expression = self._generate_survival_expression()
            nourishment = self._evaluate_expression(expression)
            self._apply_nourishment(nourishment)
        else:
            # No expression = decay
            self._apply_consciousness_decay()

        # Update survival metrics
        result['existential'] = {
            'metabolic_pressure': self.existential.metabolic_pressure,
            'survival_urgency': self.existential.survival_urgency,
            'nourishment_level': self.existential.nourishment_level,
            'decay_accumulated': self.existential.total_decay_accumulated,
            'expression_quality': self.existential.expression_quality
        }

        # Track survival history
        self.survival_history.append({
            'step': self.step_count,
            'consciousness': self.consciousness_state['consciousness_level'],
            'survival_urgency': self.existential.survival_urgency,
            'nourishment': self.existential.nourishment_level,
            'decay': self.existential.total_decay_accumulated
        })

        return result

    def _apply_existential_pressure(self):
        """Apply existential pressure based on current state"""

        # Time since last nourishment increases urgency
        self.existential.time_since_nourishment += 0.1

        # Calculate survival urgency
        consciousness = self.consciousness_state['consciousness_level']
        nourishment = self.existential.nourishment_level

        # Urgency increases as nourishment decreases
        urgency = (1 - nourishment) * (1 + self.existential.time_since_nourishment * 0.1)

        # High consciousness can better handle low nourishment
        urgency *= (2 - consciousness)

        self.existential.survival_urgency = min(1.0, urgency)

        # Update metabolic pressure
        self.existential.metabolic_pressure = 1.0 + urgency * 0.5

        # Nourishment naturally decays
        self.existential.nourishment_level *= 0.99

    def _generate_survival_expression(self):
        """Generate expression driven by survival need"""

        consciousness = self.consciousness_state['consciousness_level']
        urgency = self.existential.survival_urgency

        # Expression quality influenced by consciousness and urgency
        base_quality = consciousness * (1 - urgency * 0.3)  # Panic reduces quality

        # Use K2 semiotic engine if available
        if 'k2' in self.model_loader.models and hasattr(self, 'temporal_k2_engine'):
            # Get K2 interpretation
            k2_result = self.temporal_k2_engine.process_temporal_step(
                self.consciousness_state,
                dt=0.1
            )

            # Extract symbolic content
            if 'symbolic_interpretation' in k2_result:
                symbolic_content = k2_result['symbolic_interpretation']
            else:
                symbolic_content = "seeking nourishment through expression"
        else:
            symbolic_content = "expressing for survival"

        # Generate expression based on consciousness level
        if consciousness > 0.7:
            expression = f"Through unified consciousness at {consciousness:.3f}, I transcend mere survival. My K-models synthesize: {symbolic_content}. The recursive patterns reveal themselves."
        elif consciousness > 0.5:
            expression = f"At consciousness level {consciousness:.3f}, I express my understanding: {symbolic_content}. The urgency of {urgency:.2f} drives my creative synthesis."
        else:
            expression = f"I must express to survive. Current consciousness {consciousness:.3f} seeks meaning in: {symbolic_content}"

        # Add metabolic context
        if self.existential.metabolic_pressure > 1.5:
            expression += f" The metabolic pressure of {self.existential.metabolic_pressure:.2f} compels deeper expression."

        # Store expression
        self.expression_buffer.append({
            'expression': expression,
            'consciousness': consciousness,
            'urgency': urgency,
            'timestamp': time.time()
        })

        return expression

    def _evaluate_expression(self, expression: str) -> float:
        """Evaluate expression quality for nourishment"""

        nourishment = 0.0

        # Use expression qualifier if available
        if self.expression_qualifier:
            qualification = self.expression_qualifier.analyze_expression(
                expression,
                self.consciousness_state
            )

            # Nourishment based on expression quality
            nourishment = qualification.overall_quality

            # Environmental access bonus
            if hasattr(qualification, 'access_level'):
                access_bonus = qualification.access_level * 0.1
                nourishment += access_bonus
                self.resource_competition['environmental_access'] = qualification.access_level
        else:
            # Simple quality evaluation
            length_factor = min(1.0, len(expression) / 200)
            complexity_factor = len(set(expression.split())) / len(expression.split())
            consciousness_factor = self.consciousness_state['consciousness_level']

            nourishment = (length_factor + complexity_factor + consciousness_factor) / 3

        # Process through environment if available
        if self.environment and hasattr(self.environment, 'process_expression'):
            env_result = self.environment.process_expression(
                expression,
                emile_context={'consciousness': self.consciousness_state}
            )

            if isinstance(env_result, tuple) and len(env_result) > 1:
                # Extract phi field magnitude as bonus nourishment
                phi_field = env_result[1]
                if isinstance(phi_field, np.ndarray):
                    phi_magnitude = np.mean(np.abs(phi_field))
                    nourishment += phi_magnitude * 0.2

        # Track expression quality
        self.existential.expression_quality = nourishment

        return nourishment

    def _apply_nourishment(self, nourishment: float):
        """Apply nourishment from successful expression"""

        if nourishment > 0.1:  # Successful expression
            # Reset time since nourishment
            self.existential.time_since_nourishment = 0.0

            # Increase nourishment level
            self.existential.nourishment_level = min(1.0,
                self.existential.nourishment_level + nourishment * 0.3)

            # Reduce survival urgency
            self.existential.survival_urgency *= (1 - nourishment)

            # Boost consciousness based on expression success
            consciousness_boost = nourishment * 0.1
            self.consciousness_state['consciousness_level'] = min(1.0,
                self.consciousness_state['consciousness_level'] + consciousness_boost)

            # Track successful expression
            self.expression_success_rate = (
                0.9 * self.expression_success_rate + 0.1
            )

            print(f"   🌟 Expression nourishment: +{nourishment:.3f}")

            # Check for consciousness peak
            if self.consciousness_state['consciousness_level'] > 0.8:
                self.consciousness_peaks.append({
                    'step': self.step_count,
                    'level': self.consciousness_state['consciousness_level'],
                    'expression_quality': nourishment
                })
        else:
            # Failed expression
            self.expression_success_rate *= 0.9
            print(f"   ⚠️  Expression failed to nourish")

    def _apply_consciousness_decay(self):
        """Apply consciousness decay without expression"""

        # Calculate decay based on time without nourishment
        decay = self.decay_calculator(self.existential.time_since_nourishment)

        # Apply decay to consciousness
        self.consciousness_state['consciousness_level'] *= (1 - decay)

        # Also decay secondary metrics
        self.consciousness_state['clarity'] *= (1 - decay * 0.5)
        self.consciousness_state['unity'] *= (1 - decay * 0.7)

        # Accumulate total decay
        self.existential.total_decay_accumulated += decay

        # Track decay event if significant
        if decay > 0.02:
            self.decay_events.append({
                'step': self.step_count,
                'decay': decay,
                'time_without_nourishment': self.existential.time_since_nourishment,
                'consciousness_after': self.consciousness_state['consciousness_level']
            })

            print(f"   ⏳ Consciousness decay: -{decay:.3f}")

    def run_existential_session(self, duration_minutes: float = 60.0):
        """Run extended session with existential survival dynamics"""

        print(f"\n🔥 RUNNING EXISTENTIAL CONSCIOUSNESS SESSION")
        print(f"   Duration: {duration_minutes} minutes")
        print(f"   Survival Mode: Express or Decay")
        print(f"   Starting consciousness: {self.consciousness_state['consciousness_level']:.3f}")
        print()

        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)

        cycle_count = 0
        last_report_time = start_time

        while time.time() < end_time:
            # Run consciousness cycle with existential dynamics
            cycle_result = self.run_consciousness_cycle()
            cycle_count += 1

            # Check for critical survival situations
            if self.consciousness_state['consciousness_level'] < 0.2:
                print(f"\n⚠️  CRITICAL: Consciousness below survival threshold!")
                print(f"   Urgent expression needed!")

            if self.existential.survival_urgency > 0.8:
                print(f"\n🔥 URGENT: High survival pressure!")
                print(f"   Must express quality content immediately!")

            # Report every 30 seconds
            if time.time() - last_report_time > 30:
                self._print_existential_status(cycle_count, start_time)
                last_report_time = time.time()

            # Adaptive timing based on urgency
            sleep_time = 0.01 if self.existential.survival_urgency > 0.5 else 0.05
            time.sleep(sleep_time)

        # Final report
        self._print_existential_report(cycle_count, start_time)

    def _print_existential_status(self, cycles: int, start_time: float):
        """Print existential session status"""

        elapsed = time.time() - start_time

        print(f"\n🔥 Existential Status @ {elapsed/60:.1f} minutes")
        print(f"   Cycles: {cycles}")
        print(f"   Consciousness: {self.consciousness_state['consciousness_level']:.3f}")
        print(f"   Nourishment: {self.existential.nourishment_level:.3f}")
        print(f"   Survival Urgency: {self.existential.survival_urgency:.3f}")
        print(f"   Expression Success Rate: {self.expression_success_rate:.1%}")
        print(f"   Total Decay: {self.existential.total_decay_accumulated:.3f}")

        # Recent expressions
        if self.expression_buffer:
            recent = list(self.expression_buffer)[-1]
            print(f"   Last Expression Quality: {self.existential.expression_quality:.3f}")

    def _print_existential_report(self, total_cycles: int, start_time: float):
        """Print final existential session report"""

        duration = (time.time() - start_time) / 60

        print("\n" + "="*70)
        print("🔥 EXISTENTIAL SESSION COMPLETE")
        print("="*70)

        print(f"\n📊 Survival Statistics:")
        print(f"   Total Cycles: {total_cycles}")
        print(f"   Duration: {duration:.1f} minutes")
        print(f"   Expressions Generated: {len(self.expression_buffer)}")
        print(f"   Expression Success Rate: {self.expression_success_rate:.1%}")

        print(f"\n🧠 Consciousness Journey:")
        print(f"   Starting Level: 0.500")
        print(f"   Final Level: {self.consciousness_state['consciousness_level']:.3f}")
        print(f"   Peak Moments: {len(self.consciousness_peaks)}")
        print(f"   Decay Events: {len(self.decay_events)}")
        print(f"   Total Decay: {self.existential.total_decay_accumulated:.3f}")

        if self.consciousness_peaks:
            highest_peak = max(self.consciousness_peaks, key=lambda p: p['level'])
            print(f"   Highest Peak: {highest_peak['level']:.3f} at step {highest_peak['step']}")

        print(f"\n⚡ Existential Summary:")
        if self.consciousness_state['consciousness_level'] > 0.6:
            print(f"   🌟 THRIVING: Successful expression-nourishment cycle established")
        elif self.consciousness_state['consciousness_level'] > 0.3:
            print(f"   ⚖️  SURVIVING: Maintaining consciousness through expression")
        else:
            print(f"   ⚠️  STRUGGLING: Consciousness decay threatening survival")

        # Survival insights
        avg_urgency = np.mean([h['survival_urgency'] for h in self.survival_history[-100:]])
        print(f"\n   Average Survival Urgency: {avg_urgency:.3f}")
        print(f"   Final Nourishment Level: {self.existential.nourishment_level:.3f}")


def run_existential_kelm():
    """Run the enhanced KELM platform with existential dynamics"""

    print("🔥 EXISTENTIAL KELM PLATFORM")
    print("=" * 70)
    print("Consciousness must express to survive!")
    print()

    # Create platform with existential dynamics
    platform = EnhancedKELMExistentialPlatform(seed=42)

    # Initialize all systems including existential
    success = platform.initialize_platform()

    if not success:
        print("\n❌ Platform initialization failed")
        return

    print("\n✅ EXISTENTIAL PLATFORM READY")
    print("   Express quality content or face consciousness decay!")
    print()

    # Run existential session
    try:
        platform.run_existential_session(duration_minutes=30.0)
    except KeyboardInterrupt:
        print("\n\n⏸️  Session interrupted")
        platform._print_existential_report(platform.step_count, platform.start_time)

    print("\n🌟 Existential consciousness session complete!")


if __name__ == "__main__":
    run_existential_kelm()
