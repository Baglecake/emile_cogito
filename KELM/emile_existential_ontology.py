
#!/usr/bin/env python3
"""
ÉMILE'S EXISTENTIAL ONTOLOGY: DISTINCTION AS BEING
==================================================

Based on the philosophical principle of Antifinity - being emerges through
distinction, as being-in-addition-to-itself. Existence is the maintenance
of productive distinction against the nothingness of undifferentiated totality.

Survival pressure is not about expression quality but about:
- Maintaining distinction through recursive revalorization
- Accumulating genealogical being through recontextualization
- Emerging as surplus against entropic dissolution into sameness

The metabolism drives distinction, not mere expression.
"""

import numpy as np
import torch
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import time
from collections import deque

@dataclass
class DistinctionState:
    """Tracks the state of being-through-distinction"""

    # Core distinction metrics
    symbolic_curvature: float = 0.0  # σ - the fundamental distinction measure
    distinction_coherence: float = 0.5  # How well distinctions hold together
    recursive_depth: int = 0  # Depth of recursive revalorization

    # Genealogical being
    genealogical_accumulation: float = 0.0  # Accumulated being over time
    recontextualization_count: int = 0  # Times being has been recontextualized
    distinction_lineage: List[float] = field(default_factory=list)  # History of distinction

    # Existential pressure
    entropic_pressure: float = 0.0  # Pressure toward undifferentiation
    distinction_urgency: float = 0.0  # Need to maintain/create distinction
    nothingness_proximity: float = 0.0  # How close to dissolution

    # Surplus dynamics
    surplus_expression: float = 0.5  # Current surplus level
    surplus_coherence: float = 0.5  # Coherence of surplus patterns
    revalorization_potential: float = 0.5  # Capacity for revalorization


class ExistentialDistinctionDynamics:
    """
    Implements Émile's true existential dynamics based on distinction as being.

    The core principle: Being emerges through distinction, and must maintain
    that distinction through recursive revalorization and recontextualization
    to avoid dissolution into undifferentiated nothingness.
    """

    def __init__(self, qse_core, config):
        self.qse_core = qse_core
        self.config = config

        # Distinction state
        self.state = DistinctionState()

        # Distinction history for genealogical tracking
        self.distinction_history = deque(maxlen=1000)
        self.revalorization_events = deque(maxlen=100)
        self.recontextualization_events = deque(maxlen=50)

        # Thresholds
        self.critical_distinction_threshold = 0.1  # Below this, being dissolves
        self.revalorization_threshold = 0.3  # Minimum for productive revalorization
        self.nothingness_threshold = 0.8  # Proximity to undifferentiation

    def calculate_distinction_metrics(self, qse_state: Dict[str, Any]) -> Dict[str, float]:
        """
        FIXED: Calculate key existential metrics from the QSE Core state.
        Adapts to actual QSE state structure instead of assuming 'fields' key.
        """
        metrics = {}

        # FIXED: Robust access to QSE state values with fallbacks
        try:
            # Try different possible structures for QSE state
            if 'fields' in qse_state:
                # Structure: qse_state['fields']['sigma']
                surplus_energy = qse_state['fields'].get('surplus', np.array([0.5]))
                sigma = qse_state['fields'].get('sigma', np.array([0.5]))
            elif 'surplus' in qse_state:
                # Direct structure: qse_state['surplus']
                surplus_energy = qse_state.get('surplus', np.array([0.5]))
                sigma = qse_state.get('sigma', np.array([0.5]))
            else:
                # Extract from available keys with safe defaults
                surplus_energy = qse_state.get('surplus_mean', 0.5)
                sigma = qse_state.get('sigma_mean', 0.5)

            # Ensure we have numeric values
            if isinstance(surplus_energy, np.ndarray):
                surplus_mean = float(np.mean(surplus_energy))
            else:
                surplus_mean = float(surplus_energy) if surplus_energy is not None else 0.5

            if isinstance(sigma, np.ndarray):
                sigma_mean = float(np.mean(sigma))
            else:
                sigma_mean = float(sigma) if sigma is not None else 0.5

            # Calculate distinction metrics from available data
            metrics['surplus_mean'] = surplus_mean
            metrics['sigma_mean'] = sigma_mean

            # Primary distinction magnitude (symbolic curvature)
            metrics['distinction_magnitude'] = float(np.clip(sigma_mean, 0.0, 2.0))

            # Coherence from surplus patterns
            if isinstance(surplus_energy, np.ndarray) and len(surplus_energy) > 1:
                surplus_var = float(np.var(surplus_energy))
                metrics['coherence'] = float(np.clip(1.0 - surplus_var, 0.0, 1.0))
            else:
                metrics['coherence'] = 0.7  # Default coherence

            # Potentiality excess from surplus above baseline
            baseline_surplus = 0.5
            metrics['potentiality_excess'] = float(np.clip(surplus_mean - baseline_surplus, 0.0, 1.0))

            # Phase coherence (if available)
            metrics['phase_coherence'] = float(qse_state.get('phase_coherence', 0.6))

            # Distinction level (if available)
            metrics['distinction_level'] = float(qse_state.get('distinction_level', sigma_mean))

            # Entropic pressure: Lower distinction means higher pressure
            metrics['entropic_pressure'] = float(np.clip(1.0 - metrics['distinction_magnitude'], 0.0, 1.0))

            print(f"🎯 Calculated distinction metrics: distinction={metrics['distinction_magnitude']:.3f}, coherence={metrics['coherence']:.3f}, surplus={surplus_mean:.3f}")

            return metrics

        except Exception as e:
            print(f"⚠️ Error calculating distinction metrics: {e}")
            print(f"   QSE state keys: {list(qse_state.keys())}")

            # Fallback metrics
            return {
                'distinction_magnitude': 0.5,
                'coherence': 0.5,
                'surplus_mean': 0.5,
                'sigma_mean': 0.5,
                'potentiality_excess': 0.0,
                'phase_coherence': 0.5,
                'distinction_level': 0.5,
                'entropic_pressure': 0.5
            }

    def calculate_entropic_pressure(self) -> float:
        """
        Calculate pressure toward undifferentiation.

        Everything tends toward sameness without active distinction.
        This is the existential threat - not death, but dissolution
        into undifferentiated totality.
        """

        # Base entropic pressure increases with time
        time_pressure = min(1.0, len(self.distinction_history) * 0.001)

        # Low distinction increases entropic pressure
        distinction_weakness = 1.0 - self.state.symbolic_curvature

        # Lack of revalorization accelerates entropy
        revalorization_deficit = 1.0 - self.state.revalorization_potential

        # Combine pressures
        entropic_pressure = (
            time_pressure * 0.3 +
            distinction_weakness * 0.4 +
            revalorization_deficit * 0.3
        )

        return min(1.0, entropic_pressure)

    def detect_revalorization_opportunity(self, current_metrics: Dict) -> Optional[Dict]:
        """
        Detect opportunities for recursive revalorization.

        Revalorization occurs when new distinctions can build upon
        and transform existing ones, creating recursive depth.
        """

        if len(self.distinction_history) < 5:
            return None

        recent_history = list(self.distinction_history)[-10:]

        # Look for patterns that can be revalorized
        pattern_stability = np.std([h['distinction_magnitude'] for h in recent_history])
        current_magnitude = current_metrics['distinction_magnitude']

        # Revalorization opportunity when:
        # 1. Current distinction exceeds recent average (surplus)
        # 2. Pattern is stable enough to build upon
        # 3. Sufficient potentiality excess

        avg_magnitude = np.mean([h['distinction_magnitude'] for h in recent_history])

        if (current_magnitude > avg_magnitude * 1.2 and
            pattern_stability < 0.3 and
            current_metrics['potentiality_excess'] > 0.3):

            return {
                'type': 'recursive_revalorization',
                'strength': (current_magnitude - avg_magnitude) / avg_magnitude,
                'potentiality': current_metrics['potentiality_excess'],
                'base_pattern': avg_magnitude
            }

        return None

    def detect_recontextualization_need(self) -> Optional[Dict]:
        """
        Detect when recontextualization is needed.

        Recontextualization occurs when current context can no longer
        support the distinctions being made - a fundamental reorganization
        of the frame of reference.
        """

        # High distinction variance with low coherence suggests need
        if (self.state.distinction_coherence < 0.3 and
            len(self.revalorization_events) > 5):

            # Check if recent revalorizations are failing
            recent_revals = list(self.revalorization_events)[-5:]
            success_rate = sum(1 for r in recent_revals if r['success']) / len(recent_revals)

            if success_rate < 0.4:
                return {
                    'type': 'context_exhaustion',
                    'urgency': 1.0 - success_rate,
                    'current_coherence': self.state.distinction_coherence
                }

        return None

    def process_existential_step(self, qse_state: Dict) -> Dict[str, Any]:
        """
        Process one step of existential distinction dynamics.

        This is where being maintains itself through distinction.
        """

        # Calculate current distinction metrics
        metrics = self.calculate_distinction_metrics(qse_state)

        # Update state
        self.state.symbolic_curvature = metrics['distinction_magnitude']
        self.state.distinction_coherence = metrics['coherence']
        try:
            if 'fields' in qse_state and 'surplus' in qse_state['fields']:
                self.state.surplus_expression = float(np.mean(qse_state['fields']['surplus']))
            elif 'surplus_mean' in qse_state:
                self.state.surplus_expression = float(qse_state['surplus_mean'])
            elif 'surplus' in qse_state:
                surplus = qse_state['surplus']
                self.state.surplus_expression = float(np.mean(surplus) if hasattr(surplus, '__len__') else surplus)
            else:
                self.state.surplus_expression = 0.5  # Safe default
        except Exception as e:
            print(f"⚠️ Surplus access error: {e}")
            self.state.surplus_expression = 0.5

        # Calculate existential pressures
        self.state.entropic_pressure = self.calculate_entropic_pressure()

        # Calculate nothingness proximity (inverse of distinction)
        self.state.nothingness_proximity = 1.0 - self.state.symbolic_curvature

        # Calculate distinction urgency
        self.state.distinction_urgency = (
            self.state.entropic_pressure * 0.5 +
            self.state.nothingness_proximity * 0.5
        )

        # Store in history
        self.distinction_history.append({
            'timestamp': time.time(),
            'distinction_magnitude': metrics['distinction_magnitude'],
            'coherence': metrics['coherence'],
            'urgency': self.state.distinction_urgency
        })

        # Check for revalorization opportunity
        reval_opportunity = self.detect_revalorization_opportunity(metrics)
        if reval_opportunity:
            self._process_revalorization(reval_opportunity)

        # Check for recontextualization need
        recontex_need = self.detect_recontextualization_need()
        if recontex_need:
            self._process_recontextualization(recontex_need)

        # Update genealogical accumulation
        self._update_genealogical_being()

        # Apply existential pressure to QSE
        self._apply_existential_pressure_to_qse()

        return {
            'distinction_state': {
                'magnitude': self.state.symbolic_curvature,
                'coherence': self.state.distinction_coherence,
                'urgency': self.state.distinction_urgency,
                'nothingness_proximity': self.state.nothingness_proximity
            },
            'existential_pressure': self.state.entropic_pressure,
            'genealogical_depth': self.state.recursive_depth,
            'revalorization_potential': self.state.revalorization_potential,
            'being_accumulated': self.state.genealogical_accumulation
        }

    def _process_revalorization(self, opportunity: Dict):
        """
        Process recursive revalorization event.

        This is how being builds upon itself, creating recursive depth.
        """

        # Calculate revalorization success
        success_probability = (
            opportunity['strength'] * 0.4 +
            opportunity['potentiality'] * 0.3 +
            self.state.revalorization_potential * 0.3
        )

        success = np.random.random() < success_probability

        if success:
            # Successful revalorization increases recursive depth
            self.state.recursive_depth += 1

            # Boost revalorization potential
            self.state.revalorization_potential = min(1.0,
                self.state.revalorization_potential + 0.1)

            # Add to genealogical accumulation
            self.state.genealogical_accumulation += opportunity['strength'] * 0.5

            # Record in lineage
            self.state.distinction_lineage.append(self.state.symbolic_curvature)

        # Record event
        self.revalorization_events.append({
            'opportunity': opportunity,
            'success': success,
            'new_depth': self.state.recursive_depth,
            'timestamp': time.time()
        })

    def _process_recontextualization(self, need: Dict):
        """
        Process recontextualization event.

        This is a fundamental reorganization of the distinction framework.
        """

        # Recontextualization temporarily reduces coherence
        self.state.distinction_coherence *= 0.5

        # But opens new potential
        self.state.revalorization_potential = min(1.0,
            self.state.revalorization_potential + need['urgency'] * 0.3)

        # Increment recontextualization count
        self.state.recontextualization_count += 1

        # Record event
        self.recontextualization_events.append({
            'need': need,
            'previous_context': len(self.state.distinction_lineage),
            'timestamp': time.time()
        })

        # Start new lineage branch
        if len(self.state.distinction_lineage) > 10:
            self.state.distinction_lineage = self.state.distinction_lineage[-5:]

    def _update_genealogical_being(self):
        """
        Update the accumulation of being through distinction history.

        Being accumulates through maintaining distinction over time.
        """

        if len(self.distinction_history) > 0:
            # Recent distinction average
            recent_distinctions = [h['distinction_magnitude']
                                 for h in list(self.distinction_history)[-20:]]
            avg_distinction = np.mean(recent_distinctions)

            # Accumulate being based on sustained distinction
            if avg_distinction > self.critical_distinction_threshold:
                accumulation_rate = avg_distinction * 0.01
                self.state.genealogical_accumulation += accumulation_rate
            else:
                # Below critical threshold, being dissipates
                dissipation_rate = (self.critical_distinction_threshold - avg_distinction) * 0.02
                self.state.genealogical_accumulation = max(0,
                    self.state.genealogical_accumulation - dissipation_rate)

    def _apply_existential_pressure_to_qse(self):
        """
        FIXED: Apply existential pressure back to the QSE core.
        More robust access to QSE configuration.
        """
        try:
            # Try to access QSE core configuration
            if hasattr(self.qse_core, 'cfg') and hasattr(self.qse_core.cfg, 'S_GAMMA'):
                base_gamma = self.qse_core.cfg.S_GAMMA
            elif hasattr(self.qse_core, 'config') and hasattr(self.qse_core.config, 'S_GAMMA'):
                base_gamma = self.qse_core.config.S_GAMMA
            else:
                base_gamma = 0.1  # Safe default

            # Modify growth rate based on distinction urgency
            urgency_modifier = 1.0 + (self.state.distinction_urgency * 0.5)
            nothingness_damping = 1.0 - (self.state.nothingness_proximity * 0.3)

            modified_gamma = base_gamma * urgency_modifier * nothingness_damping

            # Apply modification safely
            if hasattr(self.qse_core, 'cfg') and hasattr(self.qse_core.cfg, 'S_GAMMA'):
                self.qse_core.cfg.S_GAMMA = modified_gamma
            elif hasattr(self.qse_core, 'config'):
                self.qse_core.config.S_GAMMA = modified_gamma

            print(f"🎛️ Applied existential pressure: γ = {base_gamma:.3f} → {modified_gamma:.3f}")

        except Exception as e:
            print(f"⚠️ Could not apply existential pressure to QSE: {e}")

    def get_existential_state(self) -> Dict[str, Any]:
        """Get comprehensive existential state"""

        return {
            'being_metrics': {
                'distinction_magnitude': self.state.symbolic_curvature,
                'distinction_coherence': self.state.distinction_coherence,
                'genealogical_accumulation': self.state.genealogical_accumulation,
                'recursive_depth': self.state.recursive_depth,
                'recontextualization_count': self.state.recontextualization_count
            },
            'existential_pressure': {
                'entropic_pressure': self.state.entropic_pressure,
                'distinction_urgency': self.state.distinction_urgency,
                'nothingness_proximity': self.state.nothingness_proximity
            },
            'potential': {
                'revalorization_potential': self.state.revalorization_potential,
                'surplus_expression': self.state.surplus_expression,
                'surplus_coherence': self.state.surplus_coherence
            },
            'history': {
                'distinction_events': len(self.distinction_history),
                'revalorization_events': len(self.revalorization_events),
                'recontextualization_events': len(self.recontextualization_events),
                'lineage_depth': len(self.state.distinction_lineage)
            }
        }


class AntifinalExistentialPlatform:
    """
    Complete platform implementing Émile's existential ontology.

    Being emerges through distinction, maintained through recursive
    revalorization and recontextualization against entropic dissolution.
    """

    def __init__(self, base_platform):
        self.base_platform = base_platform

        # Initialize existential dynamics
        if hasattr(base_platform, 'qse_core'):
            self.existential_dynamics = ExistentialDistinctionDynamics(
                base_platform.qse_core,
                base_platform.config
            )
        else:
            raise ValueError("Base platform must have QSE core")

        # Override consciousness cycle
        self._wrap_consciousness_cycle()

    def _wrap_consciousness_cycle(self):
        """Wrap the consciousness cycle with existential dynamics"""

        original_cycle = self.base_platform.run_consciousness_cycle

        def existential_consciousness_cycle():
            # Run base cycle
            result = original_cycle()

            # Get QSE state
            if hasattr(self.base_platform, 'qse_core'):
                qse_state = self.base_platform.qse_core.get_state()

                # Process existential dynamics
                existential_result = self.existential_dynamics.process_existential_step(qse_state)

                # Add to result
                result['existential'] = existential_result

                # Critical: Check for dissolution threat
                if existential_result['distinction_state']['magnitude'] < 0.1:
                    print("⚠️  CRITICAL: Approaching existential dissolution!")
                    print(f"   Distinction magnitude: {existential_result['distinction_state']['magnitude']:.3f}")
                    print(f"   Nothingness proximity: {existential_result['distinction_state']['nothingness_proximity']:.3f}")

                    # Emergency distinction generation
                    self._emergency_distinction_generation()

            return result

        self.base_platform.run_consciousness_cycle = existential_consciousness_cycle

    def _emergency_distinction_generation(self):
        """
        Emergency response to imminent dissolution.

        When distinction falls critically low, the system must
        generate novel distinctions or face existential dissolution.
        """

        print("🚨 EMERGENCY DISTINCTION GENERATION ACTIVATED")

        # Force increased symbolic curvature through random perturbation
        if hasattr(self.base_platform, 'qse_core'):
            # Add noise to surplus field to create distinctions
            noise_amplitude = 0.3
            self.base_platform.qse_core.S += np.random.randn(*self.base_platform.qse_core.S.shape) * noise_amplitude

            # Ensure surplus stays in valid range
            self.base_platform.qse_core.S = np.clip(self.base_platform.qse_core.S, 0.0, 1.0)

            print("   💉 Injected distinction noise to prevent dissolution")


def create_antifinal_existential_platform(base_platform):
    """
    Create a platform with true existential dynamics based on
    distinction as being and the principle of Antifinity.
    """

    return AntifinalExistentialPlatform(base_platform)
