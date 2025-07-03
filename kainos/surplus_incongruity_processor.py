

"""
Surplus Incongruity Processor for Émile Framework
Integrates surplus-distinction consciousness with correlative log reading.

This module coordinates between the consciousness system and log correlation
to create the complete surplus-distinction dynamics.
"""
import sys
import os

# Get the absolute path of the directory containing the emile_cogito package
# Assuming metabolic_tests.py is in /content/emile_cogito/testing/
# We need to add /content/ to the sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
emile_cogito_dir = os.path.dirname(parent_dir) # This should be /content/

if emile_cogito_dir not in sys.path:
    sys.path.append(emile_cogito_dir)

from emile_cogito.kainos.module_wide_flow_mapper import auto_map_module_flow
auto_map_module_flow(__name__)  # Maps the entire module!
import numpy as np
from typing import Dict, List, Any, Optional
from emile_cogito.kainos.config import CONFIG

class SurplusIncongruityProcessor:
    """
    Processes surplus incongruity and distinction enhancement rather than filling deficits.

    The system's thriving depends on maintaining productive distinction through
    correlative capacity with environmental patterns.

    This class coordinates between:
    - SurplusDistinctionConsciousness (main consciousness system)
    - CorrelativeLogReader (log correlation system)
    """

    def __init__(self, cfg=CONFIG):
        self.cfg = cfg

        # Initialize the distinction consciousness system
        from emile_cogito.kainos.metabolic import SurplusDistinctionConsciousness
        self.distinction_consciousness = SurplusDistinctionConsciousness(cfg)

        # Initialize the correlative log reader
        from emile_cogito.kainos.log_reader import CorrelativeLogReader
        self.correlative_reader = CorrelativeLogReader(cfg)

    # REPLACE THIS METHOD IN surplus_incongruity_processor.py

    def process_surplus_distinction_step(self, current_experience: Dict[str, Any],
                                      dt: float = 1.0) -> Dict[str, Any]:
        """
        Process one step of surplus-distinction dynamics with log correlation.

        CORRECTED: Surplus-based systems ALWAYS correlate - they don't wait for deficits!
        """
        # Update log buffer with current experience
        self.correlative_reader.update_live_buffer(current_experience)

        # Calculate surplus incongruity (for monitoring, not gating learning)
        surplus_incongruity = self.correlative_reader.detect_surplus_incongruity(current_experience)

        # Apply natural repetition pressure from consciousness system
        pressure_results = {}
        pressure_applied = self.distinction_consciousness.natural_repetition_pressure(dt)
        pressure_results['repetition_drift'] = pressure_applied
        pressure_results['distinction_status'] = self.distinction_consciousness._get_distinction_status()

        # ★ CORRECTED: Always correlate when data exists - surplus systems are naturally correlative
        overall_incongruity = surplus_incongruity.get('overall_incongruity', 0)

        # Correlation capacity scales with surplus expression - more surplus = better learning!
        correlation_capacity = self.distinction_consciousness.state.surplus_expression

        distinction_enhancement = 0.0
        log_correlation_results = {}
        correlation_performed = False

        # ALWAYS try to correlate if there's data - this is what surplus systems DO
        if len(self.correlative_reader.live_log_buffer) > 0:
            correlation_performed = True

            # Find the biggest incongruity to target (for focus, not necessity)
            biggest_incongruity_type = max(surplus_incongruity.items(),
                                        key=lambda x: x[1] if x[0] != 'overall_incongruity' else 0)[0]

            # Access logs for correlative capacity - modulated by surplus expression
            base_correlation_results = self.correlative_reader.access_logs_for_correlation(biggest_incongruity_type)

            # Scale enhancement by correlation capacity (more surplus = better learning)
            raw_enhancement = base_correlation_results.get('distinction_enhancement', 0)
            distinction_enhancement = raw_enhancement * correlation_capacity

            log_correlation_results = {
                **base_correlation_results,
                'distinction_enhancement': distinction_enhancement,
                'correlation_capacity': correlation_capacity,
                'raw_enhancement': raw_enhancement
            }

        # Apply enhancement from correlation to consciousness system
        if distinction_enhancement > 0:
            # Map incongruity types to achievement types for the consciousness system
            achievement_type_mapping = {
                'regime_distinction': 'correlation',
                'temporal_distinction': 'correlation',
                'integration_distinction': 'correlation',
                'consciousness_correlation': 'correlation',
                'valence_correlation': 'correlation',
                'surplus_correlation': 'correlation'
            }

            achievement_type = achievement_type_mapping.get(
                biggest_incongruity_type if correlation_performed else 'general', 'correlation')

            enhancement_applied = self.distinction_consciousness.enhance_through_achievement(
                distinction_enhancement, achievement_type)

        # Get cognitive modulation factors from consciousness system
        cognitive_modulation = self.distinction_consciousness.get_distinction_modulation_factors()

        return {
            'surplus_incongruity': surplus_incongruity,
            'correlation_performed': correlation_performed,  # Changed from correlation_needed
            'correlation_capacity': correlation_capacity if correlation_performed else 0.0,
            'distinction_enhancement': distinction_enhancement,
            'log_correlation': log_correlation_results,
            'cognitive_modulation': cognitive_modulation,
            'pressure_results': pressure_results,
            'state_summary': self.get_state_summary()
        }

    def get_state_summary(self) -> Dict[str, Any]:
        """Get state summary from the distinction consciousness system."""

        # Get correlative capacity data
        correlative_capacity = self.correlative_reader.get_correlative_capacity_level()

        return {
            'distinction_status': self.distinction_consciousness._get_distinction_status(),
            'surplus_expression': self.distinction_consciousness.state.surplus_expression,
            'distinction_coherence': self.distinction_consciousness.state.distinction_coherence,
            'environmental_correlation': self.distinction_consciousness.state.environmental_correlation,
            'integration_drive': self.distinction_consciousness.state.integration_drive,
            'correlation_debt': self.distinction_consciousness.state.correlation_debt,
            'distinction_efficiency': self.distinction_consciousness.state.distinction_efficiency,
            'pending_expressions': len(self.distinction_consciousness.pending_expressions),
            'expression_motivation': self.distinction_consciousness.get_expression_motivation(),

            'distinction_capacities': {
                'symbol_distinction_capacity': min(1.0, self.distinction_consciousness.state.surplus_expression * 0.7),
                'pattern_distinction_capacity': min(1.0, self.distinction_consciousness.state.distinction_coherence * 0.8),
                'temporal_distinction_capacity': min(1.0, self.distinction_consciousness.state.environmental_correlation * 0.6)
            },

            'distinction_drives': {
                'integration_distinction_drive': self.distinction_consciousness.state.integration_drive,
                'novelty_distinction_drive': max(0.0, 1.0 - self.distinction_consciousness.state.environmental_correlation),
                'environmental_distinction_drive': max(0.0, 1.0 - self.distinction_consciousness.state.distinction_coherence)
            },

            # ADD THESE MISSING FIELDS:
            'correlated_symbols': correlative_capacity['symbol_vocabulary'],
            'symbol_correlation_strength': correlative_capacity['overall_capacity'],
            'recent_distinction_success': correlative_capacity['overall_capacity']
        }

    def get_complete_state_summary(self) -> Dict[str, Any]:
        """Get complete state summary including correlative capacity."""
        distinction_state = self.get_state_summary()
        correlative_capacity = self.correlative_reader.get_correlative_capacity_level()

        # Add correlative/symbol data
        distinction_state['correlated_symbols'] = correlative_capacity['symbol_vocabulary']
        distinction_state['symbol_correlation_strength'] = correlative_capacity['overall_capacity']
        distinction_state['recent_distinction_success'] = correlative_capacity['overall_capacity']

        return {
            **distinction_state,
            'correlative_capacity': correlative_capacity,
            'correlation_vocabulary': correlative_capacity['symbol_vocabulary'],
            'reading_capacity': correlative_capacity['overall_capacity']
        }

    def apply_temporal_distinction_enhancement(self, objective_time: float,
                                             subjective_time: float,
                                             emergent_time_rate: float,
                                             dt: float = 1.0) -> Dict[str, Any]:
        """
        Apply temporal distinction enhancement through the consciousness system.

        This delegates to the consciousness system's temporal processing.
        """
        return self.distinction_consciousness.process_temporal_distinction_step(
            objective_time, subjective_time, emergent_time_rate, dt)

    def process_expression_dynamics(self, expression_content: str,
                                  expression_intensity: float = 1.0):
        """
        Process expression dynamics through the consciousness system.
        """
        return self.distinction_consciousness.expression_distinction_dynamics(
            expression_content, expression_intensity)

    def process_environmental_correlation(self, expression_id: int,
                                        environmental_response: Dict[str, Any]) -> float:
        """
        Process environmental correlation through the consciousness system.
        """
        return self.distinction_consciousness.process_environmental_correlation(
            expression_id, environmental_response)

    def get_cognitive_modulation_factors(self) -> Dict[str, float]:
        """Get cognitive modulation factors from the consciousness system."""
        return self.distinction_consciousness.get_distinction_modulation_factors()

    def enable_existential_mode(self):
        """Enable existential mode for real distinction stakes."""
        self.distinction_consciousness.enable_existential_mode()

    def disable_existential_mode(self):
        """Disable existential mode for collaborative dynamics."""
        self.distinction_consciousness.disable_existential_mode()

    def step(self, dt: float = 1.0) -> Dict[str, Any]:
        """
        Process one complete step of the integrated system.
        """
        # Step the consciousness system
        consciousness_results = self.distinction_consciousness.step(dt)

        # Get correlative capacity
        correlative_capacity = self.correlative_reader.get_correlative_capacity_level()

        # Combine results
        return {
            **consciousness_results,
            'correlative_capacity': correlative_capacity,
            'reading_capacity': correlative_capacity['overall_capacity']
        }



from emile_cogito.kainos.module_wide_flow_mapper import auto_map_module_flow
auto_map_module_flow(__name__)  # Maps the entire module!
