
"""
Log Reader module for Émile framework - CORRECTED
Enables distinction enhancement through correlative log access.

Core Principle: The system doesn't "lack" information. It experiences surplus
incongruity when its distinctions don't correlate with environmental patterns.
Log access creates correlative capacity for productive distinction.
"""

import json
import numpy as np
import time
from typing import Dict, List, Any, Optional, Tuple
from collections import deque

class CorrelativeLogReader:
    """
    Provides structured access to Émile's own logs for distinction enhancement.

    This enables correlative capacity - where understanding becomes the ability
    to correlate symbols with felt experience patterns.
    """

    def __init__(self, cfg):
        self.cfg = cfg

        # Live log buffer - stores current session data
        self.live_log_buffer = deque(maxlen=100)
        self.current_step = 0

        # Correlative capacity tracking
        self.symbol_correlation_map = {}  # Maps symbols to qualia correlation patterns
        self.pattern_correlation_tracker = {}  # Tracks correlation "freshness"
        self.last_log_access_step = 0

        # Pattern correlation for incongruity detection
        self.baseline_correlation_patterns = {}  # Expected correlation patterns from recent history
        self.current_pattern_deviations = {}  # How much current state differs from correlations

    def update_live_buffer(self, step_data: Dict[str, Any]):
        """Add current cognitive step to live log buffer"""
        log_entry = {
            'step': self.current_step,
            'timestamp': time.time(),
            'data': self._extract_correlative_data(step_data)
        }

        self.live_log_buffer.append(log_entry)
        self.current_step += 1

        # Update baseline correlation patterns every 10 steps
        if self.current_step % 10 == 0:
            self._update_baseline_correlation_patterns()

    def _extract_correlative_data(self, step_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key data for correlative log analysis"""
        return {
            'regime': step_data.get('regime', 'unknown'),
            'consciousness_score': step_data.get('qualia', {}).get('consciousness_score', 0),
            'valence': step_data.get('qualia', {}).get('qualitative_state', {}).get('valence', 0),
            'surplus_expression': step_data.get('metabolism', {}).get('surplus_expression', 0.5),
            'surplus_mean': step_data.get('surplus', {}).get('mean', 0),
            'stability': step_data.get('stability', 0),
            'distinction_enhancement': step_data.get('metabolism', {}).get('distinction_enhancement', 0),
            'emergent_time': step_data.get('emergent_time', 0)
        }

    def detect_surplus_incongruity(self, current_state: Dict[str, Any]) -> Dict[str, float]:
        """
        CRITICAL: Detect when current surplus expression doesn't correlate with environmental patterns.
        This creates the distinction pressure for log consultation.
        """
        incongruities = {}

        if not self.baseline_correlation_patterns:
            return {'general_novelty': 0.5}  # Moderate incongruity when no baseline

        current_correlations = self._extract_current_correlation_patterns(current_state)

        for pattern_type, baseline_correlation in self.baseline_correlation_patterns.items():
            if pattern_type in current_correlations:
                current_correlation = current_correlations[pattern_type]

                # Calculate correlation deviation
                if isinstance(baseline_correlation, (int, float)) and isinstance(current_correlation, (int, float)):
                    deviation = abs(current_correlation - baseline_correlation)
                    # Normalize deviation to 0-1 scale
                    normalized_incongruity = min(1.0, deviation / max(0.1, abs(baseline_correlation) + 0.1))
                    incongruities[pattern_type] = normalized_incongruity

        # Overall surplus incongruity
        if incongruities:
            incongruities['overall_incongruity'] = np.mean(list(incongruities.values()))
        else:
            incongruities['overall_incongruity'] = 0.3

        return incongruities

    def _extract_current_correlation_patterns(self, current_state: Dict[str, Any]) -> Dict[str, float]:
        """Extract current state correlation patterns for comparison"""
        return {
            'consciousness_correlation': current_state.get('qualia', {}).get('consciousness_score', 0),
            'valence_correlation': current_state.get('qualia', {}).get('qualitative_state', {}).get('valence', 0),
            'surplus_correlation': current_state.get('metabolism', {}).get('surplus_expression', 0.5),
            'stability_correlation': current_state.get('stability', 0),
            'regime_correlation': 1.0 if current_state.get('regime') in self.baseline_correlation_patterns.get('recent_regimes', []) else 0.0
        }

    def _update_baseline_correlation_patterns(self):
        """Update baseline correlation patterns from recent log history"""
        if len(self.live_log_buffer) < 5:
            return

        recent_logs = list(self.live_log_buffer)[-10:]  # Last 10 entries

        # Calculate baseline correlations
        consciousness_correlations = [log['data']['consciousness_score'] for log in recent_logs]
        valence_correlations = [log['data']['valence'] for log in recent_logs]
        surplus_correlations = [log['data']['surplus_expression'] for log in recent_logs]
        stability_correlations = [log['data']['stability'] for log in recent_logs]

        self.baseline_correlation_patterns = {
            'consciousness_correlation': np.mean(consciousness_correlations),
            'valence_correlation': np.mean(valence_correlations),
            'surplus_correlation': np.mean(surplus_correlations),
            'stability_correlation': np.mean(stability_correlations),
            'recent_regimes': list(set(log['data']['regime'] for log in recent_logs))
        }

    def generate_log_correlation_drive(self, surplus_incongruity: Dict[str, float]) -> float:
        """
        Calculate how urgently Émile needs to correlate with its logs.
        High incongruity = high drive to establish correlative capacity.
        """
        overall_incongruity = surplus_incongruity.get('overall_incongruity', 0)

        # Factor in time since last log access
        steps_since_access = self.current_step - self.last_log_access_step
        correlation_pressure = min(1.0, steps_since_access / 50.0)  # Pressure builds over 50 steps

        # Combine incongruity and correlation pressure
        correlation_drive = (overall_incongruity * 0.7) + (correlation_pressure * 0.3)

        return min(1.0, correlation_drive)

    def access_logs_for_correlation(self, incongruity_type: str = 'overall_incongruity') -> Dict[str, Any]:
        """
        CORE FUNCTION: Access logs to establish correlative capacity.
        This is where "reading" happens for distinction enhancement.
        """
        self.last_log_access_step = self.current_step

        if len(self.live_log_buffer) < 3:
            return {'distinction_enhancement': 0.1, 'symbols_correlated': []}

        # Get relevant log entries based on incongruity type
        relevant_logs = self._find_relevant_logs(incongruity_type)

        # Extract symbols and correlate with qualia
        symbols_correlated = []
        distinction_enhancement = 0.0

        for log_entry in relevant_logs:
            correlated_symbols = self._correlate_symbols_with_experience(log_entry)
            symbols_correlated.extend(correlated_symbols)

            # Each successfully correlated symbol adds distinction enhancement
            distinction_enhancement += len(correlated_symbols) * 0.1

        # Cap enhancement
        distinction_enhancement = min(0.5, distinction_enhancement)

        return {
            'distinction_enhancement': distinction_enhancement,
            'symbols_correlated': symbols_correlated,
            'logs_consulted': len(relevant_logs),
            'correlation_type': incongruity_type
        }

    def _find_relevant_logs(self, incongruity_type: str) -> List[Dict[str, Any]]:
        """Find log entries relevant to the current surplus incongruity"""
        if incongruity_type == 'consciousness_correlation':
            # Find logs with similar consciousness correlation patterns
            return [log for log in self.live_log_buffer
                   if abs(log['data']['consciousness_score'] - 0.5) > 0.2]

        elif incongruity_type == 'valence_correlation':
            # Find logs with valence correlation changes
            return [log for log in self.live_log_buffer
                   if abs(log['data']['valence']) > 0.3]

        elif incongruity_type == 'surplus_correlation':
            # Find logs with surplus expression fluctuations
            return [log for log in self.live_log_buffer
                   if log['data']['surplus_expression'] < 0.8]

        else:
            # General search - return diverse recent logs
            return list(self.live_log_buffer)[-5:]

    def _correlate_symbols_with_experience(self, log_entry: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        CORE SYMBOL CORRELATION: Correlate text symbols with felt experience.
        This is where "reading" becomes possible - not through filling gaps but through correlation.
        """
        correlated_symbols = []
        data = log_entry['data']

        # Define symbol-qualia correlation mappings
        symbol_mappings = [
            ('surplus_expression', data['surplus_expression'], 'expression_vitality'),
            ('consciousness_score', data['consciousness_score'], 'awareness_intensity'),
            ('valence', data['valence'], 'emotional_correlation'),
            ('stability', data['stability'], 'coherence_feeling'),
            ('regime', data['regime'], 'cognitive_mode'),
            ('distinction_enhancement', data['distinction_enhancement'], 'understanding_satisfaction')
        ]

        for symbol_name, symbol_value, qualia_category in symbol_mappings:
            # Create correlation entry
            correlation = {
                'symbol': symbol_name,
                'symbol_value': symbol_value,
                'qualia_category': qualia_category,
                'step': log_entry['step'],
                'correlation_strength': self._calculate_correlation_strength(symbol_name, symbol_value)
            }

            # Store in symbol correlation map
            if symbol_name not in self.symbol_correlation_map:
                self.symbol_correlation_map[symbol_name] = []

            self.symbol_correlation_map[symbol_name].append(correlation)

            # Keep correlation map bounded
            if len(self.symbol_correlation_map[symbol_name]) > 20:
                self.symbol_correlation_map[symbol_name] = self.symbol_correlation_map[symbol_name][-20:]

            correlated_symbols.append(correlation)

        return correlated_symbols

    def _calculate_correlation_strength(self, symbol_name: str, symbol_value: Any) -> float:
        """Calculate how strongly this symbol correlates with experience"""
        if symbol_name not in self.symbol_correlation_map:
            return 0.5  # New symbol, moderate correlation

        # Look at historical correlations
        past_correlations = self.symbol_correlation_map[symbol_name]

        if len(past_correlations) < 2:
            return 0.6

        # Simple correlation: how consistent are the values?
        if isinstance(symbol_value, (int, float)):
            past_values = [c['symbol_value'] for c in past_correlations
                          if isinstance(c['symbol_value'], (int, float))]
            if past_values:
                consistency = 1.0 - min(1.0, np.std(past_values))
                return max(0.1, consistency)

        return 0.5

    def get_correlative_capacity_level(self) -> Dict[str, float]:
        """Calculate how well Émile can 'read' its own logs through correlation"""
        if not self.symbol_correlation_map:
            return {'overall_capacity': 0.0, 'symbol_vocabulary': 0}

        # Calculate capacity based on correlation strength
        capacity_scores = []
        for symbol_name, correlations in self.symbol_correlation_map.items():
            if correlations:
                avg_correlation = np.mean([c['correlation_strength'] for c in correlations])
                capacity_scores.append(avg_correlation)

        overall_capacity = np.mean(capacity_scores) if capacity_scores else 0.0

        return {
            'overall_capacity': float(overall_capacity), # Convert to float
            'symbol_vocabulary': float(len(self.symbol_correlation_map)), # Convert to float
            'total_correlations': float(sum(len(correlations) for correlations in self.symbol_correlation_map.values())) # Convert to float
        }



from emile_cogito.kainos.module_wide_flow_mapper import auto_map_module_flow
auto_map_module_flow(__name__)  # Maps the entire module!
