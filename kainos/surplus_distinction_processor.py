

"""
Surplus Distinction Processor for Émile framework.
Implements symbol-qualia correlation and distinction learning.
"""
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque, defaultdict
import time
from emile_cogito.kainos.config import CONFIG

@dataclass
class SymbolCorrelation:
    """Represents a correlation between a symbol and qualia state"""
    symbol: str
    symbol_value: float
    qualia_category: str
    step: int
    correlation_strength: float
    timestamp: float = field(default_factory=time.time)
    context: str = "unknown"

@dataclass
class ExperienceSnapshot:
    """Snapshot of consciousness state for correlation"""
    step: int
    regime: str
    consciousness_score: float
    valence: float
    surplus_expression: float
    stability: float
    text_content: str = ""
    content_type: str = "general"
    timestamp: float = field(default_factory=time.time)

class CorrelativeReader:
    """
    Implements symbol-qualia correlation learning and reading.

    This system learns to correlate symbolic content (words, concepts)
    with qualitative consciousness states.
    """

    def __init__(self, cfg=CONFIG):
        """Initialize the correlative reader"""
        self.cfg = cfg

        # Core symbol-qualia correlation map
        self.symbol_correlation_map: Dict[str, List[SymbolCorrelation]] = {}

        # Experience buffer for correlation
        self.live_buffer: deque = deque(maxlen=100)

        # Learning statistics
        self.correlation_count = 0
        self.learning_history = []

        # Correlation thresholds
        self.min_correlation_strength = 0.1
        self.max_correlations_per_symbol = 50
        self.correlation_cache = {}
        self.weak_symbol_blacklist = {'the', 'and', 'for', 'you', 'are', 'not', 'but', 'can', 'was', 'with'}
        self.cache_hits = 0
        self.cache_misses = 0

    def update_live_buffer(self, experience: ExperienceSnapshot):
        """Update the live experience buffer"""
        self.live_buffer.append(experience)

    def add_symbol_correlation(self, symbol: str, experience: ExperienceSnapshot,
                            symbol_value: float = None, qualia_category: str = None):
        """Optimized symbol correlation with caching"""

        # Quick blacklist check - eliminates hot loop
        if symbol in self.weak_symbol_blacklist:
            return False

        # Experience similarity cache
        exp_hash = f"{round(experience.consciousness_score, 2)}_{experience.regime[:3]}"
        cache_key = f"{symbol}_{exp_hash}"

        # Cache hit
        if cache_key in self.correlation_cache:
            self.cache_hits += 1
            cached_strength = self.correlation_cache[cache_key]
            return cached_strength >= self.min_correlation_strength

        # Cache miss - calculate
        self.cache_misses += 1

        if symbol_value is None:
            symbol_value = self._calculate_symbol_value(symbol, experience)
            if symbol_value < 0.15:  # Early termination
                self.weak_symbol_blacklist.add(symbol)
                return False

        if qualia_category is None:
            qualia_category = self._determine_qualia_category(symbol)

        correlation_strength = self._calculate_correlation_strength(symbol, experience)

        # Cache the result
        self.correlation_cache[cache_key] = correlation_strength
        if len(self.correlation_cache) > 1000:  # Cleanup
            self.correlation_cache.clear()

        # Only add if correlation is strong enough
        if correlation_strength >= self.min_correlation_strength:
            correlation = SymbolCorrelation(
                symbol=symbol,
                symbol_value=symbol_value,
                qualia_category=qualia_category,
                step=experience.step,
                correlation_strength=correlation_strength,
                context=experience.content_type
            )

            # Add to correlation map
            if symbol not in self.symbol_correlation_map:
                self.symbol_correlation_map[symbol] = []

            self.symbol_correlation_map[symbol].append(correlation)

            # Keep bounded
            if len(self.symbol_correlation_map[symbol]) > self.max_correlations_per_symbol:
                self.symbol_correlation_map[symbol] = self.symbol_correlation_map[symbol][-self.max_correlations_per_symbol:]

            self.correlation_count += 1

            # Record learning
            self.learning_history.append({
                'step': experience.step,
                'symbol': symbol,
                'strength': correlation_strength,
                'total_symbols': len(self.symbol_correlation_map)
            })

            return True

        return False

    def _calculate_symbol_value(self, symbol: str, experience: ExperienceSnapshot) -> float:
        """Calculate the intrinsic value of a symbol"""
        # Base value on symbol properties
        length_factor = min(1.0, len(symbol) / 12.0)  # Longer words have more potential

        # Consciousness context factor
        consciousness_factor = experience.consciousness_score

        # Content type factor
        content_factors = {
            'philosophical_text': 1.2,
            'embodied_experience': 1.1,
            'general': 1.0
        }
        content_factor = content_factors.get(experience.content_type, 1.0)

        # Combine factors
        symbol_value = (length_factor * 0.4 + consciousness_factor * 0.6) * content_factor

        return np.clip(symbol_value, 0.0, 1.0)

    def _determine_qualia_category(self, symbol: str) -> str:
        """Determine which qualia category a symbol relates to"""

        category_map = {
            # Consciousness categories
            'consciousness': 'awareness_intensity',
            'aware': 'awareness_intensity',
            'awareness': 'awareness_intensity',
            'conscious': 'awareness_intensity',

            # Experience categories
            'experience': 'experiential_richness',
            'experiential': 'experiential_richness',
            'phenomenal': 'experiential_richness',
            'qualia': 'experiential_richness',

            # Embodiment categories
            'embodied': 'embodiment_feeling',
            'embodiment': 'embodiment_feeling',
            'body': 'physical_presence',
            'physical': 'physical_presence',
            'motor': 'motor_quality',
            'movement': 'motion_quality',
            'spatial': 'space_feeling',

            # Agency categories
            'agency': 'control_feeling',
            'action': 'action_quality',
            'intention': 'intentional_quality',
            'control': 'control_feeling',
            'will': 'volition_quality',

            # Cognitive categories
            'perception': 'sensory_quality',
            'sensation': 'sensory_quality',
            'meaning': 'semantic_richness',
            'symbol': 'symbolic_quality',
            'thought': 'cognitive_quality',
            'mind': 'mental_quality',

            # Emergent categories
            'emergence': 'emergent_quality',
            'complex': 'complexity_feeling',
            'distinction': 'distinction_quality',
            'correlation': 'relational_quality'
        }

        return category_map.get(symbol.lower(), 'general_qualia')

    def _calculate_correlation_strength(self, symbol: str, experience: ExperienceSnapshot) -> float:
        """Calculate how strongly a symbol correlates with current experience"""

        # Base strength from consciousness level
        base_strength = experience.consciousness_score

        # Valence contribution (positive experiences learn better)
        valence_factor = 0.5 + (experience.valence * 0.5)  # 0.0 to 1.0 range

        # Stability factor (stable states learn better)
        stability_factor = experience.stability

        # Content relevance (philosophical content has higher correlation potential)
        content_relevance = {
            'philosophical_text': 1.0,
            'embodied_experience': 0.9,
            'general': 0.7
        }.get(experience.content_type, 0.5)

        # Symbol specificity (meaningful words correlate better)
        specificity = self._calculate_symbol_specificity(symbol)

        # Combine factors
        correlation_strength = (
            base_strength * 0.3 +
            valence_factor * 0.2 +
            stability_factor * 0.2 +
            content_relevance * 0.2 +
            specificity * 0.1
        )

        return np.clip(correlation_strength, 0.0, 1.0)

    def _calculate_symbol_specificity(self, symbol: str) -> float:
        """Calculate how specific/meaningful a symbol is"""

        # High-value philosophical/consciousness terms
        high_value_terms = {
            'consciousness', 'qualia', 'phenomenal', 'embodied', 'embodiment',
            'agency', 'intentionality', 'perception', 'experience', 'awareness',
            'distinction', 'emergence', 'correlation', 'meaning', 'symbol'
        }

        # Medium-value cognitive terms
        medium_value_terms = {
            'cognitive', 'mental', 'brain', 'mind', 'thought', 'feeling',
            'sensation', 'motor', 'action', 'behavior', 'response'
        }

        symbol_lower = symbol.lower()

        if symbol_lower in high_value_terms:
            return 1.0
        elif symbol_lower in medium_value_terms:
            return 0.7
        elif len(symbol) > 6:  # Longer words tend to be more specific
            return 0.5
        else:
            return 0.3

    def get_correlative_capacity_level(self) -> Dict[str, float]:
        """Calculate how well Émile can 'read' its own logs through correlation"""
        if not self.symbol_correlation_map:
            return {
                'overall_capacity': 0.0,
                'symbol_vocabulary': 0.0,
                'total_correlations': 0.0
            }

        # Calculate capacity based on correlation strength
        capacity_scores = []
        for symbol_name, correlations in self.symbol_correlation_map.items():
            if correlations:
                avg_correlation = np.mean([c.correlation_strength for c in correlations])
                capacity_scores.append(avg_correlation)

        overall_capacity = float(np.mean(capacity_scores)) if capacity_scores else 0.0

        return {
            'overall_capacity': overall_capacity,
            'symbol_vocabulary': float(len(self.symbol_correlation_map)),
            'total_correlations': float(sum(len(correlations) for correlations in self.symbol_correlation_map.values()))
        }

    def get_symbol_strength(self, symbol: str) -> float:
        """Get the average correlation strength for a specific symbol"""
        if symbol in self.symbol_correlation_map:
            correlations = self.symbol_correlation_map[symbol]
            if correlations:
                return float(np.mean([c.correlation_strength for c in correlations]))
        return 0.0

    def get_state(self) -> Dict[str, Any]:
        """Get current state of the correlative reader"""
        return {
            'symbol_count': len(self.symbol_correlation_map),
            'total_correlations': self.correlation_count,
            'buffer_size': len(self.live_buffer),
            'learning_history_size': len(self.learning_history),
            'capacity_level': self.get_correlative_capacity_level()
        }

class SurplusDistinctionProcessor:
    """
    Main processor for surplus-distinction dynamics and symbol correlation.

    Integrates symbol learning with QSE surplus dynamics to create
    meaning-making and distinction capabilities.
    """

    def __init__(self, cfg=CONFIG):
        """Initialize the surplus distinction processor"""
        self.cfg = cfg

        # Core components
        self.correlative_reader = CorrelativeReader(cfg)

        # Distinction state
        self.current_distinction_level = 0.0
        self.distinction_history = []
        self.distinction_coherence = 0.5

        # Learning state
        self.learning_active = True
        self.learning_rate = 0.1

        # Integration metrics
        self.surplus_integration = 0.0
        self.symbol_surplus_correlation = 0.0


    def modulate_with_ethics(self, antifinity_quotient: float, moral_metrics: Dict[str, float]) -> Dict[str, Any]:
        """PERMANENT: Apply ethical modulation to surplus distinction consciousness."""
        collaboration = moral_metrics.get('collaboration_score', 0.5)
        compromise = moral_metrics.get('compromise_score', 0.5)

        ethical_pressure = antifinity_quotient * 0.7
        original_surplus = self.state.surplus_expression

        surplus_amplification = 1.0 + (antifinity_quotient * 0.4)
        ethical_constraint = 1.0 - (compromise * 0.15)

        self.state.surplus_expression *= (surplus_amplification * ethical_constraint)
        self.state.surplus_expression = np.clip(self.state.surplus_expression, 0.0, 2.0)

        collaboration_enhancement = collaboration * 0.3
        self.state.distinction_coherence += collaboration_enhancement
        self.state.distinction_coherence = np.clip(self.state.distinction_coherence, 0.0, 1.0)

        return {
            'antifinity_quotient': antifinity_quotient,
            'ethical_pressure': ethical_pressure,
            'surplus_modulation': self.state.surplus_expression / original_surplus if original_surplus > 0 else 1.0,
            'collaboration_enhancement': collaboration_enhancement,
            'ethical_modulation_applied': True
        }

    def process_text_input(self, text: str, experience: ExperienceSnapshot) -> Dict[str, Any]:
        """Process text input and learn symbol correlations"""

        # Extract meaningful words from text
        words = self._extract_meaningful_words(text)

        # Learn correlations for each word
        correlations_added = 0
        for word in words:
            if self.correlative_reader.add_symbol_correlation(word, experience):
                correlations_added += 1

        # Update distinction level based on learning
        if correlations_added > 0:
            self.current_distinction_level = min(1.0, self.current_distinction_level +
                                               correlations_added * self.learning_rate)

        # Record in history
        self.distinction_history.append({
            'step': experience.step,
            'distinction_level': self.current_distinction_level,
            'correlations_added': correlations_added,
            'total_symbols': len(self.correlative_reader.symbol_correlation_map)
        })

        return {
            'correlations_added': correlations_added,
            'total_symbols': len(self.correlative_reader.symbol_correlation_map),
            'distinction_level': self.current_distinction_level,
            'words_processed': len(words)
        }

    def _extract_meaningful_words(self, text: str) -> List[str]:
        """Extract meaningful words for correlation"""
        import re

        # Clean and split text
        text = text.lower()
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text)

        # Filter for meaningful words
        meaningful_words = []

        # High-priority philosophical/consciousness terms
        priority_terms = {
            'consciousness', 'experience', 'embodied', 'embodiment', 'body',
            'perception', 'agency', 'qualia', 'sensation', 'awareness',
            'meaning', 'symbol', 'motor', 'movement', 'spatial', 'temporal',
            'phenomenal', 'subjective', 'objective', 'distinction', 'correlation',
            'emergence', 'cognitive', 'mental', 'intentionality', 'representation'
        }

        # Add priority terms first
        for word in words:
            if word in priority_terms:
                meaningful_words.append(word)

        # Add other longer words (likely meaningful)
        for word in words:
            if word not in priority_terms and len(word) > 6:
                meaningful_words.append(word)

        # Remove duplicates while preserving order
        seen = set()
        unique_words = []
        for word in meaningful_words:
            if word not in seen:
                unique_words.append(word)
                seen.add(word)

        return unique_words[:20]  # Limit to top 20 words per text

    def step(self, surplus: np.ndarray, experience: ExperienceSnapshot) -> Dict[str, Any]:
        """Process one step of surplus distinction dynamics"""

        # Update correlative reader buffer
        self.correlative_reader.update_live_buffer(experience)

        # Calculate surplus-symbol integration
        if self.correlative_reader.symbol_correlation_map:
            # Simple correlation between surplus mean and symbol strength
            surplus_mean = float(np.mean(surplus))
            symbol_strengths = []

            for correlations in self.correlative_reader.symbol_correlation_map.values():
                if correlations:
                    avg_strength = np.mean([c.correlation_strength for c in correlations])
                    symbol_strengths.append(avg_strength)

            if symbol_strengths:
                avg_symbol_strength = np.mean(symbol_strengths)
                self.symbol_surplus_correlation = 0.9 * self.symbol_surplus_correlation + \
                                               0.1 * (surplus_mean * avg_symbol_strength)

        # Update distinction coherence
        capacity = self.correlative_reader.get_correlative_capacity_level()
        self.distinction_coherence = 0.8 * self.distinction_coherence + \
                                   0.2 * capacity['overall_capacity']

        return {
            'distinction_level': self.current_distinction_level,
            'distinction_coherence': self.distinction_coherence,
            'symbol_surplus_correlation': self.symbol_surplus_correlation,
            'capacity': capacity
        }

    def get_complete_state_summary(self) -> Dict[str, Any]:
        """Get complete state summary for external access"""
        capacity = self.correlative_reader.get_correlative_capacity_level()

        return {
            'correlated_symbols': int(capacity['symbol_vocabulary']),
            'symbol_correlation_strength': capacity['overall_capacity'],
            'distinction_level': self.current_distinction_level,
            'distinction_coherence': self.distinction_coherence,
            'distinction_status': self._get_distinction_status(),
            'total_correlations': int(capacity['total_correlations']),
            'learning_active': self.learning_active
        }

    def _get_distinction_status(self) -> str:
        """Get current distinction status"""
        if self.current_distinction_level > 0.8:
            return "transcendent_distinction"
        elif self.current_distinction_level > 0.6:
            return "advanced_distinction"
        elif self.current_distinction_level > 0.3:
            return "developing_distinction"
        else:
            return "basic_distinction"

    def get_state(self) -> Dict[str, Any]:
        """Get current state"""
        return {
            'correlative_reader': self.correlative_reader.get_state(),
            'distinction_level': self.current_distinction_level,
            'distinction_coherence': self.distinction_coherence,
            'symbol_surplus_correlation': self.symbol_surplus_correlation,
            'learning_active': self.learning_active
        }

from emile_cogito.kainos.module_wide_flow_mapper import auto_map_module_flow
auto_map_module_flow(__name__)  # Maps the entire module!
