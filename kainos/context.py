

"""
Context module for Émile framework.
Manages external environment, input encoding, and contextual processing.
"""
import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Union
import re

from emile_cogito.kainos.config import CONFIG

class Context:
    """
    Manages the external environment and context for the Émile framework.

    Handles input processing, environment simulation, and context tracking.
    """

    def __init__(self, cfg=CONFIG):
        """
        Initialize the context system.

        Args:
            cfg: Configuration parameters
        """
        self.cfg = cfg

        # Initialize grid size
        self.grid_size = cfg.GRID_SIZE

        # Current external field (Φ)
        self.phi_field = np.zeros(self.grid_size)

        # Context history
        self.context_history = []

        # Input history
        self.input_history = []

        # Context parameters
        self.complexity = 0.5  # Current environment complexity (0-1)
        self.variability = 0.3  # Current environment variability (0-1)
        self.domain = "general"  # Current contextual domain

    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode text input into a symbolic field pattern.

        Args:
            text: Text input

        Returns:
            Encoded field pattern
        """
        if not text:
            return np.zeros(self.grid_size)

        # Normalize text
        text = text.lower()

        # Extract basic features
        length = len(text)
        word_count = len(text.split())

        # Calculate complexity measures
        chars = set(text)
        char_diversity = len(chars) / max(1, length)

        # Look for patterns
        questions = text.count('?') > 0
        exclamations = text.count('!') > 0

        # Simple sentiment analysis
        positive_words = ['good', 'great', 'excellent', 'positive', 'happy', 'joy']
        negative_words = ['bad', 'terrible', 'negative', 'sad', 'unhappy', 'problem']

        sentiment = 0.5  # Neutral default
        for word in positive_words:
            if word in text:
                sentiment += 0.1
        for word in negative_words:
            if word in text:
                sentiment -= 0.1
        sentiment = np.clip(sentiment, 0.1, 0.9)

        # Create pattern array
        pattern = np.zeros(self.grid_size)

        # Base wave based on text length
        x = np.linspace(0, 4*np.pi, self.grid_size)
        base_wave = 0.4 + 0.3 * np.sin(x * (0.5 + length / 100))

        # Add sentiment modulation
        sentiment_wave = sentiment * np.sin(x * 2 + 0.5)

        # Add question/exclamation features
        if questions:
            question_wave = 0.2 * np.sin(x * 3)
            base_wave += question_wave

        if exclamations:
            exclamation_wave = 0.3 * np.sin(x * 5)
            base_wave += exclamation_wave

        # Add character-based features
        char_positions = {}
        for i, char in enumerate(text):
            if char.isalpha():
                pos = int((ord(char) - ord('a')) / 26 * self.grid_size)
                char_positions[pos] = char_positions.get(pos, 0) + 1

        for pos, count in char_positions.items():
            # Add Gaussian bump at character positions
            width = self.grid_size // 20
            for i in range(self.grid_size):
                dist = min(abs(i - pos), self.grid_size - abs(i - pos))  # Circular distance
                pattern[i] += 0.2 * count * np.exp(-dist**2 / (2 * width**2))

        # Combine patterns
        pattern = 0.6 * base_wave + 0.4 * pattern

        # Ensure valid range
        pattern = np.clip(pattern, 0.0, 1.0)

        # Record complexity for future reference
        self.complexity = 0.3 + 0.4 * char_diversity + 0.3 * min(1.0, word_count / 50)

        # Store input in history
        self.input_history.append({
            "type": "text",
            "content": text,
            "complexity": self.complexity,
            "sentiment": sentiment
        })

        return pattern

    def encode_numeric(self, data: Union[List[float], np.ndarray]) -> np.ndarray:
        """
        Encode numeric data into a symbolic field pattern.

        Args:
            data: Numeric data (list or array)

        Returns:
            Encoded field pattern
        """
        # FIX: Replace the problematic check
        # OLD: if not data:
        # NEW: Check for empty data properly
        if data is None or (hasattr(data, '__len__') and len(data) == 0):
            return np.zeros(self.grid_size)

        # Convert to numpy array if needed
        if not isinstance(data, np.ndarray):
            data = np.array(data)

        # FIX: Add additional check for empty arrays
        if data.size == 0:
            return np.zeros(self.grid_size)

        # Normalize data to [0,1] range
        data_min = data.min()
        data_max = data.max()

        if data_max > data_min:
            normalized = (data - data_min) / (data_max - data_min)
        else:
            normalized = np.ones_like(data) * 0.5

        # Interpolate to grid size
        if len(normalized) != self.grid_size:
            x_original = np.linspace(0, 1, len(normalized))
            x_new = np.linspace(0, 1, self.grid_size)

            # Simple linear interpolation
            pattern = np.interp(x_new, x_original, normalized)
        else:
            pattern = normalized.copy()

        # Add some structure to make it more interesting
        x = np.linspace(0, 2*np.pi, self.grid_size)

        # Add mild oscillation based on data variance
        variance = np.var(normalized)
        oscillation = 0.1 * variance * np.sin(x * 2)

        # Mix original pattern with oscillation
        pattern = 0.9 * pattern + 0.1 * oscillation

        # Apply smoothing (optional)
        if self.grid_size > 20:
            kernel_size = self.grid_size // 20
            kernel = np.ones(kernel_size) / kernel_size
            pattern = np.convolve(pattern, kernel, mode='same')

        # Ensure valid range
        pattern = np.clip(pattern, 0.0, 1.0)

        # Calculate complexity based on pattern features
        self.complexity = 0.2 + 0.5 * variance + 0.3 * np.mean(np.abs(np.diff(pattern)))

        # Store input in history
        self.input_history.append({
            "type": "numeric",
            "content": "numeric_data",
            "complexity": self.complexity,
            "data_points": len(data)
        })

        return pattern

    def encode_image(self, image_data: np.ndarray) -> np.ndarray:
        """
        Encode image data into a symbolic field pattern.

        Args:
            image_data: 2D or 3D image array

        Returns:
            Encoded field pattern
        """
        # Simple approach: average image to 1D
        if len(image_data.shape) == 3:  # Multi-channel image
            # Average across channels and then columns
            avg_data = np.mean(image_data, axis=(1, 2))
        elif len(image_data.shape) == 2:  # Grayscale
            # Average across columns
            avg_data = np.mean(image_data, axis=1)
        else:
            # Already 1D
            avg_data = image_data

        # Now encode the 1D vector like numeric data
        return self.encode_numeric(avg_data)

    def create_phi_field(self, input_data: Any, input_type: str = "auto") -> np.ndarray:
        """
        Create external symbolic field (Φ) from input data.

        Args:
            input_data: Input data (text, numeric, or image)
            input_type: Type of input ("text", "numeric", "image", or "auto")

        Returns:
            The phi field encoding the input
        """
        # Auto-detect input type
        if input_type == "auto":
            if isinstance(input_data, str):
                input_type = "text"
            elif isinstance(input_data, (list, np.ndarray)):
                if isinstance(input_data, np.ndarray) and len(input_data.shape) > 1:
                    input_type = "image"
                else:
                    input_type = "numeric"
            else:
                input_type = "text"  # Default

        # Encode based on type
        if input_type == "text":
            phi = self.encode_text(input_data)
        elif input_type == "numeric":
            phi = self.encode_numeric(input_data)
        elif input_type == "image":
            phi = self.encode_image(input_data)
        else:
            # Unknown type
            phi = np.zeros(self.grid_size)

        # Update current phi field
        self.phi_field = phi

        # Log context change
        self.context_history.append({
            "type": input_type,
            "complexity": self.complexity,
            "time": len(self.context_history)
        })

        return phi

    def evolve_phi(self, rate: float = 0.1, variability: float = None) -> np.ndarray:
        """
        Evolve the phi field over time based on internal dynamics.

        Args:
            rate: Rate of evolution (0-1)
            variability: Variability of evolution (defaults to self.variability)

        Returns:
            Updated phi field
        """
        if variability is None:
            variability = self.variability

        # Create noise
        noise = variability * (np.random.rand(self.grid_size) - 0.5)

        # Apply evolution
        self.phi_field = (1 - rate) * self.phi_field + rate * (self.phi_field + noise)

        # Ensure valid range
        self.phi_field = np.clip(self.phi_field, 0.0, 1.0)

        # Adjust variability based on complexity
        self.variability = min(0.8, self.variability + 0.01 * (self.complexity - 0.5))

        return self.phi_field

    def apply_action(self, action: Dict[str, Any]) -> bool:
        """
        Apply an agent's action to the environment.

        Args:
            action: Dictionary describing the action

        Returns:
            True if action was successfully applied
        """
        # Check if action is valid
        if not isinstance(action, dict) or "type" not in action:
            return False

        action_type = action.get("type", "")

        # Handle different action types
        if action_type == "modify_field":
            # Agent directly modifies part of the phi field
            if "field" in action and isinstance(action["field"], np.ndarray):
                if len(action["field"]) == self.grid_size:
                    # Apply field modification with a strength parameter
                    strength = min(1.0, max(0.0, action.get("strength", 0.5)))
                    self.phi_field = (1 - strength) * self.phi_field + strength * action["field"]
                    return True

        elif action_type == "query":
            # Agent queries a specific region - no direct effect on phi
            return True

        elif action_type == "focus":
            # Agent focuses attention on a region, slightly enhancing it
            region = action.get("region", [0, self.grid_size])
            intensity = min(0.3, max(0.0, action.get("intensity", 0.1)))

            start, end = max(0, min(region[0], self.grid_size)), min(self.grid_size, max(0, region[1]))

            # Create a Gaussian-like focus around the region
            center = (start + end) / 2
            width = max(1, (end - start) / 2)

            for i in range(self.grid_size):
                # Distance from center (with circular wrapping)
                dist = min(abs(i - center), self.grid_size - abs(i - center))
                # Gaussian falloff
                factor = intensity * np.exp(-dist**2 / (2 * width**2))
                # Enhance field in this region
                self.phi_field[i] = min(1.0, self.phi_field[i] * (1 + factor))

            return True

        # Action not recognized or failed
        return False

    def get_domain_context(self) -> Dict[str, Any]:
        """
        Get additional domain-specific context information.

        Returns:
            Dictionary with domain context
        """
        context_info = {
            "domain": self.domain,
            "complexity": self.complexity,
            "variability": self.variability,
            "history_length": len(self.context_history),
            "recent_changes": []
        }

        # Add recent changes
        if len(self.context_history) > 0:
            context_info["recent_changes"] = self.context_history[-min(3, len(self.context_history)):]

        return context_info

    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of the context system.

        Returns:
            Dictionary with context state
        """
        return {
            "phi_field": self.phi_field.copy(),
            "complexity": self.complexity,
            "variability": self.variability,
            "domain": self.domain,
            "history_length": len(self.context_history),
            "input_history_length": len(self.input_history)
        }

    def set_domain(self, domain: str) -> None:
        """
        Set the current domain for contextual processing.

        Args:
            domain: Domain identifier
        """
        self.domain = domain

        # Domain-specific adjustments could go here
        if domain == "creative":
            self.variability = min(0.8, self.variability + 0.2)
        elif domain == "analytical":
            self.variability = max(0.1, self.variability - 0.2)

from emile_cogito.kainos.module_wide_flow_mapper import auto_map_module_flow
auto_map_module_flow(__name__)  # Maps the entire module!
