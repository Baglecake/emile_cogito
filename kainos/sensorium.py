

"""
Sensorium module for Émile framework.
Implements perceptual grounding and sensorimotor interfaces.
"""
import numpy as np
from typing import Dict, List, Any, Optional, Union

from emile_cogito.kainos.config import CONFIG

class Sensorium:
    """
    Handles perception, sensory inputs, and motor outputs for the Émile framework.

    Creates a bridge between external world data and internal QSE dynamics.
    """

    def __init__(self, cfg=CONFIG):
        """
        Initialize the sensorium.

        Args:
            cfg: Configuration parameters
        """
        self.cfg = cfg
        self.grid_size = cfg.GRID_SIZE

        # Sensor configuration
        self.sensor_channels = cfg.SENSOR_CHANNELS
        self.current_input = None
        self.input_history = []

        # Motor configuration
        self.available_actions = cfg.AVAILABLE_ACTIONS
        self.motor_state = {"last_action": None, "action_history": []}

        # Internal mapping fields
        # Initialize sensor-to-surplus map with random Gaussian patterns
        self.sensor_to_surplus_map = np.zeros((self.sensor_channels, self.grid_size))
        for i in range(self.sensor_channels):
            # Create a Gaussian bump at a random position
            center = np.random.randint(0, self.grid_size)
            width = self.grid_size // 8
            for j in range(self.grid_size):
                # Circular distance to handle wrapping
                dist = min(abs(j - center), self.grid_size - abs(j - center))
                # Gaussian function
                self.sensor_to_surplus_map[i, j] = np.exp(-0.5 * (dist / width)**2)
            # Normalize
            # --- robust row normalisation ---------------------------------
            row_sums = self.sensor_to_surplus_map.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0          # avoid divide-by-zero
            self.sensor_to_surplus_map = self.sensor_to_surplus_map / row_sums
            # --------------------------------------------------------------



        # Initialize surplus-to-motor map with distinctive patterns
        self.surplus_to_motor_map = np.zeros((self.grid_size, len(self.available_actions)))
        for i, action in enumerate(self.available_actions):
            # Different pattern for each action
            if action == "shift_left":
                # Left side sensitivity
                self.surplus_to_motor_map[:self.grid_size//2, i] = np.linspace(1.0, 0.1, self.grid_size//2)
            elif action == "shift_right":
                # Right side sensitivity
                self.surplus_to_motor_map[self.grid_size//2:, i] = np.linspace(0.1, 1.0, self.grid_size - self.grid_size//2)
            elif action == "focus":
                # Center sensitivity
                center = self.grid_size // 2
                width = self.grid_size // 4
                for j in range(self.grid_size):
                    dist = min(abs(j - center), self.grid_size - abs(j - center))
                    self.surplus_to_motor_map[j, i] = np.exp(-0.5 * (dist / width)**2)
            elif action == "diffuse":
                # Sensitivity to high frequency patterns
                for j in range(self.grid_size):
                    self.surplus_to_motor_map[j, i] = 0.5 + 0.5 * np.sin(j * 8 * np.pi / self.grid_size)

            # Normalize
            if np.sum(self.surplus_to_motor_map[:, i]) > 0:
                self.surplus_to_motor_map[:, i] /= np.sum(self.surplus_to_motor_map[:, i])

    def process_sensory_input(self, input_data: np.ndarray) -> np.ndarray:
        """
        Process external sensory input into surplus field mask.

        Args:
            input_data: Input vector of sensor readings

        Returns:
            Surplus field mask for integration with QSE core
        """
        # Store input
        self.current_input = input_data
        self.input_history.append(input_data)
        if len(self.input_history) > 100:
            self.input_history = self.input_history[-100:]

        # Normalize input to [0, 1] range if needed
        input_normalized = np.clip(input_data, 0, 1)

        # Map to surplus influence - currently using a simple linear mapping
        # This will create a 'mask' of surplus influence across the grid
        S_vec = np.zeros(self.grid_size)

        # For each sensor channel, apply its influence to the surplus field
        for i, value in enumerate(input_normalized):
            if i < len(input_normalized):  # Ensure we don't exceed input dimensions
                # Scale the sensor value and map it across the field
                # This uses the sensor_to_surplus_map to determine how each sensor
                # influences different regions of the surplus field
                influence = value * self.sensor_to_surplus_map[i % self.sensor_channels]
                S_vec += influence

        # Scale to mild surplus range (0-0.4) to avoid overwhelming the system
        S_vec = np.interp(S_vec, [0, np.max(S_vec) if np.max(S_vec) > 0 else 1], [0, 0.4])

        return S_vec

    def select_action(self, surplus: np.ndarray, stability: float) -> Dict[str, Any]:
        """
        Select motor action based on current QSE state.

        Args:
            surplus: Current surplus field
            stability: Current system stability

        Returns:
            Selected action information
        """
        # Simple mapping from surplus field to action space
        action_scores = np.zeros(len(self.available_actions))

        # Calculate action scores based on surplus field
        for i, action in enumerate(self.available_actions):
            # Use the mapping to calculate how much each region of surplus
            # influences each potential action
            action_scores[i] = np.sum(surplus * self.surplus_to_motor_map[:, i])

        # Add exploration factor based on stability
        # Scale exploration by mean action score to keep it proportional
        mean_score = np.mean(action_scores) if np.mean(action_scores) > 0 else 0.1
        exploration_factor = (1.0 - stability) * mean_score * 0.5
        # Use softmax-like temperature instead of adding raw noise
        if np.sum(action_scores) > 0:
            # Convert to probability distribution
            temperature = 1.0 + exploration_factor  # Higher when stability is low
            exp_scores = np.exp(action_scores / temperature)
            action_probs = exp_scores / np.sum(exp_scores)
            # Sample from this distribution
            selected_idx = np.random.choice(len(action_scores), p=action_probs)
        else:
            # Fallback to random if all scores are zero
            selected_idx = np.random.randint(len(action_scores))

        # Select action with highest score

        selected_action = self.available_actions[selected_idx]

        # Record action
        action_info = {
            "action": selected_action,
            "action_idx": selected_idx,
            "confidence": float(action_scores[selected_idx] / np.sum(action_scores) if np.sum(action_scores) > 0 else 0)
        }
        self.motor_state["last_action"] = action_info
        self.motor_state["action_history"].append(action_info)

        return action_info

    def execute_action(self, action_info: Dict[str, Any],
                      surplus: np.ndarray) -> np.ndarray:
        """
        Execute motor action and return modified surplus field.

        Args:
            action_info: Action to execute
            surplus: Current surplus field

        Returns:
            Modified surplus field after action execution
        """
        action = action_info["action"]
        surplus_modified = surplus.copy()

        # Implement the action's effect on the surplus field
        if action == "shift_left":
            surplus_modified = np.roll(surplus_modified, -5)
        elif action == "shift_right":
            surplus_modified = np.roll(surplus_modified, 5)
        elif action == "focus":
            # Enhance the center region
            center = len(surplus) // 2
            width = len(surplus) // 10

            # Create gaussian-like focus window
            window = np.exp(-0.5 * ((np.arange(len(surplus)) - center) / width) ** 2)

            # Apply focus by increasing values in focus region
            surplus_modified += 0.1 * window * surplus_modified

        elif action == "diffuse":
            # Smooth the surplus field
            kernel_size = 5
            kernel = np.ones(kernel_size) / kernel_size
            surplus_modified = np.convolve(surplus_modified, kernel, mode='same')

        # Ensure valid surplus range
        surplus_modified = np.clip(surplus_modified, 0, 1.0)

        return surplus_modified

    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of the sensorium.

        Returns:
            Dictionary with sensorium state
        """
        return {
            "current_input": self.current_input.copy() if self.current_input is not None else None,
            "input_history_length": len(self.input_history),
            "last_action": self.motor_state["last_action"],
            "action_history_length": len(self.motor_state["action_history"]),
            "available_actions": self.available_actions
        }

from emile_cogito.kainos.module_wide_flow_mapper import auto_map_module_flow
auto_map_module_flow(__name__)  # Maps the entire module!
