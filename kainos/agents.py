
"""
Agent system module for Émile framework.
Implements recursive agent spawning, agent lineage, and contextual dynamics.
"""
import numpy as np
import uuid
import time
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field

from emile_cogito.kainos.config import CONFIG

@dataclass
class Agent:
    """
    Represents a cognitive agent within the multi-agent system.

    Each agent has its own region of influence, thresholds, and memory.
    Implements aspects of Theorem 6 (Recursive Irreducibility).
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    parent_id: Optional[str] = None
    birth_step: int = 0

    # Agent parameters
    theta_psi: float = field(default_factory=lambda: CONFIG.THETA_PSI)
    theta_phi: float = field(default_factory=lambda: CONFIG.THETA_PHI)

    # Spatial properties
    mask: Optional[np.ndarray] = None  # Region of influence
    center: int = 0  # Center position
    radius: int = 0  # Radius of influence

    # Memory and tracking
    elder_memory: Optional[np.ndarray] = None  # Genealogical memory trace
    personal_memory: List[Dict] = field(default_factory=list)  # Agent's experiences
    rupture_count: int = 0
    activation_history: List[float] = field(default_factory=list)
    child_ids: List[str] = field(default_factory=list)

    def mutate(self, mutation_strength: float = 0.05, min_gap: float = 0.05) -> None:
        """
        Mutate agent parameters by random adjustments.

        Supports diversity in the agent population through genetic-like
        variation in parameters.

        Args:
            mutation_strength: Scale of random adjustments
            min_gap: Minimum gap to enforce between thresholds
        """
        # Mutate thresholds with Gaussian noise
        self.theta_psi += mutation_strength * np.random.randn()
        self.theta_phi += mutation_strength * np.random.randn()

        # Ensure thresholds remain in valid range
        self.theta_psi = np.clip(self.theta_psi, 0.2, 0.8)
        self.theta_phi = np.clip(self.theta_phi, 0.3, 0.9)

        # Ensure theta_phi > theta_psi + min_gap to maintain meaningful distinction
        attempts = 0
        max_attempts = 5

        while (self.theta_phi - self.theta_psi < min_gap) and (attempts < max_attempts):
            # Calculate midpoint
            midpoint = (self.theta_psi + self.theta_phi) / 2

            # Adjust thresholds to enforce min_gap
            self.theta_psi = midpoint - min_gap / 2
            self.theta_phi = midpoint + min_gap / 2

            # Ensure they're still in valid range
            self.theta_psi = np.clip(self.theta_psi, 0.2, 0.8 - min_gap)
            self.theta_phi = np.clip(self.theta_phi, 0.3 + min_gap, 0.9)

            attempts += 1

        # Final check - if still invalid after max attempts, force the gap
        if self.theta_phi - self.theta_psi < min_gap:
            self.theta_phi = self.theta_psi + min_gap
            # Ensure phi is in valid range
            if self.theta_phi > 0.9:
                self.theta_phi = 0.9
                self.theta_psi = self.theta_phi - min_gap

    def update_memory(self, state: Dict[str, Any], step: int) -> None:
        """
        Update agent's personal memory with current state.

        Args:
            state: Current system state
            step: Current time step
        """
        # Create memory entry with key information
        entry = {
            "step": step,
            "regime": state.get("regime", "unknown"),
            "surplus_mean": state.get("surplus_mean", 0.0),
            "activation": self.get_activation(state.get("sigma_field"))
        }

        # Add to personal memory
        self.personal_memory.append(entry)

        # Keep memory bounded
        max_memory = 50
        if len(self.personal_memory) > max_memory:
            self.personal_memory = self.personal_memory[-max_memory:]

    def get_activation(self, sigma_field: Optional[np.ndarray]) -> float:
        """
        Calculate agent's activation level based on symbolic curvature.

        Args:
            sigma_field: Current symbolic curvature field

        Returns:
            Activation level (0-1)
        """
        if sigma_field is None or self.mask is None:
            return 0.0

        # Calculate mean absolute curvature in agent's region
        local_sigma = sigma_field * self.mask
        activation = float(np.mean(np.abs(local_sigma)))

        # Update activation history
        self.activation_history.append(activation)
        if len(self.activation_history) > 100:
            self.activation_history = self.activation_history[-100:]

        return activation

class AgentSystem:
    """
    Multi-agent system implementing recursive spawning and context dynamics.

    Implements Theorem 6 (Recursive Irreducibility) through agent lineage
    and context shifts based on distinction levels.
    """

    def __init__(self, cfg=CONFIG):
        """
        Initialize the agent system.

        Args:
            cfg: Configuration parameters
        """
        self.cfg = cfg
        self.grid_size = cfg.GRID_SIZE

        # Agent tracking
        self.agents: List[Agent] = []
        self.active_agents = 0
        self.global_context_id = 0
        self.step_count = 0

        # Shared workspace (common field influencing all agents)
        self.shared_workspace = np.zeros(cfg.GRID_SIZE)

        # Create initial agent
        self._create_initial_agent()

        # History tracking
        self.rupture_history = []
        self.context_shifts = []

    def update_agent_temporal_context(self, tau_prime: float, subjective_time: float):
        """Update agent temporal context for memory formation"""

        for agent in self.agents:
            if hasattr(agent, 'temporal_memory'):
                agent.temporal_memory.append({
                    'subjective_time': subjective_time,
                    'tau_prime': tau_prime,
                    'agent_state': agent.id,
                    'empirical_time': time.time()
                })

                # Keep bounded temporal memory
                if len(agent.temporal_memory) > 50:
                    agent.temporal_memory = agent.temporal_memory[-50:]
            else:
                agent.temporal_memory = []

    def _create_initial_agent(self) -> None:
        """Create the first agent covering the entire field."""
        # Create mask covering whole grid
        mask = np.ones(self.grid_size)

        # Create initial agent
        agent = Agent(
            id="agent-0",
            birth_step=0,
            mask=mask,
            center=self.grid_size // 2,
            radius=self.grid_size // 2,
            elder_memory=np.zeros(self.grid_size)
        )

        self.agents.append(agent)
        self.active_agents = 1

    def _spawn_new_agent(self, parent_idx: int, rupture_loc: int) -> Optional[Agent]:
        """
        Spawn a new agent from a rupture event.

        Implements part of Theorem 6, where ruptures lead to new
        recursive elements that contribute to the global field.

        Args:
            parent_idx: Index of parent agent
            rupture_loc: Location where rupture occurred

        Returns:
            Newly created agent or None if spawn failed
        """
        # Check if we can create more agents
        if self.active_agents >= self.cfg.MAX_AGENTS:
            return None

        # Get parent agent
        if parent_idx >= len(self.agents):
            return None

        parent = self.agents[parent_idx]

        # Choose a radius (1/8 to 1/3 of grid)
        radius = int(self.grid_size * (0.125 + 0.2 * np.random.random()))

        # Create circular mask using vectorized operations instead of loops
        # Precompute a distance array once and cache it for future reuse
        if not hasattr(self, "_distance_matrix"):
            self._distance_matrix = np.zeros((self.grid_size, self.grid_size))
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    # Circular distance calculation with wrapping
                    self._distance_matrix[i, j] = min(
                        abs(i - j),
                        self.grid_size - abs(i - j)
                    )

        # Use the precomputed distance matrix to create mask
        new_mask = np.zeros(self.grid_size)
        new_mask[self._distance_matrix[rupture_loc, :] <= radius] = 1.0

        # Create new agent with mutation
        new_agent = Agent(
            parent_id=parent.id,
            birth_step=self.step_count,
            theta_psi=parent.theta_psi,
            theta_phi=parent.theta_phi,
            mask=new_mask,
            center=rupture_loc,
            radius=radius,
            # Copy parent's memory as genealogical memory
            elder_memory=parent.elder_memory.copy() if parent.elder_memory is not None else None
        )

        # Mutate the new agent's parameters
        new_agent.mutate()

        # Update parent's child list and rupture count
        parent.child_ids.append(new_agent.id)
        parent.rupture_count += 1

        # Add to agent list
        self.agents.append(new_agent)
        self.active_agents += 1

        return new_agent

    def handle_ruptures(self, symbolic_fields: Dict[str, np.ndarray]) -> List[Dict[str, Any]]:
        """
        Check for rupture conditions and spawn new agents when needed.

        Implements Theorem 4 (rupture conditions) at the agent level.

        Args:
            symbolic_fields: Dictionary with symbolic field data

        Returns:
            List of rupture event details
        """
        rupture_events = []
        sigma = symbolic_fields["sigma"]

        # Process each agent for potential ruptures
        for i, agent in enumerate(self.agents):
            # Skip if no mask
            if agent.mask is None:
                continue

            # Apply mask to get agent's local region
            local_sigma = sigma * agent.mask
            max_sigma_abs = np.max(np.abs(local_sigma))

            # Check if rupture threshold is exceeded
            if max_sigma_abs > self.cfg.S_THETA_RUPTURE:
                # Find rupture location (max absolute Sigma)
                rupture_loc = np.argmax(np.abs(local_sigma))

                # Spawn new agent from this rupture
                new_agent = self._spawn_new_agent(i, rupture_loc)

                if new_agent:
                    # Record rupture event
                    rupture_events.append({
                        "parent_agent_id": agent.id,
                        "parent_agent_idx": i,
                        "new_agent_id": new_agent.id,
                        "new_agent_idx": len(self.agents) - 1,
                        "location": rupture_loc,
                        "sigma_value": float(local_sigma[rupture_loc]),
                        "step": self.step_count
                    })

                    # Store in history
                    self.rupture_history.append(rupture_events[-1])

        return rupture_events

    def process_context_shift(self, distinction_level: float) -> bool:
        """
        Check for and process context shifts based on distinction level.

        Implements Theorem 6's recontextualization aspect, where high
        distinction levels trigger a global context shift.

        Args:
            distinction_level: Current level of distinction in the system

        Returns:
            True if context shift occurred, False otherwise
        """
        # Check if distinction level exceeds threshold
        if distinction_level > self.cfg.CONTEXT_SHIFT_THRESHOLD:
            # Increment global context
            self.global_context_id += 1

            # Record context shift
            self.context_shifts.append({
                "context_id": self.global_context_id,
                "step": self.step_count,
                "distinction_level": distinction_level
            })

            return True

        return False

    def calculate_combined_fields(self, symbolic_fields: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Calculate combined symbolic fields from all agents.

        Implements the collective emergence of meaning from multiple
        recursive elements, as described in Theorem 6.

        Args:
            symbolic_fields: Raw symbolic fields calculated from surplus

        Returns:
            Modified symbolic fields taking agent system into account
        """
        grid_size = self.grid_size

        # Get original fields
        surplus = symbolic_fields["surplus"]
        sigma = symbolic_fields["sigma"]

        # Initialize combined fields
        combined_psi = np.zeros(grid_size)
        combined_phi = np.zeros(grid_size)

        # Process each agent's contribution
        for agent in self.agents:
            if agent.mask is None:
                continue

            # Apply agent-specific thresholds
            agent_psi = 1.0 / (1.0 + np.exp(-self.cfg.K_PSI * (surplus - agent.theta_psi)))
            agent_phi = np.maximum(0.0, self.cfg.K_PHI * (surplus - agent.theta_phi))

            # Add elder memory influence (genealogical memory)
            if agent.elder_memory is not None:
                elder_influence = 0.05 * agent.elder_memory
                agent_phi += elder_influence

            # Combine fields where agent is active
            mask = agent.mask
            combined_psi += agent_psi * mask
            combined_phi += agent_phi * mask

        # Add shared workspace influence
        shared_influence = self.cfg.WORKSPACE_STRENGTH * self.shared_workspace
        combined_phi += shared_influence

        # Normalize at overlapping regions
        overlap_count = np.zeros(grid_size)
        for agent in self.agents:
            if agent.mask is not None:
                overlap_count += agent.mask

        # Avoid division by zero
        overlap_count = np.maximum(overlap_count, 1.0)

        combined_psi /= overlap_count
        combined_phi /= overlap_count

        # Calculate new sigma
        combined_sigma = combined_psi - combined_phi

        # Update shared workspace (acting as a collective memory/field)
        self.shared_workspace = 0.9 * self.shared_workspace + 0.1 * combined_sigma

        return {
            "surplus": surplus,
            "psi": combined_psi,
            "phi": combined_phi,
            "sigma": combined_sigma
        }

    def update_agent_memories(self, symbolic_fields: Dict[str, np.ndarray],
                             current_state: Dict[str, Any]) -> None:
        """
        Update agent memories with current experience.

        Args:
            symbolic_fields: Current symbolic fields
            current_state: Current system state
        """
        # Current step
        step = self.step_count

        # Update each agent's memory
        for agent in self.agents:
            # Update personal memory occasionally (not every step)
            if step % 5 == 0:
                agent.update_memory(current_state, step)

            # Update elder memory (slower process)
            if step % 20 == 0 and agent.mask is not None and symbolic_fields["sigma"] is not None:
                # Create a trace of current sigma field in agent's region
                local_sigma = symbolic_fields["sigma"] * agent.mask

                # If elder memory doesn't exist, initialize it
                if agent.elder_memory is None:
                    agent.elder_memory = np.zeros_like(local_sigma)

                # Slowly integrate current patterns into elder memory
                memory_update_rate = 0.05
                agent.elder_memory = (1.0 - memory_update_rate) * agent.elder_memory + memory_update_rate * local_sigma

    def step(self, symbolic_fields: Dict[str, np.ndarray],
            current_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single step of the agent system.

        Args:
            symbolic_fields: Current symbolic fields
            current_state: Current system state

        Returns:
            Dictionary with agent system state and events
        """
        self.step_count += 1

        # 1. Calculate combined fields from all agents
        combined_fields = self.calculate_combined_fields(symbolic_fields)

        # 2. Process ruptures based on combined fields
        rupture_events = self.handle_ruptures(combined_fields)

        # 3. Update agent memories
        self.update_agent_memories(combined_fields, current_state)

        # 4. Check for context shift
        distinction_level = float(np.mean(np.abs(combined_fields["sigma"])))
        context_shifted = self.process_context_shift(distinction_level)

        # Return events and state
        return {
            "rupture_events": rupture_events,
            "context_shifted": context_shifted,
            "context_id": self.global_context_id,
            "active_agents": self.active_agents,
            "combined_fields": combined_fields,
            "distinction_level": distinction_level
        }

    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of the agent system.

        Returns:
            Dictionary with agent system state
        """
        return {
            "active_agents": self.active_agents,
            "agent_count": len(self.agents),
            "global_context_id": self.global_context_id,
            "step_count": self.step_count,
            "shared_workspace": self.shared_workspace.copy(),
            "agent_ids": [agent.id for agent in self.agents],
            "rupture_count": len(self.rupture_history)
        }

    def get_agent_lineage(self) -> Dict[str, List[str]]:
        """
        Get the parent-child relationships between agents.

        Returns:
            Dictionary mapping parent IDs to list of child IDs
        """
        lineage = {}
        for agent in self.agents:
            if agent.parent_id is not None:
                if agent.parent_id not in lineage:
                    lineage[agent.parent_id] = []
                lineage[agent.parent_id].append(agent.id)
        return lineage

    def get_agent_details(self, agent_id: Optional[str] = None) -> Any:
        """
        Get detailed information about agents.

        Args:
            agent_id: Optional specific agent ID to retrieve

        Returns:
            Dictionary with agent details or list of all agents
        """
        if agent_id is not None:
            # Find specific agent
            for agent in self.agents:
                if agent.id == agent_id:
                    return {
                        "id": agent.id,
                        "parent_id": agent.parent_id,
                        "birth_step": agent.birth_step,
                        "theta_psi": agent.theta_psi,
                        "theta_phi": agent.theta_phi,
                        "center": agent.center,
                        "radius": agent.radius,
                        "rupture_count": agent.rupture_count,
                        "child_count": len(agent.child_ids),
                        "child_ids": agent.child_ids,
                        "memory_size": len(agent.personal_memory),
                        "recent_activation": agent.activation_history[-10:] if agent.activation_history else []
                    }
            return None
        else:
            # Return summary of all agents
            return [
                {
                    "id": agent.id,
                    "parent_id": agent.parent_id,
                    "birth_step": agent.birth_step,
                    "children": len(agent.child_ids),
                    "thresholds": (agent.theta_psi, agent.theta_phi),
                    "activation": np.mean(agent.activation_history[-10:]) if agent.activation_history else 0.0
                }
                for agent in self.agents
            ]

from emile_cogito.kainos.module_wide_flow_mapper import auto_map_module_flow
auto_map_module_flow(__name__)  # Maps the entire module!
