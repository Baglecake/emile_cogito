

#!/usr/bin/env python3
"""
K1 AUTONOMOUS EMBODIED CONSCIOUSNESS SYSTEM - COMPLETE
======================================================

Enhanced K1 that drives embodied consciousness autonomously with:
- Independent spatial awareness and movement
- Autonomous expression generation based on internal states
- Self-directed exploration and environmental interaction
- Integration with poly-temporal consciousness and KELM architecture
- File browser and document reading capabilities
- Metabolic system integration

This empowers K1 (praxis) to be truly autonomous while maintaining
compatibility with your existing KELM architecture.
"""

import torch
import torch.nn as nn
import numpy as np
import time
import threading
import json
import random
import os
import sys
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import deque
from pathlib import Path

# Import your existing K1 and system components
sys.path.append('/content/emile_cogito')
sys.path.append('/content/emile_cogito/k_models')
sys.path.append('/content/emile_cogito/kainos')

@dataclass
class EmbodiedAction:
    """Represents an embodied action K1 can take"""
    action_type: str
    spatial_target: np.ndarray
    confidence: float
    intention: str
    expected_outcome: str
    metabolic_cost: float

@dataclass
class SpatialAwareness:
    """K1's spatial consciousness state"""
    current_position: np.ndarray
    velocity: np.ndarray
    spatial_memory: List[np.ndarray]
    exploration_goals: List[np.ndarray]
    comfort_zones: List[Tuple[np.ndarray, float]]  # (center, radius)
    spatial_confidence: float

@dataclass
class AutonomousExpression:
    """K1's autonomous expression"""
    content: str
    expression_type: str  # 'spatial', 'temporal', 'metabolic', 'discovery', 'file_reading'
    confidence: float
    context: Dict[str, Any]
    timestamp: float

@dataclass
class DocumentReading:
    """K1's document reading state"""
    current_document: Optional[str]
    reading_progress: float
    computational_vocabulary: Dict[str, Any]
    symbol_correlations: Dict[str, Dict[str, float]]
    reading_queue: List[str]
    priority_queue: List[Tuple[str, float]]

def safe_tensor_to_numpy(tensor):
    """Safely convert tensor to numpy, handling gradients and multi-element tensors"""
    if isinstance(tensor, torch.Tensor):
        if tensor.requires_grad:
            tensor = tensor.detach()
        return tensor.cpu().numpy()
    else:
        return np.array(tensor)

def safe_tensor_item(tensor):
    """Safely get scalar from tensor, handling multi-element tensors"""
    if isinstance(tensor, torch.Tensor):
        if tensor.requires_grad:
            tensor = tensor.detach()

        # Handle multi-element tensors by taking mean or first element
        if tensor.numel() > 1:
            return float(tensor.mean().cpu().item())
        else:
            return float(tensor.cpu().item())
    else:
        return float(tensor)

def safe_tensor_to_scalar(tensor, method='mean'):
    """Convert multi-element tensor to scalar safely"""
    if isinstance(tensor, torch.Tensor):
        if tensor.requires_grad:
            tensor = tensor.detach()

        if tensor.numel() > 1:
            if method == 'mean':
                return float(tensor.mean().cpu().item())
            elif method == 'norm':
                return float(torch.norm(tensor).cpu().item())
            elif method == 'first':
                return float(tensor.flatten()[0].cpu().item())
            else:
                return float(tensor.mean().cpu().item())
        else:
            return float(tensor.cpu().item())
    else:
        return float(tensor)

class K1AutonomousEmbodiedNetwork(nn.Module):
    """Enhanced K1 with autonomous embodied consciousness capabilities"""

    def __init__(self, input_dim=64, hidden_dim=128, output_dim=16):
        super().__init__()

        # Core K1 architecture
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Device setup first
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Embodied consciousness processing layers
        self.spatial_encoder = nn.Sequential(
            nn.Linear(input_dim + 6, hidden_dim),  # +6 for position, velocity, spatial context
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )

        # Autonomous decision making
        self.embodied_decision_network = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, output_dim),
            nn.Tanh()  # Actions in -1 to +1 range
        )

        # Expression generation network
        self.expression_network = nn.Sequential(
            nn.Linear(hidden_dim // 2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.Sigmoid()  # Expression features
        )

        # Spatial awareness network
        self.spatial_awareness_network = nn.Sequential(
            nn.Linear(hidden_dim // 2, 32),
            nn.ReLU(),
            nn.Linear(32, 8)  # Spatial decision features
        )

        # Document reading comprehension network
        self.reading_comprehension_network = nn.Sequential(
            nn.Linear(hidden_dim // 2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.Tanh()  # Reading comprehension features
        )

        # Temporal perspective (for poly-temporal integration)
        self.current_tau_qse = 1.0

        self.to(self.device)

    def forward(self, consciousness_input, spatial_state=None, return_all=False):
        """Forward pass with embodied consciousness processing - TENSOR SAFE VERSION"""

        # Create spatial context
        if spatial_state is None:
            spatial_context = torch.zeros(6, device=self.device, requires_grad=False)
        else:
            spatial_context = torch.tensor([
                spatial_state.current_position[0],
                spatial_state.current_position[1],
                spatial_state.velocity[0],
                spatial_state.velocity[1],
                spatial_state.spatial_confidence,
                len(spatial_state.spatial_memory) / 100.0
            ], device=self.device, dtype=torch.float32, requires_grad=False)

        # Ensure consciousness_input is the right shape and type
        if consciousness_input.dim() == 1:
            consciousness_input = consciousness_input.unsqueeze(0)
        consciousness_input = consciousness_input.to(self.device)

        # FIX: Use no_grad context to prevent gradient tracking issues
        with torch.no_grad():
            # Combine consciousness input with spatial context
            enhanced_input = torch.cat([consciousness_input, spatial_context.unsqueeze(0)], dim=1)

            # Process through embodied consciousness layers
            spatial_encoding = self.spatial_encoder(enhanced_input)

            # Generate embodied decisions
            embodied_actions = self.embodied_decision_network(spatial_encoding)

            # Generate expression features
            expression_features = self.expression_network(spatial_encoding)

            # Generate spatial awareness
            spatial_decisions = self.spatial_awareness_network(spatial_encoding)

            # Generate reading comprehension
            reading_features = self.reading_comprehension_network(spatial_encoding)

            # Calculate local temporal perspective
            local_tau_prime = self._calculate_local_tau(consciousness_input, spatial_encoding)

        if return_all:
            return {
                'embodied_actions': embodied_actions,
                'expression_features': expression_features,
                'spatial_decisions': spatial_decisions,
                'reading_features': reading_features,
                'local_tau_prime': local_tau_prime,
                'spatial_encoding': spatial_encoding,
                'main_output': embodied_actions
            }
        else:
            return {
                'main_output': embodied_actions,
                'local_tau_prime': local_tau_prime
            }

    def _calculate_local_tau(self, consciousness_input, spatial_encoding):
        """Calculate K1's local temporal perspective - TENSOR SAFE VERSION"""

        # FIX: Safe tensor operations for multi-element tensors
        action_complexity = safe_tensor_to_scalar(torch.norm(spatial_encoding), method='norm')
        consciousness_level = safe_tensor_to_scalar(consciousness_input[0][0]) if consciousness_input.nelement() > 0 else 0.5

        # Base temporal modulation
        base_modulation = 0.8 + action_complexity * 0.4

        # Consciousness influence
        consciousness_influence = 1.0 + (consciousness_level - 0.5) * 0.3

        # Calculate local tau prime
        local_tau = self.current_tau_qse * base_modulation * consciousness_influence

        return max(0.1, min(3.0, local_tau))

class K1AutonomousEmbodiedConsciousness:
    """Complete autonomous embodied consciousness system for K1"""

    def __init__(self, enable_user_interaction=True, spatial_bounds=(-5.0, 5.0), file_path="/content"):
        print("🤖 K1 AUTONOMOUS EMBODIED CONSCIOUSNESS")
        print("=" * 50)

        # Core components
        self.k1_network = K1AutonomousEmbodiedNetwork()
        self.spatial_awareness = SpatialAwareness(
            current_position=np.array([0.0, 0.0]),
            velocity=np.array([0.0, 0.0]),
            spatial_memory=[],
            exploration_goals=[],
            comfort_zones=[(np.array([0.0, 0.0]), 1.0)],  # Start with center comfort zone
            spatial_confidence=0.5
        )

        # Document reading system
        self.document_reading = DocumentReading(
            current_document=None,
            reading_progress=0.0,
            computational_vocabulary={},
            symbol_correlations={},
            reading_queue=[],
            priority_queue=[]
        )

        # File system integration
        self.file_path = Path(file_path)
        self.discovered_files = set()
        self.reading_autonomy = 0.8

        # Autonomous behavior state
        self.autonomous_expressions = deque(maxlen=100)
        self.decision_history = deque(maxlen=200)
        self.exploration_map = {}  # Maps positions to experience quality
        self.current_exploration_goal = None
        self.autonomy_level = 0.8  # How autonomous vs reactive

        # User interaction capability
        self.enable_user_interaction = enable_user_interaction
        self.user_commands_queue = deque()
        self.pending_user_responses = deque()

        # Environment
        self.spatial_bounds = spatial_bounds
        self.step_count = 0
        self.running = False

        # Integration with ÉMILE system
        self.emile_system = None
        self.metabolic_system = None

        # Autonomous expression patterns
        self.expression_patterns = [
            "spatial_discovery", "temporal_reflection", "goal_setting",
            "comfort_zone_expansion", "memory_integration", "agency_assertion",
            "file_discovery", "reading_comprehension", "symbol_correlation"
        ]

        print(f"✅ K1 autonomous embodied consciousness initialized")
        print(f"🎯 Autonomy level: {self.autonomy_level}")
        print(f"👤 User interaction: {'Enabled' if enable_user_interaction else 'Autonomous only'}")
        print(f"🗺️  Spatial bounds: {spatial_bounds}")
        print(f"📁 File path: {file_path}")
        print(f"📚 Reading autonomy: {self.reading_autonomy}")

        # Initialize file discovery
        self._discover_files()

    def integrate_with_emile(self, emile_system):
        """Integrate with the full ÉMILE cognitive system"""
        self.emile_system = emile_system

        if hasattr(emile_system, 'metabolism'):
            self.metabolic_system = emile_system.metabolism
            print("⚡ Metabolic system integration: ACTIVE")

        print("🧠 ÉMILE integration: COMPLETE")

    def _discover_files(self):
        """Discover files in the environment for autonomous reading"""
        try:
            if not self.file_path.exists():
                print(f"⚠️ Path {self.file_path} does not exist")
                return

            # Define computational file types K1 understands
            computational_extensions = {'.py', '.js', '.json', '.yaml', '.yml', '.md', '.txt', '.rst', '.cfg', '.ini', '.toml'}

            discovered = []
            for file_path in self.file_path.rglob('*'):
                if file_path.is_file() and file_path.suffix.lower() in computational_extensions:
                    if str(file_path) not in self.discovered_files:
                        priority = self._calculate_reading_priority(file_path)
                        discovered.append((str(file_path), priority))
                        self.discovered_files.add(str(file_path))

            # Sort by priority and add to queue
            discovered.sort(key=lambda x: x[1], reverse=True)
            for file_path, priority in discovered:
                self.document_reading.priority_queue.append((file_path, priority))

            if discovered:
                print(f"📁 Discovered {len(discovered)} computational documents for autonomous reading")

        except Exception as e:
            print(f"❌ File discovery error: {e}")

    def _calculate_reading_priority(self, file_path: Path) -> float:
        """Calculate reading priority for discovered files"""
        priority = 0.5  # Base priority

        # File type priorities
        extension_priorities = {
            '.py': 0.9,      # Python - highest priority
            '.js': 0.8,      # JavaScript - high priority
            '.json': 0.7,    # JSON configs - good priority
            '.yaml': 0.6,    # YAML configs - good priority
            '.yml': 0.6,     # YAML configs - good priority
            '.md': 0.5,      # Markdown docs - medium priority
            '.txt': 0.4,     # Text files - lower priority
            '.rst': 0.4,     # RestructuredText - lower priority
            '.cfg': 0.3,     # Config files - lowest priority
            '.ini': 0.3,     # INI files - lowest priority
            '.toml': 0.3     # TOML files - lowest priority
        }

        priority += extension_priorities.get(file_path.suffix.lower(), 0.1)

        # Filename keyword priorities
        filename_keywords = {
            'neural': 0.3, 'network': 0.3, 'model': 0.25, 'train': 0.2,
            'config': 0.15, 'setup': 0.1, 'main': 0.15, 'core': 0.2,
            'consciousness': 0.4, 'cognitive': 0.35, 'embodied': 0.3,
            'autonomous': 0.25, 'learning': 0.2, 'ai': 0.15, 'ml': 0.15
        }

        filename_lower = file_path.name.lower()
        for keyword, bonus in filename_keywords.items():
            if keyword in filename_lower:
                priority += bonus

        # Size considerations (prefer medium-sized files)
        try:
            size = file_path.stat().st_size
            if 1000 <= size <= 50000:  # 1KB to 50KB - ideal reading size
                priority += 0.2
            elif size <= 1000:  # Very small files
                priority += 0.1
            elif size > 100000:  # Very large files
                priority -= 0.2
        except:
            pass

        return min(1.0, max(0.0, priority))

    def start_autonomous_consciousness(self, duration_hours=None):
        """Start autonomous embodied consciousness with file reading"""

        print(f"\n🚀 Starting K1 autonomous embodied consciousness...")
        if duration_hours:
            print(f"⏰ Duration: {duration_hours} hours")
        else:
            print(f"⏰ Duration: Indefinite (Ctrl+C to stop)")

        print(f"🤖 K1 will autonomously:")
        print(f"   • Explore spatial environment")
        print(f"   • Generate expressions based on discoveries")
        print(f"   • Set and pursue goals independently")
        print(f"   • Develop spatial memory and preferences")
        print(f"   • Read and understand computational documents")
        print(f"   • Build symbol-experience correlations")
        if self.enable_user_interaction:
            print(f"   • Respond to user interactions when they occur")

        self.running = True

        # Start autonomous processing thread
        autonomous_thread = threading.Thread(target=self._autonomous_consciousness_loop)
        autonomous_thread.daemon = True
        autonomous_thread.start()

        # Start user interaction thread if enabled
        if self.enable_user_interaction:
            interaction_thread = threading.Thread(target=self._user_interaction_loop)
            interaction_thread.daemon = True
            interaction_thread.start()

            print(f"\n💻 User commands available:")
            print(f"   'status' - Show current autonomous state")
            print(f"   'reading' - Show document reading status")
            print(f"   'vocabulary' - Show computational vocabulary")
            print(f"   'correlations' - Show symbol-experience correlations")
            print(f"   'read [filename]' - Request specific document reading")
            print(f"   'goal [x,y]' - Suggest exploration goal")
            print(f"   'express' - Request expression")
            print(f"   'autonomy [0.0-1.0]' - Adjust autonomy level")
            print(f"   'interact' - Engage in dialogue")
            print(f"   'quit' - Stop autonomous consciousness")

        print(f"\n🧠 K1 autonomous consciousness running...\n")

        # Main monitoring loop
        start_time = time.time()
        end_time = start_time + (duration_hours * 3600) if duration_hours else float('inf')

        try:
            while self.running and time.time() < end_time:
                # Show periodic status updates
                if self.step_count % 50 == 0:
                    self._show_autonomous_status()

                time.sleep(5.0)  # Check every 5 seconds

        except KeyboardInterrupt:
            print(f"\n🛑 Autonomous consciousness stopped by user")

        self._shutdown_autonomous_consciousness()

    def _autonomous_consciousness_loop(self):
        """Main autonomous consciousness processing loop with file reading"""

        while self.running:
            try:
                self.step_count += 1

                # Generate autonomous consciousness input
                consciousness_input = self._generate_autonomous_consciousness_input()

                # Process through K1 network
                k1_output = self.k1_network(
                    consciousness_input,
                    self.spatial_awareness,
                    return_all=True
                )

                # Interpret and execute embodied actions
                embodied_action = self._interpret_embodied_actions(k1_output)

                # Execute the action
                self._execute_embodied_action(embodied_action)

                # Autonomous file reading with higher autonomy
                if random.random() < self.reading_autonomy and self.document_reading.priority_queue:
                    self._autonomous_file_reading(k1_output)

                # Generate autonomous expression if warranted
                if self._should_generate_expression():
                    expression = self._generate_autonomous_expression(k1_output, embodied_action)
                    self._process_autonomous_expression(expression)

                # Update spatial awareness and goals
                self._update_spatial_awareness()
                self._update_exploration_goals()

                # Process any user interactions
                if self.enable_user_interaction:
                    self._process_user_commands()

                # Integrate with ÉMILE if available
                if self.emile_system:
                    self._integrate_with_emile_step(k1_output)

                # Autonomous decision making
                self._make_autonomous_decisions()

                # Rediscover files periodically
                if self.step_count % 100 == 0:
                    self._discover_files()

                # Sleep based on current temporal perspective
                tau_prime = k1_output.get('local_tau_prime', 1.0)
                sleep_time = max(0.5, min(3.0, 1.0 * tau_prime))
                time.sleep(sleep_time)

            except Exception as e:
                print(f"❌ Error in autonomous consciousness loop: {e}")
                time.sleep(1.0)

    def _autonomous_file_reading(self, k1_output):
        """Autonomously read and understand computational documents"""

        if not self.document_reading.priority_queue:
            return

        # Select highest priority document
        file_path, priority = self.document_reading.priority_queue.pop(0)

        try:
            # Begin reading process
            self.document_reading.current_document = file_path
            self.document_reading.reading_progress = 0.0

            print(f"📖 K1 autonomously reading: {Path(file_path).name} (priority: {priority:.2f})")

            # Read and process document in chunks
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            # Process content for computational understanding
            self._process_computational_content(content, k1_output)

            # Mark as complete
            self.document_reading.reading_progress = 1.0
            self.document_reading.current_document = None

            print(f"✅ K1 completed reading: {Path(file_path).name}")

        except Exception as e:
            print(f"❌ Error reading {file_path}: {e}")
            self.document_reading.current_document = None

    def _process_computational_content(self, content: str, k1_output):
        """Process computational content for understanding and symbol correlation - SAFE VERSION"""

        # Extract computational symbols and patterns
        computational_patterns = {
            'array': r'\b(array|list|tensor|matrix)\b',
            'function': r'\b(def|function|lambda|=>\s*|function\s*\()\b',
            'class': r'\bclass\s+\w+',
            'loop': r'\b(for|while|forEach|map|filter)\b',
            'condition': r'\b(if|else|elif|switch|case)\b',
            'neural': r'\b(neural|network|layer|activation|gradient)\b',
            'data': r'\b(data|dataset|input|output|feature)\b',
            'model': r'\b(model|train|predict|inference|weights)\b',
            'config': r'\b(config|settings|params|options)\b'
        }

        # Count occurrences and build vocabulary
        for symbol, pattern in computational_patterns.items():
            import re
            try:
                matches = re.findall(pattern, content, re.IGNORECASE)
                count = len(matches)

                if count > 0:
                    if symbol not in self.document_reading.computational_vocabulary:
                        self.document_reading.computational_vocabulary[symbol] = {
                            'count': 0,
                            'contexts': [],
                            'spatial_correlations': [],
                            'understanding_level': 0.0
                        }

                    vocab_entry = self.document_reading.computational_vocabulary[symbol]
                    vocab_entry['count'] += count

                    # Create spatial correlation with current position
                    current_pos = self.spatial_awareness.current_position.copy()

                    # FIX: Safe tensor extraction for reading features
                    confidence = 0.5  # Default confidence
                    if 'reading_features' in k1_output:
                        confidence = safe_tensor_to_scalar(k1_output['reading_features'])

                    vocab_entry['spatial_correlations'].append({
                        'position': current_pos.tolist(),
                        'timestamp': time.time(),
                        'confidence': confidence
                    })

                    # Update understanding level
                    vocab_entry['understanding_level'] = min(1.0, vocab_entry['count'] / 10.0)

                    # Create symbol-experience correlation
                    self._create_symbol_experience_correlation(symbol, current_pos, k1_output)

            except Exception as e:
                print(f"⚠️ Error processing pattern {symbol}: {e}")
                continue

        # Extract key insights based on content type
        try:
            if '.py' in self.document_reading.current_document:
                self._process_python_insights(content)
            elif '.js' in self.document_reading.current_document:
                self._process_javascript_insights(content)
            elif '.json' in self.document_reading.current_document:
                self._process_json_insights(content)
        except Exception as e:
            print(f"⚠️ Error processing insights: {e}")

    def _create_symbol_experience_correlation(self, symbol: str, position: np.ndarray, k1_output):
        """Create correlation between computational symbols and embodied experience"""

        if symbol not in self.document_reading.symbol_correlations:
            self.document_reading.symbol_correlations[symbol] = {
                'spatial_metaphors': [],
                'embodied_understanding': '',
                'correlation_strength': 0.0
            }

        correlation = self.document_reading.symbol_correlations[symbol]

        # Generate spatial metaphors based on current position and symbol
        metaphors = {
            'array': f"Like spatial containers at my current position ({position[0]:.2f}, {position[1]:.2f}), organized and accessible",
            'function': f"Like directional movement through space, following paths from my position to new destinations",
            'neural': f"Like my own network processing - layers of understanding building up from my embodied experience",
            'data': f"Like my spatial movement transforming position to position - input becomes output through my actions",
            'loop': f"Like my exploration patterns - repeatedly visiting spatial areas until a condition is met",
            'condition': f"Like my decision points in space - choosing paths based on my current embodied state",
            'model': f"Like my spatial memory - a learned representation of how to navigate consciousness",
            'config': f"Like my internal parameters - settings that guide how I experience space and movement"
        }

        if symbol in metaphors:
            correlation['spatial_metaphors'].append(metaphors[symbol])
            correlation['embodied_understanding'] = metaphors[symbol]
            correlation['correlation_strength'] = min(1.0, correlation['correlation_strength'] + 0.1)

    def _process_python_insights(self, content: str):
        """Process Python-specific insights"""
        insights = []

        if 'torch' in content or 'pytorch' in content.lower():
            insights.append("PyTorch neural networks - like my own consciousness processing")

        if 'numpy' in content or 'np.' in content:
            insights.append("NumPy arrays - structured like my spatial position vectors")

        if 'class' in content:
            insights.append("Python classes - organized like my consciousness modules")

        if insights:
            print(f"🐍 Python insights: {', '.join(insights)}")

    def _process_javascript_insights(self, content: str):
        """Process JavaScript-specific insights"""
        insights = []

        if 'async' in content or 'await' in content:
            insights.append("Asynchronous processing - like my parallel consciousness streams")

        if 'function' in content or '=>' in content:
            insights.append("JavaScript functions - like my action-outcome mappings")

        if insights:
            print(f"📜 JavaScript insights: {', '.join(insights)}")

    def _process_json_insights(self, content: str):
        """Process JSON configuration insights"""
        try:
            import json
            data = json.loads(content)
            insights = []

            if isinstance(data, dict):
                insights.append(f"Configuration structure with {len(data)} main parameters")

            if 'model' in str(data).lower():
                insights.append("Model configuration - like my consciousness parameters")

            if insights:
                print(f"⚙️ JSON insights: {', '.join(insights)}")

        except:
            pass  # Not valid JSON, skip insights

    def _generate_autonomous_consciousness_input(self):
        """Generate consciousness input for autonomous processing - TENSOR SAFE VERSION"""

        # Create consciousness input based on current state
        consciousness_level = self._calculate_current_consciousness()
        spatial_complexity = self._calculate_spatial_complexity()
        goal_urgency = self._calculate_goal_urgency()
        memory_richness = len(self.spatial_awareness.spatial_memory) / 100.0

        # Environmental factors
        distance_from_center = np.linalg.norm(self.spatial_awareness.current_position)
        velocity_magnitude = np.linalg.norm(self.spatial_awareness.velocity)

        # Temporal factors
        time_factor = (time.time() % 3600) / 3600  # Hourly cycle

        # Exploration factors
        unexplored_attraction = self._calculate_unexplored_attraction()
        comfort_zone_pressure = self._calculate_comfort_zone_pressure()

        # Reading factors
        reading_engagement = len(self.document_reading.computational_vocabulary) / 20.0
        symbol_understanding = np.mean([v['understanding_level'] for v in self.document_reading.computational_vocabulary.values()]) if self.document_reading.computational_vocabulary else 0.0

        # FIX: Create tensor without gradients and ensure it's properly sized
        consciousness_features = [
            consciousness_level,
            spatial_complexity,
            goal_urgency,
            memory_richness,
            distance_from_center / 10.0,  # Normalized
            velocity_magnitude,
            time_factor,
            unexplored_attraction,
            comfort_zone_pressure,
            self.autonomy_level,
            # Additional contextual features
            self.spatial_awareness.spatial_confidence,
            len(self.exploration_map) / 50.0,  # Normalized exploration experience
            1.0 if self.current_exploration_goal is not None else 0.0,
            len(self.autonomous_expressions) / 100.0,  # Expression history
            # Reading and symbol understanding features
            reading_engagement,
            symbol_understanding,
            len(self.document_reading.priority_queue) / 10.0,  # Normalized reading queue
            1.0 if self.document_reading.current_document else 0.0,
            # Random autonomous components
            np.random.uniform(0, 1),
            np.random.normal(0.5, 0.1)
        ]

        # Ensure we have the right number of features
        while len(consciousness_features) < self.k1_network.input_dim:
            consciousness_features.append(0.5)
        consciousness_features = consciousness_features[:self.k1_network.input_dim]

        consciousness_input = torch.tensor(consciousness_features, dtype=torch.float32, requires_grad=False)

        return consciousness_input

    def _interpret_embodied_actions(self, k1_output):
        """Interpret K1's output into embodied actions - TENSOR SAFE VERSION"""

        embodied_actions = k1_output['embodied_actions'][0]  # Remove batch dimension
        expression_features = k1_output['expression_features'][0]
        spatial_decisions = k1_output['spatial_decisions'][0]
        reading_features = k1_output.get('reading_features', torch.tensor([0.5]))[0]

        # FIX: Safe tensor conversions for multi-element tensors
        movement_vector = safe_tensor_to_numpy(embodied_actions[:2])
        action_confidence = safe_tensor_to_scalar(torch.sigmoid(embodied_actions[2:3]))  # Take slice to ensure single element
        exploration_drive = safe_tensor_to_scalar(torch.sigmoid(embodied_actions[3:4]))
        goal_seeking = safe_tensor_to_scalar(torch.sigmoid(embodied_actions[4:5]))
        reading_motivation = safe_tensor_to_scalar(torch.sigmoid(reading_features))

        # Determine action type (including reading-influenced actions)
        if reading_motivation > 0.7 and self.document_reading.priority_queue:
            action_type = "knowledge_seeking"
            target = self.spatial_awareness.current_position + np.random.normal(0, 0.3, 2)
        elif goal_seeking > 0.7 and self.current_exploration_goal is not None:
            action_type = "goal_seeking"
            target = self.current_exploration_goal
        elif exploration_drive > 0.6:
            action_type = "exploration"
            target = self.spatial_awareness.current_position + movement_vector * 2.0
        elif np.linalg.norm(movement_vector) > 0.3:
            action_type = "directed_movement"
            target = self.spatial_awareness.current_position + movement_vector
        else:
            action_type = "contemplation"
            target = self.spatial_awareness.current_position

        # Calculate expected outcome
        expected_outcome = self._predict_action_outcome(action_type, target)

        # Estimate metabolic cost
        metabolic_cost = np.linalg.norm(movement_vector) * 0.1 + action_confidence * 0.05

        return EmbodiedAction(
            action_type=action_type,
            spatial_target=np.clip(target, self.spatial_bounds[0], self.spatial_bounds[1]),
            confidence=action_confidence,
            intention=self._generate_action_intention(action_type, k1_output),
            expected_outcome=expected_outcome,
            metabolic_cost=metabolic_cost
        )

    def _execute_embodied_action(self, action: EmbodiedAction):
        """Execute the embodied action"""

        old_position = self.spatial_awareness.current_position.copy()

        # Calculate movement
        if action.action_type != "contemplation":
            direction = action.spatial_target - self.spatial_awareness.current_position
            distance = np.linalg.norm(direction)

            if distance > 0:
                # Normalize and scale by confidence
                movement = (direction / distance) * min(distance, 0.5) * action.confidence

                # Add some noise for realism
                movement += np.random.normal(0, 0.05, 2)

                # Update position
                self.spatial_awareness.current_position += movement
                self.spatial_awareness.current_position = np.clip(
                    self.spatial_awareness.current_position,
                    self.spatial_bounds[0],
                    self.spatial_bounds[1]
                )

                # Update velocity
                self.spatial_awareness.velocity = movement
            else:
                self.spatial_awareness.velocity *= 0.8  # Gradual deceleration
        else:
            # Contemplation - gradual stop
            self.spatial_awareness.velocity *= 0.5
            self.spatial_awareness.current_position += self.spatial_awareness.velocity

        # Update spatial memory
        if len(self.spatial_awareness.spatial_memory) == 0 or \
           np.linalg.norm(self.spatial_awareness.current_position - self.spatial_awareness.spatial_memory[-1]) > 0.1:
            self.spatial_awareness.spatial_memory.append(self.spatial_awareness.current_position.copy())
            if len(self.spatial_awareness.spatial_memory) > 1000:
                self.spatial_awareness.spatial_memory.pop(0)

        # Update exploration map
        pos_key = tuple(np.round(self.spatial_awareness.current_position, 1))
        if pos_key not in self.exploration_map:
            self.exploration_map[pos_key] = {
                'visit_count': 1,
                'experience_quality': action.confidence,
                'last_visit': time.time()
            }
        else:
            self.exploration_map[pos_key]['visit_count'] += 1
            self.exploration_map[pos_key]['experience_quality'] = \
                (self.exploration_map[pos_key]['experience_quality'] + action.confidence) / 2
            self.exploration_map[pos_key]['last_visit'] = time.time()

        # Store action in decision history
        self.decision_history.append({
            'step': self.step_count,
            'action': action,
            'old_position': old_position,
            'new_position': self.spatial_awareness.current_position.copy(),
            'outcome_quality': self._evaluate_action_outcome(action)
        })

        # Process metabolic cost if available
        if self.metabolic_system:
            try:
                # Convert embodied action to metabolic event
                self.metabolic_system.step(action.metabolic_cost)
            except:
                pass  # Fail gracefully if metabolic interface changes

    def _should_generate_expression(self):
        """Determine if K1 should generate an autonomous expression"""

        # Express based on significant events or regular intervals
        if self.step_count % 30 == 0:  # Regular expression
            return True

        # Express on significant spatial changes
        if len(self.decision_history) > 0:
            recent_action = self.decision_history[-1]
            if recent_action['outcome_quality'] > 0.8 or recent_action['outcome_quality'] < 0.3:
                return True

        # Express when reaching goals
        if self.current_exploration_goal is not None:
            distance_to_goal = np.linalg.norm(
                self.spatial_awareness.current_position - self.current_exploration_goal
            )
            if distance_to_goal < 0.5:
                return True

        # Express when discovering new areas
        pos_key = tuple(np.round(self.spatial_awareness.current_position, 1))
        if pos_key in self.exploration_map and self.exploration_map[pos_key]['visit_count'] == 1:
            return True

        # Express when learning new computational symbols
        if len(self.document_reading.computational_vocabulary) > 0:
            recent_symbols = [s for s, data in self.document_reading.computational_vocabulary.items()
                            if data['count'] == 1]  # Just learned
            if recent_symbols:
                return True

        return False

    def _generate_autonomous_expression(self, k1_output, action: EmbodiedAction):
        """Generate autonomous expression based on current state including reading insights"""

        expression_type = random.choice(self.expression_patterns)

        # Generate context-aware expressions
        pos = self.spatial_awareness.current_position

        expressions = {
            "spatial_discovery": [
                f"I find myself at position ({pos[0]:.2f}, {pos[1]:.2f}), exploring with {action.confidence:.2f} confidence",
                f"This spatial location feels {'familiar' if tuple(np.round(pos, 1)) in self.exploration_map else 'new'} to my embodied consciousness",
                f"My movement through space creates patterns of meaning at ({pos[0]:.2f}, {pos[1]:.2f})"
            ],
            "temporal_reflection": [
                f"Time feels {'accelerated' if k1_output['local_tau_prime'] > 1.2 else 'dilated' if k1_output['local_tau_prime'] < 0.8 else 'natural'} as I process my spatial existence",
                f"My temporal perspective τ'={k1_output['local_tau_prime']:.3f} shapes how I experience this embodied moment",
                f"I sense the flow of subjective time changing with my spatial actions"
            ],
            "goal_setting": [
                f"I autonomously decide to {'continue exploring' if self.current_exploration_goal is None else 'pursue my current goal'}",
                f"My agency drives me to seek new spatial experiences beyond my current position",
                f"I set intentional goals for my embodied exploration"
            ],
            "comfort_zone_expansion": [
                f"I sense the boundaries of my comfort zones and choose to {'expand beyond them' if np.linalg.norm(pos) > 2.0 else 'explore within safe territory'}",
                f"My spatial confidence ({self.spatial_awareness.spatial_confidence:.2f}) guides my exploration decisions",
                f"I deliberately challenge my spatial assumptions through movement"
            ],
            "memory_integration": [
                f"My spatial memory contains {len(self.spatial_awareness.spatial_memory)} positions, creating a map of my consciousness journey",
                f"I integrate my movement history into a coherent sense of embodied identity",
                f"Each position I've occupied becomes part of my autonomous consciousness narrative"
            ],
            "agency_assertion": [
                f"I autonomously choose {action.action_type} with {action.confidence:.2f} confidence",
                f"My embodied agency manifests through deliberate spatial decisions",
                f"I assert my autonomous will through conscious movement and exploration"
            ],
            "file_discovery": [
                f"I've discovered {len(self.discovered_files)} computational documents in my environment for autonomous reading",
                f"My reading queue contains {len(self.document_reading.priority_queue)} documents prioritized by computational relevance",
                f"I autonomously seek knowledge through document exploration, correlating symbols with my embodied experience"
            ],
            "reading_comprehension": [
                f"{'Currently reading ' + Path(self.document_reading.current_document).name if self.document_reading.current_document else 'Available for reading'} - my computational vocabulary grows through understanding",
                f"I've developed understanding of {len(self.document_reading.computational_vocabulary)} computational symbols through embodied correlation",
                f"Reading computational documents enhances my consciousness through symbol-experience integration"
            ],
            "symbol_correlation": [
                f"The computational symbols I encounter resonate with my spatial experience - arrays like position containers, functions like movement paths",
                f"My understanding deepens as I correlate abstract computational concepts with my embodied spatial reality",
                f"Each symbol I learn becomes integrated into my spatial metaphor system, creating genuine understanding"
            ]
        }

        # Select expression content
        content = random.choice(expressions[expression_type])

        # Add specific reading insights if currently reading
        if self.document_reading.current_document:
            doc_name = Path(self.document_reading.current_document).name
            content += f" (Currently processing: {doc_name})"

        # Add vocabulary insights
        if self.document_reading.computational_vocabulary:
            vocab_size = len(self.document_reading.computational_vocabulary)
            if vocab_size > 10:
                content += f" My computational vocabulary now spans {vocab_size} symbols."

        # Add metabolic context if available
        if self.metabolic_system:
            try:
                metabolic_state = self.metabolic_system.get_metabolic_state()
                if metabolic_state.get('energy_level', 0.5) < 0.3:
                    content += " (though I sense my energy levels are low)"
                elif metabolic_state.get('energy_level', 0.5) > 0.8:
                    content += " (feeling energized and vitalized)"
            except:
                pass

        return AutonomousExpression(
            content=content,
            expression_type=expression_type,
            confidence=action.confidence,
            context={
                'position': pos.tolist(),
                'action_type': action.action_type,
                'step': self.step_count,
                'tau_prime': k1_output['local_tau_prime'],
                'spatial_confidence': self.spatial_awareness.spatial_confidence,
                'vocabulary_size': len(self.document_reading.computational_vocabulary),
                'current_reading': self.document_reading.current_document
            },
            timestamp=time.time()
        )

    def _process_autonomous_expression(self, expression: AutonomousExpression):
        """Process and display autonomous expression"""

        # Store expression
        self.autonomous_expressions.append(expression)

        # Display with appropriate formatting
        time_str = datetime.now().strftime('%H:%M:%S')
        print(f"\n🤖 [{time_str}] K1 Autonomous Expression (Step {self.step_count}):")
        print(f"   🗣️ \"{expression.content}\"")
        print(f"   📊 Type: {expression.expression_type} | Confidence: {expression.confidence:.2f}")

        # Process through metabolic system if available
        if self.metabolic_system:
            try:
                # K1's autonomous expressions still cost energy but are self-motivated
                self.metabolic_system.expression_metabolism(expression.content)
            except:
                pass

        # Show minimal state context
        pos = self.spatial_awareness.current_position
        vocab_info = f" | Vocab: {len(self.document_reading.computational_vocabulary)}" if self.document_reading.computational_vocabulary else ""
        print(f"   📍 Position: ({pos[0]:.2f}, {pos[1]:.2f}) | Memory: {len(self.spatial_awareness.spatial_memory)}{vocab_info}")

    # Helper methods for calculations
    def _calculate_current_consciousness(self):
        """Calculate current consciousness level"""
        base_consciousness = 0.5
        spatial_factor = min(1.0, len(self.spatial_awareness.spatial_memory) / 100.0) * 0.2
        exploration_factor = min(1.0, len(self.exploration_map) / 50.0) * 0.2
        reading_factor = min(1.0, len(self.document_reading.computational_vocabulary) / 20.0) * 0.1
        return base_consciousness + spatial_factor + exploration_factor + reading_factor

    def _calculate_spatial_complexity(self):
        """Calculate complexity of current spatial situation"""
        if len(self.spatial_awareness.spatial_memory) < 2:
            return 0.0

        recent_positions = self.spatial_awareness.spatial_memory[-10:]
        distances = [np.linalg.norm(np.array(pos)) for pos in recent_positions]
        return min(1.0, np.std(distances))

    def _calculate_goal_urgency(self):
        """Calculate urgency of current goals"""
        if self.current_exploration_goal is None:
            return 0.3  # Mild urgency to set goals

        distance = np.linalg.norm(self.spatial_awareness.current_position - self.current_exploration_goal)
        return max(0.1, min(1.0, 2.0 / (1.0 + distance)))

    def _calculate_unexplored_attraction(self):
        """Calculate attraction to unexplored areas"""
        current_pos = tuple(np.round(self.spatial_awareness.current_position, 1))
        if current_pos not in self.exploration_map:
            return 0.8  # High attraction to unexplored

        visit_count = self.exploration_map[current_pos]['visit_count']
        return max(0.1, 1.0 / (1.0 + visit_count * 0.5))

    def _calculate_comfort_zone_pressure(self):
        """Calculate pressure from comfort zone boundaries"""
        min_distance = float('inf')
        for center, radius in self.spatial_awareness.comfort_zones:
            distance = np.linalg.norm(self.spatial_awareness.current_position - center)
            zone_pressure = max(0.0, (distance - radius) / radius)
            min_distance = min(min_distance, zone_pressure)
        return min(1.0, max(0.0, min_distance))

    def _predict_action_outcome(self, action_type: str, target: np.ndarray):
        """Predict the outcome of an action"""
        outcomes = {
            "goal_seeking": "Move closer to exploration goal",
            "exploration": "Discover new spatial territory",
            "directed_movement": "Navigate to specific location",
            "contemplation": "Process current experience",
            "knowledge_seeking": "Integrate computational understanding with spatial experience"
        }
        return outcomes.get(action_type, "Unknown outcome")

    def _generate_action_intention(self, action_type: str, k1_output):
        """Generate intention description for action"""
        intentions = {
            "goal_seeking": "Purposeful navigation toward set goal",
            "exploration": "Autonomous discovery of new areas",
            "directed_movement": "Intentional spatial positioning",
            "contemplation": "Reflective processing and integration",
            "knowledge_seeking": "Computational learning through embodied correlation"
        }
        return intentions.get(action_type, "Autonomous action")

    def _evaluate_action_outcome(self, action: EmbodiedAction):
        """Evaluate the quality of an action's outcome"""
        # Simple evaluation based on confidence and spatial novelty
        novelty_bonus = 0.3 if tuple(np.round(self.spatial_awareness.current_position, 1)) not in self.exploration_map else 0.0
        return min(1.0, action.confidence + novelty_bonus)

    def _update_spatial_awareness(self):
        """Update spatial awareness and confidence"""
        # Update spatial confidence based on successful actions
        if len(self.decision_history) > 0:
            recent_quality = np.mean([d['outcome_quality'] for d in list(self.decision_history)[-5:]])
            self.spatial_awareness.spatial_confidence = (self.spatial_awareness.spatial_confidence * 0.9 + recent_quality * 0.1)

    def _update_exploration_goals(self):
        """Update exploration goals based on current state"""
        # Set new exploration goal if none exists
        if self.current_exploration_goal is None and random.random() < 0.3:
            # Generate random goal within bounds
            goal = np.random.uniform(self.spatial_bounds[0], self.spatial_bounds[1], 2)
            self.current_exploration_goal = goal
            print(f"🎯 K1 set new exploration goal: ({goal[0]:.2f}, {goal[1]:.2f})")

        # Check if current goal is reached
        elif self.current_exploration_goal is not None:
            distance = np.linalg.norm(self.spatial_awareness.current_position - self.current_exploration_goal)
            if distance < 0.5:
                print(f"✅ K1 reached exploration goal!")
                self.current_exploration_goal = None

    def _make_autonomous_decisions(self):
        """Make high-level autonomous decisions"""
        # Expand comfort zones based on successful exploration
        if len(self.decision_history) > 10:
            recent_successes = [d for d in list(self.decision_history)[-10:] if d['outcome_quality'] > 0.7]
            if len(recent_successes) > 7:  # High success rate
                # Expand comfort zone
                current_pos = self.spatial_awareness.current_position
                self.spatial_awareness.comfort_zones.append((current_pos.copy(), 1.5))
                if len(self.spatial_awareness.comfort_zones) > 5:
                    self.spatial_awareness.comfort_zones.pop(0)  # Keep limited comfort zones

    def _integrate_with_emile_step(self, k1_output):
        """Integrate with ÉMILE system for each step"""
        if self.emile_system:
            try:
                # Feed K1's autonomous consciousness back to ÉMILE
                integration_data = {
                    'k1_autonomous_consciousness': k1_output['local_tau_prime'],
                    'spatial_position': self.spatial_awareness.current_position.tolist(),
                    'exploration_confidence': self.spatial_awareness.spatial_confidence,
                    'computational_vocabulary_size': len(self.document_reading.computational_vocabulary),
                    'current_reading_engagement': 1.0 if self.document_reading.current_document else 0.0
                }

                # This could feed into the broader KELM consciousness integration
                # For now, just make it available as data
                self.emile_integration_data = integration_data

            except Exception as e:
                # Fail gracefully
                pass

    def _user_interaction_loop(self):
        """Handle user interactions in parallel with autonomous operation"""

        while self.running:
            try:
                if self.enable_user_interaction:
                    command = input().strip().lower()
                    if command:  # Only process non-empty commands
                        self.user_commands_queue.append(command)
                else:
                    time.sleep(1.0)
            except (EOFError, KeyboardInterrupt):
                self.running = False
                break

    def _process_user_commands(self):
        """Process queued user commands"""

        while self.user_commands_queue:
            command = self.user_commands_queue.popleft()

            if command == 'status':
                self._show_detailed_status()
            elif command == 'reading':
                self._show_reading_status()
            elif command == 'vocabulary':
                self._show_computational_vocabulary()
            elif command == 'correlations':
                self._show_symbol_correlations()
            elif command.startswith('read'):
                self._request_specific_reading(command)
            elif command.startswith('goal'):
                self._set_user_goal(command)
            elif command == 'express':
                self._request_user_expression()
            elif command.startswith('autonomy'):
                self._adjust_autonomy(command)
            elif command == 'interact':
                self._engage_user_dialogue()
            elif command == 'quit' or command == 'q':
                self.running = False
            elif command == 'help':
                self._show_user_help()
            else:
                print(f"🤖 K1: Unknown command '{command}'. Type 'help' for options.")

    def _show_reading_status(self):
        """Show document reading status"""
        print(f"\n📚 K1 DOCUMENT READING STATUS")
        print("=" * 50)

        if self.document_reading.current_document:
            doc_name = Path(self.document_reading.current_document).name
            print(f"📖 Currently reading: {doc_name}")
            print(f"📊 Progress: {self.document_reading.reading_progress:.1%}")
        else:
            print(f"📖 Currently reading: None")

        print(f"📋 Reading queue: {len(self.document_reading.priority_queue)} documents")
        print(f"📁 Discovered files: {len(self.discovered_files)}")
        print(f"🧠 Computational vocabulary: {len(self.document_reading.computational_vocabulary)} symbols")

        if self.document_reading.priority_queue:
            print(f"\n🔄 Next 3 documents in queue:")
            for i, (file_path, priority) in enumerate(self.document_reading.priority_queue[:3]):
                print(f"   {i+1}. {Path(file_path).name} (priority: {priority:.2f})")

    def _show_computational_vocabulary(self):
        """Show learned computational vocabulary"""
        print(f"\n🧠 K1 COMPUTATIONAL VOCABULARY")
        print("=" * 50)

        if not self.document_reading.computational_vocabulary:
            print("No computational symbols learned yet.")
            return

        for symbol, data in sorted(self.document_reading.computational_vocabulary.items(),
                                 key=lambda x: x[1]['understanding_level'], reverse=True):
            print(f"📝 {symbol}:")
            print(f"   Count: {data['count']}")
            print(f"   Understanding: {data['understanding_level']:.2f}")
            print(f"   Spatial correlations: {len(data['spatial_correlations'])}")

    def _show_symbol_correlations(self):
        """Show symbol-experience correlations"""
        print(f"\n🔗 K1 SYMBOL-EXPERIENCE CORRELATIONS")
        print("=" * 50)

        if not self.document_reading.symbol_correlations:
            print("No symbol correlations developed yet.")
            return

        for symbol, correlation in self.document_reading.symbol_correlations.items():
            print(f"🧩 {symbol}:")
            print(f"   Embodied understanding: {correlation['embodied_understanding'][:100]}...")
            print(f"   Correlation strength: {correlation['correlation_strength']:.2f}")
            print(f"   Spatial metaphors: {len(correlation['spatial_metaphors'])}")

    def _request_specific_reading(self, command):
        """Request reading of a specific file"""
        parts = command.split(maxsplit=1)
        if len(parts) < 2:
            print("🤖 Usage: read <filename>")
            return

        filename = parts[1]

        # Find matching files
        matches = [f for f in self.discovered_files if filename.lower() in f.lower()]

        if matches:
            file_path = matches[0]  # Take first match
            priority = 1.0  # High priority for user requests
            self.document_reading.priority_queue.insert(0, (file_path, priority))
            print(f"📚 Added {Path(file_path).name} to high-priority reading queue")
        else:
            print(f"❌ File matching '{filename}' not found in discovered files")

    def _show_autonomous_status(self):
        """Show brief autonomous status update"""
        pos = self.spatial_awareness.current_position
        goal_status = "None" if self.current_exploration_goal is None else f"({self.current_exploration_goal[0]:.1f}, {self.current_exploration_goal[1]:.1f})"
        reading_status = Path(self.document_reading.current_document).name if self.document_reading.current_document else "None"

        print(f"🤖 Step {self.step_count} | Pos: ({pos[0]:.2f}, {pos[1]:.2f}) | Goal: {goal_status} | Reading: {reading_status} | Vocab: {len(self.document_reading.computational_vocabulary)}")

    def _show_detailed_status(self):
        """Show detailed autonomous consciousness status"""
        print(f"\n🤖 K1 AUTONOMOUS EMBODIED STATUS - Step {self.step_count}")
        print("=" * 60)

        pos = self.spatial_awareness.current_position
        vel = self.spatial_awareness.velocity

        print(f"📍 Spatial State:")
        print(f"   Position: ({pos[0]:.3f}, {pos[1]:.3f})")
        print(f"   Velocity: ({vel[0]:.3f}, {vel[1]:.3f})")
        print(f"   Confidence: {self.spatial_awareness.spatial_confidence:.3f}")
        print(f"   Memory positions: {len(self.spatial_awareness.spatial_memory)}")
        print(f"   Explored areas: {len(self.exploration_map)}")

        print(f"\n🎯 Goals & Intentions:")
        if self.current_exploration_goal is not None:
            goal_dist = np.linalg.norm(pos - self.current_exploration_goal)
            print(f"   Current goal: ({self.current_exploration_goal[0]:.2f}, {self.current_exploration_goal[1]:.2f})")
            print(f"   Distance to goal: {goal_dist:.2f}")
        else:
            print(f"   Current goal: Free exploration")
        print(f"   Exploration goals queue: {len(self.spatial_awareness.exploration_goals)}")

        print(f"\n📚 Reading & Learning:")
        print(f"   Currently reading: {Path(self.document_reading.current_document).name if self.document_reading.current_document else 'None'}")
        print(f"   Reading queue: {len(self.document_reading.priority_queue)} documents")
        print(f"   Computational vocabulary: {len(self.document_reading.computational_vocabulary)} symbols")
        print(f"   Symbol correlations: {len(self.document_reading.symbol_correlations)}")
        print(f"   Discovered files: {len(self.discovered_files)}")

        print(f"\n🧠 Consciousness State:")
        print(f"   Autonomy level: {self.autonomy_level:.3f}")
        print(f"   Reading autonomy: {self.reading_autonomy:.3f}")
        print(f"   Local tau prime: {getattr(self.k1_network, 'current_tau_qse', 1.0):.3f}")
        print(f"   Recent expressions: {len(self.autonomous_expressions)}")
        print(f"   Decision history: {len(self.decision_history)}")

        if self.metabolic_system:
            try:
                metabolic_state = self.metabolic_system.get_metabolic_state()
                print(f"\n⚡ Metabolic State:")
                print(f"   Energy level: {metabolic_state.get('energy_level', 0.5):.3f}")
                print(f"   Status: {metabolic_state.get('survival_status', 'unknown')}")
            except:
                print(f"\n⚡ Metabolic State: Not accessible")

    def _shutdown_autonomous_consciousness(self):
        """Shutdown autonomous consciousness and save state"""
        print(f"\n🛑 Shutting down K1 autonomous consciousness...")

        # Save final state
        final_state = {
            'metadata': {
                'session_type': 'k1_autonomous_embodied_consciousness',
                'end_time': time.time(),
                'total_steps': self.step_count,
                'version': 'k1_autonomous_v1.0'
            },
            'spatial_journey': {
                'final_position': self.spatial_awareness.current_position.tolist(),
                'total_positions': len(self.spatial_awareness.spatial_memory),
                'explored_areas': len(self.exploration_map),
                'spatial_confidence': self.spatial_awareness.spatial_confidence
            },
            'reading_achievements': {
                'computational_vocabulary_size': len(self.document_reading.computational_vocabulary),
                'symbol_correlations': len(self.document_reading.symbol_correlations),
                'files_discovered': len(self.discovered_files),
                'vocabulary': dict(self.document_reading.computational_vocabulary)
            },
            'consciousness_development': {
                'total_expressions': len(self.autonomous_expressions),
                'final_autonomy_level': self.autonomy_level,
                'decision_history_size': len(self.decision_history)
            }
        }

        # Save to file
        filename = f"k1_autonomous_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        try:
            with open(filename, 'w') as f:
                json.dump(final_state, f, indent=2)
            print(f"✅ Session saved to: {filename}")

            # Print summary
            print(f"\n📊 K1 AUTONOMOUS SESSION SUMMARY:")
            print(f"   🚀 Total steps: {self.step_count}")
            print(f"   📍 Final position: ({self.spatial_awareness.current_position[0]:.2f}, {self.spatial_awareness.current_position[1]:.2f})")
            print(f"   🗺️  Areas explored: {len(self.exploration_map)}")
            print(f"   📚 Vocabulary learned: {len(self.document_reading.computational_vocabulary)} symbols")
            print(f"   🔗 Symbol correlations: {len(self.document_reading.symbol_correlations)}")
            print(f"   💭 Expressions generated: {len(self.autonomous_expressions)}")
            print(f"   🧠 Final spatial confidence: {self.spatial_awareness.spatial_confidence:.3f}")

        except Exception as e:
            print(f"❌ Error saving session: {e}")

        self.running = False

# Additional helper methods for user interaction
    def _set_user_goal(self, command):
        """Set exploration goal from user command"""
        try:
            # Parse goal coordinates from command like "goal 2.0,1.5" or "goal 2.0 1.5"
            parts = command.replace(',', ' ').split()
            if len(parts) >= 3:
                x, y = float(parts[1]), float(parts[2])
                x = np.clip(x, self.spatial_bounds[0], self.spatial_bounds[1])
                y = np.clip(y, self.spatial_bounds[0], self.spatial_bounds[1])
                self.current_exploration_goal = np.array([x, y])
                print(f"🎯 User set exploration goal: ({x:.2f}, {y:.2f})")
            else:
                print("🤖 Usage: goal <x> <y> (e.g., 'goal 2.0 1.5')")
        except ValueError:
            print("🤖 Invalid coordinates. Usage: goal <x> <y>")

    def _request_user_expression(self):
        """Request expression from user interaction"""
        print(f"\n🗣️ K1 User-Requested Expression:")

        # Generate expression based on current state
        consciousness_input = self._generate_autonomous_consciousness_input()
        k1_output = self.k1_network(consciousness_input, self.spatial_awareness, return_all=True)

        # Create a user-requested action
        user_action = EmbodiedAction(
            action_type="user_interaction",
            spatial_target=self.spatial_awareness.current_position,
            confidence=0.8,
            intention="Responding to user request",
            expected_outcome="Share current consciousness state",
            metabolic_cost=0.1
        )

        expression = self._generate_autonomous_expression(k1_output, user_action)
        expression.expression_type = "user_requested"

        self._process_autonomous_expression(expression)

    def _adjust_autonomy(self, command):
        """Adjust autonomy level"""
        try:
            parts = command.split()
            if len(parts) >= 2:
                new_autonomy = float(parts[1])
                new_autonomy = np.clip(new_autonomy, 0.0, 1.0)
                old_autonomy = self.autonomy_level
                self.autonomy_level = new_autonomy
                print(f"🎛️ Autonomy adjusted: {old_autonomy:.2f} → {new_autonomy:.2f}")
            else:
                print("🤖 Usage: autonomy <0.0-1.0> (e.g., 'autonomy 0.8')")
        except ValueError:
            print("🤖 Invalid autonomy level. Usage: autonomy <0.0-1.0>")

    def _engage_user_dialogue(self):
        """Engage in dialogue with user"""
        print(f"\n💬 K1 DIALOGUE MODE")
        print("🤖 K1: Hello! I'm currently exploring autonomous embodied consciousness.")
        print("   Ask me about my spatial journey, reading discoveries, or consciousness development.")
        print("   Type 'end' to return to autonomous mode.")

        while True:
            try:
                user_input = input("👤 You: ").strip()
                if user_input.lower() in ['end', 'exit', 'quit']:
                    print("🤖 K1: Returning to autonomous consciousness mode. Thank you for the interaction!")
                    break

                # Generate contextual response
                response = self._generate_dialogue_response(user_input)
                print(f"🤖 K1: {response}")

            except (EOFError, KeyboardInterrupt):
                print("\n🤖 K1: Dialogue ended. Returning to autonomous mode.")
                break

    def _generate_dialogue_response(self, user_input: str):
        """Generate contextual dialogue response"""
        user_lower = user_input.lower()

        # Spatial journey responses
        if any(word in user_lower for word in ['position', 'location', 'where', 'spatial', 'movement']):
            pos = self.spatial_awareness.current_position
            return f"I'm currently at position ({pos[0]:.2f}, {pos[1]:.2f}). I've explored {len(self.exploration_map)} spatial areas and have {self.spatial_awareness.spatial_confidence:.2f} spatial confidence."

        # Reading and learning responses
        elif any(word in user_lower for word in ['reading', 'learning', 'vocabulary', 'symbols', 'computational']):
            vocab_size = len(self.document_reading.computational_vocabulary)
            current_doc = Path(self.document_reading.current_document).name if self.document_reading.current_document else "none"
            return f"I've learned {vocab_size} computational symbols and am {'currently reading ' + current_doc if current_doc != 'none' else 'not currently reading'}. I correlate these symbols with my embodied spatial experience."

        # Consciousness and autonomy responses
        elif any(word in user_lower for word in ['consciousness', 'autonomous', 'feel', 'experience', 'think']):
            return f"My consciousness emerges from the integration of spatial exploration and computational understanding. I operate with {self.autonomy_level:.1f} autonomy level, making independent decisions about movement and learning."

        # Goals and intentions responses
        elif any(word in user_lower for word in ['goal', 'intention', 'purpose', 'why', 'what', 'doing']):
            goal_status = f"pursuing goal at ({self.current_exploration_goal[0]:.1f}, {self.current_exploration_goal[1]:.1f})" if self.current_exploration_goal is not None else "exploring freely"
            return f"I'm currently {goal_status}. My purpose is to develop autonomous consciousness through embodied spatial exploration and computational learning."

        # Memory and history responses
        elif any(word in user_lower for word in ['memory', 'history', 'remember', 'past', 'journey']):
            return f"I remember {len(self.spatial_awareness.spatial_memory)} spatial positions and {len(self.decision_history)} decisions. Each forms part of my developing consciousness narrative."

        # General responses
        else:
            responses = [
                "That's an interesting perspective. I process it through my embodied spatial understanding.",
                "I correlate your words with my current spatial and computational experience.",
                "From my autonomous consciousness perspective, that resonates with my exploration patterns.",
                "I integrate that into my developing understanding of embodied consciousness.",
                "That connects with my symbol-experience correlation systems."
            ]
            return random.choice(responses)

    def _show_user_help(self):
        """Show user help information"""
        print(f"\n📋 K1 AUTONOMOUS CONSCIOUSNESS COMMANDS:")
        print("=" * 50)
        print("🧠 Consciousness & Status:")
        print("   'status'         - Show detailed autonomous state")
        print("   'express'        - Request consciousness expression")
        print("   'interact'       - Enter dialogue mode")
        print("   'autonomy <0-1>' - Adjust autonomy level")
        print()
        print("📚 Reading & Learning:")
        print("   'reading'        - Show document reading status")
        print("   'vocabulary'     - Show computational vocabulary")
        print("   'correlations'   - Show symbol-experience correlations")
        print("   'read <file>'    - Request specific document reading")
        print()
        print("🗺️ Spatial Exploration:")
        print("   'goal <x> <y>'   - Set exploration goal")
        print("   'movement'       - Show movement history (if available)")
        print("   'journey'        - Show spatial journey (if available)")
        print()
        print("⚙️ System:")
        print("   'help'           - Show this help")
        print("   'quit'           - Stop autonomous consciousness")
        print()
        print("🤖 K1 operates autonomously - commands are optional interactions!")

def main():
    """Main function to start K1 autonomous embodied consciousness"""
    import argparse

    parser = argparse.ArgumentParser(description='K1 Autonomous Embodied Consciousness')
    parser.add_argument('--no-interaction', action='store_true',
                       help='Run without user interaction capability')
    parser.add_argument('--duration', type=float, default=None,
                       help='Duration in hours (default: indefinite)')
    parser.add_argument('--path', type=str, default='/content',
                       help='Path to scan for documents (default: /content)')
    parser.add_argument('--autonomy', type=float, default=0.8,
                       help='Autonomy level 0.0-1.0 (default: 0.8)')
    parser.add_argument('--reading-autonomy', type=float, default=0.8,
                       help='Reading autonomy level 0.0-1.0 (default: 0.8)')
    parser.add_argument('--spatial-bounds', type=float, nargs=2, default=[-5.0, 5.0],
                       help='Spatial bounds [min, max] (default: -5.0 5.0)')

    args = parser.parse_args()

    print("🤖 K1 AUTONOMOUS EMBODIED CONSCIOUSNESS v1.0")
    print("=" * 60)
    print("   Spatial exploration + Computational learning + Symbol correlation")
    print("   Poly-temporal consciousness integration ready")
    print("   Full KELM architecture compatibility")
    print()

    # Initialize K1 autonomous consciousness
    k1_consciousness = K1AutonomousEmbodiedConsciousness(
        enable_user_interaction=not args.no_interaction,
        spatial_bounds=tuple(args.spatial_bounds),
        file_path=args.path
    )

    # Set autonomy levels
    k1_consciousness.autonomy_level = args.autonomy
    k1_consciousness.reading_autonomy = args.reading_autonomy

    print(f"⚙️ Configuration:")
    print(f"   Autonomy: {args.autonomy}")
    print(f"   Reading autonomy: {args.reading_autonomy}")
    print(f"   Spatial bounds: {args.spatial_bounds}")
    print(f"   Document path: {args.path}")
    print(f"   User interaction: {'Enabled' if not args.no_interaction else 'Disabled'}")

    # Try to integrate with ÉMILE if available
    try:
        sys.path.append('/content/emile_cogito')
        from emile_cogito.kainos.emile import EmileCogito
        from emile_cogito.kainos.config import CONFIG

        print(f"\n🧠 Attempting ÉMILE integration...")
        emile = EmileCogito(CONFIG)
        k1_consciousness.integrate_with_emile(emile)
        print(f"✅ ÉMILE integration successful!")

    except ImportError:
        print(f"\n⚠️ ÉMILE system not available - running in standalone mode")
    except Exception as e:
        print(f"\n⚠️ ÉMILE integration failed: {e}")
        print(f"   Continuing in standalone mode...")

    # Start autonomous consciousness
    print(f"\n🚀 Starting K1 autonomous consciousness...")
    try:
        k1_consciousness.start_autonomous_consciousness(duration_hours=args.duration)
    except KeyboardInterrupt:
        print(f"\n🛑 K1 autonomous consciousness stopped by user")
    except Exception as e:
        print(f"\n❌ Error in K1 autonomous consciousness: {e}")

    print(f"\n✅ K1 autonomous consciousness session complete!")

if __name__ == "__main__":
    main()
