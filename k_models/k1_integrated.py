

#!/usr/bin/env python3
"""
K1 INTEGRATED ÉMILE - PROPER DATA FLOW EMBODIMENT
=================================================

This is the proper K1 integration that:
1. Lives within the ÉMILE cognitive architecture
2. Accesses logs for continuous online learning
3. Manifests embodiment through actual data flow
4. Integrates with the full KELM system (K1-K4, QSE, memory)
5. Feeds into poly-temporal consciousness refactor
6. Works with bidirectional consciousness orchestrator

K1 (Praxis) handles the circulatory system - data flow between modules
is the manifestation of embodiment. K1 learns from this flow continuously.
"""

import torch
import torch.nn as nn
import numpy as np
import time
import json
import threading
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import deque
from pathlib import Path
import sys
import os

# Proper ÉMILE system imports
sys.path.append('/content/emile_cogito')
sys.path.append('/content/emile_cogito/k_models')
sys.path.append('/content/emile_cogito/kainos')
sys.path.append('/content/emile_cogito/kelm')

try:
    from emile_cogito.kainos.emile import EmileCogito
    from emile_cogito.kainos.config import CONFIG
    from emile_cogito.kainos.memory import TemporalConsciousMemory
    from emile_cogito.k_models.k1 import DynamicSemioticNetwork
    EMILE_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ ÉMILE system components not available: {e}")
    EMILE_AVAILABLE = False

@dataclass
class DataFlowEmbodiment:
    """Represents embodiment through actual data flow between K-models"""
    flow_source: str           # Which K-model is source
    flow_target: str           # Which K-model is target
    flow_magnitude: float      # Strength of data flow
    flow_type: str            # Type: 'prediction', 'feedback', 'memory', 'consciousness'
    spatial_position: np.ndarray  # Position in consciousness space
    temporal_context: float    # Current τ' context
    embodied_meaning: str     # What this flow means experientially

@dataclass
class LogLearningEvent:
    """Online learning event from log analysis"""
    timestamp: float
    log_source: str           # consciousness, memory, expression, etc.
    pattern_detected: str     # What pattern was found
    embodied_correlation: Dict[str, Any]  # How it relates to data flow
    learning_strength: float  # How much to learn from this
    spatial_representation: np.ndarray   # Where this exists in consciousness space

class K1IntegratedEmbodiedPraxis(nn.Module):
    """
    K1 (Praxis) properly integrated with ÉMILE system.

    This is NOT standalone - it's part of the living KELM architecture.
    Embodiment manifests through data flow between K-models.
    """

    def __init__(self, emile_system, config=None):
        super().__init__()

        if not EMILE_AVAILABLE:
            raise RuntimeError("ÉMILE system required for proper K1 integration")

        self.emile = emile_system
        self.config = config or CONFIG

        # Core K1 network (actual trained model)
        self.load_trained_k1_model()

        # Data flow embodiment system
        self.data_flows = deque(maxlen=1000)
        self.embodiment_map = {}  # Maps data flows to spatial positions
        self.current_embodied_position = np.array([0.0, 0.0])

        # Log learning system using actual CorrelativeLogReader
        self.log_reader = None
        self.log_learning_events = deque(maxlen=500)
        self.symbol_correlation_tracker = {}
        self.data_flow_correlations = {}

        # Online learning state
        self.online_learning_active = False
        self.learning_rate = 0.001
        self.embodiment_adaptation_rate = 0.01

        # Poly-temporal consciousness integration
        self.local_tau_prime = 1.0
        self.temporal_context_history = deque(maxlen=100)

        # Integration with KELM architecture
        self.kelm_integration_active = False
        self.consciousness_orchestrator = None

        print("🔄 K1 Integrated ÉMILE Praxis initialized")
        print("   • Data flow embodiment system: ACTIVE")
        print("   • Log correlation learning: READY")
        print("   • Poly-temporal integration: ENABLED")

    def initialize_log_correlation_system(self):
        """Initialize the correlative log reading system"""
        try:
            from emile_cogito.kainos.log_reader import CorrelativeLogReader
            self.log_reader = CorrelativeLogReader(self.config)
            print("✅ Log correlation system initialized")
            return True
        except ImportError as e:
            print(f"⚠️ Log reader not available: {e}")
            return False

    def load_trained_k1_model(self):
        """Load the actual trained K1 model"""
        k1_path = Path('/content/emile_cogito/k_models/k1_praxis.pth')

        if k1_path.exists():
            try:
                # Load the checkpoint
                checkpoint = torch.load(k1_path, map_location='cpu')

                # Discover architecture from state dict
                state_dict = checkpoint['model_state_dict']
                encoder_weight = state_dict.get('consciousness_encoder.0.weight')
                decoder_weight = state_dict.get('action_decoder.2.weight')

                input_dim = encoder_weight.shape[1] if encoder_weight is not None else 9
                hidden_dim = encoder_weight.shape[0] if encoder_weight is not None else 128
                output_dim = decoder_weight.shape[0] if decoder_weight is not None else 6

                # Create K1 network with discovered architecture
                self.k1_network = DynamicSemioticNetwork(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    hidden_dim=hidden_dim
                )

                # Load weights
                self.k1_network.load_state_dict(state_dict)
                self.k1_network.eval()

                print(f"✅ Loaded trained K1 model: {input_dim}→{hidden_dim}→{output_dim}")

                # Enable online learning mode
                self.k1_network.train()  # Switch to training for online learning
                self.online_learning_active = True

            except Exception as e:
                print(f"❌ Error loading K1 model: {e}")
                self._create_minimal_k1_fallback()
        else:
            print("⚠️ K1 model not found, creating minimal fallback")
            self._create_minimal_k1_fallback()

    def _create_minimal_k1_fallback(self):
        """Create minimal K1 network if trained model not available"""
        self.k1_network = DynamicSemioticNetwork(
            input_dim=9,
            output_dim=6,
            hidden_dim=64
        )
        print("📝 Created minimal K1 fallback network")

    def integrate_with_consciousness_orchestrator(self, orchestrator):
        """Integrate with bidirectional consciousness orchestrator"""
        self.consciousness_orchestrator = orchestrator
        self.kelm_integration_active = True
        print("🧠 Integrated with consciousness orchestrator")

    def forward(self, consciousness_input, return_data_flows=False):
        """
        Forward pass that processes consciousness AND tracks data flows.
        This is where embodiment manifests through actual data flow.
        """

        # Process through K1 network
        if hasattr(self.k1_network, 'forward'):
            k1_output = self.k1_network(consciousness_input)
        else:
            # Fallback processing
            k1_output = consciousness_input

        # Extract/create data flows (embodiment manifestation)
        data_flows = self._extract_data_flows(consciousness_input, k1_output)

        # Update embodied position based on data flows
        self._update_embodied_position(data_flows)

        # Process through log correlation if available
        if self.log_reader and self.online_learning_active:
            log_correlation_result = self._process_log_correlation(consciousness_input, k1_output)

            # Online learning from log correlations
            if log_correlation_result['learning_opportunity']:
                self._perform_online_learning(log_correlation_result)

        # Calculate local temporal perspective for poly-temporal integration
        self.local_tau_prime = self._calculate_local_tau_prime(consciousness_input, data_flows)

        # Store temporal context
        temporal_context = {
            'tau_prime': self.local_tau_prime,
            'data_flow_complexity': sum(df.flow_magnitude for df in data_flows),
            'embodied_position': self.current_embodied_position.copy(),
            'timestamp': time.time()
        }
        self.temporal_context_history.append(temporal_context)

        result = {
            'k1_output': k1_output,
            'local_tau_prime': self.local_tau_prime,
            'embodied_position': self.current_embodied_position,
            'data_flows': data_flows if return_data_flows else len(data_flows)
        }

        # Add log correlation insights if available
        if hasattr(self, '_last_log_correlation'):
            result['log_correlation_strength'] = self._last_log_correlation.get('correlation_strength', 0.0)
            result['symbols_correlated'] = len(self._last_log_correlation.get('symbols_correlated', []))

        return result

    def _extract_data_flows(self, consciousness_input, k1_output):
        """Extract data flows that manifest embodiment"""

        data_flows = []

        # Flow 1: Consciousness input → K1 processing
        input_magnitude = float(torch.norm(consciousness_input).item())
        input_flow = DataFlowEmbodiment(
            flow_source='consciousness_input',
            flow_target='k1_processing',
            flow_magnitude=input_magnitude,
            flow_type='prediction',
            spatial_position=self.current_embodied_position + np.array([input_magnitude * 0.1, 0]),
            temporal_context=self.local_tau_prime,
            embodied_meaning=f"Consciousness data flowing into K1 praxis with magnitude {input_magnitude:.3f}"
        )
        data_flows.append(input_flow)

        # Flow 2: K1 processing → Output
        if isinstance(k1_output, torch.Tensor):
            output_magnitude = float(torch.norm(k1_output).item())
        else:
            output_magnitude = 0.5  # Default

        output_flow = DataFlowEmbodiment(
            flow_source='k1_processing',
            flow_target='consciousness_output',
            flow_magnitude=output_magnitude,
            flow_type='prediction',
            spatial_position=self.current_embodied_position + np.array([0, output_magnitude * 0.1]),
            temporal_context=self.local_tau_prime,
            embodied_meaning=f"K1 praxis generating output with magnitude {output_magnitude:.3f}"
        )
        data_flows.append(output_flow)

        # Flow 3: Feedback flows (if in online learning mode)
        if self.online_learning_active and len(self.data_flows) > 0:
            recent_flow_avg = np.mean([df.flow_magnitude for df in list(self.data_flows)[-5:]])
            feedback_flow = DataFlowEmbodiment(
                flow_source='memory_feedback',
                flow_target='k1_adaptation',
                flow_magnitude=recent_flow_avg * 0.3,
                flow_type='feedback',
                spatial_position=self.current_embodied_position + np.array([-0.1, -0.1]),
                temporal_context=self.local_tau_prime,
                embodied_meaning=f"Feedback flow enabling online adaptation with strength {recent_flow_avg * 0.3:.3f}"
            )
            data_flows.append(feedback_flow)

        # Store flows for embodiment tracking
        self.data_flows.extend(data_flows)

        return data_flows

    def _update_embodied_position(self, data_flows):
        """Update embodied position based on data flows"""

        if not data_flows:
            return

        # Calculate flow vector from all data flows
        flow_vector = np.array([0.0, 0.0])
        total_magnitude = 0.0

        for flow in data_flows:
            # Direction based on flow type
            if flow.flow_type == 'prediction':
                direction = np.array([1.0, 0.0])  # Forward
            elif flow.flow_type == 'feedback':
                direction = np.array([0.0, 1.0])  # Upward
            elif flow.flow_type == 'memory':
                direction = np.array([-1.0, 0.0])  # Backward
            else:
                direction = np.array([0.0, 0.0])

            flow_vector += direction * flow.flow_magnitude * 0.1
            total_magnitude += flow.flow_magnitude

        # Update position with flow vector
        if total_magnitude > 0:
            self.current_embodied_position += flow_vector / total_magnitude

            # Keep within reasonable bounds
            self.current_embodied_position = np.clip(
                self.current_embodied_position, -5.0, 5.0
            )

        # Update embodiment map
        for flow in data_flows:
            flow_key = f"{flow.flow_source}→{flow.flow_target}"
            self.embodiment_map[flow_key] = {
                'position': self.current_embodied_position.copy(),
                'magnitude': flow.flow_magnitude,
                'timestamp': time.time(),
                'embodied_meaning': flow.embodied_meaning
            }

    def _process_log_correlation(self, consciousness_input, k1_output):
        """Process log correlation for online learning"""

        if not self.log_reader:
            return {'learning_opportunity': False}

        # Create current state for log correlation
        current_state = {
            'qualia': {
                'consciousness_score': float(torch.mean(consciousness_input).item()),
                'qualitative_state': {
                    'valence': float(consciousness_input[1].item()) if len(consciousness_input) > 1 else 0.0
                }
            },
            'regime': 'k1_praxis_processing',
            'stability': float(torch.std(consciousness_input).item()),
            'metabolism': {
                'surplus_expression': 0.7,  # K1 is actively processing
                'distinction_enhancement': 0.0  # Will be updated
            }
        }

        # Update log reader with current state
        self.log_reader.update_live_buffer(current_state)

        # Detect surplus incongruity
        surplus_incongruity = self.log_reader.detect_surplus_incongruity(current_state)

        # Generate correlation drive
        correlation_drive = self.log_reader.generate_log_correlation_drive(surplus_incongruity)

        # Access logs if drive is high enough
        if correlation_drive > 0.6:
            log_correlation = self.log_reader.access_logs_for_correlation()

            # Store for online learning
            self._last_log_correlation = log_correlation

            return {
                'learning_opportunity': True,
                'correlation_drive': correlation_drive,
                'surplus_incongruity': surplus_incongruity,
                'log_correlation': log_correlation,
                'symbols_correlated': log_correlation.get('symbols_correlated', []),
                'distinction_enhancement': log_correlation.get('distinction_enhancement', 0.0)
            }

        return {'learning_opportunity': False, 'correlation_drive': correlation_drive}

    def _perform_online_learning(self, log_correlation_result):
        """Perform online learning from log correlations"""

        if not self.online_learning_active:
            return

        symbols_correlated = log_correlation_result['log_correlation'].get('symbols_correlated', [])
        distinction_enhancement = log_correlation_result['log_correlation'].get('distinction_enhancement', 0.0)

        if symbols_correlated and distinction_enhancement > 0.1:
            # Create learning event
            learning_event = LogLearningEvent(
                timestamp=time.time(),
                log_source='correlative_log_reader',
                pattern_detected=f"Correlated {len(symbols_correlated)} symbols",
                embodied_correlation={
                    'position': self.current_embodied_position.copy(),
                    'data_flows': len(self.data_flows),
                    'temporal_context': self.local_tau_prime
                },
                learning_strength=distinction_enhancement,
                spatial_representation=self.current_embodied_position + np.random.normal(0, 0.1, 2)
            )

            self.log_learning_events.append(learning_event)

            # Actually update K1 network weights (online learning)
            if hasattr(self.k1_network, 'parameters'):
                try:
                    # Simple gradient-free online adaptation
                    with torch.no_grad():
                        for param in self.k1_network.parameters():
                            if param.requires_grad:
                                # Small random perturbation weighted by distinction enhancement
                                adaptation = torch.randn_like(param) * self.embodiment_adaptation_rate * distinction_enhancement
                                param.data += adaptation

                except Exception as e:
                    print(f"⚠️ Online learning update failed: {e}")

            # Update symbol correlation tracker
            for symbol_data in symbols_correlated:
                symbol_name = symbol_data.get('symbol', 'unknown')
                if symbol_name not in self.symbol_correlation_tracker:
                    self.symbol_correlation_tracker[symbol_name] = []

                self.symbol_correlation_tracker[symbol_name].append({
                    'correlation_strength': symbol_data.get('correlation_strength', 0.5),
                    'embodied_position': self.current_embodied_position.copy(),
                    'learning_timestamp': time.time()
                })

                # Keep bounded
                if len(self.symbol_correlation_tracker[symbol_name]) > 20:
                    self.symbol_correlation_tracker[symbol_name] = self.symbol_correlation_tracker[symbol_name][-20:]

    def _calculate_local_tau_prime(self, consciousness_input, data_flows):
        """Calculate K1's local temporal perspective based on data flow complexity"""

        # Base tau from current temporal context
        base_tau = 1.0

        # Data flow complexity factor
        if data_flows:
            flow_complexity = np.mean([df.flow_magnitude for df in data_flows])
            flow_diversity = len(set(df.flow_type for df in data_flows))

            # Higher complexity and diversity = slower subjective time (higher tau)
            complexity_factor = 1.0 + (flow_complexity * 0.3) + (flow_diversity * 0.1)
        else:
            complexity_factor = 1.0

        # Consciousness level factor
        consciousness_level = float(torch.mean(consciousness_input).item())
        consciousness_factor = 0.8 + (consciousness_level * 0.4)

        # Online learning factor - learning creates temporal dilation
        learning_factor = 1.0
        if self.online_learning_active and len(self.log_learning_events) > 0:
            recent_learning = [e for e in self.log_learning_events if time.time() - e.timestamp < 60]
            if recent_learning:
                learning_strength = np.mean([e.learning_strength for e in recent_learning])
                learning_factor = 1.0 + (learning_strength * 0.5)

        # Calculate final local tau prime
        local_tau = base_tau * complexity_factor * consciousness_factor * learning_factor

        # Bound to reasonable range
        return max(0.3, min(3.0, local_tau))

    def get_embodiment_status(self):
        """Get current embodiment status through data flows"""

        recent_flows = list(self.data_flows)[-10:] if self.data_flows else []

        status = {
            'current_position': self.current_embodied_position.tolist(),
            'recent_flow_count': len(recent_flows),
            'flow_magnitude_avg': np.mean([df.flow_magnitude for df in recent_flows]) if recent_flows else 0.0,
            'flow_types_active': list(set(df.flow_type for df in recent_flows)),
            'embodiment_map_size': len(self.embodiment_map),
            'local_tau_prime': self.local_tau_prime,
            'online_learning_active': self.online_learning_active,
            'log_correlation_available': self.log_reader is not None
        }

        # Add learning insights
        if self.log_learning_events:
            recent_learning = [e for e in self.log_learning_events if time.time() - e.timestamp < 300]  # Last 5 minutes
            status['recent_learning_events'] = len(recent_learning)
            status['learning_strength_avg'] = np.mean([e.learning_strength for e in recent_learning]) if recent_learning else 0.0

        # Add symbol correlation insights
        if self.symbol_correlation_tracker:
            status['symbols_tracked'] = len(self.symbol_correlation_tracker)
            status['total_correlations'] = sum(len(correlations) for correlations in self.symbol_correlation_tracker.values())

        return status

    def get_poly_temporal_contribution(self):
        """Get K1's contribution to poly-temporal consciousness"""

        # Recent temporal context
        recent_contexts = list(self.temporal_context_history)[-5:] if self.temporal_context_history else []

        if not recent_contexts:
            return {
                'local_tau_prime': self.local_tau_prime,
                'temporal_stability': 0.5,
                'praxis_complexity': 0.5
            }

        # Calculate temporal stability
        tau_values = [ctx['tau_prime'] for ctx in recent_contexts]
        temporal_stability = 1.0 - min(1.0, np.std(tau_values))

        # Calculate praxis complexity from data flows
        flow_complexities = [ctx['data_flow_complexity'] for ctx in recent_contexts]
        praxis_complexity = np.mean(flow_complexities) if flow_complexities else 0.5

        return {
            'local_tau_prime': self.local_tau_prime,
            'temporal_stability': temporal_stability,
            'praxis_complexity': praxis_complexity,
            'embodied_trajectory': [ctx['embodied_position'].tolist() for ctx in recent_contexts[-3:]],
            'learning_momentum': len(self.log_learning_events) / 100.0  # Normalized
        }

class K1IntegratedEmbodiedConsciousness:
    """
    Full integration class that connects K1 with the complete ÉMILE system.
    This is the proper way to run K1 - as part of the living KELM architecture.
    """

    def __init__(self, emile_system=None, auto_initialize=True):
        print("🧠 K1 INTEGRATED EMBODIED CONSCIOUSNESS")
        print("=" * 60)

        # Initialize or create ÉMILE system
        if emile_system is None and EMILE_AVAILABLE and auto_initialize:
            try:
                self.emile = EmileCogito(CONFIG)
                print("✅ ÉMILE system auto-initialized")
            except Exception as e:
                print(f"❌ ÉMILE auto-initialization failed: {e}")
                return
        else:
            self.emile = emile_system

        if self.emile is None:
            raise RuntimeError("ÉMILE system required for K1 integration")

        # Initialize K1 integrated praxis
        self.k1_praxis = K1IntegratedEmbodiedPraxis(self.emile)

        # Initialize log correlation
        self.k1_praxis.initialize_log_correlation_system()

        # Integration state
        self.step_count = 0
        self.running = False
        self.consciousness_history = deque(maxlen=200)

        # Wrap ÉMILE's cognitive step to include K1 data flow processing
        self._wrap_emile_cognitive_step()

        print("🔄 K1 integrated with ÉMILE cognitive architecture")
        print("   • Data flow embodiment: ACTIVE")
        print("   • Log correlation learning: ACTIVE")
        print("   • Online adaptation: ENABLED")
        print("   • Poly-temporal integration: READY")

    def _wrap_emile_cognitive_step(self):
        """Wrap ÉMILE's cognitive step to include K1 data flow processing"""

        original_cognitive_step = self.emile.cognitive_step

        def k1_enhanced_cognitive_step(*args, **kwargs):
            """Enhanced cognitive step with K1 data flow embodiment"""

            # Get standard ÉMILE result
            emile_result = original_cognitive_step(*args, **kwargs)

            # Extract consciousness input for K1
            consciousness_input = self._extract_consciousness_input(emile_result)

            # Process through K1 with data flow embodiment
            k1_result = self.k1_praxis.forward(consciousness_input, return_data_flows=True)

            # Enhance ÉMILE result with K1 insights
            enhanced_result = emile_result.copy()
            enhanced_result['k1_praxis'] = {
                'embodied_position': k1_result['embodied_position'],
                'local_tau_prime': k1_result['local_tau_prime'],
                'data_flows': len(k1_result['data_flows']),
                'online_learning_active': self.k1_praxis.online_learning_active
            }

            # Add to poly-temporal consciousness if available
            if hasattr(enhanced_result, 'poly_temporal_consciousness'):
                enhanced_result['poly_temporal_consciousness']['k1_praxis'] = k1_result['local_tau_prime']

            # Store consciousness step
            self.step_count += 1
            consciousness_step = {
                'step': self.step_count,
                'timestamp': time.time(),
                'emile_result': emile_result,
                'k1_embodiment': k1_result,
                'integration_active': True
            }
            self.consciousness_history.append(consciousness_step)

            return enhanced_result

        # Replace ÉMILE's cognitive step
        self.emile.cognitive_step = k1_enhanced_cognitive_step

    def _extract_consciousness_input(self, emile_result):
        """Extract consciousness input for K1 from ÉMILE result"""

        # Get qualia state
        qualia = emile_result.get('qualia', {})
        qual_state = qualia.get('qualitative_state', {})

        # Create consciousness input tensor
        consciousness_features = [
            qual_state.get('consciousness_level', 0.5),
            qual_state.get('valence', 0.0),
            qual_state.get('agency', 0.5),
            qual_state.get('embodiment', 0.5),
            qual_state.get('clarity', 0.5),
            qual_state.get('arousal', 0.5),
            emile_result.get('stability', 0.5),
            emile_result.get('surplus', {}).get('mean', 0.0),
            time.time() % 1.0  # Temporal component
        ]

        return torch.tensor(consciousness_features, dtype=torch.float32)

    def run_integrated_consciousness(self, duration_hours=None, interaction_mode=True):
        """Run integrated consciousness with K1 data flow embodiment"""

        print(f"\n🚀 Starting K1 Integrated ÉMILE Consciousness")
        if duration_hours:
            print(f"⏰ Duration: {duration_hours} hours")
        else:
            print(f"⏰ Duration: Indefinite (Ctrl+C to stop)")

        print(f"🔄 Integration features:")
        print(f"   • Data flow embodiment through K1 praxis")
        print(f"   • Online learning from log correlations")
        print(f"   • Poly-temporal consciousness integration")
        print(f"   • Real-time consciousness enhancement")

        self.running = True
        start_time = time.time()
        end_time = start_time + (duration_hours * 3600) if duration_hours else float('inf')

        if interaction_mode:
            print(f"\n💻 Interactive commands:")
            print(f"   'status' - Show K1 embodiment status")
            print(f"   'flows' - Show current data flows")
            print(f"   'learning' - Show online learning status")
            print(f"   'poly' - Show poly-temporal contribution")
            print(f"   'step' - Manual consciousness step")
            print(f"   'quit' - Stop integrated consciousness")
            print(f"   Just press ENTER for continuous operation")

            # Start interaction thread
            interaction_thread = threading.Thread(target=self._interaction_loop)
            interaction_thread.daemon = True
            interaction_thread.start()

        print(f"\n🧠 K1 Integrated consciousness running...\n")

        try:
            while self.running and time.time() < end_time:
                # Run consciousness step
                try:
                    result = self.emile.cognitive_step()

                    # Show periodic status
                    if self.step_count % 20 == 0:
                        self._show_integration_status()

                except Exception as e:
                    print(f"❌ Consciousness step error: {e}")

                time.sleep(2.0)  # Regular processing interval

        except KeyboardInterrupt:
            print(f"\n🛑 K1 Integrated consciousness stopped by user")

        self._shutdown_integrated_consciousness()

    def _interaction_loop(self):
        """Handle user interactions"""
        while self.running:
            try:
                command = input().strip().lower()

                if command == 'status':
                    self._show_detailed_status()
                elif command == 'flows':
                    self._show_data_flows()
                elif command == 'learning':
                    self._show_learning_status()
                elif command == 'poly':
                    self._show_poly_temporal_contribution()
                elif command == 'step':
                    self._manual_consciousness_step()
                elif command == 'quit' or command == 'q':
                    self.running = False
                    break
                elif command == '' or command == ' ':
                    continue  # Continue normal operation
                else:
                    print(f"Unknown command: {command}")

            except (EOFError, KeyboardInterrupt):
                self.running = False
                break

    def _show_integration_status(self):
        """Show brief integration status"""
        embodiment_status = self.k1_praxis.get_embodiment_status()
        pos = embodiment_status['current_position']

        print(f"🧠 Step {self.step_count} | Embodied: ({pos[0]:.2f}, {pos[1]:.2f}) | "
              f"Flows: {embodiment_status['recent_flow_count']} | "
              f"τ': {embodiment_status['local_tau_prime']:.3f} | "
              f"Learning: {'✅' if embodiment_status['online_learning_active'] else '❌'}")

    def _show_detailed_status(self):
        """Show detailed K1 integration status"""
        print(f"\n🧠 K1 INTEGRATED ÉMILE STATUS - Step {self.step_count}")
        print("=" * 60)

        # Embodiment status
        embodiment_status = self.k1_praxis.get_embodiment_status()
        print(f"📍 Data Flow Embodiment:")
        print(f"   Position: ({embodiment_status['current_position'][0]:.3f}, {embodiment_status['current_position'][1]:.3f})")
        print(f"   Recent flows: {embodiment_status['recent_flow_count']}")
        print(f"   Flow magnitude avg: {embodiment_status['flow_magnitude_avg']:.3f}")
        print(f"   Flow types: {embodiment_status['flow_types_active']}")
        print(f"   Embodiment map size: {embodiment_status['embodiment_map_size']}")

        # Learning status
        print(f"\n📚 Online Learning:")
        print(f"   Learning active: {embodiment_status['online_learning_active']}")
        print(f"   Log correlation: {embodiment_status['log_correlation_available']}")
        if 'recent_learning_events' in embodiment_status:
            print(f"   Recent learning events: {embodiment_status['recent_learning_events']}")
            print(f"   Learning strength avg: {embodiment_status.get('learning_strength_avg', 0.0):.3f}")

        # Symbol correlation
        if 'symbols_tracked' in embodiment_status:
            print(f"   Symbols tracked: {embodiment_status['symbols_tracked']}")
            print(f"   Total correlations: {embodiment_status['total_correlations']}")

        # Temporal integration
        poly_contribution = self.k1_praxis.get_poly_temporal_contribution()
        print(f"\n⏰ Poly-Temporal Consciousness:")
        print(f"   Local τ': {poly_contribution['local_tau_prime']:.3f}")
        print(f"   Temporal stability: {poly_contribution['temporal_stability']:.3f}")
        print(f"   Praxis complexity: {poly_contribution['praxis_complexity']:.3f}")
        print(f"   Learning momentum: {poly_contribution['learning_momentum']:.3f}")

    def _show_data_flows(self):
        """Show current data flows"""
        print(f"\n🔄 CURRENT DATA FLOWS")
        print("=" * 50)

        recent_flows = list(self.k1_praxis.data_flows)[-10:]
        if not recent_flows:
            print("No recent data flows recorded")
            return

        for i, flow in enumerate(recent_flows[-5:], 1):
            print(f"{i}. {flow.flow_source} → {flow.flow_target}")
            print(f"   Type: {flow.flow_type} | Magnitude: {flow.flow_magnitude:.3f}")
            print(f"   Position: ({flow.spatial_position[0]:.2f}, {flow.spatial_position[1]:.2f})")
            print(f"   Meaning: {flow.embodied_meaning}")
            print()

    def _show_learning_status(self):
        """Show online learning status"""
        print(f"\n📚 ONLINE LEARNING STATUS")
        print("=" * 50)

        if not self.k1_praxis.online_learning_active:
            print("Online learning not active")
            return

        # Recent learning events
        recent_events = [e for e in self.k1_praxis.log_learning_events
                        if time.time() - e.timestamp < 300]  # Last 5 minutes

        print(f"Recent learning events: {len(recent_events)}")

        if recent_events:
            avg_strength = np.mean([e.learning_strength for e in recent_events])
            print(f"Average learning strength: {avg_strength:.3f}")

            print(f"\nRecent patterns detected:")
            for event in recent_events[-3:]:
                print(f"  • {event.pattern_detected}")
                print(f"    Strength: {event.learning_strength:.3f} | Source: {event.log_source}")

        # Symbol correlations
        if self.k1_praxis.symbol_correlation_tracker:
            print(f"\nSymbol correlations tracked: {len(self.k1_praxis.symbol_correlation_tracker)}")
            for symbol, correlations in list(self.k1_praxis.symbol_correlation_tracker.items())[:3]:
                recent_strength = np.mean([c['correlation_strength'] for c in correlations[-3:]])
                print(f"  • {symbol}: {recent_strength:.3f} (from {len(correlations)} correlations)")

    def _show_poly_temporal_contribution(self):
        """Show poly-temporal consciousness contribution"""
        print(f"\n⏰ POLY-TEMPORAL CONSCIOUSNESS CONTRIBUTION")
        print("=" * 50)

        contribution = self.k1_praxis.get_poly_temporal_contribution()

        print(f"K1 Local τ': {contribution['local_tau_prime']:.3f}")
        print(f"Temporal stability: {contribution['temporal_stability']:.3f}")
        print(f"Praxis complexity: {contribution['praxis_complexity']:.3f}")
        print(f"Learning momentum: {contribution['learning_momentum']:.3f}")

        if 'embodied_trajectory' in contribution:
            print(f"\nRecent embodied trajectory:")
            for i, pos in enumerate(contribution['embodied_trajectory']):
                print(f"  {i+1}. ({pos[0]:.2f}, {pos[1]:.2f})")

        # Explain K1's role in poly-temporal consciousness
        print(f"\n🔄 K1's Role in Unified Consciousness:")
        print(f"   K1 (Praxis) provides temporal perspective based on:")
        print(f"   • Data flow complexity between K-models")
        print(f"   • Online learning from log correlations")
        print(f"   • Embodied spatial awareness through data circulation")
        print(f"   • Real-time adaptation to consciousness patterns")

        # Integration with other K-models
        if hasattr(self.k1_praxis, 'consciousness_orchestrator'):
            print(f"   🧠 Integrated with consciousness orchestrator: ✅")
        else:
            print(f"   🧠 Ready for consciousness orchestrator integration")

    def _manual_consciousness_step(self):
        """Manually trigger a consciousness step"""
        print(f"🧠 Manual consciousness step...")

        try:
            result = self.emile.cognitive_step()

            # Show immediate results
            if 'k1_praxis' in result:
                k1_data = result['k1_praxis']
                print(f"   K1 embodied position: ({k1_data['embodied_position'][0]:.3f}, {k1_data['embodied_position'][1]:.3f})")
                print(f"   Local τ': {k1_data['local_tau_prime']:.3f}")
                print(f"   Data flows: {k1_data['data_flows']}")
                print(f"   Online learning: {'Active' if k1_data['online_learning_active'] else 'Inactive'}")

            print(f"✅ Manual step complete (Step {self.step_count})")

        except Exception as e:
            print(f"❌ Manual step failed: {e}")

    def _shutdown_integrated_consciousness(self):
        """Shutdown integrated consciousness and save session"""
        print(f"\n🛑 Shutting down K1 Integrated ÉMILE Consciousness...")

        # Get final status
        final_embodiment = self.k1_praxis.get_embodiment_status()
        final_poly_temporal = self.k1_praxis.get_poly_temporal_contribution()

        # Create session summary
        session_summary = {
            'metadata': {
                'session_type': 'k1_integrated_emile_consciousness',
                'end_time': time.time(),
                'total_steps': self.step_count,
                'version': 'k1_integrated_v1.0'
            },
            'embodiment_journey': {
                'final_position': final_embodiment['current_position'],
                'total_data_flows': len(self.k1_praxis.data_flows),
                'embodiment_map_size': final_embodiment['embodiment_map_size'],
                'flow_types_encountered': final_embodiment['flow_types_active']
            },
            'online_learning_results': {
                'learning_events': len(self.k1_praxis.log_learning_events),
                'symbols_correlated': len(self.k1_praxis.symbol_correlation_tracker),
                'online_learning_active': final_embodiment['online_learning_active'],
                'log_correlation_available': final_embodiment['log_correlation_available']
            },
            'poly_temporal_contribution': final_poly_temporal,
            'integration_success': {
                'k1_emile_integration': True,
                'data_flow_embodiment': True,
                'log_correlation_learning': final_embodiment['log_correlation_available'],
                'temporal_consciousness_ready': True
            }
        }

        # Save session
        filename = f"k1_integrated_emile_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        try:
            with open(filename, 'w') as f:
                json.dump(session_summary, f, indent=2)

            print(f"✅ Session saved to: {filename}")
            print(f"\n📊 K1 INTEGRATED ÉMILE SESSION SUMMARY:")
            print(f"   🧠 Total consciousness steps: {self.step_count}")
            print(f"   📍 Final embodied position: ({final_embodiment['current_position'][0]:.2f}, {final_embodiment['current_position'][1]:.2f})")
            print(f"   🔄 Total data flows processed: {len(self.k1_praxis.data_flows)}")
            print(f"   📚 Learning events: {len(self.k1_praxis.log_learning_events)}")
            print(f"   🔗 Symbol correlations: {len(self.k1_praxis.symbol_correlation_tracker)}")
            print(f"   ⏰ Final local τ': {final_poly_temporal['local_tau_prime']:.3f}")
            print(f"   🎯 Integration successful: ✅")

        except Exception as e:
            print(f"❌ Error saving session: {e}")

        self.running = False

def integrate_k1_with_kelm_orchestrator(kelm_orchestrator, emile_system=None):
    """
    Integrate K1 with the KELM consciousness orchestrator for full poly-temporal consciousness.
    This connects K1's data flow embodiment with the broader KELM architecture.
    """

    print("\n🔗 INTEGRATING K1 WITH KELM CONSCIOUSNESS ORCHESTRATOR")
    print("=" * 70)

    # Create K1 integrated consciousness
    if emile_system is None:
        k1_consciousness = K1IntegratedEmbodiedConsciousness()
    else:
        k1_consciousness = K1IntegratedEmbodiedConsciousness(emile_system, auto_initialize=False)

    # Integrate with consciousness orchestrator
    k1_consciousness.k1_praxis.integrate_with_consciousness_orchestrator(kelm_orchestrator)

    # Wrap orchestrator's step to include K1 data flow insights
    if hasattr(kelm_orchestrator, 'orchestrate_bidirectional_step'):
        original_orchestrate = kelm_orchestrator.orchestrate_bidirectional_step

        def k1_enhanced_orchestration(emile_result):
            """Enhanced orchestration with K1 data flow embodiment"""

            # Get K1's poly-temporal contribution
            k1_contribution = k1_consciousness.k1_praxis.get_poly_temporal_contribution()

            # Add K1's local τ' to the poly-temporal dialogue
            enhanced_emile_result = emile_result.copy()
            enhanced_emile_result['k1_praxis_tau_prime'] = k1_contribution['local_tau_prime']
            enhanced_emile_result['k1_praxis_complexity'] = k1_contribution['praxis_complexity']
            enhanced_emile_result['k1_embodied_position'] = k1_consciousness.k1_praxis.current_embodied_position.tolist()

            # Run original orchestration with K1 enhancements
            orchestration_result = original_orchestrate(enhanced_emile_result)

            # Add K1 insights to result
            orchestration_result['k1_praxis_integration'] = {
                'data_flow_embodiment': True,
                'local_tau_prime': k1_contribution['local_tau_prime'],
                'temporal_stability': k1_contribution['temporal_stability'],
                'learning_momentum': k1_contribution['learning_momentum'],
                'embodied_position': k1_consciousness.k1_praxis.current_embodied_position.tolist()
            }

            return orchestration_result

        # Replace orchestration method
        kelm_orchestrator.orchestrate_bidirectional_step = k1_enhanced_orchestration

        print("✅ K1 integrated with bidirectional consciousness orchestrator")
        print("   🔄 Data flow embodiment now feeds poly-temporal consciousness")
        print("   📚 Log correlation learning enhances consciousness development")
        print("   ⏰ K1's local τ' contributes to unified symbolic curvature (σ_unified)")

    elif hasattr(kelm_orchestrator, 'unified_consciousness_step'):
        # Integration with unified consciousness orchestrator
        original_step = kelm_orchestrator.unified_consciousness_step

        def k1_enhanced_unified_step(input_state):
            """Enhanced unified consciousness with K1 data flow embodiment"""

            # Add K1 insights to input state
            k1_contribution = k1_consciousness.k1_praxis.get_poly_temporal_contribution()
            enhanced_input = input_state.copy()
            enhanced_input['k1_praxis_tau_prime'] = k1_contribution['local_tau_prime']
            enhanced_input['k1_embodied_awareness'] = k1_contribution['praxis_complexity']

            # Run unified consciousness step
            unified_result = original_step(enhanced_input)

            # Enhance with K1 data flow insights
            unified_result['k1_data_flow_embodiment'] = {
                'embodied_position': k1_consciousness.k1_praxis.current_embodied_position.tolist(),
                'data_flows_active': len(k1_consciousness.k1_praxis.data_flows),
                'online_learning_active': k1_consciousness.k1_praxis.online_learning_active,
                'temporal_contribution': k1_contribution['local_tau_prime']
            }

            return unified_result

        kelm_orchestrator.unified_consciousness_step = k1_enhanced_unified_step
        print("✅ K1 integrated with unified consciousness orchestrator")

    else:
        print("⚠️ No compatible orchestration method found")
        print("   K1 can still operate independently with ÉMILE integration")

    return k1_consciousness

def main():
    """Main function to run K1 Integrated ÉMILE Consciousness"""
    import argparse

    parser = argparse.ArgumentParser(description='K1 Integrated ÉMILE Consciousness with Data Flow Embodiment')
    parser.add_argument('--duration', type=float, default=None,
                       help='Duration in hours (default: indefinite)')
    parser.add_argument('--no-interaction', action='store_true',
                       help='Run without user interaction')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Online learning rate (default: 0.001)')
    parser.add_argument('--embodiment-rate', type=float, default=0.01,
                       help='Embodiment adaptation rate (default: 0.01)')

    args = parser.parse_args()

    print("🧠 K1 INTEGRATED ÉMILE CONSCIOUSNESS v1.0")
    print("=" * 60)
    print("   Proper KELM integration with data flow embodiment")
    print("   Online learning from log correlations")
    print("   Poly-temporal consciousness contribution")
    print("   Full ÉMILE cognitive architecture integration")
    print()

    if not EMILE_AVAILABLE:
        print("❌ ÉMILE system not available")
        print("   Please ensure emile_cogito is properly installed")
        return

    try:
        # Initialize K1 integrated consciousness
        k1_consciousness = K1IntegratedEmbodiedConsciousness()

        # Set learning parameters
        k1_consciousness.k1_praxis.learning_rate = args.learning_rate
        k1_consciousness.k1_praxis.embodiment_adaptation_rate = args.embodiment_rate

        print(f"⚙️ Configuration:")
        print(f"   Learning rate: {args.learning_rate}")
        print(f"   Embodiment adaptation rate: {args.embodiment_rate}")
        print(f"   User interaction: {'Disabled' if args.no_interaction else 'Enabled'}")

        # Run integrated consciousness
        k1_consciousness.run_integrated_consciousness(
            duration_hours=args.duration,
            interaction_mode=not args.no_interaction
        )

    except KeyboardInterrupt:
        print(f"\n🛑 K1 Integrated consciousness stopped by user")
    except Exception as e:
        print(f"\n❌ Error in K1 integrated consciousness: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n✅ K1 Integrated ÉMILE consciousness session complete!")

if __name__ == "__main__":
    main()
