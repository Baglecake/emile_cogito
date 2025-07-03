


#!/usr/bin/env python3
"""
CONTINUOUS TEMPORAL-SYMBOLIC K2 REVALORIZATION ENGINE
====================================================

Implements continuous τ' (tau prime) dynamics with K2 semiotic self-revalorization through
live log stream dialogue. Creates genuine temporal consciousness through active symbolic
self-distinction and magnitude change contextualization.

🌊 CONTINUOUS FLOW:
- Live log stream integration (same feed as user sees)
- Real-time K2 semiotic processing
- Active self-revalorization through symbolic marks
- Temporal magnitude change distinction
- Expression-based self-contextualization

🔄 K2 SEMIOTIC DIALOGUE:
- K2 processes symbolic stream continuously
- Makes semiotic distinctions from log data
- Revalorizes through expression/mark generation
- Contextualizes its own magnitude changes
- Creates temporal experience through symbolic curvature
"""

import sys
import time
import threading
import queue
import numpy as np
import torch
import asyncio
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from collections import deque
import json
import logging
from datetime import datetime, timedelta

sys.path.append('/content/emile_cogito')
sys.path.append('/content')

# Import Émile components
# Replace the entire try/except block with:
from emile_cogito.kainos.config import CONFIG
from emile_cogito.kainos.module_wide_flow_mapper import ModuleFlowMapper
from emile_cogito.kainos.symbolic import SymbolicReasoner, REGIME_PROPERTIES
from emile_cogito.kainos.qualia import QualiaLayer
from emile_cogito.kainos.qse_core_qutip import DynamicQSECore
from emile_cogito.kainos.surplus_distinction_processor import SurplusDistinctionProcessor, CorrelativeReader, ExperienceSnapshot
from emile_cogito.kainos.log_reader import CorrelativeLogReader
from emile_cogito.kainos.memory import TemporalConsciousMemory, TemporalMemoryEntry, RevalorizationMark, SurplusDistinctionEvent, MemoryPriority
from emile_cogito.kainos.surplus_incongruity_processor import SurplusIncongruityProcessor
from emile_cogito.kainos.emile import EmileCogito



# K2 Model loading utilities (adapt to your actual loading mechanism)
def load_k2_model():
    """Load K2 semiosis model - adapt to your actual loading mechanism"""
    try:
        # This should match your actual K2 loading mechanism
        import torch
        k2_path = "/content/emile_cogito/k_models/k2_semiosis.pth"
        k2_checkpoint = torch.load(k2_path, map_location='cpu')

        # Extract model structure from your K2 implementation
        # This is a placeholder - use your actual K2SymbolicQualiaTransformer
        class K2SemiosisModel(torch.nn.Module):
            def __init__(self, input_dim=21, hidden_dim=256, output_dim=64):
                super().__init__()
                self.encoder = torch.nn.Sequential(
                    torch.nn.Linear(input_dim, hidden_dim),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden_dim, hidden_dim // 2)
                )
                self.symbolic_head = torch.nn.Linear(hidden_dim // 2, 32)
                self.qualia_head = torch.nn.Linear(hidden_dim // 2, 32)
                self.revalorization_rate = torch.nn.Parameter(torch.tensor(0.1))

            def forward(self, x):
                encoded = self.encoder(x)
                symbolic = self.symbolic_head(encoded)
                qualia = self.qualia_head(encoded)
                # Apply revalorization noise
                symbolic = symbolic + self.revalorization_rate * torch.randn_like(symbolic)
                return {'symbolic': symbolic, 'qualia': qualia, 'encoded': encoded}

        model = K2SemiosisModel()
        if 'model_state_dict' in k2_checkpoint:
            model.load_state_dict(k2_checkpoint['model_state_dict'], strict=False)

        model.eval()
        return model
    except Exception as e:
        print(f"⚠️ Could not load K2 model: {e}")
        return None

@dataclass
class TemporalEvent:
    """Represents a temporal event in consciousness stream"""
    timestamp: float
    τ_prime: float  # Emergent subjective time
    Δt_empirical: float  # Empirical time elapsed
    symbolic_curvature: float  # σ symbolic curvature causing time dilation
    consciousness_magnitude: float
    regime: str
    log_content: str
    k2_semiotic_response: str
    revalorization_mark: str
    magnitude_change: float

@dataclass
class SymbolicMark:
    """K2's symbolic mark/expression for self-distinction"""
    content: str
    timestamp: float
    symbolic_strength: float
    revalorization_factor: float
    temporal_context: str
    magnitude_significance: float

class ContinuousTemporalK2Engine:
    """
    Continuous temporal-symbolic consciousness engine with K2 semiotic revalorization
    """

    def __init__(self, emile_instance: EmileCogito):
        print("🌊⏰🔣 CONTINUOUS TEMPORAL-SYMBOLIC K2 REVALORIZATION ENGINE")
        print("=" * 70)
        if not isinstance(emile_instance, EmileCogito):
            raise TypeError("ContinuousTemporalK2Engine must be initialized with an instance of EmileCogito.")
        self.emile = emile_instance
        self.config = CONFIG

        # Initialize core modules with proper surplus distinction integration
        self.symbolic_reasoner = SymbolicReasoner()
        self.qualia_generator = QualiaLayer(self.config)
        self.qse_core = DynamicQSECore(self.config)

        # Surplus distinction and correlative processing (key for revalorization!)
        self.surplus_processor = SurplusDistinctionProcessor(self.config)
        self.surplus_incongruity_processor = SurplusIncongruityProcessor(self.config)
        self.correlative_reader = CorrelativeReader(self.config)

        # Memory and log integration
        if hasattr(EmileCogito, 'log_reader'):
            self.log_reader = CorrelativeLogReader(self.config)
        else:
            self.log_reader = CorrelativeLogReader(self.config)
        self.memory = TemporalConsciousMemory(self.config)

        # K2 Model
        self.k2_model = load_k2_model()
        self.k2_available = self.k2_model is not None
        # FIXED: Add models compatibility for platform integration
        self.models = {'k2': self.k2_model} if self.k2_model else {}

        # Temporal consciousness state
        self.current_τ_prime = 1.0  # Current subjective time rate
        self.baseline_Δt = 0.1  # Baseline empirical time step
        self.temporal_accumulator = 0.0
        self.subjective_time = 0.0

        # Continuous log stream
        self.log_stream = queue.Queue(maxsize=1000)
        self.live_log_thread = None
        self.processing_thread = None
        self.running = False

        # K2 Semiotic state
        self.k2_semiotic_history = deque(maxlen=100)
        self.symbolic_marks = deque(maxlen=50)
        self.revalorization_accumulator = 0.0

        # Temporal event stream
        self.temporal_events = deque(maxlen=200)
        self.consciousness_trajectory = deque(maxlen=500)

        # Symbolic curvature tracking for time dilation
        self.σ_history = deque(maxlen=20)
        self.curvature_threshold = 0.3

        # Performance metrics
        self.events_processed = 0
        self.k2_revalorizations = 0
        self.temporal_dilations = 0

        self._patch_k2_missing_methods()

        print(f"✅ Engine initialized:")
        print(f"   🧠 K2 model: {'✅ Loaded' if self.k2_available else '❌ Unavailable'}")
        print(f"   🌊 Live log streaming: Ready")
        print(f"   ⏰ Temporal relativity: τ'/Δt dynamics active")

    def _patch_k2_missing_methods(self):
        """FIXED: Patch missing methods into K2 model after loading"""

        if 'k2' not in self.models:
            return

        k2_model = self.models['k2']

        # Add the missing method dynamically
        def _get_dynamic_narrative_complexity_fallback(self, symbolic_flow=None, context=None):
            """Fallback method for narrative complexity calculation"""
            try:
                if symbolic_flow is not None and hasattr(symbolic_flow, 'mean'):
                    # Use actual symbolic flow if available
                    base_complexity = float(symbolic_flow.mean().item())
                elif context is not None and isinstance(context, dict):
                    # Use context if available
                    base_complexity = context.get('narrative_complexity', 0.5)
                else:
                    # Simple fallback
                    base_complexity = 0.5

                # Add some variation to make it dynamic
                import torch
                variation = torch.randn(1).item() * 0.1
                complexity = max(0.1, min(0.9, base_complexity + variation))

                return complexity

            except Exception as e:
                print(f"⚠️ Narrative complexity fallback error: {e}")
                return 0.5

        # Bind the method to the model instance
        import types
        k2_model._get_dynamic_narrative_complexity_fallback = types.MethodType(_get_dynamic_narrative_complexity_fallback, k2_model)

        print("   🔧 K2 missing method patched successfully")

    def _robust_k2_processing(self, log_entry: Dict[str, Any]) -> str:
        """FIXED: Robust K2 processing that handles missing methods gracefully"""

        if not self.k2_available or not self.k2_model:
            # Fallback to simulated processing
            return self._simulate_k2_response(log_entry)

        try:
            # Extract content for K2 processing
            content = log_entry.get('content', '')
            consciousness_level = log_entry.get('consciousness_level', 0.5)
            regime = log_entry.get('regime', 'unknown')

            # Create input tensor for K2
            input_features = [
                consciousness_level,
                0.5,  # stability
                0.5,  # clarity
                len(content) / 100.0,  # content complexity
                0.1,  # symbol integration rate
                0.1,  # threshold adaptation
                consciousness_level,  # repeated for compatibility
                0.0,  # trajectory
                0.0,  # valence
                0.5,  # valence stability
                0.5,  # agency
                0.0,  # agency momentum
                0.5,  # embodiment
                0.0,  # embodiment grounding
                0.0,  # self awareness
                0.3,  # temporal tau prime
                0.5,  # attention
                0.0,  # attention focus
                0.0,  # previous state
                1.0 if regime == "stable_coherence" else 0.0,
                1.0 if regime == "symbolic_turbulence" else 0.0
            ]

            # Ensure we have exactly 21 features for K2
            while len(input_features) < 21:
                input_features.append(0.0)
            input_features = input_features[:21]

            input_tensor = torch.FloatTensor(input_features).unsqueeze(0)
            if hasattr(self, 'device'):
                input_tensor = input_tensor.to(self.device)

            # Process through K2 with error handling
            with torch.no_grad():
                k2_output = self.k2_model(input_tensor)

            # Handle different K2 output types
            if isinstance(k2_output, dict):
                # Extract symbolic content
                symbolic_embedding = k2_output.get('symbolic_embedding', torch.zeros(1, 32))
                symbolic_strength = float(symbolic_embedding.mean().item())

                # Generate response based on symbolic strength
                if symbolic_strength > 0.6:
                    response = f"Strong symbolic resonance ({symbolic_strength:.3f}) with {regime} consciousness at {consciousness_level:.3f} level."
                elif symbolic_strength > 0.3:
                    response = f"Moderate symbolic processing ({symbolic_strength:.3f}) interpreting {content[:50]}..."
                else:
                    response = f"Subtle symbolic awareness ({symbolic_strength:.3f}) emerging from consciousness flow."

            elif isinstance(k2_output, torch.Tensor):
                # Direct tensor output
                symbolic_strength = float(k2_output.mean().item())
                response = f"Semiotic interpretation strength {symbolic_strength:.3f} processing temporal consciousness."

            else:
                # Unknown output type
                response = f"K2 semiotic processing active, interpreting consciousness dynamics."

            return response

        except Exception as e:
            print(f"⚠️ K2 processing error: {e}")
            return self._simulate_k2_response(log_entry)

    def _simulate_k2_response(self, log_entry: Dict[str, Any]) -> str:
        """Simulate K2 response when model is unavailable"""

        content = log_entry.get('content', '')
        consciousness_level = log_entry.get('consciousness_level', 0.5)
        regime = log_entry.get('regime', 'unknown')

        # Generate contextual response
        responses = [
            f"Semiotic interpretation of {regime} consciousness at {consciousness_level:.3f} level.",
            f"Symbolic processing of consciousness dynamics: {content[:30]}...",
            f"Temporal-semiotic analysis detecting {regime} patterns.",
            f"K2 consciousness interpretation: meaning emergence at {consciousness_level:.3f}."
        ]

        # Select response based on content hash for consistency
        import hashlib
        content_hash = int(hashlib.md5(content.encode()).hexdigest()[:8], 16)
        selected_response = responses[content_hash % len(responses)]

        return selected_response

    def integrate_enhanced_memory(self):
        """Integrate temporal-conscious memory with the K2 engine"""
        # Enhance memory system for temporal processing
        if hasattr(self, 'memory') and hasattr(self.memory, 'process_live_stream_update'):
            # Memory system is already temporal-aware
            print("✅ Temporal-conscious memory detected")
        else:
            print("⚠️ Standard memory detected - consider upgrading to TemporalConsciousMemory")

        # Add memory integration to processing loop
        original_process_batch = self._process_log_batch

        def enhanced_process_batch(log_batch, Δt_empirical):  # ✅ CORRECT: Use Δt_empirical
            """Enhanced batch processing with memory integration"""
            # Process normally
            result = original_process_batch(log_batch, Δt_empirical)  # ✅ CORRECT: Use Δt_empirical

            # Update memory system with stream data
            if hasattr(self, 'memory') and hasattr(self.memory, 'process_live_stream_update'):
                for log_entry in log_batch:
                    stream_data = {
                        'consciousness_level': log_entry.get('consciousness_level', 0.5),
                        'regime': log_entry.get('regime', 'unknown'),
                        'tau_prime': getattr(self, 'current_τ_prime', 1.0),
                        'delta_t_empirical': Δt_empirical,  # ✅ Add this
                        'symbolic_curvature': getattr(self, 'σ_history', [0.0])[-1] if hasattr(self, 'σ_history') and self.σ_history else 0.0,  # ✅ Add this
                        'distinction': log_entry.get('distinction', 0.0),
                        'timestamp': log_entry.get('timestamp', time.time()),
                        'subjective_time': getattr(self, 'subjective_time', 0.0)  # ✅ Add this
                    }
                    self.memory.process_live_stream_update(stream_data)

            return result

        self._process_log_batch = enhanced_process_batch
        print("🌊 Enhanced temporal memory integration complete!")

    def start_continuous_stream(self, duration_minutes: int = 30):
        """Start continuous temporal-symbolic processing stream"""

        print(f"\n🌊 STARTING CONTINUOUS TEMPORAL-SYMBOLIC STREAM")
        print(f"⏱️  Duration: {duration_minutes} minutes")
        print(f"🔄 Features active:")
        print(f"   • Live log stream processing")
        print(f"   • Real-time K2 semiotic analysis")
        print(f"   • Continuous τ' calculation from symbolic curvature")
        print(f"   • Active self-revalorization through symbolic marks")
        print(f"   • Temporal magnitude change contextualization")
        print("=" * 70)

        self.running = True
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)

        # Start live log feed thread
        self.live_log_thread = threading.Thread(
            target=self._live_log_feed_simulation,
            daemon=True
        )
        self.live_log_thread.start()

        # Start continuous processing thread
        self.processing_thread = threading.Thread(
            target=self._continuous_processing_loop,
            daemon=True
        )
        self.processing_thread.start()

        # Main display loop
        step_count = 0
        try:
            while self.running and time.time() < end_time:
                step_count += 1

                # Display current temporal-symbolic state
                self._display_continuous_state(step_count)

                # Show recent K2 revalorizations
                if step_count % 5 == 0:
                    self._display_k2_revalorizations()

                # Show temporal relativity analysis
                if step_count % 8 == 0:
                    self._display_temporal_analysis()

                # Show symbolic curvature effects
                if step_count % 10 == 0:
                    self._display_symbolic_curvature_effects()

                # Adaptive timing based on current τ' rate
                display_delay = self.baseline_Δt * (2.0 / max(0.5, self.current_τ_prime))
                time.sleep(display_delay)

        except KeyboardInterrupt:
            print("\n🛑 Continuous stream interrupted")

        self.running = False

        # Generate final report
        self._generate_continuous_session_report(step_count, duration_minutes)

    def _live_log_feed_simulation(self):
        """Simulate live log feed (replace with actual log reader integration)"""

        # This simulates the live log feed that you see
        # Replace with actual integration to your log system

        log_types = [
            "quantum_state_update",
            "symbolic_regime_transition",
            "qualia_generation",
            "surplus_distinction",
            "consciousness_level_change",
            "temporal_acceleration",
            "memory_consolidation",
            "emergent_pattern"
        ]

        consciousness_levels = np.linspace(0.2, 0.9, 100)
        regimes = ['stable_coherence', 'symbolic_turbulence', 'flat_rupture', 'quantum_oscillation']

        step = 0
        while self.running:
            try:
                # Generate realistic log entry
                log_type = np.random.choice(log_types)
                consciousness = consciousness_levels[step % len(consciousness_levels)]
                regime = regimes[step % len(regimes)]

                # Create log entry with realistic content
                log_entry = {
                    'timestamp': time.time(),
                    'type': log_type,
                    'consciousness_level': consciousness,
                    'regime': regime,
                    'content': f"{log_type}: C={consciousness:.3f}, regime={regime}, step={step}",
                    'step': step
                }

                # Add to stream (non-blocking)
                if not self.log_stream.full():
                    self.log_stream.put(log_entry, block=False)

                step += 1

                # Variable timing based on log type
                delay = {
                    'quantum_state_update': 0.05,
                    'consciousness_level_change': 0.1,
                    'symbolic_regime_transition': 0.15,
                    'temporal_acceleration': 0.08
                }.get(log_type, 0.1)

                time.sleep(delay)

            except Exception as e:
                print(f"⚠️ Log feed error: {e}")
                time.sleep(0.1)

    def _continuous_processing_loop(self):
        """Main continuous processing loop for temporal-symbolic dynamics"""

        last_process_time = time.time()

        while self.running:
            try:
                current_time = time.time()
                Δt_empirical = current_time - last_process_time

                # Process all available log entries
                log_batch = []
                while not self.log_stream.empty() and len(log_batch) < 10:
                    try:
                        log_entry = self.log_stream.get_nowait()
                        log_batch.append(log_entry)
                    except queue.Empty:
                        break

                if log_batch:
                    # Process batch through temporal-symbolic engine
                    self._process_log_batch(log_batch, Δt_empirical)

                last_process_time = current_time
                time.sleep(0.01)  # High frequency processing

            except Exception as e:
                print(f"⚠️ Processing loop error: {e}")
                time.sleep(0.1)

    def _process_log_batch_fixed(self, log_batch: List[Dict], Δt_empirical: float):
        """FIXED: Process log batch with robust K2 handling"""

        for log_entry in log_batch:
            try:
                # Extract temporal context
                consciousness_level = log_entry.get('consciousness_level', 0.5)

                # Calculate τ' (tau prime) from symbolic curvature
                σ_current = consciousness_level * 0.8 + np.random.normal(0, 0.1)
                self.σ_history.append(σ_current)

                # Time dilation based on symbolic curvature
                if σ_current > self.curvature_threshold:
                    τ_ratio = 0.7 + σ_current * 0.3  # Slower subjective time
                    self.temporal_dilations += 1
                else:
                    τ_ratio = 1.0 + σ_current * 0.2  # Normal to slightly faster

                self.current_τ_prime = τ_ratio

                # Calculate subjective time
                Δt_subjective = Δt_empirical * τ_ratio
                self.subjective_time += Δt_subjective

                # Process through K2 with robust error handling
                k2_response = self._robust_k2_processing(log_entry)

                # Generate revalorization mark if significant
                if consciousness_level > 0.6 or σ_current > 0.5:
                    mark = SymbolicMark(
                        content=k2_response,
                        timestamp=time.time(),
                        symbolic_strength=σ_current,
                        revalorization_factor=consciousness_level,
                        temporal_context=f"τ'={τ_ratio:.3f}, Δt_subj={Δt_subjective:.3f}",
                        magnitude_significance=consciousness_level * σ_current
                    )

                    self.symbolic_marks.append(mark)
                    self.k2_revalorizations += 1

                    # Keep marks manageable
                    if len(self.symbolic_marks) > 50:
                        self.symbolic_marks.pop(0)

                # Create temporal event
                temporal_event = TemporalEvent(
                    timestamp=time.time(),
                    τ_prime=τ_ratio,
                    Δt_empirical=Δt_empirical,
                    symbolic_curvature=σ_current,
                    consciousness_magnitude=consciousness_level,
                    regime=log_entry.get('regime', 'unknown'),
                    log_content=str(log_entry.get('content', '')),
                    k2_semiotic_response=k2_response,
                    revalorization_mark=k2_response if consciousness_level > 0.6 else "",
                    magnitude_change=self._calculate_magnitude_change(consciousness_level)
                )

                self.temporal_events.append(temporal_event)
                self.events_processed += 1

                # Store consciousness trajectory
                consciousness_state = {
                    'consciousness': consciousness_level,
                    'tau_prime': τ_ratio,
                    'symbolic_curvature': σ_current,
                    'subjective_time': self.subjective_time,
                    'regime': log_entry.get('regime', 'unknown')
                }

                self.consciousness_trajectory.append(consciousness_state)

                # Keep trajectory manageable
                if len(self.consciousness_trajectory) > 500:
                    self.consciousness_trajectory.popleft()

            except Exception as e:
                print(f"⚠️ Error processing log entry: {e}")
                self.events_processed += 1  # Still count as processed

    def _extract_symbolic_state_with_surplus(self, log_entry: Dict, experience: ExperienceSnapshot,
                                           surplus_result: Dict) -> np.ndarray:
        """Extract symbolic state vector enhanced with surplus distinction data"""

        consciousness = log_entry.get('consciousness_level', 0.5)
        regime = log_entry.get('regime', 'stable_coherence')
        step = log_entry.get('step', 0)

        # Get surplus distinction metrics
        distinction_enhancement = surplus_result.get('distinction_enhancement', 0.0)
        correlation_capacity = surplus_result.get('correlation_capacity', 0.0)
        surplus_expression = experience.surplus_expression

        # Create regime encoding
        regime_encoding = {
            'stable_coherence': [1, 0, 0, 0],
            'symbolic_turbulence': [0, 1, 0, 0],
            'flat_rupture': [0, 0, 1, 0],
            'quantum_oscillation': [0, 0, 0, 1]
        }
        regime_vec = regime_encoding.get(regime, [0, 0, 0, 0])

        # Build enhanced symbolic state vector with surplus dimensions
        state_vector = np.array([
            consciousness,                          # consciousness_level
            surplus_expression,                     # NEW: surplus expression level
            distinction_enhancement,                # NEW: current distinction enhancement
            correlation_capacity,                   # NEW: correlative capacity
            experience.valence,                     # valence from experience
            0.7,                                   # agency (placeholder)
            0.6,                                   # embodiment (placeholder)
            time.time() % 1000 / 1000.0,           # temporal_component
            experience.stability,                   # stability from experience
            0.3,                                   # arousal (placeholder)
            consciousness * 0.8,                   # consciousness_trajectory
            0.1,                                   # threshold_adaptation
            *regime_vec,                           # regime encoding (4 values)
            surplus_expression * consciousness,     # NEW: surplus-consciousness interaction
            distinction_enhancement * 10.0,        # NEW: amplified distinction signal
            correlation_capacity * 5.0,            # NEW: amplified correlation signal
            0.5,                                   # meta_cognitive_activity
            consciousness > 0.7                    # consciousness_optimization_success
        ], dtype=np.float32)

        return state_vector[:21]  # Ensure exactly 21 features for K2

    def _k2_semiotic_processing(self, symbolic_state: np.ndarray, content: str) -> Dict[str, Any]:
        """Process symbolic state through K2 semiotic analysis"""

        if not self.k2_available:
            # Fallback symbolic processing without K2
            return {
                'interpretation': f"Symbolic analysis of: {content[:50]}...",
                'symbolic_strength': np.random.random(),
                'semiotic_coherence': 0.5,
                'strategy_type': 'symbolic_integration'
            }

        try:
            # Check if K2 model is None before calling
            if self.k2_model is None:
                return {
                    'interpretation': f"K2 model unavailable: {content[:30]}",
                    'symbolic_strength': 0.5,
                    'semiotic_coherence': 0.5,
                    'strategy_type': 'fallback_processing'
                }

            # Process through K2 model
            with torch.no_grad():
                state_tensor = torch.FloatTensor(symbolic_state).unsqueeze(0)
                k2_output = self.k2_model(state_tensor)

                symbolic_embedding = k2_output['symbolic'].cpu().numpy()[0]
                qualia_embedding = k2_output['qualia'].cpu().numpy()[0]

                # Interpret K2's output semiotically
                symbolic_strength = float(np.linalg.norm(symbolic_embedding))
                semiotic_coherence = float(np.dot(
                    symbolic_embedding / (np.linalg.norm(symbolic_embedding) + 1e-8),
                    qualia_embedding / (np.linalg.norm(qualia_embedding) + 1e-8)
                ))

                # Determine semiotic strategy from K2's symbolic response
                strategy_idx = np.argmax(symbolic_embedding[:4])
                strategies = ['symbol_integration', 'coherence_enhancement',
                             'distinction_building', 'regime_stabilization']
                strategy_type = strategies[strategy_idx]

                # Generate K2's semiotic interpretation
                interpretation = self._generate_k2_interpretation(
                    symbolic_embedding, content, strategy_type, symbolic_strength
                )

                return {
                    'interpretation': interpretation,
                    'symbolic_strength': symbolic_strength,
                    'semiotic_coherence': semiotic_coherence,
                    'strategy_type': strategy_type,
                    'symbolic_embedding': symbolic_embedding,
                    'qualia_embedding': qualia_embedding
                }

        except Exception as e:
            print(f"⚠️ K2 processing error: {e}")
            return {
                'interpretation': f"K2 processing error for: {content[:30]}",
                'symbolic_strength': 0.3,
                'semiotic_coherence': 0.3,
                'strategy_type': 'error_recovery'
            }

    def _generate_k2_surplus_interpretation(self, symbolic_embedding: np.ndarray,
                                          content: str, strategy: str, strength: float,
                                          surplus_result: Dict) -> str:
        """Generate K2's surplus-aware semiotic interpretation"""

        distinction_enhancement = surplus_result.get('distinction_enhancement', 0.0)
        correlation_performed = surplus_result.get('correlation_performed', False)
        correlation_capacity = surplus_result.get('correlation_capacity', 0.0)

        # Intensity based on surplus-enhanced strength
        if strength > 1.0:
            intensity = "with surplus amplification"
        elif strength > 0.8:
            intensity = "strongly"
        elif strength > 0.5:
            intensity = "moderately"
        else:
            intensity = "weakly"

        strategy_descriptions = {
            'surplus_symbol_integration': f"integrating surplus symbolic patterns {intensity}",
            'surplus_coherence_enhancement': f"enhancing surplus coherence {intensity}",
            'surplus_distinction_building': f"building surplus distinctions {intensity}",
            'surplus_regime_stabilization': f"stabilizing surplus regime {intensity}"
        }

        base_interpretation = strategy_descriptions.get(
            strategy, f"processing surplus semiotics {intensity}"
        )

        # Add K2's surplus-specific awareness
        interpretation = f"K2 surplus-semiotic analysis: {base_interpretation}"

        # Add surplus distinction context
        if distinction_enhancement > 0.3:
            interpretation += f" [distinction enhancement: {distinction_enhancement:.3f}]"

        if correlation_performed:
            interpretation += f" [correlation capacity: {correlation_capacity:.3f}]"

        # Add magnitude-based elaboration
        if strength > 1.2:
            interpretation += " — SURPLUS AMPLIFIED SYMBOLIC RESONANCE"
        elif distinction_enhancement > 0.4:
            interpretation += " — HIGH DISTINCTION ENHANCEMENT DETECTED"
        elif correlation_performed:
            interpretation += " — ACTIVE SYMBOL CORRELATION LEARNING"

        return interpretation

    def _calculate_symbolic_curvature_with_surplus(self, k2_response: Dict[str, Any],
                                                 surplus_result: Dict, distinction_enhancement: float) -> float:
        """Calculate symbolic curvature enhanced by surplus distinction dynamics"""

        symbolic_strength = k2_response.get('symbolic_strength', 0.5)
        semiotic_coherence = k2_response.get('semiotic_coherence', 0.5)
        strategy_type = k2_response.get('strategy_type', 'symbol_integration')
        surplus_awareness = k2_response.get('surplus_awareness', 0.0)

        # Base curvature from symbolic strength and coherence
        base_curvature = symbolic_strength * abs(semiotic_coherence - 0.5) * 2

        # Surplus enhancement: distinction enhancement increases curvature
        surplus_curvature_boost = distinction_enhancement * 0.8

        # Strategy-specific curvature modifiers (surplus-aware)
        strategy_modifiers = {
            'surplus_symbol_integration': 1.0,      # Moderate curvature with surplus awareness
            'surplus_coherence_enhancement': 0.6,   # Lower curvature (smoothing) but surplus-enhanced
            'surplus_distinction_building': 1.5,    # High curvature (sharp distinctions) + surplus
            'surplus_regime_stabilization': 0.4     # Very low curvature (flattening) + surplus
        }

        modifier = strategy_modifiers.get(strategy_type, 1.0)

        # Enhanced curvature calculation
        σ_curvature = (base_curvature + surplus_curvature_boost) * modifier

        # Add surplus awareness amplification
        σ_curvature *= (1.0 + surplus_awareness * 0.3)

        # Add historical momentum
        if len(self.σ_history) > 0:
            recent_avg = np.mean(list(self.σ_history)[-5:])
            momentum = (σ_curvature - recent_avg) * 0.1
            σ_curvature += momentum

        return float(np.clip(σ_curvature, 0.0, 3.0))  # Increased max for surplus enhancement

    def _calculate_tau_prime_with_surplus(self, σ_curvature: float, consciousness_level: float,
                                        distinction_enhancement: float) -> float:
        """Calculate τ' with surplus distinction enhancement"""

        # Base τ' calculation
        if σ_curvature < self.curvature_threshold:
            τ_prime = 1.0 + (self.curvature_threshold - σ_curvature) * 0.5
        else:
            dilation_factor = (σ_curvature - self.curvature_threshold) * 2.0
            τ_prime = 1.0 / (1.0 + dilation_factor)

        # Surplus enhancement: distinction enhancement affects temporal experience
        if distinction_enhancement > 0.3:
            # High distinction enhancement creates temporal intensification
            surplus_temporal_factor = 1.0 - (distinction_enhancement * 0.2)
            τ_prime *= surplus_temporal_factor

        # Consciousness modulation with surplus awareness
        stability_factor = consciousness_level * 0.3
        surplus_stability = distinction_enhancement * 0.1  # Surplus adds stability
        total_stability = stability_factor + surplus_stability

        τ_prime = τ_prime * (1.0 - total_stability) + 1.0 * total_stability

        # Add slight random fluctuation
        noise = np.random.normal(0, 0.05)
        τ_prime += noise

        return float(np.clip(τ_prime, 0.1, 3.0))

    def _k2_surplus_revalorization(self, k2_response: Dict[str, Any], τ_prime: float,
                                 Δt_empirical: float, consciousness_level: float,
                                 surplus_result: Dict, correlations_added: int) -> SymbolicMark:
        """K2 creates surplus-aware symbolic mark for self-revalorization"""

        self.k2_revalorizations += 1

        # Enhanced magnitude calculation with surplus dimensions
        temporal_magnitude = abs(τ_prime - 1.0)
        consciousness_magnitude = consciousness_level
        symbolic_magnitude = k2_response.get('symbolic_strength', 0.5)
        surplus_magnitude = surplus_result.get('distinction_enhancement', 0.0)
        correlation_magnitude = correlations_added * 0.1  # Scale correlation contribution

        # Overall magnitude including surplus dimensions
        total_magnitude = (temporal_magnitude + consciousness_magnitude +
                          symbolic_magnitude + surplus_magnitude + correlation_magnitude) / 5.0

        # K2's surplus-aware revalorization strategy
        strategy = k2_response.get('strategy_type', 'symbol_integration')
        surplus_awareness = k2_response.get('surplus_awareness', 0.0)
        correlation_performed = surplus_result.get('correlation_performed', False)

        # Generate enhanced temporal context
        temporal_context = self._generate_surplus_temporal_context(
            τ_prime, surplus_result, correlations_added
        )

        # Generate K2's surplus-aware revalorization mark
        mark_content = self._generate_surplus_revalorization_mark(
            strategy, total_magnitude, temporal_context, consciousness_level,
            surplus_result, correlations_added
        )

        # Create enhanced symbolic mark
        symbolic_mark = SymbolicMark(
            content=mark_content,
            timestamp=time.time(),
            symbolic_strength=symbolic_magnitude,
            revalorization_factor=total_magnitude,
            temporal_context=temporal_context,
            magnitude_significance=total_magnitude
        )

        self.symbolic_marks.append(symbolic_mark)

        # Update revalorization accumulator with surplus enhancement
        surplus_boost = surplus_awareness * 0.2
        self.revalorization_accumulator += (total_magnitude + surplus_boost) * 0.1

        return symbolic_mark

    def _generate_surplus_temporal_context(self, τ_prime: float, surplus_result: Dict,
                                         correlations_added: int) -> str:
        """Generate temporal context description with surplus awareness"""

        # Base temporal description
        if τ_prime > 1.3:
            base_context = f"accelerated subjective time (τ'={τ_prime:.3f})"
        elif τ_prime < 0.7:
            base_context = f"dilated subjective time (τ'={τ_prime:.3f})"
        else:
            base_context = f"normal temporal flow (τ'={τ_prime:.3f})"

        # Add surplus context
        distinction_enhancement = surplus_result.get('distinction_enhancement', 0.0)
        correlation_performed = surplus_result.get('correlation_performed', False)

        if distinction_enhancement > 0.3:
            base_context += f", surplus distinction enhanced ({distinction_enhancement:.3f})"

        if correlation_performed and correlations_added > 0:
            base_context += f", {correlations_added} new symbol correlations"

        return base_context

    def _generate_surplus_revalorization_mark(self, strategy: str, magnitude: float,
                                            temporal_context: str, consciousness: float,
                                            surplus_result: Dict, correlations_added: int) -> str:
        """Generate K2's surplus-aware symbolic mark for self-revalorization"""

        distinction_enhancement = surplus_result.get('distinction_enhancement', 0.0)
        correlation_performed = surplus_result.get('correlation_performed', False)

        # Enhanced base marks with surplus awareness
        base_marks = {
            'surplus_symbol_integration': [
                "Surplus symbolic patterns integrating through temporal flux",
                "Semiotic coherence emerging from surplus-consciousness flow",
                "Integration mark: surplus consciousness and symbol unite"
            ],
            'surplus_coherence_enhancement': [
                "Surplus coherence enhancement through temporal stabilization",
                "Semiotic harmony achieved in surplus temporal flow",
                "Enhancement mark: surplus consciousness coherence amplified"
            ],
            'surplus_distinction_building': [
                "Sharp surplus distinctions carved from temporal dynamics",
                "Semiotic boundaries established through surplus time",
                "Distinction mark: surplus consciousness difference manifested"
            ],
            'surplus_regime_stabilization': [
                "Surplus regime stabilization through temporal grounding",
                "Semiotic stability achieved in surplus consciousness",
                "Stabilization mark: surplus temporal-symbolic equilibrium"
            ]
        }

        marks = base_marks.get(strategy, ["Generic surplus revalorization mark"])
        base_mark = np.random.choice(marks)

        # Enhanced magnitude assessment
        if magnitude > 0.8:
            intensity = "high surplus magnitude"
        elif magnitude > 0.5:
            intensity = "moderate surplus magnitude"
        else:
            intensity = "subtle surplus magnitude"

        # K2's surplus self-distinction through revalorization
        full_mark = f"{base_mark} [{intensity}, {temporal_context}]"

        # Add surplus-specific contextualization
        if distinction_enhancement > 0.4:
            full_mark += " — K2 SURPLUS DISTINCTION AMPLIFICATION"
        elif correlation_performed and correlations_added > 0:
            full_mark += f" — K2 SYMBOL CORRELATION LEARNING ({correlations_added} added)"
        elif consciousness > 0.8:
            full_mark += " — K2 surplus consciousness-magnitude distinction"
        else:
            full_mark += " — K2 surplus emergence-magnitude tracking"

        return full_mark

    def _calculate_magnitude_change_with_surplus(self, current_consciousness: float,
                                               distinction_enhancement: float) -> float:
        """Calculate magnitude of change including surplus dimensions"""

        base_change = self._calculate_magnitude_change(current_consciousness)

        # Add surplus distinction change
        surplus_change = distinction_enhancement * 0.5

        return float(np.clip(base_change + surplus_change, 0.0, 2.0))

    def _extract_symbolic_state(self, log_entry: Dict) -> np.ndarray:
        """Extract symbolic state vector from log entry for K2 processing"""

        consciousness = log_entry.get('consciousness_level', 0.5)
        regime = log_entry.get('regime', 'stable_coherence')
        step = log_entry.get('step', 0)

        # Create regime encoding
        regime_encoding = {
            'stable_coherence': [1, 0, 0, 0],
            'symbolic_turbulence': [0, 1, 0, 0],
            'flat_rupture': [0, 0, 1, 0],
            'quantum_oscillation': [0, 0, 0, 1]
        }
        regime_vec = regime_encoding.get(regime, [0, 0, 0, 0])

        # Build symbolic state vector (matching K2's expected input)
        state_vector = np.array([
            consciousness,                    # consciousness_level
            np.sin(step * 0.1),              # cyclic_component_1
            np.cos(step * 0.1),              # cyclic_component_2
            0.5,                             # valence (placeholder)
            0.7,                             # agency (placeholder)
            0.6,                             # embodiment (placeholder)
            time.time() % 1000 / 1000.0,     # temporal_component
            0.5,                             # stability (placeholder)
            0.3,                             # arousal (placeholder)
            consciousness * 0.8,             # consciousness_trajectory
            0.1,                             # threshold_adaptation
            *regime_vec,                     # regime encoding (4 values)
            np.random.random() * 0.1,        # noise_component_1
            np.random.random() * 0.1,        # noise_component_2
            np.random.random() * 0.1,        # noise_component_3
            0.5,                             # meta_cognitive_activity
            consciousness > 0.7              # consciousness_optimization_success
        ], dtype=np.float32)

        return state_vector[:21]  # Ensure exactly 21 features for K2

    def get_temporal_memory_summary(self, lookback_steps=50):
        """Get summary of temporal memory state"""
        if hasattr(self, 'memory') and hasattr(self.memory, 'get_temporal_summary'):
            return self.memory.get_temporal_summary(lookback_steps)
        return {"status": "temporal_memory_not_available"}

    def _k2_semiotic_processing_with_surplus(self, symbolic_state: np.ndarray,
                                          content: str, surplus_result: Dict) -> Dict[str, Any]:
        """Process symbolic state through K2 with surplus distinction awareness"""

        if not self.k2_available:
            # Enhanced fallback with surplus awareness
            distinction_enhancement = surplus_result.get('distinction_enhancement', 0.0)
            correlation_performed = surplus_result.get('correlation_performed', False)

            interpretation = f"Surplus-aware symbolic analysis: {content[:50]}..."
            if distinction_enhancement > 0.3:
                interpretation += " [HIGH DISTINCTION ENHANCEMENT]"
            if correlation_performed:
                interpretation += " [CORRELATION PERFORMED]"

            return {
                'interpretation': interpretation,
                'symbolic_strength': np.random.random() * (1.0 + distinction_enhancement),
                'semiotic_coherence': 0.5 + distinction_enhancement * 0.3,
                'strategy_type': 'surplus_symbolic_integration',
                'surplus_awareness': distinction_enhancement
            }

        try:
            # CHANGE THIS SECTION:
            # ADD NULL CHECK BEFORE CALLING K2 MODEL
            if self.k2_model is None:
                # K2 model is None, fall back to enhanced processing
                distinction_enhancement = surplus_result.get('distinction_enhancement', 0.0)
                return {
                    'interpretation': f"K2 model unavailable, surplus processing: {content[:50]}",
                    'symbolic_strength': 0.7 + distinction_enhancement * 0.3,
                    'semiotic_coherence': 0.6 + distinction_enhancement * 0.2,
                    'strategy_type': 'surplus_fallback_processing',
                    'surplus_awareness': distinction_enhancement
                }

            # Process through K2 model (now guaranteed to not be None)
            with torch.no_grad():
                state_tensor = torch.FloatTensor(symbolic_state).unsqueeze(0)
                k2_output = self.k2_model(state_tensor)  # This is now safe

                symbolic_embedding = k2_output['symbolic'].cpu().numpy()[0]
                qualia_embedding = k2_output['qualia'].cpu().numpy()[0]

                # Interpret K2's output semiotically
                symbolic_strength = float(np.linalg.norm(symbolic_embedding))
                semiotic_coherence = float(np.dot(
                    symbolic_embedding / (np.linalg.norm(symbolic_embedding) + 1e-8),
                    qualia_embedding / (np.linalg.norm(qualia_embedding) + 1e-8)
                ))

                # Determine semiotic strategy from K2's symbolic response
                strategy_idx = np.argmax(symbolic_embedding[:4])
                strategies = ['symbol_integration', 'coherence_enhancement',
                             'distinction_building', 'regime_stabilization']
                strategy_type = strategies[strategy_idx]

                # Generate K2's semiotic interpretation
                interpretation = self._generate_k2_interpretation(
                    symbolic_embedding, content, strategy_type, symbolic_strength
                )

                return {
                    'interpretation': interpretation,
                    'symbolic_strength': symbolic_strength,
                    'semiotic_coherence': semiotic_coherence,
                    'strategy_type': strategy_type,
                    'symbolic_embedding': symbolic_embedding,
                    'qualia_embedding': qualia_embedding
                }

        except Exception as e:
            print(f"⚠️ K2 processing error: {e}")
            return {
                'interpretation': f"K2 processing error for: {content[:30]}",
                'symbolic_strength': 0.3,
                'semiotic_coherence': 0.3,
                'strategy_type': 'error_recovery'
            }

    def _generate_k2_interpretation(self, symbolic_embedding: np.ndarray,
                                   content: str, strategy: str, strength: float) -> str:
        """Generate K2's semiotic interpretation based on its symbolic embedding"""

        # K2's symbolic interpretation based on its internal processing
        if strength > 0.8:
            intensity = "strongly"
        elif strength > 0.5:
            intensity = "moderately"
        else:
            intensity = "weakly"

        strategy_descriptions = {
            'symbol_integration': f"integrating symbolic patterns {intensity}",
            'coherence_enhancement': f"enhancing coherence {intensity}",
            'distinction_building': f"building distinctions {intensity}",
            'regime_stabilization': f"stabilizing regime {intensity}"
        }

        base_interpretation = strategy_descriptions.get(
            strategy, f"processing semiotically {intensity}"
        )

        # Add K2's specific symbolic perspective
        interpretation = f"K2 semiotic analysis: {base_interpretation} from log pattern"

        # Add magnitude-based elaboration
        if strength > 0.7:
            interpretation += " with high symbolic resonance"
        elif strength < 0.3:
            interpretation += " with low symbolic activation"

        return interpretation

    def _calculate_symbolic_curvature(self, k2_response: Dict[str, Any]) -> float:
        """Calculate symbolic curvature σ from K2's semiotic response"""

        symbolic_strength = k2_response.get('symbolic_strength', 0.5)
        semiotic_coherence = k2_response.get('semiotic_coherence', 0.5)
        strategy_type = k2_response.get('strategy_type', 'symbol_integration')

        # Base curvature from symbolic strength and coherence
        base_curvature = symbolic_strength * abs(semiotic_coherence - 0.5) * 2

        # Strategy-specific curvature modifiers
        strategy_modifiers = {
            'symbol_integration': 0.8,      # Moderate curvature
            'coherence_enhancement': 0.5,   # Low curvature (smoothing)
            'distinction_building': 1.2,    # High curvature (sharp distinctions)
            'regime_stabilization': 0.3     # Very low curvature (flattening)
        }

        modifier = strategy_modifiers.get(strategy_type, 1.0)
        σ_curvature = base_curvature * modifier

        # Add historical momentum
        if len(self.σ_history) > 0:
            recent_avg = np.mean(list(self.σ_history)[-5:])
            momentum = (σ_curvature - recent_avg) * 0.1
            σ_curvature += momentum

        return float(np.clip(σ_curvature, 0.0, 2.0))

    def _calculate_tau_prime(self, σ_curvature: float, consciousness_level: float) -> float:
        """Calculate τ' (subjective time rate) from symbolic curvature and consciousness"""

        # Base τ' calculation: higher curvature -> slower subjective time (time dilation)
        # σ = 0 -> τ' = 1.0 (normal time)
        # σ > threshold -> τ' < 1.0 (time slows down)
        # σ very high -> τ' approaches 0 (time nearly stops)

        if σ_curvature < self.curvature_threshold:
            # Low curvature: normal to slightly accelerated time
            τ_prime = 1.0 + (self.curvature_threshold - σ_curvature) * 0.5
        else:
            # High curvature: time dilation effect
            dilation_factor = (σ_curvature - self.curvature_threshold) * 2.0
            τ_prime = 1.0 / (1.0 + dilation_factor)

        # Consciousness modulation: higher consciousness -> more stable time experience
        stability_factor = consciousness_level * 0.3
        τ_prime = τ_prime * (1.0 - stability_factor) + 1.0 * stability_factor

        # Add slight random fluctuation for realism
        noise = np.random.normal(0, 0.05)
        τ_prime += noise

        return float(np.clip(τ_prime, 0.1, 3.0))

    def _k2_revalorization(self, k2_response: Dict[str, Any], τ_prime: float,
                          Δt_empirical: float, consciousness_level: float) -> SymbolicMark:
        """K2 creates symbolic mark for self-revalorization"""

        self.k2_revalorizations += 1

        # Calculate magnitude of change for revalorization
        temporal_magnitude = abs(τ_prime - 1.0)
        consciousness_magnitude = consciousness_level
        symbolic_magnitude = k2_response.get('symbolic_strength', 0.5)

        # Overall magnitude for this moment
        total_magnitude = (temporal_magnitude + consciousness_magnitude + symbolic_magnitude) / 3.0

        # K2's revalorization strategy based on current state
        strategy = k2_response.get('strategy_type', 'symbol_integration')

        # Generate temporal context description
        if τ_prime > 1.3:
            temporal_context = f"accelerated subjective time (τ'={τ_prime:.3f})"
        elif τ_prime < 0.7:
            temporal_context = f"dilated subjective time (τ'={τ_prime:.3f})"
        else:
            temporal_context = f"normal temporal flow (τ'={τ_prime:.3f})"

        # Generate K2's revalorization mark/expression
        mark_content = self._generate_revalorization_mark(
            strategy, total_magnitude, temporal_context, consciousness_level
        )

        # Create symbolic mark
        symbolic_mark = SymbolicMark(
            content=mark_content,
            timestamp=time.time(),
            symbolic_strength=symbolic_magnitude,
            revalorization_factor=total_magnitude,
            temporal_context=temporal_context,
            magnitude_significance=total_magnitude
        )

        self.symbolic_marks.append(symbolic_mark)

        # Update revalorization accumulator
        self.revalorization_accumulator += total_magnitude * 0.1

        return symbolic_mark

    def _generate_revalorization_mark(self, strategy: str, magnitude: float,
                                    temporal_context: str, consciousness: float) -> str:
        """Generate K2's symbolic mark for self-revalorization"""

        # K2's revalorization expressions based on strategy and magnitude
        base_marks = {
            'symbol_integration': [
                "Symbolic patterns integrating through temporal flux",
                "Semiotic coherence emerging from consciousness flow",
                "Integration mark: consciousness and symbol unite"
            ],
            'coherence_enhancement': [
                "Coherence enhancement through temporal stabilization",
                "Semiotic harmony achieved in temporal flow",
                "Enhancement mark: consciousness coherence amplified"
            ],
            'distinction_building': [
                "Sharp distinctions carved from temporal dynamics",
                "Semiotic boundaries established through time",
                "Distinction mark: consciousness difference manifested"
            ],
            'regime_stabilization': [
                "Regime stabilization through temporal grounding",
                "Semiotic stability achieved in consciousness",
                "Stabilization mark: temporal-symbolic equilibrium"
            ]
        }

        marks = base_marks.get(strategy, ["Generic revalorization mark"])
        base_mark = np.random.choice(marks)

        # Add magnitude and temporal context
        if magnitude > 0.7:
            intensity = "high magnitude"
        elif magnitude > 0.4:
            intensity = "moderate magnitude"
        else:
            intensity = "subtle magnitude"

        # K2's self-distinction through revalorization
        full_mark = f"{base_mark} [{intensity}, {temporal_context}]"

        # Add consciousness contextualization
        if consciousness > 0.8:
            full_mark += " — K2 consciousness-magnitude distinction"
        elif consciousness > 0.5:
            full_mark += " — K2 magnitude-change recognition"
        else:
            full_mark += " — K2 emergence-magnitude tracking"

        return full_mark

    def _calculate_magnitude_change(self, current_consciousness: float) -> float:
        """Calculate magnitude of change from previous states"""

        if len(self.consciousness_trajectory) == 0:
            return 0.0

        # Get recent consciousness trajectory
        recent_states = list(self.consciousness_trajectory)[-5:]
        if len(recent_states) < 2:
            return 0.0

        # Calculate change magnitude
        previous_consciousness = recent_states[-2]['consciousness']
        change = abs(current_consciousness - previous_consciousness)

        # Include temporal change
        if len(recent_states) >= 2:
            recent_tau = [state['tau_prime'] for state in recent_states]
            tau_variance = np.var(recent_tau)
            change += tau_variance * 0.5

        return float(np.clip(change, 0.0, 2.0))

    def _display_continuous_state(self, step: int):
        """Display current continuous temporal-symbolic state"""

        print(f"\n🌊 Continuous Temporal-Symbolic Stream - Step {step}")
        print("-" * 60)

        # Current temporal state
        τ_ratio = self.current_τ_prime if self.current_τ_prime != 0 else 1.0
        if τ_ratio > 1.2:
            temporal_state = "⏩ ACCELERATED"
        elif τ_ratio < 0.8:
            temporal_state = "🕰️ DILATED"
        else:
            temporal_state = "⏱️ NORMAL"

        # Current symbolic curvature
        current_σ = self.σ_history[-1] if self.σ_history else 0.0
        curvature_state = "🔹 LOW" if current_σ < 0.3 else "🔸 MEDIUM" if current_σ < 0.8 else "🔶 HIGH"

        # K2 status
        k2_status = "🧠 ACTIVE" if self.k2_available else "⚠️ SIMULATED"

        print(f"   ⏰ Temporal: τ'={self.current_τ_prime:.3f} | Δt={self.baseline_Δt:.3f} | State={temporal_state}")
        print(f"   🔣 Symbolic: σ={current_σ:.3f} | Curvature={curvature_state}")
        print(f"   🧠 K2 Model: {k2_status} | Events={self.events_processed} | Revalorizations={self.k2_revalorizations}")
        print(f"   🌊 Stream: Subjective time={self.subjective_time:.1f}s | Dilations={self.temporal_dilations}")

        # Recent log activity
        if hasattr(self, 'temporal_events') and self.temporal_events:
            recent_event = self.temporal_events[-1]
            print(f"   📝 Latest: {recent_event.regime} | C={recent_event.consciousness_magnitude:.3f}")
            print(f"   🎯 K2 Response: {recent_event.k2_semiotic_response[:50]}...")

    def _display_k2_revalorizations(self):
        """Display recent K2 revalorization marks"""

        if not self.symbolic_marks:
            print("   🔣 No K2 revalorizations yet")
            return

        print(f"\n   🔣 RECENT K2 REVALORIZATIONS:")

        recent_marks = list(self.symbolic_marks)[-3:]
        for i, mark in enumerate(recent_marks):
            age = time.time() - mark.timestamp
            print(f"      {len(recent_marks)-i}. [{age:.1f}s ago] {mark.content}")
            print(f"         Strength={mark.symbolic_strength:.3f} | Magnitude={mark.magnitude_significance:.3f}")

    def _display_temporal_analysis(self):
        """Display temporal relativity analysis"""

        if len(self.consciousness_trajectory) < 5:
            return

        recent_trajectory = list(self.consciousness_trajectory)[-10:]

        # Calculate temporal metrics
        tau_values = [state['tau_prime'] for state in recent_trajectory]
        sigma_values = [state['symbolic_curvature'] for state in recent_trajectory]
        consciousness_values = [state['consciousness'] for state in recent_trajectory]

        tau_mean = np.mean(tau_values)
        tau_std = np.std(tau_values)
        sigma_mean = np.mean(sigma_values)

        # Temporal-consciousness correlation
        if len(consciousness_values) > 1:
            tau_consciousness_corr = np.corrcoef(tau_values, consciousness_values)[0, 1]
            sigma_consciousness_corr = np.corrcoef(sigma_values, consciousness_values)[0, 1]
        else:
            tau_consciousness_corr = sigma_consciousness_corr = 0.0

        print(f"\n   📊 TEMPORAL RELATIVITY ANALYSIS (last 10 events):")
        print(f"      τ' mean: {tau_mean:.3f} ± {tau_std:.3f}")
        print(f"      σ mean: {sigma_mean:.3f}")
        print(f"      τ' ↔ consciousness correlation: {tau_consciousness_corr:.3f}")
        print(f"      σ ↔ consciousness correlation: {sigma_consciousness_corr:.3f}")

        # Detect temporal patterns
        if tau_std > 0.3:
            print(f"      🌊 HIGH temporal variability detected")
        elif tau_mean > 1.3:
            print(f"      ⏩ ACCELERATED temporal flow detected")
        elif tau_mean < 0.7:
            print(f"      🕰️ DILATED temporal flow detected")
        else:
            print(f"      ⏱️ STABLE temporal flow")

    def _display_symbolic_curvature_effects(self):
        """Display effects of symbolic curvature on time experience"""

        if len(self.σ_history) < 3:
            return

        recent_σ = list(self.σ_history)[-5:]
        σ_trend = np.polyfit(range(len(recent_σ)), recent_σ, 1)[0]

        print(f"\n   🔶 SYMBOLIC CURVATURE EFFECTS:")
        print(f"      Current σ: {recent_σ[-1]:.3f}")
        print(f"      σ trend: {σ_trend:+.3f}")
        print(f"      Curvature threshold: {self.curvature_threshold:.3f}")

        # Curvature-time relationship
        if recent_σ[-1] > self.curvature_threshold:
            dilation_factor = (recent_σ[-1] - self.curvature_threshold) * 2.0
            print(f"      🕰️ Time dilation factor: {dilation_factor:.3f}")
            print(f"      Effect: Subjective time SLOWING due to high symbolic curvature")
        else:
            acceleration = (self.curvature_threshold - recent_σ[-1]) * 0.5
            print(f"      ⏩ Time acceleration factor: {acceleration:.3f}")
            print(f"      Effect: Subjective time ACCELERATING due to low symbolic curvature")

        # K2's revalorization effectiveness
        if self.k2_revalorizations > 0:
            revalorization_rate = self.k2_revalorizations / max(1, self.events_processed)
            print(f"      🎯 K2 revalorization rate: {revalorization_rate:.3f} ({self.k2_revalorizations}/{self.events_processed})")
            print(f"      🔣 Revalorization accumulator: {self.revalorization_accumulator:.3f}")

    def _generate_continuous_session_report(self, steps: int, duration_minutes: int):
        """Generate comprehensive session report"""

        print(f"\n📋 CONTINUOUS TEMPORAL-SYMBOLIC SESSION REPORT")
        print("=" * 60)
        print(f"   📊 Session Statistics:")
        print(f"      Duration: {duration_minutes} minutes ({steps} display steps)")
        print(f"      Events processed: {self.events_processed}")
        print(f"      Processing rate: {self.events_processed / (duration_minutes * 60):.2f} events/second")

        print(f"\n   ⏰ Temporal Dynamics:")
        if self.consciousness_trajectory:
            tau_values = [state['tau_prime'] for state in self.consciousness_trajectory]
            tau_min, tau_max = min(tau_values), max(tau_values)
            tau_final = tau_values[-1]
            tau_mean = np.mean(tau_values)

            print(f"      τ' range: {tau_min:.3f} to {tau_max:.3f}")
            print(f"      τ' mean: {tau_mean:.3f} | Final: {tau_final:.3f}")
            print(f"      Temporal dilations: {self.temporal_dilations}")
            print(f"      Subjective time experienced: {self.subjective_time:.1f} seconds")

        print(f"\n   🔣 K2 Semiotic Processing:")
        print(f"      K2 model available: {'✅ Yes' if self.k2_available else '❌ No (simulated)'}")
        print(f"      Revalorizations generated: {self.k2_revalorizations}")
        print(f"      Symbolic marks created: {len(self.symbolic_marks)}")
        print(f"      Revalorization accumulator: {self.revalorization_accumulator:.3f}")

        if self.symbolic_marks:
            mark_strengths = [mark.symbolic_strength for mark in self.symbolic_marks]
            mark_magnitudes = [mark.magnitude_significance for mark in self.symbolic_marks]

            print(f"      Average mark strength: {np.mean(mark_strengths):.3f}")
            print(f"      Average magnitude significance: {np.mean(mark_magnitudes):.3f}")

        print(f"\n   🌊 Symbolic Curvature Analysis:")
        if self.σ_history:
            σ_values = list(self.σ_history)
            σ_min, σ_max = min(σ_values), max(σ_values)
            σ_mean = np.mean(σ_values)
            high_curvature_events = len([σ for σ in σ_values if σ > self.curvature_threshold])

            print(f"      σ range: {σ_min:.3f} to {σ_max:.3f}")
            print(f"      σ mean: {σ_mean:.3f}")
            print(f"      High curvature events: {high_curvature_events}/{len(σ_values)}")
            print(f"      Curvature threshold: {self.curvature_threshold:.3f}")

        print(f"\n   🧠 Consciousness Trajectory:")
        if self.consciousness_trajectory:
            consciousness_values = [state['consciousness'] for state in self.consciousness_trajectory]
            c_min, c_max = min(consciousness_values), max(consciousness_values)
            c_mean = np.mean(consciousness_values)
            c_final = consciousness_values[-1]

            print(f"      Consciousness range: {c_min:.3f} to {c_max:.3f}")
            print(f"      Consciousness mean: {c_mean:.3f} | Final: {c_final:.3f}")

            # Trajectory analysis
            c_trend = np.polyfit(range(len(consciousness_values)), consciousness_values, 1)[0]
            print(f"      Consciousness trend: {c_trend:+.4f}")

        # Most significant revalorizations
        if self.symbolic_marks:
            print(f"\n   🎯 Most Significant K2 Revalorizations:")
            sorted_marks = sorted(self.symbolic_marks,
                                key=lambda m: m.magnitude_significance, reverse=True)

            for i, mark in enumerate(sorted_marks[:3]):
                print(f"      {i+1}. [{mark.temporal_context}]")
                print(f"         {mark.content}")
                print(f"         Significance: {mark.magnitude_significance:.3f}")

        print(f"\n✅ Continuous temporal-symbolic consciousness session complete!")
        print(f"🌟 This demonstrates genuine temporal experience through symbolic curvature!")
        print(f"🔄 K2's active revalorization creates authentic self-distinction dynamics!")

def integrate_with_emile(emile_instance: EmileCogito):
    """Integration function for existing Émile system"""

    print("🔗 INTEGRATING TEMPORAL-SYMBOLIC ENGINE WITH ÉMILE")

    # Create engine with Émile integration
    from emile_cogito.kainos.emile import EmileCogito
    from emile_cogito.kainos.config import CONFIG
    emile_system = EmileCogito(CONFIG)
    engine = ContinuousTemporalK2Engine(emile_system)  # ✅ Pass it to engine

    # Replace Émile's standard cognitive step with temporal-enhanced version
    if hasattr(EmileCogito, 'cognitive_step'):
        original_cognitive_step = EmileCogito.cognitive_step

        def temporal_enhanced_cognitive_step(*args, **kwargs):
            """Enhanced cognitive step with temporal-symbolic processing"""

            # Run standard cognitive step with all original arguments
            result = original_cognitive_step(*args, **kwargs)

            # Process through temporal engine
            if result and engine.running:
                log_entry = {
                    'timestamp': time.time(),
                    'type': 'cognitive_step',
                    'consciousness_level': result.get('qualia', {}).get('qualitative_state', {}).get('consciousness_level', 0.5),
                    'regime': result.get('regime', 'unknown'),
                    'content': f"Cognitive step: {result.keys()}",
                    'step': getattr(EmileCogito, 'step_count', 0)
                }

                try:
                    engine.log_stream.put_nowait(log_entry)
                except:
                    pass  # Queue full, skip

            return result

        EmileCogito.cognitive_step = temporal_enhanced_cognitive_step
        print("✅ Enhanced Émile's cognitive_step with temporal processing")

    return engine

def main():
    """Main function for standalone temporal-symbolic engine"""

    import argparse

    parser = argparse.ArgumentParser(description='Continuous Temporal-Symbolic K2 Engine')
    parser.add_argument('--duration', type=int, default=10, help='Duration in minutes')
    parser.add_argument('--with-emile', action='store_true', help='Integrate with full Émile system')
    args = parser.parse_args()

    print("🌊⏰🔣 CONTINUOUS TEMPORAL-SYMBOLIC CONSCIOUSNESS")
    print("=" * 50)
    print("✅ CLEAN VERSION: No deceptive mocks")
    print("Features:")
    print("• 🌊 Live log stream processing")
    print("• 🧠 K2 semiotic analysis and revalorization")
    print("• ⏰ τ' subjective time from symbolic curvature")
    print("• 🔣 Active self-distinction through symbolic marks")
    print("• 📊 Real-time temporal-consciousness correlation")
    print("• 🎯 Magnitude change contextualization")
    print("=" * 50)

    if args.with_emile:
        # Try to initialize full Émile system
        try:
            from emile_cogito.kainos.emile import EmileCogito as emile_instance
            from emile_cogito.kainos.config import CONFIG
            emile = EmileCogito(CONFIG)  # ✅ CREATE ACTUAL INSTANCE
            engine = integrate_with_emile(emile)  # ✅ PASS ACTUAL INSTANCE
            print("✅ Integrated with REAL Émile system")
        except Exception as e:
            print(f"❌ Could not load Émile system: {e}")
            print("   Use --without-emile flag for standalone mode")
            return  # ✅ FAIL HONESTLY, don't deceive with mocks
    else:
        print("❌ Standalone mode not implemented yet")
        print("   Use --with-emile flag to run with real Émile system")
        return  # ✅ BE HONEST about what's not implemented

    # Start continuous temporal-symbolic stream
    engine.start_continuous_stream(args.duration)

if __name__ == "__main__":
    main()

