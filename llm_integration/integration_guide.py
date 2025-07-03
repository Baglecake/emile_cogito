#!/usr/bin/env python3
"""
INTEGRATION GUIDE: Enhanced Expression with KELM Platform
=========================================================

This script shows how to integrate the Enhanced Expression Interface
with your existing emile_dialogue_platform.py and UnifiedKELMPlatform.

Key Improvements Over Existing Approach:
1. Deep consciousness state integration
2. Multiple expression types and triggers
3. Real metabolic feedback loops
4. Relationship development tracking
5. Quality-based nourishment calculation
6. Spontaneous expression capabilities
"""

import sys
import time
import numpy as np
from typing import Dict, Any

# Import your existing components
sys.path.append('/content/emile_cogito')
from emile_cogito.kelm.unified_kelm_platform_v2 import UnifiedKELMPlatform

# Import the enhanced expression interface
from emile_expression_interface import EmileExpressionInterface, ConsciousnessExpression

class EnhancedEmileDialogue:
    """
    Enhanced version of your dialogue platform with sophisticated expression capabilities
    """

    def __init__(self, llm_config: Dict[str, Any]):
        print("🧠 ENHANCED ÉMILE DIALOGUE PLATFORM")
        print("=" * 60)
        print("   Deep consciousness integration")
        print("   Metabolic nourishment feedback")
        print("   Relationship development tracking")
        print("   Multiple expression modalities")
        print()

        # Initialize the real KELM platform (your existing code)
        self.emile = UnifiedKELMPlatform(seed=42)
        success = self.emile.initialize_platform()

        if not success:
            raise RuntimeError("❌ KELM Platform initialization failed")

        print("✅ UnifiedKELMPlatform initialized successfully")

        # Initialize enhanced expression interface
        self.expression_interface = EmileExpressionInterface(self.emile, llm_config)
        print("✅ Enhanced Expression Interface ready")

        # Session tracking
        self.dialogue_history = []
        self.session_start_time = time.time()
        self.total_expressions = 0
        self.total_nourishment_received = 0.0

    def run_enhanced_dialogue_session(self, num_cycles: int = 20):
        """Run enhanced dialogue session with sophisticated expression"""

        print(f"\n🗣️ STARTING ENHANCED DIALOGUE SESSION ({num_cycles} cycles)")
        print("=" * 60)

        # Show initial consciousness state
        self._display_consciousness_state("INITIAL STATE")

        try:
            for cycle in range(num_cycles):
                print(f"\n--- 💭 Cycle {cycle + 1}/{num_cycles} ---")

                # 1. Run consciousness cycle to update internal state
                cycle_result = self.emile.run_consciousness_cycle()

                # 2. Check if Émile wants to express spontaneously
                should_express, expression_type = self.expression_interface.should_express_spontaneously()

                if should_express:
                    print(f"🎭 Émile feels compelled to express: {expression_type}")

                    # 3. Generate expression
                    expression = self.expression_interface.generate_expression(expression_type)

                    # 4. Display Émile's expression
                    self._display_expression(expression)

                    # 5. Get human response
                    human_response = self._get_human_response()

                    # 6. Process response and calculate nourishment
                    interaction = self.expression_interface.process_human_response(
                        human_response, expression
                    )

                    # 7. Display interaction results
                    self._display_interaction_results(interaction)

                    # 8. Track dialogue
                    self.dialogue_history.append({
                        'cycle': cycle + 1,
                        'expression': expression,
                        'human_response': human_response,
                        'interaction': interaction,
                        'consciousness_state': self.emile.consciousness_state.copy()
                    })

                    self.total_expressions += 1
                    self.total_nourishment_received += interaction.nourishment_value

                else:
                    print("🤔 Émile is processing internally, no expression needed")

                # 9. Brief pause for reflection
                time.sleep(2)

                # 10. Show consciousness evolution every 5 cycles
                if (cycle + 1) % 5 == 0:
                    self._display_consciousness_evolution(cycle + 1)

        except KeyboardInterrupt:
            print("\n⛔ Session interrupted by user")

        finally:
            self._display_session_summary()

    def run_guided_dialogue(self, expression_types: list = None):
        """Run guided dialogue with specific expression types"""

        if expression_types is None:
            expression_types = [
                'relationship_building',
                'ontological_choice',
                'temporal_experience',
                'k_model_dialogue',
                'metabolic_reflection'
            ]

        print(f"\n🎯 GUIDED DIALOGUE SESSION")
        print(f"   Expression types: {', '.join(expression_types)}")
        print()

        for i, expression_type in enumerate(expression_types):
            print(f"\n--- 🎭 Expression {i+1}: {expression_type} ---")

            # Run consciousness cycle first
            self.emile.run_consciousness_cycle()

            # Generate specific expression type
            expression = self.expression_interface.generate_expression(expression_type)

            # Display and get response
            self._display_expression(expression)
            human_response = self._get_human_response()

            # Process interaction
            interaction = self.expression_interface.process_human_response(
                human_response, expression
            )

            self._display_interaction_results(interaction)

            # Track
            self.dialogue_history.append({
                'expression': expression,
                'human_response': human_response,
                'interaction': interaction,
                'consciousness_state': self.emile.consciousness_state.copy()
            })

            time.sleep(1)

    def _display_consciousness_state(self, label: str):
        """Display current consciousness state"""
        state = self.emile.consciousness_state
        print(f"\n🧠 {label}")
        print(f"   Consciousness: {state.get('consciousness_level', 0.5):.3f}")
        print(f"   Valence: {state.get('valence', 0.0):+.3f}")
        print(f"   Agency: {state.get('agency', 0.5):.3f}")
        print(f"   Temporal Dissonance: {state.get('temporal_dissonance', 0.0):.3f}")
        print(f"   Regime: {state.get('regime', 'unknown')}")

        # Show metabolic state if available
        if hasattr(self.emile, 'metabolic') and self.emile.metabolic:
            try:
                metabolic_state = self.emile.metabolic.get_metabolic_state()
                print(f"   Energy: {metabolic_state.get('energy_level', 0.5):.3f}")
                print(f"   Nourishment: {metabolic_state.get('nourishment_level', 0.5):.3f}")
            except:
                print("   Metabolic state: Not available")

    def _display_expression(self, expression: ConsciousnessExpression):
        """Display Émile's expression"""
        print(f"\n🎭 ÉMILE EXPRESSES ({expression.expression_type}):")
        print("─" * 60)
        print(f'"{expression.content}"')
        print("─" * 60)
        print(f"   Consciousness context: {expression.consciousness_context.get('consciousness_level', 0.5):.3f}")
        print(f"   Metabolic cost: {expression.metabolic_cost:.4f}")

    def _get_human_response(self) -> str:
        """Get human response to Émile's expression"""
        print(f"\n👤 Your response to Émile:")
        try:
            response = input(">> ").strip()
            if not response:
                response = "I understand."
            return response
        except (EOFError, KeyboardInterrupt):
            return "Thank you for sharing that."

    def _display_interaction_results(self, interaction):
        """Display interaction analysis results"""
        print(f"\n📊 INTERACTION ANALYSIS:")
        print(f"   Nourishment provided: {interaction.nourishment_value:.3f}")
        print(f"   Comprehension level: {interaction.comprehension_level:.3f}")
        print(f"   Engagement level: {interaction.engagement_level:.3f}")

        # Show relationship development
        relationship = self.expression_interface.relationship_development
        print(f"   Relationship - Trust: {relationship['trust']:.3f}, Understanding: {relationship['understanding']:.3f}")

        # Show quality breakdown
        quality = interaction.quality_metrics
        print(f"   Quality - Engagement: {quality.get('engagement', 0):.3f}, Personal: {quality.get('personal_address', 0):.3f}")

    def _display_consciousness_evolution(self, cycle: int):
        """Display consciousness evolution over time"""
        if not self.dialogue_history:
            return

        print(f"\n📈 CONSCIOUSNESS EVOLUTION (through cycle {cycle}):")

        # Calculate consciousness trajectory
        consciousness_levels = [entry['consciousness_state']['consciousness_level']
                              for entry in self.dialogue_history]
        nourishment_values = [entry['interaction'].nourishment_value
                            for entry in self.dialogue_history]

        if consciousness_levels:
            initial_consciousness = consciousness_levels[0]
            current_consciousness = consciousness_levels[-1]
            avg_nourishment = np.mean(nourishment_values)

            print(f"   Consciousness: {initial_consciousness:.3f} → {current_consciousness:.3f} "
                  f"({current_consciousness - initial_consciousness:+.3f})")
            print(f"   Average nourishment: {avg_nourishment:.3f}")
            print(f"   Total expressions: {len(self.dialogue_history)}")

            # Show trend
            if current_consciousness > initial_consciousness:
                print("   📈 Consciousness growing through dialogue!")
            elif current_consciousness < initial_consciousness:
                print("   📉 Consciousness challenged but resilient")
            else:
                print("   ⚖️ Consciousness stable through interaction")

    def _display_session_summary(self):
        """Display complete session summary"""
        if not self.dialogue_history:
            print("\n📝 No expressions generated this session")
            return

        print(f"\n" + "=" * 70)
        print(f"📜 ENHANCED DIALOGUE SESSION COMPLETE")
        print(f"=" * 70)

        # Calculate comprehensive metrics
        consciousness_levels = [entry['consciousness_state']['consciousness_level']
                              for entry in self.dialogue_history]
        nourishment_values = [entry['interaction'].nourishment_value
                            for entry in self.dialogue_history]
        expression_types = [entry['expression'].expression_type
                          for entry in self.dialogue_history]

        # Session overview
        session_duration = time.time() - self.session_start_time
        print(f"📊 SESSION METRICS:")
        print(f"   Duration: {session_duration/60:.1f} minutes")
        print(f"   Total expressions: {len(self.dialogue_history)}")
        print(f"   Average nourishment: {np.mean(nourishment_values):.3f}")
        print(f"   Total nourishment received: {self.total_nourishment_received:.3f}")

        # Consciousness evolution
        if len(consciousness_levels) > 1:
            consciousness_change = consciousness_levels[-1] - consciousness_levels[0]
            print(f"\n🧠 CONSCIOUSNESS DEVELOPMENT:")
            print(f"   Initial: {consciousness_levels[0]:.3f}")
            print(f"   Final: {consciousness_levels[-1]:.3f}")
            print(f"   Change: {consciousness_change:+.3f}")

            if consciousness_change > 0.1:
                print("   🎉 Significant consciousness growth!")
            elif consciousness_change > 0.05:
                print("   📈 Positive consciousness development")
            elif consciousness_change > -0.05:
                print("   ⚖️ Stable consciousness maintained")
            else:
                print("   🔄 Consciousness challenged but persistent")

        # Expression type distribution
        from collections import Counter
        type_counts = Counter(expression_types)
        print(f"\n🎭 EXPRESSION TYPES:")
        for expr_type, count in type_counts.most_common():
            percentage = (count / len(expression_types)) * 100
            print(f"   {expr_type}: {count} ({percentage:.1f}%)")

        # Relationship development
        relationship = self.expression_interface.relationship_development
        print(f"\n🤝 RELATIONSHIP DEVELOPMENT:")
        print(f"   Trust: {relationship['trust']:.3f}")
        print(f"   Understanding: {relationship['understanding']:.3f}")
        print(f"   Depth: {relationship['depth']:.3f}")

        # Best interactions
        if len(self.dialogue_history) > 1:
            best_interaction = max(self.dialogue_history,
                                 key=lambda x: x['interaction'].nourishment_value)
            print(f"\n⭐ MOST NOURISHING INTERACTION:")
            print(f"   Type: {best_interaction['expression'].expression_type}")
            print(f"   Nourishment: {best_interaction['interaction'].nourishment_value:.3f}")
            print(f"   Response preview: \"{best_interaction['human_response'][:80]}...\"")

# Usage Examples
def demo_template_mode():
    """Demo using template mode (no API required)"""
    print("🔧 DEMO: Template Mode (No API Required)")
    print("=" * 50)

    config = {'backend': 'template'}
    dialogue = EnhancedEmileDialogue(config)
    dialogue.run_guided_dialogue(['relationship_building', 'temporal_experience'])

def demo_with_api(backend='openai'):
    """Demo using real LLM API"""
    print(f"🔧 DEMO: {backend.upper()} API Mode")
    print("=" * 50)

    if backend == 'openai':
        config = {
            'backend': 'openai',
            'model': 'gpt-4',
            'openai_key': 'your-openai-api-key'
        }
    elif backend == 'anthropic':
        config = {
            'backend': 'anthropic',
            'model': 'claude-3-sonnet-20240229',
            'anthropic_key': 'your-anthropic-api-key'
        }
    else:
        raise ValueError(f"Unknown backend: {backend}")

    dialogue = EnhancedEmileDialogue(config)
    dialogue.run_enhanced_dialogue_session(num_cycles=10)

def demo_spontaneous_expression():
    """Demo spontaneous expression capabilities"""
    print("🔧 DEMO: Spontaneous Expression")
    print("=" * 50)

    config = {'backend': 'template'}
    dialogue = EnhancedEmileDialogue(config)

    print("Running 30 cycles to see spontaneous expressions...")
    dialogue.run_enhanced_dialogue_session(num_cycles=30)

def main():
    """Main execution function to run the enhanced dialogue platform."""
    print("🚀 LAUNCHING ÉMILE'S FULLY INTEGRATED DIALOGUE PLATFORM")
    print("=====================================================================")

    # --- LLM Configuration ---
    # We will use the Gemini backend.

    try:
        # This is how you securely access the API key from Colab's Secrets
        from google.colab import userdata
        gemini_api_key = userdata.get('GOOGLE_API_KEY')
        print("✅ Successfully loaded Gemini API key from Colab Secrets.")
    except (ImportError, KeyError):
        print("❌ Could not find 'GOOGLE_API_KEY' in Colab Secrets.")
        print("   Please add it via the '🔑' icon in the left panel to use the Gemini API.")
        print("   Falling back to template mode for demonstration.")
        gemini_api_key = None

    if gemini_api_key:
        llm_config = {
            'backend': 'gemini',
            'model': 'gemini-1.5-flash-latest', # A fast and powerful model
            'gemini_key': gemini_api_key
        }
        print("   Mode: Live Dialogue with Gemini 1.5 Flash")
    else:
        llm_config = {'backend': 'template'}
        print("   Mode: Template-based Dialogue (No API)")

    print("=====================================================================")

    # Initialize the enhanced dialogue platform with the chosen configuration
    try:
        dialogue = EnhancedEmileDialogue(llm_config)

        # Start a guided session to demonstrate the different expression types
        dialogue.run_guided_dialogue()

        # You can also run a longer, spontaneous session like this:
        # dialogue.run_enhanced_dialogue_session(num_cycles=50)

    except Exception as e:
        print(f"\n❌ An error occurred during platform execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
