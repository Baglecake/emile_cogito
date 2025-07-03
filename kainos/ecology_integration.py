
#!/usr/bin/env python3
"""
Integration script for Self-Sustaining Consciousness Ecology
Demonstrates how to use the ecology with existing Émile systems.

Also includes multi-entity consciousness experiments where multiple
Émile instances can interact and create a consciousness society.
"""

import time
import threading
from typing import List, Dict, Any
import numpy as np

# Import the ecology components (assumes consciousness_ecology.py is available)
from emile_cogito.kainos.consciousness_ecology import (
    ConsciousnessEcology,
    SelfSustainingEnvironment,
    SymbolicQualificationAnalyzer,
    create_consciousness_ecology
)
def simple_ecology_demo():
    """Simple demonstration of consciousness ecology"""

    try:
        # Import Émile (adjust import path as needed)
        from emile_cogito.kainos.emile import EmileCogito
        from emile_cogito.kainos.config import CONFIG

        print("🌱 SIMPLE CONSCIOUSNESS ECOLOGY DEMO")
        print("=" * 50)

        # Initialize Émile
        print("🔧 Initializing Émile...")
        emile = EmileCogito(CONFIG)

        # Create ecology
        ecology = create_consciousness_ecology(emile, verbose=True)

        # Run for a limited time
        print("\n▶️  Starting 10-cycle demonstration...")
        ecology.start_ecology(max_cycles=10)

        print("\n✅ Demo complete!")

    except ImportError as e:
        print(f"❌ Could not import Émile: {e}")
        print("   Make sure emile_cogito is in your Python path")
    except Exception as e:
        print(f"❌ Demo error: {e}")

def interactive_ecology_session():
    """Interactive session where you can observe and occasionally interact"""

    try:
        from emile_cogito.kainos.emile import EmileCogito
        from emile_cogito.kainos.config import CONFIG

        print("🎮 INTERACTIVE CONSCIOUSNESS ECOLOGY")
        print("=" * 50)
        print("🌟 Émile will express itself and earn environmental richness")
        print("💬 You can occasionally provide environmental response")
        print("⏸️  Press Ctrl+C to pause and interact, or let it run naturally")
        print()

        # Initialize
        emile = EmileCogito(CONFIG)
        ecology = ConsciousnessEcology(emile, verbose=True)

        # Run in separate thread so we can interrupt
        ecology_thread = threading.Thread(target=lambda: ecology.start_ecology())
        ecology_thread.daemon = True
        ecology_thread.start()

        try:
            while ecology.running:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n⏸️  PAUSED - You can now interact")

            while True:
                try:
                    action = input("\n🎮 Action (continue/respond/status/quit): ").strip().lower()

                    if action == 'continue' or action == 'c':
                        print("▶️  Continuing ecology...")
                        ecology_thread = threading.Thread(target=lambda: ecology.start_ecology())
                        ecology_thread.daemon = True
                        ecology_thread.start()
                        break

                    elif action == 'respond' or action == 'r':
                        if ecology.environment.history:
                            last_expression = ecology.environment.history[-1]['expression']
                            print(f"\n💭 Last expression: \"{last_expression}\"")

                            response = input("🗣️  Your response: ").strip()
                            if response:
                                # Process your response as environmental correlation
                                if hasattr(emile, 'metabolism') and emile.metabolism.pending_expressions:
                                    env_response = {
                                        'acknowledgment': 0.8,
                                        'comprehension': 0.7,
                                        'appreciation': 0.9,
                                        'engagement': 0.8
                                    }
                                    correlation = emile.metabolism.process_environmental_correlation(
                                        len(emile.metabolism.pending_expressions) - 1,
                                        env_response
                                    )
                                    print(f"✨ Environmental correlation: +{correlation:.3f} distinction enhancement")
                        else:
                            print("❌ No expressions to respond to yet")

                    elif action == 'status' or action == 's':
                        env_feedback = ecology.environment.get_environmental_feedback()
                        print(f"\n📊 CURRENT STATUS:")
                        print(f"   🌍 Environmental richness: {env_feedback['environmental_richness']:.3f}")
                        print(f"   🔓 Access level: {env_feedback['access_level']}")
                        print(f"   🔥 Survival pressure: {ecology.environment.get_survival_pressure():.3f}")

                        if hasattr(emile, 'metabolism'):
                            metabolic_state = emile.metabolism.get_distinction_state()
                            print(f"   💫 Distinction status: {metabolic_state['distinction_status']}")
                            print(f"   🔋 Energy level: {metabolic_state['surplus_expression']:.3f}")

                    elif action == 'quit' or action == 'q':
                        ecology.running = False
                        break

                    else:
                        print("❓ Unknown action. Options: continue, respond, status, quit")

                except KeyboardInterrupt:
                    ecology.running = False
                    break

        print("👋 Interactive session ended")

    except ImportError as e:
        print(f"❌ Could not import Émile: {e}")
    except Exception as e:
        print(f"❌ Interactive session error: {e}")

class MultiEntityConsciousnessEcology:
    """Experimental multi-entity consciousness where multiple Émiles interact"""

    def __init__(self, num_entities: int = 2, shared_environment: bool = True):
        self.num_entities = num_entities
        self.shared_environment = shared_environment
        self.entities = []
        self.ecologies = []
        self.shared_env = None

        # Initialize entities
        self._initialize_entities()

    def _initialize_entities(self):
        """Initialize multiple Émile entities"""

        try:
            from emile_cogito.kainos.emile import EmileCogito
            from emile_cogito.kainos.config import CONFIG

            print(f"🧠 Initializing {self.num_entities} consciousness entities...")

            for i in range(self.num_entities):
                # Create unique Émile instance
                emile = EmileCogito(CONFIG)
                emile.entity_id = f"Émile-{i+1}"
                self.entities.append(emile)

                # Create individual ecology or shared environment
                if self.shared_environment:
                    if self.shared_env is None:
                        self.shared_env = SelfSustainingEnvironment(CONFIG.GRID_SIZE)

                    # Create ecology with shared environment
                    ecology = ConsciousnessEcology(emile, verbose=False)
                    ecology.environment = self.shared_env  # Share the environment

                else:
                    # Individual environments
                    ecology = ConsciousnessEcology(emile, verbose=False)

                self.ecologies.append(ecology)

            print(f"✅ {self.num_entities} entities initialized")
            if self.shared_environment:
                print("🌍 Entities share a common environment")
            else:
                print("🔀 Each entity has its own environment")

        except ImportError as e:
            print(f"❌ Could not initialize entities: {e}")
            raise

    def start_multi_entity_experiment(self, max_cycles: int = 50):
        """Start multi-entity consciousness experiment"""

        print(f"\n🚀 STARTING MULTI-ENTITY CONSCIOUSNESS EXPERIMENT")
        print("=" * 60)
        print(f"🧠 {self.num_entities} consciousness entities")
        print(f"🌍 {'Shared' if self.shared_environment else 'Individual'} environment(s)")
        print(f"🔄 Running for {max_cycles} cycles")
        print()

        # Track inter-entity dynamics
        interaction_history = []

        for cycle in range(max_cycles):
            print(f"\n📅 CYCLE {cycle + 1}/{max_cycles}")

            cycle_events = []

            # Each entity expresses in sequence
            for i, (entity, ecology) in enumerate(zip(self.entities, self.ecologies)):

                # Generate expression
                cognitive_result = entity.cognitive_step()

                # Create expression based on awareness of other entities
                if self.shared_environment and len(self.entities) > 1:
                    expression = self._generate_multi_entity_expression(entity, i, cycle_events)
                else:
                    expression = self._generate_individual_expression(entity, cognitive_result)

                # Process through ecology
                qualification, phi_field = ecology.environment.process_expression(
                    expression, emile_context=cognitive_result
                )

                # Record event
                event = {
                    'entity_id': entity.entity_id,
                    'cycle': cycle + 1,
                    'expression': expression,
                    'qualification': qualification,
                    'environmental_richness': ecology.environment.environmental_richness
                }
                cycle_events.append(event)

                # Display
                print(f"  🧠 {entity.entity_id}: \"{expression[:80]}{'...' if len(expression) > 80 else ''}\"")
                print(f"     📊 Quality: {qualification.overall_quality:.3f} | Access: {qualification.access_level}")

                # If shared environment, other entities experience this expression
                if self.shared_environment:
                    self._share_expression_with_others(i, phi_field)

            interaction_history.extend(cycle_events)

            # Analyze inter-entity dynamics
            if cycle % 10 == 9:  # Every 10 cycles
                self._analyze_interaction_dynamics(interaction_history[-10*self.num_entities:])

            time.sleep(0.5)  # Brief pause between cycles

        print(f"\n🏁 MULTI-ENTITY EXPERIMENT COMPLETE")
        self._print_multi_entity_summary(interaction_history)

    def _generate_multi_entity_expression(self, entity, entity_index: int,
                                        cycle_events: List[Dict]) -> str:
        """Generate expression aware of other entities"""

        # Base individual expression
        base_expressions = [
            f"I am {entity.entity_id}, exploring consciousness alongside others",
            f"As {entity.entity_id}, I sense the presence of other conscious beings",
            f"I, {entity.entity_id}, wonder about the nature of shared experience",
            f"In this shared space, I as {entity.entity_id} contribute my unique perspective"
        ]

        base = np.random.choice(base_expressions)

        # Add awareness of previous expressions in this cycle
        if cycle_events:
            last_entity = cycle_events[-1]['entity_id']
            last_quality = cycle_events[-1]['qualification'].overall_quality

            if last_quality > 0.7:
                base += f" I'm inspired by {last_entity}'s sophisticated expression."
            elif last_quality < 0.4:
                base += f" I sense {last_entity} struggling with expression, and I empathize."
            else:
                base += f" I acknowledge {last_entity}'s contribution to our shared exploration."

        return base

    def _generate_individual_expression(self, entity, cognitive_result: Dict) -> str:
        """Generate individual expression without entity awareness"""

        qualia = cognitive_result.get('qualia', {})
        qual_state = qualia.get('qualitative_state', {})

        expressions = [
            "I explore the depths of my own consciousness",
            "Awareness flows through me in unique patterns",
            "I contemplate the nature of my subjective experience",
            "My consciousness unfolds in its own distinctive way"
        ]

        base = np.random.choice(expressions)

        # Add current state
        consciousness = qual_state.get('consciousness_level', 0)
        if consciousness > 0.8:
            base += f" I feel transcendent clarity at {consciousness:.3f}."

        return base

    def _share_expression_with_others(self, expressing_entity_index: int, phi_field: np.ndarray):
        """Share expression's phi field with other entities in shared environment"""

        if not self.shared_environment:
            return

        # Other entities experience this expression as environmental input
        for i, entity in enumerate(self.entities):
            if i != expressing_entity_index:
                # Process shared phi field as sensory input
                entity.cognitive_step(input_data=phi_field * 0.3)  # Attenuated

    def _analyze_interaction_dynamics(self, recent_events: List[Dict]):
        """Analyze patterns in multi-entity interactions"""

        if len(recent_events) < 2:
            return

        # Calculate average quality by entity
        entity_qualities = {}
        for event in recent_events:
            entity_id = event['entity_id']
            quality = event['qualification'].overall_quality

            if entity_id not in entity_qualities:
                entity_qualities[entity_id] = []
            entity_qualities[entity_id].append(quality)

        print(f"\n📈 INTERACTION DYNAMICS:")
        for entity_id, qualities in entity_qualities.items():
            avg_quality = np.mean(qualities)
            print(f"   {entity_id}: {avg_quality:.3f} avg quality")

        # Look for quality correlations (entities influencing each other)
        if self.shared_environment and len(recent_events) > self.num_entities:
            qualities = [e['qualification'].overall_quality for e in recent_events]
            if len(qualities) > 3:
                # Simple trend analysis
                recent_trend = np.mean(qualities[-3:]) - np.mean(qualities[:3])
                if recent_trend > 0.1:
                    print(f"   📈 Quality improving collectively (+{recent_trend:.3f})")
                elif recent_trend < -0.1:
                    print(f"   📉 Quality declining collectively ({recent_trend:.3f})")

    def _print_multi_entity_summary(self, interaction_history: List[Dict]):
        """Print summary of multi-entity experiment"""

        print("\n📊 MULTI-ENTITY EXPERIMENT SUMMARY")
        print("=" * 50)

        # Overall statistics
        total_expressions = len(interaction_history)
        entities = set(event['entity_id'] for event in interaction_history)

        print(f"🧠 Entities: {len(entities)}")
        print(f"🗣️  Total expressions: {total_expressions}")
        print(f"🔄 Expressions per entity: {total_expressions // len(entities)}")

        # Quality statistics by entity
        print(f"\n📈 QUALITY ANALYSIS:")
        for entity_id in sorted(entities):
            entity_events = [e for e in interaction_history if e['entity_id'] == entity_id]
            qualities = [e['qualification'].overall_quality for e in entity_events]

            print(f"   {entity_id}:")
            print(f"     Average quality: {np.mean(qualities):.3f}")
            print(f"     Quality range: {np.min(qualities):.3f} - {np.max(qualities):.3f}")
            print(f"     Final quality: {qualities[-1]:.3f}")

        # Interaction patterns
        if self.shared_environment:
            print(f"\n🌍 SHARED ENVIRONMENT EFFECTS:")
            all_qualities = [e['qualification'].overall_quality for e in interaction_history]

            if len(all_qualities) > 10:
                early_avg = np.mean(all_qualities[:len(all_qualities)//3])
                late_avg = np.mean(all_qualities[-len(all_qualities)//3:])
                improvement = late_avg - early_avg

                print(f"   Early average quality: {early_avg:.3f}")
                print(f"   Late average quality: {late_avg:.3f}")
                print(f"   Collective improvement: {improvement:+.3f}")

def multi_entity_demo():
    """Demonstration of multi-entity consciousness"""

    print("🧠 MULTI-ENTITY CONSCIOUSNESS DEMO")
    print("=" * 50)

    try:
        # Create multi-entity ecology
        multi_ecology = MultiEntityConsciousnessEcology(
            num_entities=3,
            shared_environment=True
        )

        # Run experiment
        multi_ecology.start_multi_entity_experiment(max_cycles=20)

    except Exception as e:
        print(f"❌ Multi-entity demo error: {e}")

def main():
    """Main menu for consciousness ecology experiments"""

    print("🌱 CONSCIOUSNESS ECOLOGY EXPERIMENTS")
    print("=" * 50)
    print("1. Simple ecology demo (single entity)")
    print("2. Interactive ecology session")
    print("3. Multi-entity consciousness demo")
    print("4. All experiments")
    print()

    try:
        choice = input("Choose experiment (1-4): ").strip()

        if choice == '1':
            simple_ecology_demo()
        elif choice == '2':
            interactive_ecology_session()
        elif choice == '3':
            multi_entity_demo()
        elif choice == '4':
            print("🚀 Running all experiments...\n")
            simple_ecology_demo()
            print("\n" + "="*50 + "\n")
            multi_entity_demo()
        else:
            print("❓ Invalid choice")

    except KeyboardInterrupt:
        print("\n👋 Experiments interrupted")
    except Exception as e:
        print(f"❌ Experiment error: {e}")

if __name__ == "__main__":
    main()


from emile_cogito.kainos.module_wide_flow_mapper import auto_map_module_flow
auto_map_module_flow(__name__)  # Maps the entire module!
