

#!/usr/bin/env python3
"""
MODULE-WIDE DATA FLOW MAPPER
============================

Automatically maps data flow across an ENTIRE module without decorating individual methods.
Just add 2 lines at the top of any module and it tracks ALL method calls automatically!

Usage:
    from module_wide_flow_mapper import auto_map_module_flow
    auto_map_module_flow(__name__)  # Maps the entire module!

Features:
- 🔄 Automatic method discovery and wrapping
- 📊 Complete module data flow visualization
- 🧠 Inter-method data flow tracking
- 📈 Module-level flow statistics
- 🎯 Smart filtering (ignores private methods, properties, etc.)
"""

import sys
import inspect
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Set, Optional
import numpy as np
from collections import defaultdict, deque

class ModuleFlowMapper:
    """Maps data flow across an entire module automatically"""

    def __init__(self, module_name: str, log_dir: str = "module_flow_maps"):
        self.module_name = module_name.split('.')[-1]
        self.full_module_name = module_name
        self.log_dir = Path(log_dir) / self.module_name
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Data flow tracking
        self.method_flows = {}
        self.inter_method_flows = []
        self.call_stack = deque(maxlen=50)  # Track call chains
        self.data_lineage = defaultdict(list)  # Track where data comes from

        # Statistics
        self.total_calls = 0
        self.method_call_counts = defaultdict(int)
        self.data_transformations = []

        # Setup logging
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.flow_map_file = self.log_dir / f"{self.module_name}_flow_map_{timestamp}.jsonl"
        self.module_summary_file = self.log_dir / f"{self.module_name}_module_summary_{timestamp}.md"
        self.flow_diagram_file = self.log_dir / f"{self.module_name}_flow_diagram_{timestamp}.txt"

        self._init_logs()

    def _init_logs(self):
        """Initialize module flow logs"""

        with self.flow_map_file.open("w") as f:
            f.write(json.dumps({
                "event": "module_flow_mapping_init",
                "module": self.full_module_name,
                "timestamp": datetime.now().isoformat(),
                "description": "Module-wide data flow mapping initialized"
            }) + "\n")

        with self.module_summary_file.open("w") as f:
            f.write(f"# 🗺️ Module Flow Map - {self.module_name}\n\n")
            f.write(f"**Module:** {self.full_module_name}\n")
            f.write(f"**Started:** {datetime.now():%Y-%m-%d %H:%M:%S}\n")
            f.write(f"**Purpose:** Complete module data flow visualization\n\n")
            f.write("This document tracks how data flows through the entire module.\n\n")
            f.write("---\n\n")

    def _should_track_method(self, method_name: str, method_obj: Any) -> bool:
        """Determine if a method should be tracked"""

        # Skip private methods
        if method_name.startswith('_') and not method_name.startswith('__'):
            return False

        # Skip dunder methods except important ones
        important_dunders = {'__init__', '__call__', '__enter__', '__exit__'}
        if method_name.startswith('__') and method_name not in important_dunders:
            return False

        # Skip properties, descriptors
        if isinstance(method_obj, (property, staticmethod, classmethod)):
            return False

        # Must be callable
        if not callable(method_obj):
            return False

        return True

    def _analyze_module_data(self, data: Any, data_name: str = "data") -> Dict[str, Any]:
        """Analyze data with module-aware context"""

        analysis = {
            "name": data_name,
            "type": type(data).__name__,
            "module_relevance": "unknown",
            "consciousness_indicators": [],
            "data_complexity": "low"
        }

        # Detect consciousness-related data
        consciousness_keywords = [
            "consciousness", "qualia", "awareness", "experience", "embodied",
            "agency", "valence", "arousal", "clarity", "regime", "surplus",
            "distinction", "symbolic", "phenomenal"
        ]

        data_str = str(data).lower()
        for keyword in consciousness_keywords:
            if keyword in data_str or keyword in data_name.lower():
                analysis["consciousness_indicators"].append(keyword)

        if analysis["consciousness_indicators"]:
            analysis["module_relevance"] = "consciousness_related"

        # Analyze complexity
        try:
            if isinstance(data, dict):
                analysis["complexity_score"] = len(data)
                analysis["data_complexity"] = "high" if len(data) > 10 else "medium" if len(data) > 3 else "low"
                analysis["structure"] = {
                    "keys": list(data.keys())[:5],
                    "nested_levels": self._calculate_nesting_depth(data)
                }

            elif isinstance(data, np.ndarray):
                analysis["complexity_score"] = data.size
                analysis["data_complexity"] = "high" if data.size > 100 else "medium" if data.size > 10 else "low"
                analysis["structure"] = {
                    "shape": list(data.shape),
                    "dtype": str(data.dtype),
                    "stats": {
                        "mean": (float(np.real(np.mean(data))) if np.iscomplexobj(data) else float(np.mean(data))) if data.size > 0 else None,
                        "std": float(np.std(data)) if data.size > 0 else None
                    }
                }

            elif isinstance(data, (list, tuple)):
                analysis["complexity_score"] = len(data)
                analysis["data_complexity"] = "high" if len(data) > 50 else "medium" if len(data) > 5 else "low"
                analysis["structure"] = {
                    "length": len(data),
                    "item_types": [type(item).__name__ for item in data[:3]]
                }

            elif isinstance(data, str):
                analysis["complexity_score"] = len(data)
                analysis["data_complexity"] = "high" if len(data) > 200 else "medium" if len(data) > 50 else "low"
                analysis["structure"] = {
                    "length": len(data),
                    "word_count": len(data.split()) if len(data) < 1000 else "too_long"
                }

        except Exception as e:
            analysis["analysis_error"] = str(e)

        return analysis

    def _calculate_nesting_depth(self, obj: Any, current_depth: int = 0, max_depth: int = 10) -> int:
        """Calculate nesting depth of data structures"""
        if current_depth >= max_depth:
            return current_depth

        if isinstance(obj, dict):
            if not obj:
                return current_depth
            return max(self._calculate_nesting_depth(v, current_depth + 1, max_depth) for v in obj.values())
        elif isinstance(obj, (list, tuple)):
            if not obj:
                return current_depth
            return max(self._calculate_nesting_depth(item, current_depth + 1, max_depth) for item in obj)
        else:
            return current_depth

    def _track_method_call(self, method_name: str, class_name: str, inputs: Dict, outputs: Any, execution_time: float):
        """Track a method call in the module flow"""

        self.total_calls += 1
        self.method_call_counts[method_name] += 1
        call_id = f"{class_name}.{method_name}_{self.total_calls}"

        # Analyze inputs and outputs
        input_analyses = {}
        for param_name, param_value in inputs.items():
            input_analyses[param_name] = self._analyze_module_data(param_value, param_name)

        output_analysis = self._analyze_module_data(outputs, "result")

        # Create flow record
        flow_record = {
            "timestamp": datetime.now().isoformat(),
            "call_id": call_id,
            "module": self.module_name,
            "class": class_name,
            "method": method_name,
            "execution_time_ms": round(execution_time * 1000, 3),
            "inputs": input_analyses,
            "output": output_analysis,
            "call_context": {
                "total_calls_so_far": self.total_calls,
                "method_call_count": self.method_call_counts[method_name],
                "call_stack_depth": len(self.call_stack)
            },
            "module_flow_insights": self._generate_flow_insights(input_analyses, output_analysis, method_name)
        }

        # Track call in stack
        self.call_stack.append({
            "call_id": call_id,
            "method": f"{class_name}.{method_name}",
            "timestamp": time.time()
        })

        # Store flow record
        self.method_flows[call_id] = flow_record

        # Log to file
        with self.flow_map_file.open("a") as f:
            f.write(json.dumps(flow_record, default=str) + "\n")

        # Update module summary
        self._update_module_summary(flow_record)

        return flow_record

    def _generate_flow_insights(self, inputs: Dict, output: Dict, method_name: str) -> List[str]:
        """Generate insights about this method's data flow"""

        insights = []

        # Consciousness processing detection
        consciousness_indicators = output.get("consciousness_indicators", [])
        if consciousness_indicators:
            insights.append(f"🧠 Consciousness processing: {', '.join(consciousness_indicators)}")

        # Data complexity changes
        input_complexities = [inp.get("complexity_score", 0) for inp in inputs.values()]
        output_complexity = output.get("complexity_score", 0)

        if input_complexities:
            total_input_complexity = sum(input_complexities)
            if output_complexity > total_input_complexity * 1.5:
                insights.append(f"📈 Data enrichment: {total_input_complexity} → {output_complexity}")
            elif output_complexity < total_input_complexity * 0.5:
                insights.append(f"📉 Data reduction: {total_input_complexity} → {output_complexity}")

        # Type transformations
        input_types = [inp["type"] for inp in inputs.values()]
        output_type = output["type"]

        if output_type not in input_types:
            insights.append(f"🔄 Type transformation: {input_types} → {output_type}")

        # Method-specific insights
        if "cognitive" in method_name.lower():
            insights.append("🧩 Cognitive processing operation")
        elif "generate" in method_name.lower():
            insights.append("✨ Generative operation")
        elif "process" in method_name.lower():
            insights.append("⚙️ Processing operation")
        elif "step" in method_name.lower():
            insights.append("👣 Step-wise operation")

        return insights

    def _update_module_summary(self, flow_record: Dict):
        """Update the module summary with new flow information"""

        if self.total_calls % 5 == 0:  # Update every 5 calls to avoid too much I/O
            with self.module_summary_file.open("a") as f:
                timestamp = datetime.now().strftime("%H:%M:%S")
                method = f"{flow_record['class']}.{flow_record['method']}"

                f.write(f"### 🔄 [{timestamp}] {method}\n\n")

                # Input summary
                f.write("**📥 Inputs:**\n")
                for param_name, analysis in flow_record["inputs"].items():
                    relevance = analysis.get("module_relevance", "unknown")
                    indicators = analysis.get("consciousness_indicators", [])
                    relevance_icon = "🧠" if relevance == "consciousness_related" else "📊"

                    f.write(f"- {relevance_icon} **{param_name}**: {analysis['type']}")
                    if indicators:
                        f.write(f" ({', '.join(indicators)})")
                    f.write("\n")

                # Output summary
                output = flow_record["output"]
                output_relevance = output.get("module_relevance", "unknown")
                output_icon = "🧠" if output_relevance == "consciousness_related" else "📤"
                f.write(f"\n{output_icon} **Output**: {output['type']}")
                if output.get("consciousness_indicators"):
                    f.write(f" ({', '.join(output['consciousness_indicators'])})")
                f.write("\n")

                # Insights
                if flow_record["module_flow_insights"]:
                    f.write("\n**💡 Flow Insights:**\n")
                    for insight in flow_record["module_flow_insights"]:
                        f.write(f"- {insight}\n")

                f.write(f"\n**⚡ Performance:** {flow_record['execution_time_ms']}ms\n\n")
                f.write("---\n\n")

    def wrap_method(self, class_obj: Any, method_name: str, original_method: Any):
        """Wrap a method to track its data flow - FIXED VERSION"""

        # REMOVED: This method is no longer used in the fixed map_module
        # The wrapping is now done directly in map_module to avoid recursion issues
        # This method is kept for compatibility but shouldn't be called

        def wrapped_method(*args, **kwargs):
            print(f"⚠️ WARNING: Using deprecated wrap_method - should use new inline wrapping")
            return original_method(*args, **kwargs)

        return wrapped_method

    def map_module(self, module_obj: Any):
        """Map data flow for an entire module - FIXED VERSION (keeps all your existing functionality)"""
        mapped_methods = []

        #print(f"🔍 DEBUG: Mapping module {self.full_module_name}")
        #print(f"🔍 DEBUG: Module object: {module_obj}")

        # FIXED: Better class detection for consciousness modules
        for name in dir(module_obj):
            obj = getattr(module_obj, name)

            # Skip built-in attributes
            if name.startswith('__') and name.endswith('__'):
                continue

            #print(f"🔍 DEBUG: Examining {name} (type: {type(obj).__name__})")

            # FIXED: More robust class detection
            if inspect.isclass(obj):
                obj_module = getattr(obj, '__module__', None)
                #print(f"🔍 DEBUG: Class {name} belongs to module: {obj_module}")
                #print(f"🔍 DEBUG: Target module: {self.full_module_name}")

                # FIXED: Handle module name matching more robustly
                is_our_class = False

                # Method 1: Direct module name match
                if obj_module == self.full_module_name:
                    is_our_class = True
                    print(f"📍 MATCH: {name} (direct module match)")

                # Method 2: Source file match (for edge cases)
                if not is_our_class:
                    try:
                        module_source = inspect.getfile(module_obj)
                        class_source = inspect.getfile(obj)
                        if module_source == class_source:
                            is_our_class = True
                            print(f"📍 MATCH: {name} (source file match)")
                    except (TypeError, OSError):
                        pass

                if is_our_class:
                    print(f"✅ MAPPING CLASS: {name}")
                    class_methods_mapped = 0

                    # Map all methods in the class
                    for method_name in dir(obj):
                        try:
                            method_obj = getattr(obj, method_name)

                            if self._should_track_method(method_name, method_obj):
                                print(f"   🔧 Wrapping method: {method_name}")

                                # KEEP YOUR EXISTING WRAPPER CREATION CODE HERE
                                # This is the same sophisticated wrapper from your original mapper
                                try:
                                    original_method = method_obj

                                    def create_method_wrapper(orig_method, m_name, cls_name):
                                        def wrapped_method(*args, **kwargs):
                                            start_time = time.time()

                                            # Capture inputs (skip 'self') - YOUR ORIGINAL LOGIC
                                            try:
                                                sig = inspect.signature(orig_method)
                                                bound_args = sig.bind(*args, **kwargs)
                                                bound_args.apply_defaults()
                                                inputs = dict(bound_args.arguments)

                                                # Remove 'self' if present
                                                if 'self' in inputs:
                                                    inputs.pop('self')

                                            except Exception:
                                                # Fallback if signature inspection fails
                                                inputs = {"args": args[1:] if len(args) > 1 else [], "kwargs": kwargs}

                                            # Execute original method
                                            try:
                                                result = orig_method(*args, **kwargs)
                                                success = True
                                            except Exception as e:
                                                result = None
                                                success = False
                                                # Log the error but re-raise
                                                execution_time = time.time() - start_time
                                                self._track_method_call(m_name, cls_name, inputs, f"ERROR: {e}", execution_time)
                                                raise

                                            # Track successful call - USES YOUR EXISTING _track_method_call
                                            if success:
                                                execution_time = time.time() - start_time
                                                self._track_method_call(m_name, cls_name, inputs, result, execution_time)

                                            return result

                                        return wrapped_method

                                    # Create the wrapper
                                    wrapped = create_method_wrapper(original_method, method_name, name)

                                    # Replace the method
                                    setattr(obj, method_name, wrapped)

                                    mapped_methods.append(f"{name}.{method_name}")
                                    class_methods_mapped += 1

                                except Exception as wrap_error:
                                    print(f"   ❌ Failed to wrap {method_name}: {wrap_error}")
                                    continue

                        except Exception as method_error:
                            print(f"   ❌ Error accessing method {method_name}: {method_error}")
                            continue

                    if class_methods_mapped > 0:
                        print(f"   ✅ Mapped {class_methods_mapped} methods in class {name}")
                    else:
                        print(f"   ⚠️ No methods mapped in class {name}")
                else:
                    print(f"   ⏭️ Skipping class {name} (belongs to {obj_module})")

            # ALSO check for module-level functions (KEEP YOUR ORIGINAL LOGIC)
            elif inspect.isfunction(obj):
                obj_module = getattr(obj, '__module__', None)
                if obj_module == self.full_module_name and self._should_track_method(name, obj):
                    print(f"📍 MAPPING FUNCTION: {name}")
                    try:
                        # KEEP YOUR EXISTING FUNCTION WRAPPER CODE
                        original_func = obj

                        def create_function_wrapper(orig_func, func_name):
                            def wrapped_function(*args, **kwargs):
                                start_time = time.time()

                                # Capture inputs - YOUR ORIGINAL LOGIC
                                try:
                                    sig = inspect.signature(orig_func)
                                    bound_args = sig.bind(*args, **kwargs)
                                    bound_args.apply_defaults()
                                    inputs = dict(bound_args.arguments)
                                except Exception:
                                    inputs = {"args": args, "kwargs": kwargs}

                                # Execute function
                                try:
                                    result = orig_func(*args, **kwargs)
                                    success = True
                                except Exception as e:
                                    result = None
                                    success = False
                                    execution_time = time.time() - start_time
                                    self._track_method_call(func_name, "Module", inputs, f"ERROR: {e}", execution_time)
                                    raise

                                # Track successful call
                                if success:
                                    execution_time = time.time() - start_time
                                    self._track_method_call(func_name, "Module", inputs, result, execution_time)

                                return result

                            return wrapped_function

                        wrapped = create_function_wrapper(original_func, name)
                        setattr(module_obj, name, wrapped)
                        mapped_methods.append(f"module.{name}")

                    except Exception as func_error:
                        print(f"   ❌ Failed to wrap function {name}: {func_error}")
                        continue

        print(f"🗺️ Module flow mapping complete: {len(mapped_methods)} methods mapped")

        if len(mapped_methods) == 0:
            print("⚠️ NO METHODS MAPPED - Debug info:")
            print(f"   Module name: {self.full_module_name}")
            print(f"   Classes found: {[name for name in dir(module_obj) if inspect.isclass(getattr(module_obj, name))]}")
            print(f"   Functions found: {[name for name in dir(module_obj) if inspect.isfunction(getattr(module_obj, name))]}")

        # KEEP YOUR EXISTING LOGGING CODE
        with self.flow_map_file.open("a") as f:
            f.write(json.dumps({
                "event": "module_mapping_complete",
                "timestamp": datetime.now().isoformat(),
                "methods_mapped": mapped_methods,
                "total_methods": len(mapped_methods),
                "debug_info": {
                    "module_name": self.full_module_name,
                    "classes_in_module": [name for name in dir(module_obj) if inspect.isclass(getattr(module_obj, name))]
                }
            }, default=str) + "\n")

        return mapped_methods

    def generate_module_flow_diagram(self):
        """Generate a text-based flow diagram of the module"""

        if not self.method_flows:
            return

        # Analyze call patterns
        method_calls = defaultdict(int)
        class_calls = defaultdict(int)
        data_types_flow = defaultdict(set)

        for flow_record in self.method_flows.values():
            method_name = flow_record["method"]
            class_name = flow_record["class"]

            method_calls[method_name] += 1
            class_calls[class_name] += 1

            # Track data types that flow through each method
            for input_analysis in flow_record["inputs"].values():
                data_types_flow[method_name].add(input_analysis["type"])

            data_types_flow[method_name].add(flow_record["output"]["type"])

        # Generate diagram
        with self.flow_diagram_file.open("w") as f:
            f.write(f"MODULE FLOW DIAGRAM - {self.module_name}\n")
            f.write("=" * 50 + "\n\n")

            f.write("📊 CALL FREQUENCY:\n")
            for method, count in sorted(method_calls.items(), key=lambda x: x[1], reverse=True):
                bar = "█" * min(20, count)
                f.write(f"  {method:20} {bar} ({count} calls)\n")

            f.write("\n🏗️ CLASS ACTIVITY:\n")
            for class_name, count in sorted(class_calls.items(), key=lambda x: x[1], reverse=True):
                f.write(f"  {class_name}: {count} method calls\n")

            f.write("\n🔄 DATA TYPE FLOWS:\n")
            for method, types in data_types_flow.items():
                f.write(f"  {method}: {' → '.join(sorted(types))}\n")

            f.write(f"\n📈 TOTAL STATISTICS:\n")
            f.write(f"  Total method calls: {self.total_calls}\n")
            f.write(f"  Unique methods called: {len(method_calls)}\n")
            f.write(f"  Active classes: {len(class_calls)}\n")

    def get_module_statistics(self) -> Dict[str, Any]:
        """Get comprehensive module flow statistics"""

        stats = {
            "module": self.module_name,
            "total_calls": self.total_calls,
            "unique_methods": len(self.method_call_counts),
            "method_call_frequency": dict(self.method_call_counts),
            "consciousness_methods": [],
            "data_transformations": len(self.data_transformations),
            "call_stack_info": {
                "current_depth": len(self.call_stack),
                "recent_calls": [call["method"] for call in list(self.call_stack)[-5:]]
            }
        }

        # Find consciousness-related methods
        for call_id, flow_record in self.method_flows.items():
            if flow_record["output"].get("consciousness_indicators"):
                stats["consciousness_methods"].append(flow_record["method"])

        stats["consciousness_methods"] = list(set(stats["consciousness_methods"]))

        return stats

# ========================================================================
# MAIN API FUNCTIONS
# ========================================================================

_module_mappers = {}  # Global registry of module mappers

def auto_map_module_flow(module_name: str) -> ModuleFlowMapper:
    """
    MAIN FUNCTION: Automatically map data flow for an entire module.

    Usage:
        from emile_cogito.module_wide_flow_mapper import auto_map_module_flow
        auto_map_module_flow(__name__)  # Maps the entire current module!

    Args:
        module_name: The module name (usually __name__)

    Returns:
        ModuleFlowMapper instance for this module
    """

    if module_name in _module_mappers:
        return _module_mappers[module_name]

    # Create mapper
    mapper = ModuleFlowMapper(module_name)

    # Get the actual module object
    module_obj = sys.modules[module_name]

    # Map the entire module
    mapped_methods = mapper.map_module(module_obj)

    # Store in registry
    _module_mappers[module_name] = mapper

    print(f"🗺️ AUTO-MAPPED MODULE: {module_name}")
    print(f"   📍 {len(mapped_methods)} methods now tracked")
    print(f"   📁 Logs: module_flow_maps/{mapper.module_name}/")

    return mapper

def get_module_mapper(module_name: str) -> Optional[ModuleFlowMapper]:
    """Get the flow mapper for a module (if it exists)"""
    return _module_mappers.get(module_name)

def generate_all_flow_diagrams():
    """Generate flow diagrams for all mapped modules"""
    for mapper in _module_mappers.values():
        mapper.generate_module_flow_diagram()
        print(f"📊 Generated flow diagram for {mapper.module_name}")

def get_all_module_stats() -> Dict[str, Dict]:
    """Get statistics for all mapped modules"""
    return {name: mapper.get_module_statistics() for name, mapper in _module_mappers.items()}

# ========================================================================
# USAGE EXAMPLES
# ========================================================================

def create_example_module():
    """Create an example module to demonstrate module-wide mapping"""

    example_code = '''
# example_consciousness_module.py
"""
Example consciousness module demonstrating module-wide flow mapping.
"""

# Add these 2 lines to ANY module for complete flow mapping:
from emile_cogito.module_wide_flow_mapper import auto_map_module_flow
auto_map_module_flow(__name__)  # Maps the entire module automatically!

import numpy as np

class ConsciousnessProcessor:
    """Example consciousness processor"""

    def __init__(self, config):
        self.config = config
        self.state = {"consciousness": 0.5, "regime": "emerging"}

    def process_sensory_input(self, sensory_data, context=None):
        """Process sensory input - automatically tracked!"""
        consciousness_boost = np.mean(sensory_data) * 0.3
        self.state["consciousness"] = min(1.0, self.state["consciousness"] + consciousness_boost)

        return {
            "consciousness_level": self.state["consciousness"],
            "qualia_richness": np.std(sensory_data),
            "sensory_integration": len(sensory_data),
            "context_influence": len(context) if context else 0
        }

    def generate_response(self, consciousness_state):
        """Generate response - automatically tracked!"""
        level = consciousness_state["consciousness_level"]

        if level > 0.8:
            return {"response": "transcendent awareness", "confidence": 0.95}
        elif level > 0.5:
            return {"response": "conscious awareness", "confidence": 0.7}
        else:
            return {"response": "dim awareness", "confidence": 0.3}

class QualiaGenerator:
    """Example qualia generator"""

    def generate_qualia(self, consciousness_state, sensory_input):
        """Generate qualitative experience - automatically tracked!"""
        return {
            "valence": consciousness_state["consciousness_level"] * 0.8,
            "arousal": np.mean(sensory_input) if len(sensory_input) > 0 else 0,
            "clarity": consciousness_state.get("qualia_richness", 0.5),
            "phenomenal_binding": len(sensory_input) * 0.1
        }

# Module-level function - also automatically tracked!
def integrate_consciousness_qualia(consciousness_result, qualia_result):
    """Integrate consciousness and qualia - automatically tracked!"""
    return {
        "integrated_consciousness": consciousness_result["consciousness_level"],
        "integrated_experience": {
            "cognitive": consciousness_result,
            "phenomenal": qualia_result
        },
        "unity_score": consciousness_result["consciousness_level"] * qualia_result["valence"]
    }

# Everything is now automatically tracked! No decorators needed!
'''

    with open("example_consciousness_module.py", "w") as f:
        f.write(example_code)

    print("📝 Created example_consciousness_module.py")
    print("🎯 This shows how to map an entire module with just 2 lines!")

if __name__ == "__main__":
    print("🗺️ MODULE-WIDE DATA FLOW MAPPER")
    print("=" * 50)
    print("Map data flow across ENTIRE modules with just 2 lines!")
    print("Perfect for understanding consciousness system architecture 🧠")
    print()

    # Create examples and documentation
    create_example_module()

    print("\n🎯 USAGE:")
    print("Add these 2 lines to ANY module:")
    print("  from emile_cogito.module_wide_flow_mapper import auto_map_module_flow")
    print("  auto_map_module_flow(__name__)")
    print("\nThen every method in that module is automatically tracked! 🚀")
