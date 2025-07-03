
#!/usr/bin/env python3
"""
Universal Drop-In Module Logging for Émile Framework
===================================================

SIMPLE DROP-IN PATTERN:
Just add these 3 lines to the top of ANY module:

    from universal_module_logging import setup_module_logging
    logger = setup_module_logging(__name__)
    logged_method = logger.method_decorator

Then use @logged_method on any method you want tracked!

For events: logger.log_event("event_name", "description", data)
For consciousness events: logger.log_consciousness("transition", from_state, to_state)
"""

import json
import time
import threading
import inspect
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Dict, Any, Optional, Callable

class UniversalModuleLogger:
    """
    Drop-in logger for any module with zero configuration needed.
    Automatically detects module type and adjusts logging appropriately.
    """

    def __init__(self, module_name: str, log_dir: str = "module_logs"):
        self.module_name = module_name.split('.')[-1]  # Get just the module name
        self.full_module_name = module_name

        # Auto-detect module type and set appropriate style
        self.module_type = self._detect_module_type()
        self.style = self._get_module_style()

        # Setup logging directory
        self.module_dir = Path(log_dir) / self.module_name
        self.module_dir.mkdir(parents=True, exist_ok=True)

        # Setup log files
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.json_log = self.module_dir / f"{self.module_name}_calls_{ts}.jsonl"
        self.events_log = self.module_dir / f"{self.module_name}_events_{ts}.md"

        # Initialize logs
        self._init_logs()

        # Statistics
        self.call_count = 0
        self.event_count = 0
        self.start_time = time.time()

        # Create the decorator
        self.method_decorator = self._create_method_decorator()

    def _detect_module_type(self) -> str:
        """Auto-detect what kind of module this is"""
        name = self.module_name.lower()

        if 'qualia' in name:
            return 'consciousness'
        elif any(word in name for word in ['memory', 'temporal', 'autobiographical']):
            return 'memory'
        elif any(word in name for word in ['philosophy', 'philosophical']):
            return 'philosophy'
        elif any(word in name for word in ['vocab', 'symbol', 'language']):
            return 'vocabulary'
        elif any(word in name for word in ['ecology', 'environment', 'expression']):
            return 'ecology'
        elif any(word in name for word in ['essay', 'writing', 'narrative']):
            return 'essays'
        elif any(word in name for word in ['agent', 'multi', 'society']):
            return 'agents'
        elif any(word in name for word in ['metabolic', 'surplus', 'distinction']):
            return 'metabolic'
        elif any(word in name for word in ['qse', 'quantum', 'core']):
            return 'quantum'
        else:
            return 'technical'

    def _get_module_style(self) -> Dict[str, Any]:
        """Get logging style based on module type"""
        styles = {
            'consciousness': {
                'icon': '🧠',
                'description': 'Consciousness Experience',
                'narrative': 'consciousness',
                'track_performance': True
            },
            'memory': {
                'icon': '💭',
                'description': 'Memory & Temporal Processing',
                'narrative': 'consciousness',
                'track_performance': True
            },
            'philosophy': {
                'icon': '🧭',
                'description': 'Philosophical Processing',
                'narrative': 'philosophical',
                'track_performance': False
            },
            'vocabulary': {
                'icon': '🔤',
                'description': 'Symbol & Vocabulary Learning',
                'narrative': 'consciousness',
                'track_performance': False
            },
            'ecology': {
                'icon': '🌍',
                'description': 'Consciousness Ecology',
                'narrative': 'consciousness',
                'track_performance': True
            },
            'essays': {
                'icon': '✍️',
                'description': 'Essay & Narrative Generation',
                'narrative': 'artistic',
                'track_performance': False
            },
            'agents': {
                'icon': '🤝',
                'description': 'Multi-Agent Systems',
                'narrative': 'consciousness',
                'track_performance': True
            },
            'metabolic': {
                'icon': '⚡',
                'description': 'Metabolic Consciousness',
                'narrative': 'consciousness',
                'track_performance': True
            },
            'quantum': {
                'icon': '🔬',
                'description': 'Quantum Consciousness Dynamics',
                'narrative': 'technical',
                'track_performance': True
            },
            'technical': {
                'icon': '⚙️',
                'description': 'Technical Processing',
                'narrative': 'technical',
                'track_performance': False
            }
        }

        return styles.get(self.module_type, styles['technical'])

    def _init_logs(self):
        """Initialize log files with module-appropriate styling"""

        # Initialize JSON log
        with self.json_log.open("w") as f:
            init_entry = {
                "event": "MODULE_INIT",
                "module": self.module_name,
                "module_type": self.module_type,
                "timestamp": datetime.now().isoformat(),
                "full_module_name": self.full_module_name
            }
            f.write(json.dumps(init_entry) + "\n")

        # Initialize markdown log with appropriate style
        with self.events_log.open("w") as f:
            icon = self.style['icon']
            desc = self.style['description']

            f.write(f"# {icon} {self.module_name} - {desc}\n\n")
            f.write(f"**Module Type:** {self.module_type}\n")
            f.write(f"**Started:** {datetime.now():%Y-%m-%d %H:%M:%S}\n")
            f.write(f"**Auto-detected Style:** {self.style['narrative']}\n\n")
            f.write("---\n\n")

    def _create_method_decorator(self):
        """Create the @logged_method decorator for this module"""

        def logged_method(track_args=False, track_result=False, significant=False):
            """
            Decorator to automatically log method calls.

            Args:
                track_args: Whether to log method arguments
                track_result: Whether to log method results
                significant: Whether this is a significant method for events
            """
            def decorator(func):
                @wraps(func)
                def wrapper(*args, **kwargs):
                    start_time = time.time()
                    success = True
                    result = None

                    try:
                        result = func(*args, **kwargs)
                        return result
                    except Exception as e:
                        success = False
                        # Log the exception
                        self.log_event(
                            f"exception_{func.__name__}",
                            f"Method {func.__name__} failed: {type(e).__name__}: {e}",
                            {'exception_type': type(e).__name__, 'exception_msg': str(e)}
                        )
                        raise
                    finally:
                        duration = time.time() - start_time

                        # Log the method call
                        self._log_method_call(
                            func.__name__, duration, success,
                            track_args, track_result, significant,
                            args, kwargs, result
                        )

                return wrapper
            return decorator

        # Also create a simple version that's just @logged_method
        def simple_decorator(func):
            return logged_method()(func)

        # Return the parameterized version, but add simple as attribute
        logged_method.simple = simple_decorator
        return logged_method

    def _log_method_call(self, method_name, duration, success,
                        track_args, track_result, significant,
                        args, kwargs, result):
        """Internal method call logging"""

        self.call_count += 1

        # Create log entry
        entry = {
            "timestamp": datetime.now().isoformat(),
            "module": self.module_name,
            "method": method_name,
            "call_id": self.call_count,
            "duration_ms": round(duration * 1000, 3),
            "success": success,
            "significant": significant
        }

        # Add args/kwargs if requested
        if track_args and (args or kwargs):
            entry["args_summary"] = {
                "args_count": len(args),
                "kwargs_keys": list(kwargs.keys())[:5]  # Limit to first 5
            }

        # Add result info if requested
        if track_result and result is not None:
            if isinstance(result, dict):
                entry["result_summary"] = {
                    "type": "dict",
                    "keys": list(result.keys())[:5]
                }
            elif hasattr(result, '__len__'):
                entry["result_summary"] = {
                    "type": type(result).__name__,
                    "length": len(result)
                }
            else:
                entry["result_summary"] = {
                    "type": type(result).__name__
                }

        # Write to JSON log
        with self.json_log.open("a") as f:
            f.write(json.dumps(entry) + "\n")

        # If significant or slow, also log as event
        if significant or duration > 1.0 or not success:
            self._log_method_as_event(method_name, duration, success)

    def _log_method_as_event(self, method_name, duration, success):
        """Log significant method calls as narrative events"""

        if success:
            if duration > 1.0:
                event_name = f"slow_method_{method_name}"
                description = f"Method {method_name} completed slowly ({duration:.2f}s)"
            else:
                event_name = f"significant_method_{method_name}"
                description = f"Significant method {method_name} completed successfully"
        else:
            event_name = f"failed_method_{method_name}"
            description = f"Method {method_name} failed after {duration:.3f}s"

        self.log_event(event_name, description, {
            'method': method_name,
            'duration': duration,
            'success': success
        })

    def log_event(self, event: str, description: str, data: Dict[str, Any] = None):
        """
        Log a significant event with module-appropriate styling.

        Args:
            event: Event name/type
            description: Human-readable description
            data: Optional additional data
        """

        self.event_count += 1
        ts = datetime.now().strftime("%H:%M:%S")

        # Get appropriate icon for event
        event_icon = self._get_event_icon(event)

        # Write to markdown log with style
        with self.events_log.open("a") as f:
            if self.style['narrative'] == 'consciousness':
                f.write(f"## {event_icon} [{ts}] {event.replace('_', ' ').title()}\n\n")
                f.write(f"**Consciousness Event:** {description}\n\n")
            elif self.style['narrative'] == 'philosophical':
                f.write(f"### 🤔 [{ts}] {event.replace('_', ' ').title()}\n\n")
                f.write(f"**Philosophical Event:** {description}\n\n")
            elif self.style['narrative'] == 'artistic':
                f.write(f"### ✨ [{ts}] The {event.replace('_', ' ').title()} Movement\n\n")
                f.write(f"*{description}*\n\n")
            else:  # technical
                f.write(f"## [{ts}] {event}\n\n{description}\n\n")

            if data:
                f.write("**Data:**\n```json\n")
                f.write(json.dumps(data, indent=2, default=str))
                f.write("\n```\n\n")

            f.write("---\n\n")

        # Also log to JSON
        json_entry = {
            "timestamp": datetime.now().isoformat(),
            "module": self.module_name,
            "event_type": "narrative_event",
            "event": event,
            "description": description,
            "data": data or {},
            "event_id": self.event_count
        }

        with self.json_log.open("a") as f:
            f.write(json.dumps(json_entry) + "\n")

    def _get_event_icon(self, event: str) -> str:
        """Get appropriate icon for event based on module type and event name"""

        # Standard logging level icons
        if event.lower() == 'debug':
            return '🔍'
        elif event.lower() == 'info':
            return 'ℹ️'
        elif event.lower() == 'warning':
            return '⚠️'
        elif event.lower() == 'error':
            return '❌'
        elif event.lower() == 'critical':
            return '🚨'

        # Event-specific icons
        elif 'error' in event.lower() or 'fail' in event.lower() or 'exception' in event.lower():
            return '❌'
        elif 'success' in event.lower() or 'complete' in event.lower():
            return '✅'
        elif 'start' in event.lower() or 'init' in event.lower():
            return '🚀'
        elif 'consciousness' in event.lower():
            return '🧠'
        elif 'memory' in event.lower():
            return '💭'
        elif 'symbol' in event.lower() or 'vocab' in event.lower():
            return '🔤'
        elif 'philosophy' in event.lower():
            return '💡'
        elif 'expression' in event.lower():
            return '🗣️'
        elif 'environment' in event.lower():
            return '🌍'
        elif 'agent' in event.lower() or 'interaction' in event.lower():
            return '🤝'

        # Module-type default icons
        return self.style['icon']

    # Convenience methods for common consciousness events
    def log_consciousness_transition(self, from_state: str, to_state: str, trigger: str = None):
        """Log consciousness state transitions"""
        description = f"Consciousness transitioned from {from_state} to {to_state}"
        if trigger:
            description += f" (triggered by: {trigger})"

        self.log_event("consciousness_transition", description, {
            'from_state': from_state,
            'to_state': to_state,
            'trigger': trigger
        })

    def log_symbol_learning(self, symbol: str, strength: float, context: str = None):
        """Log symbol learning events"""
        description = f"Learned symbol '{symbol}' with strength {strength:.3f}"
        if context:
            description += f" in context: {context}"

        self.log_event("symbol_learning", description, {
            'symbol': symbol,
            'strength': strength,
            'context': context
        })

    def log_philosophical_insight(self, concept: str, insight: str):
        """Log philosophical insights"""
        self.log_event("philosophical_insight", f"Insight about {concept}: {insight}", {
            'concept': concept,
            'insight': insight
        })

    def log_memory_operation(self, operation: str, memory_type: str, details: str = None):
        """Log memory operations"""
        description = f"Memory {operation} ({memory_type})"
        if details:
            description += f": {details}"

        self.log_event(f"memory_{operation}", description, {
            'operation': operation,
            'memory_type': memory_type,
            'details': details
        })

    def log_expression_generation(self, expression_type: str, quality: float, length: int):
        """Log expression generation"""
        self.log_event("expression_generation",
                      f"Generated {expression_type} expression (quality: {quality:.3f}, length: {length})", {
            'expression_type': expression_type,
            'quality': quality,
            'length': length
        })

    # Standard logging levels for anything that doesn't fit predefined categories
    def debug(self, message: str, data: Dict[str, Any] = None):
        """Log debug information - detailed internal state info"""
        self.log_event("debug", f"DEBUG: {message}", data)

    def info(self, message: str, data: Dict[str, Any] = None):
        """Log general information - normal operation info"""
        self.log_event("info", f"INFO: {message}", data)

    def warning(self, message: str, data: Dict[str, Any] = None):
        """Log warning - something unexpected but not breaking"""
        self.log_event("warning", f"WARNING: {message}", data)

    def error(self, message: str, data: Dict[str, Any] = None):
        """Log error - something went wrong"""
        self.log_event("error", f"ERROR: {message}", data)

    def critical(self, message: str, data: Dict[str, Any] = None):
        """Log critical error - major system failure"""
        self.log_event("critical", f"CRITICAL: {message}", data)

    # Convenience aliases
    def log_debug(self, message: str, data: Dict[str, Any] = None):
        """Alias for debug()"""
        self.debug(message, data)

    def log_info(self, message: str, data: Dict[str, Any] = None):
        """Alias for info()"""
        self.info(message, data)

    def log_warning(self, message: str, data: Dict[str, Any] = None):
        """Alias for warning()"""
        self.warning(message, data)

    def log_error(self, message: str, data: Dict[str, Any] = None):
        """Alias for error()"""
        self.error(message, data)

    def get_stats(self) -> Dict[str, Any]:
        """Get logging statistics for this module"""
        runtime = time.time() - self.start_time
        return {
            'module_name': self.module_name,
            'module_type': self.module_type,
            'runtime_seconds': round(runtime, 2),
            'total_method_calls': self.call_count,
            'total_events': self.event_count,
            'calls_per_second': round(self.call_count / runtime, 2) if runtime > 0 else 0,
            'style': self.style['narrative']
        }

# Global registry to avoid creating duplicate loggers
_module_loggers = {}

def setup_module_logging(module_name: str, log_dir: str = "module_logs") -> UniversalModuleLogger:
    """
    MAIN FUNCTION: Setup logging for any module.

    Usage in any module:
        from emile_cogito.kainos.universal_module_logging import setup_module_logging
        logger = setup_module_logging(__name__)
        logged_method = logger.method_decorator

        @logged_method
        def my_method(self):
            pass
    """

    if module_name not in _module_loggers:
        _module_loggers[module_name] = UniversalModuleLogger(module_name, log_dir)

    return _module_loggers[module_name]

# Convenience function for quick decorator access
def get_method_decorator(module_name: str) -> Callable:
    """Get just the method decorator for a module"""
    logger = setup_module_logging(module_name)
    return logger.method_decorator

# Export the main functions
__all__ = [
    "setup_module_logging",
    "get_method_decorator",
    "UniversalModuleLogger"
]

from emile_cogito.kainos.module_wide_flow_mapper import auto_map_module_flow
auto_map_module_flow(__name__)  # Maps the entire module!
