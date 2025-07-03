
#!/usr/bin/env python3
"""
Universal Module Logging System for Émile Framework
==================================================
Every module gets automatic logging of all method calls and events.
"""

import os
import json
import time
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path
from functools import wraps

class UniversalModuleLogger:
    """Logger that every module gets"""

    def __init__(self, module_name: str, log_dir: str = "module_logs"):
        self.module_name = module_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Create module-specific directory
        self.module_log_dir = self.log_dir / module_name
        self.module_log_dir.mkdir(parents=True, exist_ok=True)

        # Create log files
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.json_log = self.module_log_dir / f"{module_name}_calls_{timestamp}.jsonl"
        self.narrative_log = self.module_log_dir / f"{module_name}_events_{timestamp}.md"

        # Initialize narrative log
        with open(self.narrative_log, 'w') as f:
            f.write(f"# {module_name.title()} Module Log\n\n")
            f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        self.call_count = 0

    def log_method_call(self, method_name: str, duration: float, success: bool, **kwargs):
        """Log a method call"""
        self.call_count += 1

        entry = {
            'timestamp': datetime.now().isoformat(),
            'module': self.module_name,
            'method': method_name,
            'call_id': self.call_count,
            'duration_ms': duration * 1000,
            'success': success,
            **kwargs
        }

        with open(self.json_log, 'a') as f:
            f.write(json.dumps(entry, default=str) + '\n')

    def log_event(self, event: str, description: str, data: Dict[str, Any] = None):
        """Log a significant event"""
        timestamp = datetime.now().strftime('%H:%M:%S')

        with open(self.narrative_log, 'a') as f:
            f.write(f"## [{timestamp}] {event}\n\n")
            f.write(f"{description}\n\n")

            if data:
                f.write("**Data:**\n")
                for key, value in data.items():
                    f.write(f"- {key}: {value}\n")
                f.write("\n")

            f.write("---\n\n")

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

def logged_method(func):
    """Decorator to automatically log method calls"""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        # Only log if the instance has a module_logger
        if not hasattr(self, 'module_logger'):
            return func(self, *args, **kwargs)

        start_time = time.time()
        success = True

        try:
            result = func(self, *args, **kwargs)
            return result
        except Exception as e:
            success = False
            raise
        finally:
            duration = time.time() - start_time
            self.module_logger.log_method_call(
                func.__name__, duration, success,
                args_count=len(args), kwargs_keys=list(kwargs.keys())
            )

    return wrapper

class LoggedModule:
    """Base class that modules inherit to get automatic logging"""

    def __init__(self, module_name: str, log_dir: str = "module_logs"):
        self.module_logger = UniversalModuleLogger(module_name, log_dir)
        self.module_name = module_name

    def log_event(self, event: str, description: str, data: Dict[str, Any] = None):
        """Log an event in this module"""
        self.module_logger.log_event(event, description, data)


from emile_cogito.kainos.module_wide_flow_mapper import auto_map_module_flow
auto_map_module_flow(__name__)  # Maps the entire module!
