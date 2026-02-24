import time
from typing import Any, Dict
from collections import defaultdict


class MetricsCollector:
    """Metrics collector for memory operations."""
    
    def __init__(self):
        self._counters: Dict[str, int] = defaultdict(int)
        self._timers: Dict[str, list] = defaultdict(list)
        self._gauges: Dict[str, float] = {}
        self._start_time = time.time()
    
    def increment(self, name: str, value: int = 1) -> None:
        """Increment a counter."""
        self._counters[name] += value
    
    def decrement(self, name: str, value: int = 1) -> None:
        """Decrement a counter."""
        self._counters[name] -= value
    
    def gauge(self, name: str, value: float) -> None:
        """Set a gauge value."""
        self._gauges[name] = value
    
    def timing(self, name: str, duration_ms: float) -> None:
        """Record timing in milliseconds."""
        self._timers[name].append(duration_ms)
    
    def timer(self, name: str):
        """Context manager for timing."""
        return Timer(self, name)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get all stats."""
        uptime = time.time() - self._start_time
        
        timer_stats = {}
        for name, values in self._timers.items():
            if values:
                timer_stats[name] = {
                    "count": len(values),
                    "avg_ms": sum(values) / len(values),
                    "min_ms": min(values),
                    "max_ms": max(values),
                }
        
        return {
            "uptime_seconds": uptime,
            "counters": dict(self._counters),
            "gauges": dict(self._gauges),
            "timers": timer_stats,
        }
    
    def reset(self) -> None:
        """Reset all metrics."""
        self._counters.clear()
        self._timers.clear()
        self._gauges.clear()


class Timer:
    """Context manager for timing."""
    
    def __init__(self, collector: MetricsCollector, name: str):
        self._collector = collector
        self._name = name
        self._start = 0
    
    def __enter__(self):
        self._start = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration_ms = (time.time() - self._start) * 1000
        self._collector.timing(self._name, duration_ms)


class RequestMetrics:
    """Request-level metrics."""
    
    def __init__(self):
        self._requests = 0
        self._errors = 0
        self._total_duration = 0.0
        self._by_endpoint: Dict[str, int] = defaultdict(int)
        self._by_status: Dict[int, int] = defaultdict(int)
    
    def record_request(
        self,
        endpoint: str,
        status: int,
        duration_ms: float,
    ) -> None:
        """Record a request."""
        self._requests += 1
        self._by_endpoint[endpoint] += 1
        self._by_status[status] += 1
        self._total_duration += duration_ms
        if status >= 400:
            self._errors += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get request stats."""
        avg_duration = (
            self._total_duration / self._requests if self._requests > 0 else 0
        )
        error_rate = (
            self._errors / self._requests if self._requests > 0 else 0
        )
        
        return {
            "total_requests": self._requests,
            "total_errors": self._errors,
            "error_rate": error_rate,
            "avg_duration_ms": avg_duration,
            "by_endpoint": dict(self._by_endpoint),
            "by_status": dict(self._by_status),
        }


metrics = MetricsCollector()
request_metrics = RequestMetrics()
