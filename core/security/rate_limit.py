import time
from typing import Dict, Optional
from collections import defaultdict

from ..integrations_config import db_settings


class RateLimiter:
    """Rate limiter for API requests."""
    
    def __init__(
        self,
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000,
    ):
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self._requests: Dict[str, list] = defaultdict(list)
    
    def _cleanup_old_requests(self, key: str, now: float) -> None:
        """Remove old requests from tracking."""
        hour_ago = now - 3600
        # Keep only last-hour events; minute window is derived on demand.
        self._requests[key] = [t for t in self._requests[key] if t > hour_ago]
    
    def is_allowed(self, key: str = "default") -> bool:
        """Check if request is allowed."""
        now = time.time()
        
        self._cleanup_old_requests(key, now)
        
        requests_recent = self._requests[key]

        minute_count = sum(1 for t in requests_recent if t > now - 60)
        if minute_count >= self.requests_per_minute:
            return False

        hour_count = len(requests_recent)
        if hour_count >= self.requests_per_hour:
            return False
        
        self._requests[key].append(now)
        return True
    
    def get_remaining(self, key: str = "default") -> Dict[str, int]:
        """Get remaining requests."""
        now = time.time()
        
        self._cleanup_old_requests(key, now)
        
        minute_count = sum(1 for t in self._requests[key] if t > now - 60)
        hour_count = len(self._requests[key])
        
        return {
            "requests_per_minute": max(0, self.requests_per_minute - minute_count),
            "requests_per_hour": max(0, self.requests_per_hour - hour_count),
        }
    
    def reset(self, key: Optional[str] = None) -> None:
        """Reset rate limit for key."""
        if key:
            self._requests[key] = []
        else:
            self._requests.clear()


class DistributedRateLimiter:
    """Redis-based distributed rate limiter."""
    
    def __init__(
        self,
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000,
    ):
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self._local = RateLimiter(requests_per_minute, requests_per_hour)
    
    def is_allowed(self, key: str = "default") -> bool:
        """Check if request is allowed (with Redis fallback)."""
        if db_settings.redis_enabled:
            return self._redis_is_allowed(key)
        return self._local.is_allowed(key)
    
    def _redis_is_allowed(self, key: str) -> bool:
        """Redis-based rate limiting."""
        try:
            import redis
            client = redis.Redis(
                host=db_settings.redis_host,
                port=db_settings.redis_port,
                db=db_settings.redis_db,
                decode_responses=True,
            )
            
            minute_key = f"rate:{key}:minute"
            hour_key = f"rate:{key}:hour"
            
            pipe = client.pipeline()
            pipe.incr(minute_key)
            pipe.expire(minute_key, 60)
            pipe.incr(hour_key)
            pipe.expire(hour_key, 3600)
            results = pipe.execute()
            
            minute_count = results[0]
            hour_count = results[2]
            
            if minute_count > self.requests_per_minute:
                return False
            if hour_count > self.requests_per_hour:
                return False
            
            return True
        
        except Exception:
            return self._local.is_allowed(key)
    
    def get_remaining(self, key: str = "default") -> Dict[str, int]:
        """Get remaining requests."""
        if db_settings.redis_enabled:
            try:
                import redis
                client = redis.Redis(
                    host=db_settings.redis_host,
                    port=db_settings.redis_port,
                    db=db_settings.redis_db,
                    decode_responses=True,
                )
                
                minute_count = int(client.get(f"rate:{key}:minute") or 0)
                hour_count = int(client.get(f"rate:{key}:hour") or 0)
                
                return {
                    "requests_per_minute": max(0, self.requests_per_minute - minute_count),
                    "requests_per_hour": max(0, self.requests_per_hour - hour_count),
                }
            except Exception:
                pass
        
        return self._local.get_remaining(key)


rate_limiter = RateLimiter()
distributed_rate_limiter = DistributedRateLimiter()
