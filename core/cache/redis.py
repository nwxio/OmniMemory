import json
import hashlib
from typing import Any, Optional

from ..integrations_config import db_settings


class RedisCache:
    """Redis cache for search results and LLM responses."""
    
    def __init__(self):
        self._client = None
        self._enabled = db_settings.redis_enabled
    
    def _get_client(self):
        """Lazy load Redis client."""
        if not self._enabled:
            return None
        
        if self._client is None:
            try:
                import redis
                self._client = redis.Redis(
                    host=db_settings.redis_host,
                    port=db_settings.redis_port,
                    db=db_settings.redis_db,
                    password=db_settings.redis_password,
                    decode_responses=True,
                )
            except Exception:
                self._enabled = False
                return None
        
        return self._client
    
    def _hash_key(self, prefix: str, *args) -> str:
        """Generate cache key."""
        key_data = ":".join(str(a) for a in args)
        hash_part = hashlib.md5(key_data.encode()).hexdigest()[:12]
        return f"omnimind:{prefix}:{hash_part}"
    
    def is_enabled(self) -> bool:
        return self._enabled and self._get_client() is not None
    
    async def get(self, key: str) -> Optional[str]:
        """Get value from cache."""
        client = self._get_client()
        if not client:
            return None
        
        try:
            return client.get(key)
        except Exception:
            return None
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[int] = None,
    ) -> bool:
        """Set value in cache."""
        client = self._get_client()
        if not client:
            return False
        
        try:
            if isinstance(value, (dict, list)):
                value = json.dumps(value)
            
            if ttl_seconds:
                client.setex(key, ttl_seconds, value)
            else:
                client.set(key, value)
            return True
        except Exception:
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        client = self._get_client()
        if not client:
            return False
        
        try:
            client.delete(key)
            return True
        except Exception:
            return False
    
    async def clear_prefix(self, prefix: str) -> int:
        """Clear all keys with prefix."""
        client = self._get_client()
        if not client:
            return 0
        
        try:
            pattern = f"omnimind:{prefix}:*"
            keys = client.keys(pattern)
            if keys:
                return client.delete(*keys)
            return 0
        except Exception:
            return 0
    
    def close(self) -> None:
        """Close Redis connection."""
        if self._client:
            self._client.close()
            self._client = None


redis_cache = RedisCache()
