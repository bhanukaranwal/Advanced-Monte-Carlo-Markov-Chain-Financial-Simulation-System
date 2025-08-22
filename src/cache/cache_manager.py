"""
Cache management for MCMF system
"""

import json
import pickle
import redis
from typing import Any, Optional, Dict, List
import hashlib
from datetime import timedelta
import logging

from ..config.settings import settings
from ..utils.exceptions import MCMFException

logger = logging.getLogger(__name__)

class CacheManager:
    """Redis-based cache manager"""
    
    def __init__(self):
        self.redis_client = redis.from_url(
            settings.redis.url,
            max_connections=settings.redis.max_connections,
            decode_responses=False  # We'll handle encoding ourselves
        )
        self.default_ttl = 3600  # 1 hour
        
    def _make_key(self, namespace: str, key: str) -> str:
        """Create namespaced cache key"""
        return f"{settings.project_name}:{namespace}:{key}"
        
    def _serialize(self, data: Any) -> bytes:
        """Serialize data for caching"""
        if isinstance(data, (dict, list)):
            return json.dumps(data).encode('utf-8')
        else:
            return pickle.dumps(data)
            
    def _deserialize(self, data: bytes, json_compatible: bool = True) -> Any:
        """Deserialize cached data"""
        if json_compatible:
            return json.loads(data.decode('utf-8'))
        else:
            return pickle.loads(data)
            
    def set(self, namespace: str, key: str, value: Any, ttl: Optional[int] = None, json_compatible: bool = True):
        """Set cache value"""
        cache_key = self._make_key(namespace, key)
        serialized_value = self._serialize(value)
        
        if ttl is None:
            ttl = self.default_ttl
            
        try:
            self.redis_client.setex(cache_key, ttl, serialized_value)
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            
    def get(self, namespace: str, key: str, json_compatible: bool = True) -> Optional[Any]:
        """Get cache value"""
        cache_key = self._make_key(namespace, key)
        
        try:
            data = self.redis_client.get(cache_key)
            if data is not None:
                return self._deserialize(data, json_compatible)
            return None
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None
            
    def delete(self, namespace: str, key: str):
        """Delete cache value"""
        cache_key = self._make_key(namespace, key)
        
        try:
            self.redis_client.delete(cache_key)
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            
    def exists(self, namespace: str, key: str) -> bool:
        """Check if cache key exists"""
        cache_key = self._make_key(namespace, key)
        
        try:
            return self.redis_client.exists(cache_key) > 0
        except Exception as e:
            logger.error(f"Cache exists error: {e}")
            return False
            
    def flush_namespace(self, namespace: str):
        """Delete all keys in namespace"""
        pattern = self._make_key(namespace, "*")
        
        try:
            keys = self.redis_client.keys(pattern)
            if keys:
                self.redis_client.delete(*keys)
        except Exception as e:
            logger.error(f"Cache flush error: {e}")
            
    def get_or_set(self, namespace: str, key: str, factory_func: callable, ttl: Optional[int] = None, json_compatible: bool = True) -> Any:
        """Get value from cache or set using factory function"""
        value = self.get(namespace, key, json_compatible)
        
        if value is None:
            value = factory_func()
            self.set(namespace, key, value, ttl, json_compatible)
            
        return value

class SimulationCache:
    """Specialized cache for simulation results"""
    
    def __init__(self, cache_manager: CacheManager):
        self.cache = cache_manager
        self.namespace = "simulations"
        
    def _make_simulation_key(self, simulation_type: str, parameters: Dict) -> str:
        """Create cache key for simulation"""
        param_hash = hashlib.md5(json.dumps(parameters, sort_keys=True).encode()).hexdigest()
        return f"{simulation_type}:{param_hash}"
        
    def cache_simulation_result(self, simulation_type: str, parameters: Dict, result: Dict, ttl: int = 7200):
        """Cache simulation result"""
        key = self._make_simulation_key(simulation_type, parameters)
        self.cache.set(self.namespace, key, result, ttl, json_compatible=False)
        
    def get_simulation_result(self, simulation_type: str, parameters: Dict) -> Optional[Dict]:
        """Get cached simulation result"""
        key = self._make_simulation_key(simulation_type, parameters)
        return self.cache.get(self.namespace, key, json_compatible=False)

class ESGCache:
    """Specialized cache for ESG data"""
    
    def __init__(self, cache_manager: CacheManager):
        self.cache = cache_manager
        self.namespace = "esg_data"
        
    def cache_esg_scores(self, symbol: str, scores: Dict, ttl: int = 86400):  # 24 hours
        """Cache ESG scores for symbol"""
        self.cache.set(self.namespace, symbol, scores, ttl)
        
    def get_esg_scores(self, symbol: str) -> Optional[Dict]:
        """Get cached ESG scores"""
        return self.cache.get(self.namespace, symbol)

class MarketDataCache:
    """Specialized cache for market data"""
    
    def __init__(self, cache_manager: CacheManager):
        self.cache = cache_manager
        self.namespace = "market_data"
        
    def cache_price_data(self, symbol: str, timeframe: str, data: List[Dict], ttl: int = 300):  # 5 minutes
        """Cache price data"""
        key = f"{symbol}:{timeframe}"
        self.cache.set(self.namespace, key, data, ttl)
        
    def get_price_data(self, symbol: str, timeframe: str) -> Optional[List[Dict]]:
        """Get cached price data"""
        key = f"{symbol}:{timeframe}"
        return self.cache.get(self.namespace, key)

# Global cache instances
cache_manager = CacheManager()
simulation_cache = SimulationCache(cache_manager)
esg_cache = ESGCache(cache_manager)
market_data_cache = MarketDataCache(cache_manager)
