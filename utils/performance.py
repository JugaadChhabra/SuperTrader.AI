"""
Performance Optimization Framework - Phase 5

Advanced performance optimization for SuperTrader.AI with caching strategies,
memory optimization, and production deployment optimizations.
"""

import time
import logging
import threading
from typing import Dict, Any, List, Optional, Callable, Union
from functools import wraps, lru_cache
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from collections import defaultdict, OrderedDict
import hashlib
import pickle
import os
from pathlib import Path

from utils.config import get_config
from utils.monitoring_integration import track_performance, get_health_monitor

logger = logging.getLogger(__name__)


@dataclass
class CacheConfig:
    """Configuration for caching system"""
    max_size: int = 1000
    ttl_seconds: int = 300  # 5 minutes default TTL
    enable_disk_cache: bool = True
    cache_dir: str = "cache"
    max_disk_size_mb: int = 500


@dataclass
class PerformanceMetrics:
    """Performance tracking metrics"""
    call_count: int = 0
    total_time: float = 0.0
    avg_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    
    def update(self, execution_time: float, cache_hit: bool = False):
        """Update metrics with new execution"""
        self.call_count += 1
        self.total_time += execution_time
        self.avg_time = self.total_time / self.call_count
        self.min_time = min(self.min_time, execution_time)
        self.max_time = max(self.max_time, execution_time)
        
        if cache_hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
    
    @property
    def cache_hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        return (self.cache_hits / total * 100) if total > 0 else 0.0


class AdvancedCache:
    """
    Advanced caching system with TTL, LRU eviction, and disk persistence
    """
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.memory_cache = OrderedDict()
        self.cache_times = {}
        self.access_counts = defaultdict(int)
        self._lock = threading.RLock()
        
        # Setup disk cache
        if self.config.enable_disk_cache:
            self.cache_dir = Path(self.config.cache_dir)
            self.cache_dir.mkdir(exist_ok=True)
    
    def _generate_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Generate cache key from function signature"""
        key_data = f"{func_name}_{str(args)}_{str(sorted(kwargs.items()))}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _is_expired(self, key: str) -> bool:
        """Check if cache entry is expired"""
        if key not in self.cache_times:
            return True
        
        age = time.time() - self.cache_times[key]
        return age > self.config.ttl_seconds
    
    def _evict_lru(self):
        """Evict least recently used item"""
        if not self.memory_cache:
            return
        
        # Find LRU item
        lru_key = min(self.access_counts.keys(), key=lambda k: self.access_counts[k])
        
        # Remove from all caches
        self.memory_cache.pop(lru_key, None)
        self.cache_times.pop(lru_key, None)
        self.access_counts.pop(lru_key, None)
    
    def _save_to_disk(self, key: str, value: Any):
        """Save cache entry to disk"""
        if not self.config.enable_disk_cache:
            return
        
        try:
            cache_file = self.cache_dir / f"{key}.pkl"
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'value': value,
                    'timestamp': time.time()
                }, f)
        except Exception as e:
            logger.warning(f"Failed to save cache to disk: {e}")
    
    def _load_from_disk(self, key: str) -> Optional[Any]:
        """Load cache entry from disk"""
        if not self.config.enable_disk_cache:
            return None
        
        try:
            cache_file = self.cache_dir / f"{key}.pkl"
            if not cache_file.exists():
                return None
            
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            
            # Check if disk cache is expired
            age = time.time() - data['timestamp']
            if age > self.config.ttl_seconds:
                cache_file.unlink()  # Delete expired file
                return None
            
            return data['value']
            
        except Exception as e:
            logger.warning(f"Failed to load cache from disk: {e}")
            return None
    
    def get(self, func_name: str, args: tuple, kwargs: dict) -> tuple[Any, bool]:
        """Get cached value, returns (value, cache_hit)"""
        key = self._generate_key(func_name, args, kwargs)
        
        with self._lock:
            # Check memory cache first
            if key in self.memory_cache and not self._is_expired(key):
                self.access_counts[key] += 1
                return self.memory_cache[key], True
            
            # Check disk cache
            disk_value = self._load_from_disk(key)
            if disk_value is not None:
                # Move to memory cache
                self.set(func_name, args, kwargs, disk_value)
                return disk_value, True
            
            return None, False
    
    def set(self, func_name: str, args: tuple, kwargs: dict, value: Any):
        """Set cached value"""
        key = self._generate_key(func_name, args, kwargs)
        
        with self._lock:
            # Check if we need to evict
            if len(self.memory_cache) >= self.config.max_size:
                self._evict_lru()
            
            # Store in memory
            self.memory_cache[key] = value
            self.cache_times[key] = time.time()
            self.access_counts[key] = 1
            
            # Store on disk asynchronously
            if self.config.enable_disk_cache:
                threading.Thread(
                    target=self._save_to_disk, 
                    args=(key, value), 
                    daemon=True
                ).start()
    
    def clear(self):
        """Clear all caches"""
        with self._lock:
            self.memory_cache.clear()
            self.cache_times.clear()
            self.access_counts.clear()
            
        # Clear disk cache
        if self.config.enable_disk_cache and self.cache_dir.exists():
            for cache_file in self.cache_dir.glob("*.pkl"):
                try:
                    cache_file.unlink()
                except Exception as e:
                    logger.warning(f"Failed to delete cache file {cache_file}: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            return {
                'memory_entries': len(self.memory_cache),
                'max_size': self.config.max_size,
                'ttl_seconds': self.config.ttl_seconds,
                'total_access_count': sum(self.access_counts.values()),
                'unique_keys': len(self.access_counts)
            }


class PerformanceOptimizer:
    """
    Main performance optimization coordinator
    """
    
    def __init__(self):
        self.config = get_config()
        self.cache_config = CacheConfig(
            max_size=getattr(self.config.performance, 'cache_max_size', 1000),
            ttl_seconds=getattr(self.config.performance, 'cache_ttl_seconds', 300),
            enable_disk_cache=getattr(self.config.performance, 'enable_disk_cache', True)
        )
        
        self.cache = AdvancedCache(self.cache_config)
        self.performance_metrics = {}
        self._optimization_enabled = True
        
        logger.info("Performance optimizer initialized")
    
    def cached_function(self, cache_key_func: Optional[Callable] = None):
        """
        Decorator for caching function results with performance tracking
        """
        def decorator(func):
            func_name = f"{func.__module__}.{func.__name__}"
            
            if func_name not in self.performance_metrics:
                self.performance_metrics[func_name] = PerformanceMetrics()
            
            @wraps(func)
            def wrapper(*args, **kwargs):
                if not self._optimization_enabled:
                    return func(*args, **kwargs)
                
                # Generate cache key
                cache_args = args
                cache_kwargs = kwargs
                
                if cache_key_func:
                    try:
                        cache_args, cache_kwargs = cache_key_func(*args, **kwargs)
                    except Exception as e:
                        logger.warning(f"Cache key function failed: {e}")
                
                # Try to get from cache
                start_time = time.time()
                cached_result, cache_hit = self.cache.get(func_name, cache_args, cache_kwargs)
                
                if cache_hit:
                    execution_time = time.time() - start_time
                    self.performance_metrics[func_name].update(execution_time, cache_hit=True)
                    return cached_result
                
                # Execute function
                with track_performance('performance_optimizer', f'execute_{func_name}'):
                    result = func(*args, **kwargs)
                
                execution_time = time.time() - start_time
                
                # Cache the result
                self.cache.set(func_name, cache_args, cache_kwargs, result)
                
                # Update metrics
                self.performance_metrics[func_name].update(execution_time, cache_hit=False)
                
                return result
            
            return wrapper
        return decorator
    
    def batch_process(self, items: List[Any], func: Callable, batch_size: int = 100) -> List[Any]:
        """
        Process items in batches for better performance
        """
        results = []
        
        with track_performance('performance_optimizer', 'batch_processing'):
            for i in range(0, len(items), batch_size):
                batch = items[i:i + batch_size]
                batch_results = [func(item) for item in batch]
                results.extend(batch_results)
                
                # Yield control occasionally
                if i % (batch_size * 10) == 0:
                    time.sleep(0.001)  # Small yield
        
        return results
    
    def optimize_dataframe_operations(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize DataFrame operations for better performance
        """
        with track_performance('performance_optimizer', 'dataframe_optimization'):
            # Convert to optimal data types
            optimized_df = df.copy()
            
            for col in optimized_df.select_dtypes(include=['float64']).columns:
                try:
                    optimized_df[col] = pd.to_numeric(optimized_df[col], downcast='float')
                except Exception:
                    pass
            
            for col in optimized_df.select_dtypes(include=['int64']).columns:
                try:
                    optimized_df[col] = pd.to_numeric(optimized_df[col], downcast='integer')
                except Exception:
                    pass
            
            return optimized_df
    
    def memory_efficient_indicator_calculation(
        self, 
        data: pd.DataFrame, 
        indicator_func: Callable, 
        chunk_size: int = 1000
    ) -> pd.Series:
        """
        Calculate indicators in chunks to save memory
        """
        if len(data) <= chunk_size:
            return indicator_func(data)
        
        results = []
        
        with track_performance('performance_optimizer', 'chunked_indicator_calculation'):
            for i in range(0, len(data), chunk_size):
                chunk = data.iloc[i:i + chunk_size]
                chunk_result = indicator_func(chunk)
                results.append(chunk_result)
        
        return pd.concat(results, ignore_index=True)
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'optimization_enabled': self._optimization_enabled,
            'cache_stats': self.cache.get_stats(),
            'function_performance': {}
        }
        
        for func_name, metrics in self.performance_metrics.items():
            report['function_performance'][func_name] = {
                'call_count': metrics.call_count,
                'avg_time_ms': metrics.avg_time * 1000,
                'min_time_ms': metrics.min_time * 1000,
                'max_time_ms': metrics.max_time * 1000,
                'cache_hit_rate': metrics.cache_hit_rate,
                'total_time_saved_ms': metrics.cache_hits * metrics.avg_time * 1000
            }
        
        return report
    
    def clear_caches(self):
        """Clear all caches"""
        self.cache.clear()
        logger.info("Performance caches cleared")
    
    def enable_optimization(self):
        """Enable performance optimizations"""
        self._optimization_enabled = True
        logger.info("Performance optimizations enabled")
    
    def disable_optimization(self):
        """Disable performance optimizations (for debugging)"""
        self._optimization_enabled = False
        logger.info("Performance optimizations disabled")


# Global optimizer instance
_optimizer_instance = None

def get_performance_optimizer() -> PerformanceOptimizer:
    """Get global performance optimizer instance"""
    global _optimizer_instance
    if _optimizer_instance is None:
        _optimizer_instance = PerformanceOptimizer()
    return _optimizer_instance


# Convenient decorators
def cached(cache_key_func: Optional[Callable] = None):
    """Convenient caching decorator"""
    optimizer = get_performance_optimizer()
    return optimizer.cached_function(cache_key_func)


def phase4_feature_cache_key(*args, **kwargs):
    """Cache key generator for Phase 4 features"""
    # Only cache based on data hash, not full dataframe
    if args and hasattr(args[0], 'index'):
        data_hash = hash(str(args[0].index[-1]) + str(len(args[0])))
        return (data_hash,), {}
    return args, kwargs


# Memory optimization utilities
def optimize_memory_usage():
    """Optimize system memory usage"""
    import gc
    
    with track_performance('performance_optimizer', 'memory_optimization'):
        # Force garbage collection
        collected = gc.collect()
        
        # Log memory optimization
        logger.info(f"Memory optimization: collected {collected} objects")
        
        return collected


def profile_function(func):
    """Decorator to profile function performance"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        import cProfile
        import io
        import pstats
        
        pr = cProfile.Profile()
        pr.enable()
        
        result = func(*args, **kwargs)
        
        pr.disable()
        
        # Generate profile report
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
        ps.print_stats(10)  # Top 10 functions
        
        logger.info(f"Profile for {func.__name__}:\n{s.getvalue()}")
        
        return result
    
    return wrapper


if __name__ == "__main__":
    # Test performance optimization
    optimizer = get_performance_optimizer()
    
    @cached(phase4_feature_cache_key)
    def expensive_calculation(data: pd.DataFrame) -> float:
        # Simulate expensive calculation
        time.sleep(0.1)
        return data['close'].mean() if 'close' in data.columns else 0.0
    
    # Create test data
    test_data = pd.DataFrame({
        'close': np.random.randn(100) + 100,
        'volume': np.random.randint(1000, 10000, 100)
    })
    
    # Test caching
    print("Testing performance optimization...")
    
    # First call (cache miss)
    start = time.time()
    result1 = expensive_calculation(test_data)
    time1 = time.time() - start
    print(f"First call: {time1:.3f}s, result: {result1:.2f}")
    
    # Second call (cache hit)
    start = time.time()
    result2 = expensive_calculation(test_data)
    time2 = time.time() - start
    print(f"Second call: {time2:.3f}s, result: {result2:.2f}")
    
    # Generate report
    report = optimizer.get_performance_report()
    print("\nPerformance Report:")
    for func, metrics in report['function_performance'].items():
        print(f"{func}: {metrics['call_count']} calls, {metrics['cache_hit_rate']:.1f}% cache hit rate")
    
    print(f"Cache stats: {report['cache_stats']}")