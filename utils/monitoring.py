"""
Pipeline Health Monitoring - Phase 5

Comprehensive monitoring system for SuperTrader.AI with real-time health checks,
performance tracking, and Phase 4 feature monitoring.
"""

import logging
import time
import threading
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
from collections import defaultdict, deque
import json
import os

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logging.warning("psutil not available - system metrics will be limited")

from utils.config import get_config

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for a component"""
    latency_ms: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_pct: float = 0.0
    success_rate: float = 100.0
    error_count: int = 0
    last_update: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'latency_ms': self.latency_ms,
            'memory_usage_mb': self.memory_usage_mb,
            'cpu_usage_pct': self.cpu_usage_pct,
            'success_rate': self.success_rate,
            'error_count': self.error_count,
            'last_update': self.last_update.isoformat()
        }


@dataclass
class HealthStatus:
    """Health status for a component"""
    status: str = "unknown"  # healthy, warning, critical, unknown
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    last_check: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'status': self.status,
            'message': self.message,
            'details': self.details,
            'last_check': self.last_check.isoformat()
        }


class ComponentMonitor:
    """Monitor for individual system components"""
    
    def __init__(self, name: str, check_interval: int = 60):
        self.name = name
        self.check_interval = check_interval
        self.metrics = PerformanceMetrics()
        self.health = HealthStatus()
        self.history = deque(maxlen=100)  # Keep last 100 data points
        self._lock = threading.Lock()
        
    def update_metrics(self, **kwargs):
        """Update performance metrics"""
        with self._lock:
            for key, value in kwargs.items():
                if hasattr(self.metrics, key):
                    setattr(self.metrics, key, value)
            self.metrics.last_update = datetime.now()
            
            # Add to history
            self.history.append({
                'timestamp': datetime.now(),
                'metrics': self.metrics.to_dict()
            })
    
    def update_health(self, status: str, message: str = "", **details):
        """Update health status"""
        with self._lock:
            self.health.status = status
            self.health.message = message
            self.health.details = details
            self.health.last_check = datetime.now()
    
    def get_status(self) -> Dict[str, Any]:
        """Get current component status"""
        with self._lock:
            return {
                'name': self.name,
                'metrics': self.metrics.to_dict(),
                'health': self.health.to_dict(),
                'history_count': len(self.history)
            }


class FeatureGenerationMonitor(ComponentMonitor):
    """Specialized monitor for feature generation pipeline"""
    
    def __init__(self):
        super().__init__("feature_generation", check_interval=30)
        self.phase4_metrics = {
            'pcr_integration': {'success_count': 0, 'failure_count': 0, 'avg_latency': 0.0},
            'multi_timeframe': {'success_count': 0, 'failure_count': 0, 'avg_latency': 0.0},
            'enhanced_oi': {'success_count': 0, 'failure_count': 0, 'avg_latency': 0.0},
            'market_regime': {'success_count': 0, 'failure_count': 0, 'avg_latency': 0.0},
            'microstructure': {'success_count': 0, 'failure_count': 0, 'avg_latency': 0.0}
        }
        
    def record_feature_generation(self, feature_type: str, success: bool, latency_ms: float):
        """Record feature generation attempt"""
        if feature_type in self.phase4_metrics:
            metrics = self.phase4_metrics[feature_type]
            if success:
                metrics['success_count'] += 1
            else:
                metrics['failure_count'] += 1
            
            # Update average latency (exponential moving average)
            alpha = 0.1
            metrics['avg_latency'] = alpha * latency_ms + (1 - alpha) * metrics['avg_latency']
    
    def get_phase4_status(self) -> Dict[str, Any]:
        """Get Phase 4 specific metrics"""
        status = {}
        for feature_type, metrics in self.phase4_metrics.items():
            total = metrics['success_count'] + metrics['failure_count']
            success_rate = (metrics['success_count'] / total * 100) if total > 0 else 0.0
            
            status[feature_type] = {
                'success_rate': success_rate,
                'avg_latency_ms': metrics['avg_latency'],
                'total_attempts': total,
                'status': 'healthy' if success_rate > 95 else 'warning' if success_rate > 80 else 'critical'
            }
        
        return status


class DataPipelineMonitor(ComponentMonitor):
    """Monitor for data pipeline health"""
    
    def __init__(self):
        super().__init__("data_pipeline", check_interval=60)
        self.data_quality_metrics = {
            'spike_detections': 0,
            'missing_data_pct': 0.0,
            'validation_failures': 0,
            'rollover_detections': 0
        }
    
    def record_data_quality(self, **metrics):
        """Record data quality metrics"""
        self.data_quality_metrics.update(metrics)
        
        # Update overall health based on data quality
        missing_pct = self.data_quality_metrics.get('missing_data_pct', 0)
        validation_failures = self.data_quality_metrics.get('validation_failures', 0)
        
        if missing_pct > 10 or validation_failures > 5:
            self.update_health('critical', f"Data quality issues: {missing_pct:.1f}% missing, {validation_failures} validation failures")
        elif missing_pct > 5 or validation_failures > 2:
            self.update_health('warning', f"Data quality degraded: {missing_pct:.1f}% missing, {validation_failures} validation failures")
        else:
            self.update_health('healthy', "Data quality good")


class RLAgentMonitor(ComponentMonitor):
    """Monitor for RL agent performance"""
    
    def __init__(self):
        super().__init__("rl_agent", check_interval=30)
        self.decision_metrics = {
            'total_decisions': 0,
            'phase4_enhanced_decisions': 0,
            'confidence_scores': deque(maxlen=100),
            'action_distribution': defaultdict(int)
        }
    
    def record_decision(self, action: str, confidence: float, phase4_enhanced: bool):
        """Record RL agent decision"""
        self.decision_metrics['total_decisions'] += 1
        if phase4_enhanced:
            self.decision_metrics['phase4_enhanced_decisions'] += 1
        
        self.decision_metrics['confidence_scores'].append(confidence)
        self.decision_metrics['action_distribution'][action] += 1
    
    def get_decision_quality(self) -> Dict[str, Any]:
        """Get decision quality metrics"""
        confidence_scores = list(self.decision_metrics['confidence_scores'])
        avg_confidence = np.mean(confidence_scores) if confidence_scores else 0.0
        
        phase4_usage_pct = 0.0
        if self.decision_metrics['total_decisions'] > 0:
            phase4_usage_pct = (self.decision_metrics['phase4_enhanced_decisions'] / 
                              self.decision_metrics['total_decisions'] * 100)
        
        return {
            'avg_confidence': avg_confidence,
            'phase4_usage_pct': phase4_usage_pct,
            'action_distribution': dict(self.decision_metrics['action_distribution']),
            'total_decisions': self.decision_metrics['total_decisions']
        }


class SystemHealthMonitor:
    """
    Comprehensive system health monitoring for SuperTrader.AI
    Phase 5 Enhanced with real-time monitoring and alerting
    """
    
    def __init__(self):
        self.config = get_config()
        
        # Component monitors
        self.monitors = {
            'feature_generation': FeatureGenerationMonitor(),
            'data_pipeline': DataPipelineMonitor(), 
            'rl_agent': RLAgentMonitor(),
            'system': ComponentMonitor("system", check_interval=30)
        }
        
        # System-wide metrics
        self.system_metrics = {
            'uptime_seconds': 0,
            'total_memory_mb': 0,
            'available_memory_mb': 0,
            'cpu_usage_pct': 0,
            'disk_usage_pct': 0
        }
        
        # Health monitoring thread
        self._monitoring_thread = None
        self._stop_monitoring = threading.Event()
        self._start_time = datetime.now()
        
        logger.info("System health monitor initialized")
    
    def start_monitoring(self):
        """Start background health monitoring"""
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            logger.warning("Monitoring already running")
            return
        
        self._stop_monitoring.clear()
        self._monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitoring_thread.start()
        
        logger.info("Health monitoring started")
    
    def stop_monitoring(self):
        """Stop background health monitoring"""
        self._stop_monitoring.set()
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5)
        
        logger.info("Health monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while not self._stop_monitoring.is_set():
            try:
                self._update_system_metrics()
                self._check_component_health()
                
                # Sleep for the configured interval
                time.sleep(self.config.monitoring.health_check_frequency_seconds)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(10)  # Short sleep on error
    
    def _update_system_metrics(self):
        """Update system-wide metrics"""
        try:
            if PSUTIL_AVAILABLE:
                # Memory usage
                memory = psutil.virtual_memory()
                self.system_metrics['total_memory_mb'] = memory.total / 1024 / 1024
                self.system_metrics['available_memory_mb'] = memory.available / 1024 / 1024
                
                # CPU usage
                self.system_metrics['cpu_usage_pct'] = psutil.cpu_percent(interval=1)
                
                # Disk usage
                disk = psutil.disk_usage('/')
                self.system_metrics['disk_usage_pct'] = (disk.used / disk.total) * 100
                
                # Update system monitor
                self.monitors['system'].update_metrics(
                    memory_usage_mb=self.system_metrics['total_memory_mb'] - self.system_metrics['available_memory_mb'],
                    cpu_usage_pct=self.system_metrics['cpu_usage_pct']
                )
            else:
                # Fallback system metrics without psutil
                self.system_metrics['total_memory_mb'] = 8192  # Default assumption
                self.system_metrics['available_memory_mb'] = 4096
                self.system_metrics['cpu_usage_pct'] = 0.0
                self.system_metrics['disk_usage_pct'] = 0.0
            
            # Uptime (always available)
            self.system_metrics['uptime_seconds'] = (datetime.now() - self._start_time).total_seconds()
            
        except Exception as e:
            logger.error(f"Failed to update system metrics: {e}")
    
    def _check_component_health(self):
        """Check health of all components"""
        try:
            if PSUTIL_AVAILABLE:
                # Check memory usage
                memory_usage_pct = ((self.system_metrics['total_memory_mb'] - 
                                   self.system_metrics['available_memory_mb']) / 
                                   self.system_metrics['total_memory_mb'] * 100)
                
                if memory_usage_pct > 90:
                    self.monitors['system'].update_health('critical', f"High memory usage: {memory_usage_pct:.1f}%")
                elif memory_usage_pct > 80:
                    self.monitors['system'].update_health('warning', f"Elevated memory usage: {memory_usage_pct:.1f}%")
                else:
                    self.monitors['system'].update_health('healthy', f"System resources normal")
                
                # Check CPU usage
                if self.system_metrics['cpu_usage_pct'] > 95:
                    self.monitors['system'].update_health('critical', f"High CPU usage: {self.system_metrics['cpu_usage_pct']:.1f}%")
                elif self.system_metrics['cpu_usage_pct'] > 80:
                    self.monitors['system'].update_health('warning', f"Elevated CPU usage: {self.system_metrics['cpu_usage_pct']:.1f}%")
            else:
                # Basic health check without system metrics
                uptime_hours = self.system_metrics['uptime_seconds'] / 3600
                if uptime_hours > 24:
                    self.monitors['system'].update_health('healthy', f"System running for {uptime_hours:.1f} hours")
                else:
                    self.monitors['system'].update_health('healthy', "System operational")
                
        except Exception as e:
            logger.error(f"Failed to check component health: {e}")
    
    def get_overall_health(self) -> Dict[str, Any]:
        """Get overall system health status"""
        component_statuses = {}
        overall_status = "healthy"
        
        for name, monitor in self.monitors.items():
            status = monitor.get_status()
            component_statuses[name] = status
            
            # Determine overall status
            if status['health']['status'] == 'critical':
                overall_status = 'critical'
            elif status['health']['status'] == 'warning' and overall_status == 'healthy':
                overall_status = 'warning'
        
        return {
            'overall_status': overall_status,
            'timestamp': datetime.now().isoformat(),
            'uptime_seconds': self.system_metrics['uptime_seconds'],
            'system_metrics': self.system_metrics,
            'components': component_statuses,
            'phase4_status': self.monitors['feature_generation'].get_phase4_status()
        }
    
    def record_feature_generation(self, feature_type: str, success: bool, latency_ms: float):
        """Record feature generation event"""
        self.monitors['feature_generation'].record_feature_generation(feature_type, success, latency_ms)
    
    def record_data_quality(self, **metrics):
        """Record data quality metrics"""
        self.monitors['data_pipeline'].record_data_quality(**metrics)
    
    def record_rl_decision(self, action: str, confidence: float, phase4_enhanced: bool):
        """Record RL agent decision"""
        self.monitors['rl_agent'].record_decision(action, confidence, phase4_enhanced)
    
    def export_health_report(self, filepath: str = None) -> str:
        """Export comprehensive health report"""
        if filepath is None:
            filepath = f"health_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            health_data = self.get_overall_health()
            
            # Add detailed component metrics
            for name, monitor in self.monitors.items():
                if hasattr(monitor, 'get_phase4_status'):
                    health_data['components'][name]['phase4_metrics'] = monitor.get_phase4_status()
                elif hasattr(monitor, 'get_decision_quality'):
                    health_data['components'][name]['decision_quality'] = monitor.get_decision_quality()
            
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w') as f:
                json.dump(health_data, f, indent=2)
            
            logger.info(f"Health report exported to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Failed to export health report: {e}")
            return ""


# Global health monitor instance
_health_monitor_instance = None

def get_health_monitor() -> SystemHealthMonitor:
    """Get global health monitor instance (singleton pattern)"""
    global _health_monitor_instance
    if _health_monitor_instance is None:
        _health_monitor_instance = SystemHealthMonitor()
    return _health_monitor_instance


# Decorator for monitoring function performance
def monitor_performance(component: str, feature_type: str = None):
    """Decorator to automatically monitor function performance"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            success = True
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                raise e
            finally:
                latency_ms = (time.time() - start_time) * 1000
                
                monitor = get_health_monitor()
                if feature_type:
                    monitor.record_feature_generation(feature_type, success, latency_ms)
                else:
                    monitor.monitors[component].update_metrics(
                        latency_ms=latency_ms,
                        success_rate=100.0 if success else 0.0
                    )
        
        return wrapper
    return decorator


if __name__ == "__main__":
    # Test health monitoring system
    monitor = SystemHealthMonitor()
    monitor.start_monitoring()
    
    # Simulate some activity
    import time
    
    for i in range(5):
        # Simulate feature generation
        monitor.record_feature_generation('pcr_integration', True, 50.0)
        monitor.record_feature_generation('multi_timeframe', True, 120.0)
        
        # Simulate RL decisions
        monitor.record_rl_decision('long', 0.85, True)
        
        time.sleep(2)
    
    # Get health status
    health = monitor.get_overall_health()
    print("System Health Status:")
    print(json.dumps(health, indent=2))
    
    # Export report
    report_path = monitor.export_health_report()
    print(f"Health report exported: {report_path}")
    
    monitor.stop_monitoring()