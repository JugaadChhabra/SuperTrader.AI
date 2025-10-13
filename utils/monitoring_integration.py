"""
Monitoring Integration - Phase 5

Integration layer to connect monitoring system with existing SuperTrader.AI components.
Provides seamless integration without requiring major refactoring.
"""

import logging
import time
from typing import Dict, Any, Optional
from functools import wraps
import traceback
import numpy as np

from utils.monitoring import get_health_monitor, monitor_performance

logger = logging.getLogger(__name__)


class MonitoringIntegration:
    """
    Integration layer for connecting monitoring to existing components
    """
    
    def __init__(self):
        self.health_monitor = get_health_monitor()
        self._initialized = False
    
    def initialize(self):
        """Initialize monitoring integration"""
        if self._initialized:
            return
        
        try:
            self.health_monitor.start_monitoring()
            self._initialized = True
            logger.info("Monitoring integration initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize monitoring: {e}")
    
    def shutdown(self):
        """Shutdown monitoring integration"""
        try:
            if self._initialized:
                self.health_monitor.stop_monitoring()
                self._initialized = False
                logger.info("Monitoring integration shutdown complete")
        except Exception as e:
            logger.error(f"Error during monitoring shutdown: {e}")


# Global integration instance
_integration_instance = None

def get_monitoring_integration() -> MonitoringIntegration:
    """Get global monitoring integration instance"""
    global _integration_instance
    if _integration_instance is None:
        _integration_instance = MonitoringIntegration()
    return _integration_instance


# Feature generation monitoring decorators
def monitor_phase4_feature(feature_type: str):
    """Decorator for monitoring Phase 4 feature generation"""
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
                logger.error(f"Phase 4 feature '{feature_type}' generation failed: {e}")
                raise e
            finally:
                latency_ms = (time.time() - start_time) * 1000
                
                try:
                    monitor = get_health_monitor()
                    monitor.record_feature_generation(feature_type, success, latency_ms)
                except Exception as e:
                    logger.warning(f"Failed to record monitoring data: {e}")
        
        return wrapper
    return decorator


def monitor_data_quality(func):
    """Decorator for monitoring data quality"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            
            # Extract data quality metrics if available
            if hasattr(result, 'get') and isinstance(result, dict):
                quality_metrics = {}
                
                # Check for common data quality indicators
                if 'missing_data_pct' in result:
                    quality_metrics['missing_data_pct'] = result['missing_data_pct']
                if 'validation_failures' in result:
                    quality_metrics['validation_failures'] = result['validation_failures']
                if 'spike_detections' in result:
                    quality_metrics['spike_detections'] = result['spike_detections']
                
                if quality_metrics:
                    monitor = get_health_monitor()
                    monitor.record_data_quality(**quality_metrics)
            
            return result
            
        except Exception as e:
            logger.error(f"Data quality monitoring failed: {e}")
            raise e
    
    return wrapper


def monitor_rl_decision(func):
    """Decorator for monitoring RL agent decisions"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            
            # Extract decision information
            if isinstance(result, dict):
                action = result.get('action', 'unknown')
                confidence = result.get('confidence', 0.0)
                phase4_enhanced = result.get('phase4_enhanced', False)
                
                monitor = get_health_monitor()
                monitor.record_rl_decision(action, confidence, phase4_enhanced)
            
            return result
            
        except Exception as e:
            logger.error(f"RL decision monitoring failed: {e}")
            raise e
    
    return wrapper


class PerformanceTracker:
    """
    Context manager for tracking performance of code blocks
    """
    
    def __init__(self, component: str, operation: str, feature_type: str = None):
        self.component = component
        self.operation = operation
        self.feature_type = feature_type
        self.start_time = None
        self.success = True
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.success = False
        
        latency_ms = (time.time() - self.start_time) * 1000
        
        try:
            monitor = get_health_monitor()
            if self.feature_type:
                monitor.record_feature_generation(self.feature_type, self.success, latency_ms)
            else:
                monitor.monitors[self.component].update_metrics(
                    latency_ms=latency_ms,
                    success_rate=100.0 if self.success else 0.0
                )
        except Exception as e:
            logger.warning(f"Failed to record performance metrics: {e}")


class DataQualityChecker:
    """
    Helper class for data quality checks and monitoring
    """
    
    @staticmethod
    def check_data_completeness(data, expected_columns: list) -> Dict[str, Any]:
        """Check data completeness and return quality metrics"""
        import pandas as pd
        
        if not isinstance(data, pd.DataFrame):
            return {'validation_failures': 1, 'missing_data_pct': 100.0}
        
        quality_metrics = {
            'missing_data_pct': 0.0,
            'validation_failures': 0,
            'spike_detections': 0
        }
        
        try:
            # Check missing data
            missing_pct = (data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100
            quality_metrics['missing_data_pct'] = missing_pct
            
            # Check for expected columns
            missing_columns = set(expected_columns) - set(data.columns)
            if missing_columns:
                quality_metrics['validation_failures'] += len(missing_columns)
            
            # Check for data spikes (simple outlier detection)
            numeric_columns = data.select_dtypes(include=['number']).columns
            for col in numeric_columns:
                if col in data.columns:
                    q99 = data[col].quantile(0.99)
                    q01 = data[col].quantile(0.01)
                    outliers = ((data[col] > q99 * 3) | (data[col] < q01 * 3)).sum()
                    quality_metrics['spike_detections'] += outliers
            
        except Exception as e:
            logger.error(f"Data quality check failed: {e}")
            quality_metrics['validation_failures'] += 1
        
        return quality_metrics
    
    @staticmethod
    def validate_phase4_features(features: Dict[str, Any]) -> Dict[str, Any]:
        """Validate Phase 4 features and return quality metrics"""
        quality_metrics = {
            'missing_features': 0,
            'invalid_values': 0,
            'feature_completeness_pct': 0.0
        }
        
        expected_phase4_categories = [
            'pcr_integration', 'multi_timeframe', 'enhanced_oi', 
            'market_regime', 'microstructure'
        ]
        
        try:
            total_expected = len(expected_phase4_categories)
            available_features = 0
            
            for category in expected_phase4_categories:
                if category in features:
                    available_features += 1
                    
                    # Check for NaN or infinite values
                    feature_data = features[category]
                    if isinstance(feature_data, dict):
                        for key, value in feature_data.items():
                            if value is None or (isinstance(value, float) and 
                                              (np.isnan(value) or np.isinf(value))):
                                quality_metrics['invalid_values'] += 1
                    elif feature_data is None:
                        quality_metrics['invalid_values'] += 1
                else:
                    quality_metrics['missing_features'] += 1
            
            quality_metrics['feature_completeness_pct'] = (available_features / total_expected) * 100
            
        except Exception as e:
            logger.error(f"Phase 4 feature validation failed: {e}")
            quality_metrics['missing_features'] = len(expected_phase4_categories)
        
        return quality_metrics


# Utility functions for easy integration
def start_monitoring():
    """Start monitoring system"""
    integration = get_monitoring_integration()
    integration.initialize()


def stop_monitoring():
    """Stop monitoring system"""
    integration = get_monitoring_integration()
    integration.shutdown()


def get_system_health() -> Dict[str, Any]:
    """Get current system health status"""
    try:
        monitor = get_health_monitor()
        return monitor.get_overall_health()
    except Exception as e:
        logger.error(f"Failed to get system health: {e}")
        return {'overall_status': 'unknown', 'error': str(e)}


def export_health_report(filepath: str = None) -> str:
    """Export health report"""
    try:
        monitor = get_health_monitor()
        return monitor.export_health_report(filepath)
    except Exception as e:
        logger.error(f"Failed to export health report: {e}")
        return ""


def track_performance(component: str, operation: str, feature_type: str = None):
    """Create performance tracker context manager"""
    return PerformanceTracker(component, operation, feature_type)


# Example usage patterns for integration
if __name__ == "__main__":
    # Example: Using decorators
    @monitor_phase4_feature('pcr_integration')
    def calculate_pcr_features():
        # Simulate PCR calculation
        time.sleep(0.1)
        return {'pcr_ratio': 1.2, 'pcr_sentiment': 0.65}
    
    @monitor_rl_decision
    def make_trading_decision():
        # Simulate RL decision
        return {
            'action': 'long',
            'confidence': 0.85,
            'phase4_enhanced': True
        }
    
    # Example: Using context manager
    def process_market_data():
        with track_performance('data_pipeline', 'market_data_processing'):
            # Simulate data processing
            time.sleep(0.05)
            return {'processed_bars': 100}
    
    # Example: Data quality checking
    import pandas as pd
    
    def validate_data():
        # Create sample data
        data = pd.DataFrame({
            'close': [100, 101, 102, 103],
            'volume': [1000, 1100, 1200, 1300]
        })
        
        quality_metrics = DataQualityChecker.check_data_completeness(
            data, ['close', 'volume', 'open']
        )
        
        monitor = get_health_monitor()
        monitor.record_data_quality(**quality_metrics)
    
    # Run examples
    start_monitoring()
    
    try:
        calculate_pcr_features()
        make_trading_decision()
        process_market_data()
        validate_data()
        
        # Get health status
        health = get_system_health()
        print("System Health:", health['overall_status'])
        
    finally:
        stop_monitoring()