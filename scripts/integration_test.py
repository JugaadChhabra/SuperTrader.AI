"""
Integration Testing Suite - Phase 5

Comprehensive testing to validate all phases (1-4) working together with
Phase 5 enhancements (monitoring, configuration, and RL agent updates).
"""

import sys
import os
import time
import logging
import traceback
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.config import get_config
from utils.logging import setup_logging
from utils.monitoring_integration import (
    start_monitoring, stop_monitoring, get_system_health, 
    export_health_report, DataQualityChecker
)
from data.loaders import DataLoader
from indicators.technical import EnhancedTechnicalIndicators
from models.intraday_environment import IntradayTradingEnv

logger = logging.getLogger(__name__)


class IntegrationTestSuite:
    """
    Comprehensive integration test suite for SuperTrader.AI
    Tests all phases working together with Phase 5 enhancements
    """
    
    def __init__(self):
        self.config = None
        self.test_results = {}
        self.start_time = time.time()
        
        # Test data requirements
        self.min_data_rows = 1000
        self.required_features = [
            'close', 'volume', 'high', 'low', 'open'
        ]
        
        # Phase 4 feature categories to test
        self.phase4_categories = [
            'pcr_integration', 'multi_timeframe', 'enhanced_oi',
            'market_regime', 'microstructure'
        ]
    
    def setup_test_environment(self) -> bool:
        """Setup test environment and validate prerequisites"""
        try:
            logger.info("Setting up integration test environment...")
            
            # Setup logging
            setup_logging()
            
            # Load configuration
            self.config = get_config()
            logger.info(f"Configuration loaded: {len(self.config.__dict__)} settings")
            
            # Start monitoring
            start_monitoring()
            logger.info("Monitoring system started")
            
            # Validate data directory
            data_dir = Path("data")
            if not data_dir.exists():
                logger.error("Data directory not found")
                return False
            
            self.test_results['setup'] = {'status': 'passed', 'details': 'Environment setup successful'}
            return True
            
        except Exception as e:
            logger.error(f"Test environment setup failed: {e}")
            self.test_results['setup'] = {'status': 'failed', 'error': str(e)}
            return False
    
    def test_configuration_system(self) -> bool:
        """Test configuration management system"""
        try:
            logger.info("Testing configuration system...")
            
            # Test configuration loading
            assert self.config is not None, "Configuration not loaded"
            
            # Test Phase 4 configuration
            assert hasattr(self.config, 'phase4'), "Phase 4 configuration missing"
            assert hasattr(self.config.phase4, 'pcr_integration'), "PCR configuration missing"
            assert hasattr(self.config.phase4, 'multi_timeframe'), "Multi-timeframe configuration missing"
            
            # Test trading configuration
            assert hasattr(self.config, 'trading'), "Trading configuration missing"
            assert self.config.trading.initial_capital > 0, "Invalid initial capital"
            
            # Test data configuration
            assert hasattr(self.config, 'data'), "Data configuration missing"
            assert self.config.data.symbol in ['NIFTY', 'BANKNIFTY', 'FINNIFTY'], "Invalid symbol"
            
            # Test monitoring configuration
            assert hasattr(self.config, 'monitoring'), "Monitoring configuration missing"
            assert self.config.monitoring.enabled, "Monitoring should be enabled"
            
            logger.info("Configuration system tests passed")
            self.test_results['configuration'] = {'status': 'passed', 'details': 'All configuration tests passed'}
            return True
            
        except Exception as e:
            logger.error(f"Configuration system test failed: {e}")
            self.test_results['configuration'] = {'status': 'failed', 'error': str(e)}
            return False
    
    def test_data_pipeline(self) -> bool:
        """Test data loading and validation pipeline"""
        try:
            logger.info("Testing data pipeline...")
            
            # Test data file existence
            data_file = Path("data/NIFTY_historical_data_5min.csv")
            if not data_file.exists():
                logger.warning("Using synthetic data for testing")
                data = self._create_synthetic_data()
            else:
                loader = DataLoader(str(data_file))
                data = loader.load_data()
            
            # Validate data structure
            assert isinstance(data, pd.DataFrame), "Data must be DataFrame"
            assert len(data) >= self.min_data_rows, f"Insufficient data rows: {len(data)}"
            
            # Check required columns
            missing_cols = set(self.required_features) - set(data.columns)
            assert len(missing_cols) == 0, f"Missing columns: {missing_cols}"
            
            # Data quality checks
            quality_metrics = DataQualityChecker.check_data_completeness(
                data, self.required_features
            )
            
            assert quality_metrics['missing_data_pct'] < 10, f"Too much missing data: {quality_metrics['missing_data_pct']:.1f}%"
            
            logger.info(f"Data pipeline tests passed - {len(data)} rows, {quality_metrics['missing_data_pct']:.1f}% missing")
            self.test_results['data_pipeline'] = {
                'status': 'passed', 
                'details': f"{len(data)} rows loaded, {quality_metrics['missing_data_pct']:.1f}% missing data",
                'quality_metrics': quality_metrics
            }
            
            # Store data for subsequent tests
            self.test_data = data
            return True
            
        except Exception as e:
            logger.error(f"Data pipeline test failed: {e}")
            self.test_results['data_pipeline'] = {'status': 'failed', 'error': str(e)}
            return False
    
    def test_phase4_indicators(self) -> bool:
        """Test Phase 4 technical indicators"""
        try:
            logger.info("Testing Phase 4 indicators...")
            
            # Initialize indicators
            indicators = EnhancedTechnicalIndicators()
            
            # Test with subset of data for performance
            test_data = self.test_data.tail(500).copy()
            
            # Calculate Phase 4 indicators
            enhanced_data = indicators.add_enhanced_indicators(test_data)
            
            # Validate Phase 4 features
            phase4_features = {}
            
            # Test PCR integration features
            pcr_features = ['pcr_percentile', 'pcr_oversold', 'pcr_overbought']
            for feature in pcr_features:
                if feature in enhanced_data.columns:
                    phase4_features[feature] = enhanced_data[feature].dropna()
            
            # Test multi-timeframe features
            mtf_features = ['momentum_confluence_score', 'trend_confluence_score']
            for feature in mtf_features:
                if feature in enhanced_data.columns:
                    phase4_features[feature] = enhanced_data[feature].dropna()
            
            # Test market regime features
            regime_features = ['market_regime_strong_trend', 'volatility_regime_high']
            for feature in regime_features:
                if feature in enhanced_data.columns:
                    phase4_features[feature] = enhanced_data[feature].dropna()
            
            # Validate feature quality
            quality_metrics = DataQualityChecker.validate_phase4_features(phase4_features)
            
            assert quality_metrics['feature_completeness_pct'] > 50, f"Low feature completeness: {quality_metrics['feature_completeness_pct']:.1f}%"
            
            logger.info(f"Phase 4 indicators test passed - {len(phase4_features)} feature types, {quality_metrics['feature_completeness_pct']:.1f}% complete")
            self.test_results['phase4_indicators'] = {
                'status': 'passed',
                'details': f"{len(phase4_features)} feature types generated",
                'quality_metrics': quality_metrics
            }
            
            # Store enhanced data for RL tests
            self.enhanced_data = enhanced_data
            return True
            
        except Exception as e:
            logger.error(f"Phase 4 indicators test failed: {e}")
            self.test_results['phase4_indicators'] = {'status': 'failed', 'error': str(e)}
            return False
    
    def test_rl_environment(self) -> bool:
        """Test RL environment with Phase 5 enhancements"""
        try:
            logger.info("Testing RL environment...")
            
            # Create environment
            env = IntradayTradingEnv(
                data=self.enhanced_data,
                initial_capital=self.config.trading.initial_capital,
                lot_size=self.config.trading.lot_size,
                max_lots=self.config.trading.max_lots
            )
            
            # Test environment initialization
            assert env.observation_space.shape[0] == 68, f"Expected 68 features, got {env.observation_space.shape[0]}"
            
            # Test episode execution
            state = env.reset()
            assert isinstance(state, np.ndarray), "State must be numpy array"
            assert len(state) == 68, f"State length mismatch: {len(state)}"
            
            # Test multiple steps
            total_reward = 0.0
            episode_steps = 0
            phase4_detected = False
            
            for step in range(min(100, len(self.enhanced_data) // 2)):
                action = np.random.choice(3)  # Random action for testing
                
                next_state, reward, done, info = env.step(action)
                
                # Validate step results
                assert isinstance(next_state, np.ndarray), "Next state must be numpy array"
                assert len(next_state) == 68, f"Next state length mismatch: {len(next_state)}"
                assert isinstance(reward, (int, float)), "Reward must be numeric"
                assert isinstance(done, bool), "Done must be boolean"
                assert isinstance(info, dict), "Info must be dictionary"
                
                # Check Phase 4 enhancement detection
                if 'phase4_enhanced' in info and info['phase4_enhanced']:
                    phase4_detected = True
                
                total_reward += reward
                episode_steps += 1
                
                if done:
                    break
            
            # Validate episode results
            episode_metrics = env.get_episode_metrics()
            assert isinstance(episode_metrics, dict), "Episode metrics must be dictionary"
            
            logger.info(f"RL environment test passed - {episode_steps} steps, Phase 4 detected: {phase4_detected}")
            self.test_results['rl_environment'] = {
                'status': 'passed',
                'details': f"{episode_steps} steps executed, total reward: {total_reward:.2f}",
                'phase4_detected': phase4_detected,
                'episode_metrics': episode_metrics
            }
            return True
            
        except Exception as e:
            logger.error(f"RL environment test failed: {e}")
            self.test_results['rl_environment'] = {'status': 'failed', 'error': str(e)}
            return False
    
    def test_monitoring_system(self) -> bool:
        """Test monitoring and health checking system"""
        try:
            logger.info("Testing monitoring system...")
            
            # Get system health
            health_status = get_system_health()
            
            # Validate health response
            assert isinstance(health_status, dict), "Health status must be dictionary"
            assert 'overall_status' in health_status, "Missing overall status"
            assert health_status['overall_status'] in ['healthy', 'warning', 'critical', 'unknown'], "Invalid status"
            
            # Check component statuses
            if 'components' in health_status:
                for component_name, component_status in health_status['components'].items():
                    assert 'health' in component_status, f"Component {component_name} missing health info"
                    assert 'metrics' in component_status, f"Component {component_name} missing metrics"
            
            # Test health report export
            report_path = export_health_report()
            if report_path:
                report_file = Path(report_path)
                assert report_file.exists(), "Health report file not created"
            
            logger.info(f"Monitoring system test passed - Status: {health_status['overall_status']}")
            self.test_results['monitoring'] = {
                'status': 'passed',
                'details': f"System status: {health_status['overall_status']}",
                'health_status': health_status
            }
            return True
            
        except Exception as e:
            logger.error(f"Monitoring system test failed: {e}")
            self.test_results['monitoring'] = {'status': 'failed', 'error': str(e)}
            return False
    
    def test_end_to_end_integration(self) -> bool:
        """Test complete end-to-end data flow"""
        try:
            logger.info("Testing end-to-end integration...")
            
            # Simulate complete trading pipeline
            # 1. Data loading (already tested)
            # 2. Feature generation (already tested) 
            # 3. RL decision making
            # 4. Monitoring integration
            
            # Create environment with monitoring
            env = IntradayTradingEnv(
                data=self.enhanced_data.tail(200),  # Smaller subset for E2E test
                initial_capital=100000,  # Smaller capital for testing
                lot_size=25
            )
            
            state = env.reset()
            
            # Run mini-episode with monitoring
            decisions_made = 0
            monitoring_events = 0
            
            for step in range(50):  # Short episode for testing
                # Make decision (simulate simple strategy)
                action = 1  # Hold action for stability
                if step % 10 == 0:  # Occasional trade
                    action = 2 if np.random.random() > 0.5 else 0
                
                next_state, reward, done, info = env.step(action)
                decisions_made += 1
                
                # Check if monitoring captured the decision
                if 'phase4_enhanced' in info:
                    monitoring_events += 1
                
                if done:
                    break
            
            # Validate integration results
            assert decisions_made > 0, "No decisions made during E2E test"
            assert monitoring_events > 0, "No monitoring events captured"
            
            # Get final system health
            final_health = get_system_health()
            
            logger.info(f"End-to-end integration test passed - {decisions_made} decisions, {monitoring_events} monitoring events")
            self.test_results['end_to_end'] = {
                'status': 'passed',
                'details': f"{decisions_made} decisions made, {monitoring_events} monitoring events",
                'final_health': final_health['overall_status']
            }
            return True
            
        except Exception as e:
            logger.error(f"End-to-end integration test failed: {e}")
            self.test_results['end_to_end'] = {'status': 'failed', 'error': str(e)}
            return False
    
    def cleanup_test_environment(self):
        """Cleanup test environment"""
        try:
            logger.info("Cleaning up test environment...")
            
            # Stop monitoring
            stop_monitoring()
            
            # Export final health report
            report_path = export_health_report("integration_test_final_report.json")
            if report_path:
                logger.info(f"Final health report exported: {report_path}")
            
            logger.info("Test environment cleanup complete")
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
    
    def _create_synthetic_data(self) -> pd.DataFrame:
        """Create synthetic market data for testing"""
        np.random.seed(42)  # Reproducible data
        
        dates = pd.date_range('2024-01-01 09:15:00', periods=self.min_data_rows, freq='5min')
        
        # Generate realistic price data
        base_price = 22000.0
        returns = np.random.normal(0, 0.01, self.min_data_rows)
        prices = base_price * np.exp(np.cumsum(returns))
        
        data = pd.DataFrame({
            'open': prices + np.random.normal(0, 5, self.min_data_rows),
            'high': prices + np.abs(np.random.normal(10, 5, self.min_data_rows)),
            'low': prices - np.abs(np.random.normal(10, 5, self.min_data_rows)),
            'close': prices,
            'volume': np.random.randint(100000, 1000000, self.min_data_rows),
        }, index=dates)
        
        return data
    
    def generate_test_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        total_time = time.time() - self.start_time
        
        # Count test results
        passed_tests = sum(1 for result in self.test_results.values() if result.get('status') == 'passed')
        failed_tests = sum(1 for result in self.test_results.values() if result.get('status') == 'failed')
        total_tests = len(self.test_results)
        
        report = {
            'test_summary': {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': failed_tests,
                'success_rate': (passed_tests / total_tests * 100) if total_tests > 0 else 0,
                'total_time_seconds': total_time
            },
            'test_results': self.test_results,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'overall_status': 'PASS' if failed_tests == 0 else 'FAIL'
        }
        
        return report
    
    def run_all_tests(self) -> bool:
        """Run complete integration test suite"""
        try:
            logger.info("="*50)
            logger.info("Starting SuperTrader.AI Integration Test Suite")
            logger.info("="*50)
            
            # Run tests in sequence
            tests = [
                ('Setup', self.setup_test_environment),
                ('Configuration System', self.test_configuration_system),
                ('Data Pipeline', self.test_data_pipeline),
                ('Phase 4 Indicators', self.test_phase4_indicators),
                ('RL Environment', self.test_rl_environment),
                ('Monitoring System', self.test_monitoring_system),
                ('End-to-End Integration', self.test_end_to_end_integration)
            ]
            
            all_passed = True
            
            for test_name, test_func in tests:
                logger.info(f"Running test: {test_name}")
                try:
                    result = test_func()
                    status = "PASS" if result else "FAIL"
                    logger.info(f"Test {test_name}: {status}")
                    
                    if not result:
                        all_passed = False
                        
                except Exception as e:
                    logger.error(f"Test {test_name} crashed: {e}")
                    logger.error(traceback.format_exc())
                    all_passed = False
                
                logger.info("-" * 30)
            
            # Generate final report
            report = self.generate_test_report()
            
            logger.info("="*50)
            logger.info("Integration Test Results")
            logger.info("="*50)
            logger.info(f"Total Tests: {report['test_summary']['total_tests']}")
            logger.info(f"Passed: {report['test_summary']['passed_tests']}")
            logger.info(f"Failed: {report['test_summary']['failed_tests']}")
            logger.info(f"Success Rate: {report['test_summary']['success_rate']:.1f}%")
            logger.info(f"Total Time: {report['test_summary']['total_time_seconds']:.1f}s")
            logger.info(f"Overall Status: {report['overall_status']}")
            
            # Save report
            report_file = f"integration_test_report_{time.strftime('%Y%m%d_%H%M%S')}.json"
            import json
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Detailed report saved: {report_file}")
            
            return all_passed
            
        except Exception as e:
            logger.error(f"Integration test suite failed: {e}")
            logger.error(traceback.format_exc())
            return False
        
        finally:
            self.cleanup_test_environment()


def main():
    """Main integration test entry point"""
    test_suite = IntegrationTestSuite()
    success = test_suite.run_all_tests()
    
    exit_code = 0 if success else 1
    sys.exit(exit_code)


if __name__ == "__main__":
    main()