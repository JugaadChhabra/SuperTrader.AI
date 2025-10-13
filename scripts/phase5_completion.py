"""
Phase 5 Completion and Validation Script

Final validation and completion script for Phase 5 of SuperTrader.AI.
Validates all Phase 5 enhancements and provides comprehensive system status.
"""

import sys
import os
import time
import json
from pathlib import Path
from datetime import datetime
import logging
from typing import Dict, Any, List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import Phase 5 components
from utils.config import get_config
from utils.logging import setup_logging
from utils.monitoring import SystemHealthMonitor, get_health_monitor
from utils.monitoring_integration import start_monitoring, stop_monitoring, get_system_health
from utils.performance import get_performance_optimizer, cached
from utils.production import ProductionDeploymentManager, setup_production_deployment

logger = logging.getLogger(__name__)


class Phase5Validator:
    """
    Comprehensive validator for Phase 5 implementation
    """
    
    def __init__(self):
        self.validation_results = {}
        self.start_time = time.time()
        
        # Setup logging
        setup_logging()
        self.logger = logging.getLogger(__name__)
        self.logger.info("Phase 5 validation started")
    
    def validate_configuration_system(self) -> bool:
        """Validate enhanced configuration management system"""
        try:
            self.logger.info("Validating configuration system...")
            
            # Test configuration loading
            config = get_config()
            
            # Validate Phase 4 configuration exists
            assert hasattr(config, 'phase4'), "Phase 4 configuration missing"
            assert hasattr(config.phase4, 'pcr_integration'), "PCR integration config missing"
            assert hasattr(config.phase4, 'multi_timeframe'), "Multi-timeframe config missing"
            assert hasattr(config.phase4, 'enhanced_oi'), "Enhanced OI config missing"
            assert hasattr(config.phase4, 'market_regime'), "Market regime config missing"
            assert hasattr(config.phase4, 'microstructure'), "Microstructure config missing"
            
            # Validate Phase 5 performance configuration
            assert hasattr(config, 'performance'), "Performance configuration missing"
            assert hasattr(config.performance, 'cache_max_size'), "Cache configuration missing"
            assert hasattr(config.performance, 'enable_memory_optimization'), "Memory optimization config missing"
            
            # Validate monitoring configuration
            assert hasattr(config, 'monitoring'), "Monitoring configuration missing"
            assert config.monitoring.enabled, "Monitoring should be enabled"
            
            self.validation_results['configuration_system'] = {
                'status': 'PASS',
                'details': 'All configuration components validated successfully'
            }
            return True
            
        except Exception as e:
            self.logger.error(f"Configuration system validation failed: {e}")
            self.validation_results['configuration_system'] = {
                'status': 'FAIL',
                'error': str(e)
            }
            return False
    
    def validate_monitoring_system(self) -> bool:
        """Validate pipeline health monitoring system"""
        try:
            self.logger.info("Validating monitoring system...")
            
            # Test monitoring system initialization
            monitor = get_health_monitor()
            assert isinstance(monitor, SystemHealthMonitor), "Health monitor initialization failed"
            
            # Start monitoring
            start_monitoring()
            time.sleep(2)  # Allow monitoring to initialize
            
            # Test health status retrieval
            health_status = get_system_health()
            assert isinstance(health_status, dict), "Health status should be dictionary"
            assert 'overall_status' in health_status, "Overall status missing"
            assert 'components' in health_status, "Component statuses missing"
            
            # Test monitoring integration
            from utils.monitoring_integration import DataQualityChecker
            
            # Test data quality checking
            import pandas as pd
            test_data = pd.DataFrame({
                'close': [100, 101, 102, 103, 104],
                'volume': [1000, 1100, 1200, 1300, 1400]
            })
            
            quality_metrics = DataQualityChecker.check_data_completeness(
                test_data, ['close', 'volume']
            )
            assert isinstance(quality_metrics, dict), "Quality metrics should be dictionary"
            
            # Stop monitoring
            stop_monitoring()
            
            self.validation_results['monitoring_system'] = {
                'status': 'PASS',
                'details': f'Monitoring system operational, status: {health_status["overall_status"]}',
                'health_status': health_status
            }
            return True
            
        except Exception as e:
            self.logger.error(f"Monitoring system validation failed: {e}")
            self.validation_results['monitoring_system'] = {
                'status': 'FAIL',
                'error': str(e)
            }
            return False
    
    def validate_performance_optimization(self) -> bool:
        """Validate performance optimization framework"""
        try:
            self.logger.info("Validating performance optimization...")
            
            # Test performance optimizer
            optimizer = get_performance_optimizer()
            
            # Test caching decorator
            call_count = 0
            
            @cached()
            def test_cached_function(x: int) -> int:
                nonlocal call_count
                call_count += 1
                time.sleep(0.01)  # Simulate work
                return x * 2
            
            # First call (cache miss)
            result1 = test_cached_function(5)
            assert result1 == 10, "Function result incorrect"
            assert call_count == 1, "Function should be called once"
            
            # Second call (cache hit)
            result2 = test_cached_function(5)
            assert result2 == 10, "Cached result incorrect"
            assert call_count == 1, "Function should not be called again (cache hit)"
            
            # Test performance report
            perf_report = optimizer.get_performance_report()
            assert isinstance(perf_report, dict), "Performance report should be dictionary"
            assert 'function_performance' in perf_report, "Function performance metrics missing"
            
            # Test memory optimization
            from utils.performance import optimize_memory_usage
            collected = optimize_memory_usage()
            assert isinstance(collected, int), "Memory optimization should return collection count"
            
            self.validation_results['performance_optimization'] = {
                'status': 'PASS',
                'details': 'Performance optimization framework operational',
                'performance_report': perf_report
            }
            return True
            
        except Exception as e:
            self.logger.error(f"Performance optimization validation failed: {e}")
            self.validation_results['performance_optimization'] = {
                'status': 'FAIL',
                'error': str(e)
            }
            return False
    
    def validate_rl_agent_enhancements(self) -> bool:
        """Validate RL agent observation space updates"""
        try:
            self.logger.info("Validating RL agent enhancements...")
            
            # Test enhanced intraday environment
            from models.intraday_environment import IntradayTradingEnv
            import numpy as np
            import pandas as pd
            
            # Create test data with Phase 4 features
            test_data = pd.DataFrame({
                'open': np.random.randn(100) + 100,
                'high': np.random.randn(100) + 101,
                'low': np.random.randn(100) + 99,
                'close': np.random.randn(100) + 100,
                'volume': np.random.randint(1000, 10000, 100),
                # Add some Phase 4 feature columns
                'pcr_percentile': np.random.uniform(0, 100, 100),
                'momentum_confluence_score': np.random.uniform(0, 3, 100),
                'market_regime_strong_trend': np.random.choice([0, 1], 100)
            })
            
            # Test environment creation
            env = IntradayTradingEnv(data=test_data, initial_capital=100000)
            
            # Test observation space
            assert hasattr(env, 'observation_space'), "Observation space missing"
            assert env.observation_space.shape[0] == 68, f"Expected 68 features, got {env.observation_space.shape[0]}"
            
            # Test state generation
            state = env.reset()
            assert isinstance(state, np.ndarray), "State should be numpy array"
            assert len(state) == 68, f"State length should be 68, got {len(state)}"
            
            # Test Phase 4 feature detection
            assert hasattr(env, '_detect_phase4_features'), "Phase 4 feature detection method missing"
            
            self.validation_results['rl_agent_enhancements'] = {
                'status': 'PASS',
                'details': f'RL agent enhanced with 68-feature observation space',
                'observation_space_size': env.observation_space.shape[0]
            }
            return True
            
        except Exception as e:
            self.logger.error(f"RL agent enhancements validation failed: {e}")
            self.validation_results['rl_agent_enhancements'] = {
                'status': 'FAIL',
                'error': str(e)
            }
            return False
    
    def validate_integration_testing(self) -> bool:
        """Validate integration testing framework"""
        try:
            self.logger.info("Validating integration testing framework...")
            
            # Check integration test script exists
            integration_test_path = Path("scripts/integration_test.py")
            assert integration_test_path.exists(), "Integration test script missing"
            
            # Test integration test class
            from scripts.integration_test import IntegrationTestSuite
            
            test_suite = IntegrationTestSuite()
            assert hasattr(test_suite, 'run_all_tests'), "Integration test suite incomplete"
            assert hasattr(test_suite, 'test_configuration_system'), "Configuration test missing"
            assert hasattr(test_suite, 'test_monitoring_system'), "Monitoring test missing"
            assert hasattr(test_suite, 'test_rl_environment'), "RL environment test missing"
            
            self.validation_results['integration_testing'] = {
                'status': 'PASS',
                'details': 'Integration testing framework complete',
                'test_script_path': str(integration_test_path)
            }
            return True
            
        except Exception as e:
            self.logger.error(f"Integration testing validation failed: {e}")
            self.validation_results['integration_testing'] = {
                'status': 'FAIL',
                'error': str(e)
            }
            return False
    
    def validate_production_deployment(self) -> bool:
        """Validate production deployment configuration"""
        try:
            self.logger.info("Validating production deployment...")
            
            # Test production deployment manager
            deployment_manager = ProductionDeploymentManager("staging")
            
            # Validate configuration
            validation = deployment_manager.validate_configuration()
            assert isinstance(validation, dict), "Validation result should be dictionary"
            assert 'valid' in validation, "Validation status missing"
            
            # Test deployment file generation (in test mode)
            test_output_dir = "test_deployment"
            deployment_manager.generate_deployment_files(test_output_dir)
            
            # Check generated files
            test_dir = Path(test_output_dir)
            expected_files = [
                "Dockerfile", "docker-compose.yml", "supertrader.service",
                "nginx.conf", ".env.template", "monitoring.yml"
            ]
            
            for file_name in expected_files:
                file_path = test_dir / file_name
                assert file_path.exists(), f"Deployment file {file_name} not generated"
            
            # Clean up test files
            import shutil
            if test_dir.exists():
                shutil.rmtree(test_dir)
            
            self.validation_results['production_deployment'] = {
                'status': 'PASS',
                'details': 'Production deployment configuration complete',
                'validation_result': validation
            }
            return True
            
        except Exception as e:
            self.logger.error(f"Production deployment validation failed: {e}")
            self.validation_results['production_deployment'] = {
                'status': 'FAIL',
                'error': str(e)
            }
            return False
    
    def generate_phase5_report(self) -> Dict[str, Any]:
        """Generate comprehensive Phase 5 completion report"""
        total_time = time.time() - self.start_time
        
        # Count validation results
        passed_validations = sum(1 for result in self.validation_results.values() 
                               if result.get('status') == 'PASS')
        failed_validations = sum(1 for result in self.validation_results.values() 
                               if result.get('status') == 'FAIL')
        total_validations = len(self.validation_results)
        
        # Phase 5 feature summary
        phase5_features = [
            "RL Agent Observation Space Updates (32→68 features)",
            "Enhanced Configuration Management System",
            "Pipeline Health Monitoring with Real-time Metrics",
            "Complete System Integration Testing Framework",
            "Performance Optimization with Advanced Caching",
            "Production Deployment Configuration"
        ]
        
        report = {
            'phase5_completion': {
                'version': '5.0.0',
                'completion_date': datetime.now().isoformat(),
                'validation_summary': {
                    'total_validations': total_validations,
                    'passed_validations': passed_validations,
                    'failed_validations': failed_validations,
                    'success_rate': (passed_validations / total_validations * 100) if total_validations > 0 else 0,
                    'validation_time_seconds': total_time
                },
                'features_implemented': phase5_features,
                'overall_status': 'COMPLETE' if failed_validations == 0 else 'PARTIAL'
            },
            'validation_details': self.validation_results,
            'next_steps': self._get_next_steps(),
            'deployment_readiness': self._assess_deployment_readiness()
        }
        
        return report
    
    def _get_next_steps(self) -> List[str]:
        """Get recommended next steps based on validation results"""
        next_steps = []
        
        # Check for any failed validations
        failed_components = [component for component, result in self.validation_results.items() 
                           if result.get('status') == 'FAIL']
        
        if failed_components:
            next_steps.append(f"Fix failed validations: {', '.join(failed_components)}")
        
        # General next steps for production
        if not failed_components:
            next_steps.extend([
                "Run comprehensive integration tests (scripts/integration_test.py)",
                "Setup production environment variables (.env.template)",
                "Generate SSL certificates for HTTPS",
                "Configure monitoring alerts and notifications",
                "Perform load testing with realistic trading volumes",
                "Setup database backups and disaster recovery",
                "Deploy to staging environment for final validation",
                "Schedule production deployment"
            ])
        
        return next_steps
    
    def _assess_deployment_readiness(self) -> Dict[str, Any]:
        """Assess readiness for production deployment"""
        failed_validations = sum(1 for result in self.validation_results.values() 
                               if result.get('status') == 'FAIL')
        
        readiness_score = ((len(self.validation_results) - failed_validations) / 
                          len(self.validation_results) * 100) if self.validation_results else 0
        
        if readiness_score == 100:
            readiness_level = "READY"
            readiness_message = "All Phase 5 validations passed. System ready for production deployment."
        elif readiness_score >= 80:
            readiness_level = "MOSTLY_READY"
            readiness_message = "Most validations passed. Address remaining issues before production."
        elif readiness_score >= 60:
            readiness_level = "NEEDS_WORK"
            readiness_message = "Several validations failed. Significant work needed before deployment."
        else:
            readiness_level = "NOT_READY"
            readiness_message = "Multiple critical issues. Not ready for production deployment."
        
        return {
            'level': readiness_level,
            'score': readiness_score,
            'message': readiness_message,
            'failed_components': [component for component, result in self.validation_results.items() 
                                if result.get('status') == 'FAIL']
        }
    
    def run_complete_validation(self) -> bool:
        """Run complete Phase 5 validation"""
        try:
            self.logger.info("="*60)
            self.logger.info("SuperTrader.AI Phase 5 Completion Validation")
            self.logger.info("="*60)
            
            # Run all validations
            validations = [
                ("Configuration System", self.validate_configuration_system),
                ("Monitoring System", self.validate_monitoring_system),
                ("Performance Optimization", self.validate_performance_optimization),
                ("RL Agent Enhancements", self.validate_rl_agent_enhancements),
                ("Integration Testing", self.validate_integration_testing),
                ("Production Deployment", self.validate_production_deployment)
            ]
            
            all_passed = True
            
            for validation_name, validation_func in validations:
                self.logger.info(f"Running validation: {validation_name}")
                try:
                    result = validation_func()
                    status = "PASS" if result else "FAIL"
                    self.logger.info(f"Validation {validation_name}: {status}")
                    
                    if not result:
                        all_passed = False
                        
                except Exception as e:
                    self.logger.error(f"Validation {validation_name} crashed: {e}")
                    all_passed = False
                
                self.logger.info("-" * 40)
            
            # Generate final report
            report = self.generate_phase5_report()
            
            # Display results
            self.logger.info("="*60)
            self.logger.info("Phase 5 Validation Results")
            self.logger.info("="*60)
            summary = report['phase5_completion']['validation_summary']
            self.logger.info(f"Total Validations: {summary['total_validations']}")
            self.logger.info(f"Passed: {summary['passed_validations']}")
            self.logger.info(f"Failed: {summary['failed_validations']}")
            self.logger.info(f"Success Rate: {summary['success_rate']:.1f}%")
            self.logger.info(f"Overall Status: {report['phase5_completion']['overall_status']}")
            
            # Deployment readiness
            readiness = report['deployment_readiness']
            self.logger.info(f"Deployment Readiness: {readiness['level']} ({readiness['score']:.1f}%)")
            self.logger.info(f"Message: {readiness['message']}")
            
            # Save detailed report
            report_file = f"phase5_completion_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            self.logger.info(f"Detailed report saved: {report_file}")
            
            # Next steps
            if report['next_steps']:
                self.logger.info("\nNext Steps:")
                for i, step in enumerate(report['next_steps'], 1):
                    self.logger.info(f"{i}. {step}")
            
            return all_passed
            
        except Exception as e:
            self.logger.error(f"Phase 5 validation failed: {e}")
            return False


def main():
    """Main Phase 5 validation entry point"""
    validator = Phase5Validator()
    success = validator.run_complete_validation()
    
    if success:
        print("\n🎉 Phase 5 Implementation COMPLETE! 🎉")
        print("\nSuperTrader.AI is now enhanced with:")
        print("✅ Expanded RL observation space (68 features)")
        print("✅ Comprehensive configuration management")
        print("✅ Real-time pipeline health monitoring")
        print("✅ Complete integration testing framework")
        print("✅ Advanced performance optimization")
        print("✅ Production deployment configuration")
        print("\nSystem is ready for production deployment!")
    else:
        print("\n⚠️  Phase 5 Implementation Incomplete")
        print("Please address the failed validations before proceeding to production.")
    
    exit_code = 0 if success else 1
    sys.exit(exit_code)


if __name__ == "__main__":
    main()