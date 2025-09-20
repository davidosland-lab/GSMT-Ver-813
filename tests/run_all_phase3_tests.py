#!/usr/bin/env python3
"""
Phase 3 Master Test Runner
=========================

Comprehensive test orchestrator for all Phase 3 extension components.
Runs all test suites and generates unified test report.
"""

import asyncio
import logging
import json
import sys
import os
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import all test suites
from tests.test_phase3_p005_advanced_feature_engineering import run_comprehensive_feature_engineering_tests
from tests.test_phase3_p006_reinforcement_learning import run_comprehensive_rl_tests
from tests.test_phase3_p007_advanced_risk_management import run_comprehensive_risk_management_tests
from tests.test_phase3_extended_unified_predictor import run_comprehensive_extended_predictor_tests
from tests.test_phase3_performance_benchmarking import run_performance_benchmark_suite

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Phase3TestOrchestrator:
    """Orchestrates all Phase 3 test suites"""
    
    def __init__(self):
        self.test_results = {}
        self.execution_summary = {}
        self.start_time = None
    
    async def run_all_tests(self):
        """Run all Phase 3 test suites"""
        logger.info("🚀 PHASE 3 COMPREHENSIVE TEST SUITE EXECUTION")
        logger.info("=" * 80)
        logger.info("Testing Phase 3 Extensions: P3-005, P3-006, P3-007 + Integration")
        logger.info("=" * 80)
        
        self.start_time = datetime.now()
        
        # Define test suites
        test_suites = [
            {
                'name': 'P3-005 Advanced Feature Engineering',
                'module': 'Feature Engineering Pipeline',
                'function': run_comprehensive_feature_engineering_tests,
                'description': 'Multi-modal feature fusion and engineering'
            },
            {
                'name': 'P3-006 Reinforcement Learning',
                'module': 'RL Integration Framework', 
                'function': run_comprehensive_rl_tests,
                'description': 'Adaptive model selection and learning'
            },
            {
                'name': 'P3-007 Advanced Risk Management',
                'module': 'Risk Management Framework',
                'function': run_comprehensive_risk_management_tests,
                'description': 'VaR, position sizing, and risk controls'
            },
            {
                'name': 'Extended Unified Predictor Integration',
                'module': 'End-to-End Integration',
                'function': run_comprehensive_extended_predictor_tests,
                'description': 'Complete system integration testing'
            },
            {
                'name': 'Performance Benchmarking',
                'module': 'Performance Validation',
                'function': run_performance_benchmark_suite,
                'description': 'Performance, scalability, and accuracy validation'
            }
        ]
        
        # Execute all test suites
        for i, suite in enumerate(test_suites, 1):
            await self._run_test_suite(suite, i, len(test_suites))
        
        # Generate final report
        return self._generate_final_report()
    
    async def _run_test_suite(self, suite, current, total):
        """Run individual test suite"""
        logger.info(f"\n{'='*20} TEST SUITE {current}/{total} {'='*20}")
        logger.info(f"🧪 {suite['name']}")
        logger.info(f"📦 Module: {suite['module']}")
        logger.info(f"📝 Description: {suite['description']}")
        logger.info(f"{'='*60}")
        
        suite_start = datetime.now()
        
        try:
            # Run the test suite
            success, results = await suite['function']()
            suite_end = datetime.now()
            
            execution_time = (suite_end - suite_start).total_seconds()
            
            self.test_results[suite['name']] = {
                'success': success,
                'results': results,
                'execution_time': execution_time,
                'module': suite['module'],
                'description': suite['description'],
                'timestamp': suite_start.isoformat()
            }
            
            # Log suite completion
            status = "✅ PASSED" if success else "❌ FAILED"
            logger.info(f"\n🎯 {suite['name']}: {status}")
            logger.info(f"⏱️ Execution time: {execution_time:.2f}s")
            
            if isinstance(results, dict):
                successful_tests = sum(1 for r in results.values() 
                                     if isinstance(r, dict) and r.get('status') == 'SUCCESS')
                total_tests = len([r for r in results.values() 
                                 if isinstance(r, dict) and 'status' in r])
                
                if total_tests > 0:
                    logger.info(f"📊 Test Results: {successful_tests}/{total_tests} passed ({successful_tests/total_tests*100:.1f}%)")
            
        except Exception as e:
            suite_end = datetime.now()
            execution_time = (suite_end - suite_start).total_seconds()
            
            logger.error(f"❌ {suite['name']} EXECUTION FAILED: {e}")
            
            self.test_results[suite['name']] = {
                'success': False,
                'results': {'execution_error': str(e)},
                'execution_time': execution_time,
                'module': suite['module'],
                'description': suite['description'],
                'timestamp': suite_start.isoformat(),
                'error': str(e)
            }
        
        logger.info(f"{'='*60}")
    
    def _generate_final_report(self):
        """Generate comprehensive final report"""
        end_time = datetime.now()
        total_execution_time = (end_time - self.start_time).total_seconds()
        
        # Calculate overall statistics
        total_suites = len(self.test_results)
        successful_suites = sum(1 for result in self.test_results.values() 
                               if result['success'])
        success_rate = successful_suites / total_suites if total_suites > 0 else 0
        
        # Detailed test statistics
        total_individual_tests = 0
        successful_individual_tests = 0
        
        for suite_name, suite_result in self.test_results.items():
            if isinstance(suite_result.get('results'), dict):
                suite_tests = suite_result['results']
                for test_name, test_result in suite_tests.items():
                    if isinstance(test_result, dict) and 'status' in test_result:
                        total_individual_tests += 1
                        if test_result.get('status') == 'SUCCESS':
                            successful_individual_tests += 1
        
        individual_success_rate = (successful_individual_tests / total_individual_tests 
                                 if total_individual_tests > 0 else 0)
        
        # Create execution summary
        self.execution_summary = {
            'execution_metadata': {
                'start_time': self.start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'total_execution_time': total_execution_time,
                'test_environment': 'Phase 3 Development Environment'
            },
            'suite_statistics': {
                'total_suites': total_suites,
                'successful_suites': successful_suites,
                'failed_suites': total_suites - successful_suites,
                'suite_success_rate': success_rate
            },
            'test_statistics': {
                'total_individual_tests': total_individual_tests,
                'successful_individual_tests': successful_individual_tests,
                'failed_individual_tests': total_individual_tests - successful_individual_tests,
                'individual_success_rate': individual_success_rate
            },
            'component_status': self._assess_component_status(),
            'quality_assessment': self._assess_overall_quality(),
            'recommendations': self._generate_recommendations()
        }
        
        # Log final results
        self._log_final_results()
        
        # Save comprehensive report
        self._save_comprehensive_report()
        
        return success_rate >= 0.8, self.execution_summary
    
    def _assess_component_status(self):
        """Assess status of each Phase 3 component"""
        component_status = {}
        
        component_mapping = {
            'P3-005 Advanced Feature Engineering': 'P3-005_Feature_Engineering',
            'P3-006 Reinforcement Learning': 'P3-006_Reinforcement_Learning', 
            'P3-007 Advanced Risk Management': 'P3-007_Risk_Management',
            'Extended Unified Predictor Integration': 'Integration_Layer',
            'Performance Benchmarking': 'Performance_Validation'
        }
        
        for suite_name, component_name in component_mapping.items():
            if suite_name in self.test_results:
                result = self.test_results[suite_name]
                
                component_status[component_name] = {
                    'status': 'OPERATIONAL' if result['success'] else 'NEEDS_ATTENTION',
                    'execution_time': result['execution_time'],
                    'module': result['module'],
                    'description': result['description']
                }
                
                if not result['success']:
                    component_status[component_name]['issues'] = result.get('error', 'Unknown error')
        
        return component_status
    
    def _assess_overall_quality(self):
        """Assess overall Phase 3 quality"""
        suite_success_rate = self.execution_summary['suite_statistics']['suite_success_rate']
        individual_success_rate = self.execution_summary['test_statistics']['individual_success_rate']
        
        # Quality thresholds
        if suite_success_rate >= 0.95 and individual_success_rate >= 0.90:
            quality = "EXCELLENT"
            quality_score = 95
        elif suite_success_rate >= 0.8 and individual_success_rate >= 0.80:
            quality = "GOOD"
            quality_score = 80
        elif suite_success_rate >= 0.6 and individual_success_rate >= 0.70:
            quality = "ACCEPTABLE"
            quality_score = 70
        else:
            quality = "NEEDS_IMPROVEMENT"
            quality_score = 50
        
        return {
            'overall_quality': quality,
            'quality_score': quality_score,
            'suite_success_rate': suite_success_rate,
            'individual_success_rate': individual_success_rate,
            'ready_for_production': quality in ['EXCELLENT', 'GOOD']
        }
    
    def _generate_recommendations(self):
        """Generate recommendations based on test results"""
        recommendations = []
        
        quality = self.execution_summary['quality_assessment']['overall_quality']
        
        if quality == 'EXCELLENT':
            recommendations.extend([
                "✅ Phase 3 extensions are fully operational and ready for production deployment",
                "🚀 All components demonstrate excellent performance and reliability",
                "📊 Consider enabling all Phase 3 features in production environment",
                "🎯 Monitor performance metrics in production for continued optimization"
            ])
        elif quality == 'GOOD':
            recommendations.extend([
                "✅ Phase 3 extensions are operational with minor issues",
                "🔧 Review failed tests and address any identified issues",
                "📈 Consider gradual rollout of Phase 3 features",
                "🎯 Implement additional monitoring for components with lower success rates"
            ])
        elif quality == 'ACCEPTABLE':
            recommendations.extend([
                "⚠️ Phase 3 extensions have some significant issues",
                "🔧 Address failed test suites before production deployment",
                "📋 Focus on improving reliability of core components",
                "🎯 Implement comprehensive error handling and fallback mechanisms"
            ])
        else:
            recommendations.extend([
                "❌ Phase 3 extensions require significant work before deployment",
                "🔧 Address all failed test suites and underlying issues",
                "📋 Review architecture and implementation for fundamental problems",
                "⏳ Delay production deployment until quality improves"
            ])
        
        # Component-specific recommendations
        failed_suites = [name for name, result in self.test_results.items() 
                        if not result['success']]
        
        if failed_suites:
            recommendations.append(f"🎯 Priority focus areas: {', '.join(failed_suites)}")
        
        return recommendations
    
    def _log_final_results(self):
        """Log comprehensive final results"""
        logger.info("\n" + "=" * 80)
        logger.info("🎯 PHASE 3 COMPREHENSIVE TEST EXECUTION COMPLETE")
        logger.info("=" * 80)
        
        # Execution summary
        exec_summary = self.execution_summary
        logger.info(f"\n⏱️ EXECUTION SUMMARY:")
        logger.info(f"   Total execution time: {exec_summary['execution_metadata']['total_execution_time']:.2f}s")
        logger.info(f"   Test suites executed: {exec_summary['suite_statistics']['total_suites']}")
        logger.info(f"   Individual tests run: {exec_summary['test_statistics']['total_individual_tests']}")
        
        # Suite results
        logger.info(f"\n📊 SUITE RESULTS:")
        logger.info(f"   Successful suites: {exec_summary['suite_statistics']['successful_suites']}")
        logger.info(f"   Failed suites: {exec_summary['suite_statistics']['failed_suites']}")
        logger.info(f"   Suite success rate: {exec_summary['suite_statistics']['suite_success_rate']:.1%}")
        
        # Individual test results  
        logger.info(f"\n🧪 INDIVIDUAL TEST RESULTS:")
        logger.info(f"   Successful tests: {exec_summary['test_statistics']['successful_individual_tests']}")
        logger.info(f"   Failed tests: {exec_summary['test_statistics']['failed_individual_tests']}")
        logger.info(f"   Individual success rate: {exec_summary['test_statistics']['individual_success_rate']:.1%}")
        
        # Quality assessment
        quality_assessment = exec_summary['quality_assessment']
        logger.info(f"\n🎯 QUALITY ASSESSMENT:")
        logger.info(f"   Overall quality: {quality_assessment['overall_quality']}")
        logger.info(f"   Quality score: {quality_assessment['quality_score']}/100")
        logger.info(f"   Production ready: {'✅ YES' if quality_assessment['ready_for_production'] else '❌ NO'}")
        
        # Component status
        logger.info(f"\n🧩 COMPONENT STATUS:")
        for component, status in exec_summary['component_status'].items():
            status_icon = "✅" if status['status'] == 'OPERATIONAL' else "⚠️"
            logger.info(f"   {status_icon} {component}: {status['status']}")
        
        # Recommendations
        logger.info(f"\n💡 RECOMMENDATIONS:")
        for recommendation in exec_summary['recommendations']:
            logger.info(f"   {recommendation}")
        
        # Final verdict
        overall_success = quality_assessment['ready_for_production']
        if overall_success:
            logger.info(f"\n🎉 PHASE 3 EXTENSIONS: READY FOR DEPLOYMENT!")
            logger.info("✅ All components meet production quality standards")
        else:
            logger.info(f"\n⚠️ PHASE 3 EXTENSIONS: REQUIRE ATTENTION")
            logger.info("🔧 Address identified issues before production deployment")
        
        logger.info("=" * 80)
    
    def _save_comprehensive_report(self):
        """Save comprehensive test report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create comprehensive report
        comprehensive_report = {
            'test_execution_summary': self.execution_summary,
            'detailed_test_results': self.test_results,
            'metadata': {
                'report_generated': datetime.now().isoformat(),
                'phase': 'Phase 3 Extensions Testing',
                'components_tested': ['P3-005', 'P3-006', 'P3-007', 'Integration', 'Performance'],
                'test_environment': 'Development'
            }
        }
        
        # Save main report
        main_report_file = f'phase3_comprehensive_test_report_{timestamp}.json'
        with open(main_report_file, 'w') as f:
            json.dump(comprehensive_report, f, indent=2, default=str)
        
        # Save execution summary
        summary_file = f'phase3_execution_summary_{timestamp}.json'
        with open(summary_file, 'w') as f:
            json.dump(self.execution_summary, f, indent=2, default=str)
        
        logger.info(f"\n💾 REPORTS SAVED:")
        logger.info(f"   📋 Comprehensive report: {main_report_file}")
        logger.info(f"   📊 Execution summary: {summary_file}")
        
        return main_report_file, summary_file

async def main():
    """Main execution function"""
    try:
        orchestrator = Phase3TestOrchestrator()
        success, summary = await orchestrator.run_all_tests()
        
        logger.info(f"\n🏁 Test execution {'COMPLETED SUCCESSFULLY' if success else 'COMPLETED WITH ISSUES'}")
        
        return 0 if success else 1
        
    except Exception as e:
        logger.error(f"❌ Test orchestrator failed: {e}")
        return 1

if __name__ == "__main__":
    # Create tests directory if it doesn't exist
    os.makedirs('tests', exist_ok=True)
    
    # Run the comprehensive test suite
    exit_code = asyncio.run(main())
    
    logger.info(f"\nTest orchestrator exit code: {exit_code}")
    sys.exit(exit_code)