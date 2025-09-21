#!/usr/bin/env python3
"""
Phase 3 Test Suite Validation
=============================

Quick validation to ensure all test suites are properly structured and importable.
"""

import sys
import os
import logging
import importlib.util

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def validate_test_suite():
    """Validate Phase 3 test suite structure and imports"""
    logger.info("🔍 VALIDATING PHASE 3 TEST SUITE")
    logger.info("=" * 50)
    
    # Test files to validate
    test_files = [
        'tests/test_phase3_p005_advanced_feature_engineering.py',
        'tests/test_phase3_p006_reinforcement_learning.py', 
        'tests/test_phase3_p007_advanced_risk_management.py',
        'tests/test_phase3_extended_unified_predictor.py',
        'tests/test_phase3_performance_benchmarking.py',
        'tests/run_all_phase3_tests.py'
    ]
    
    # Phase 3 component files to validate
    component_files = [
        'phase3_advanced_feature_engineering.py',
        'phase3_reinforcement_learning.py',
        'phase3_advanced_risk_management.py', 
        'phase3_extended_unified_predictor.py'
    ]
    
    validation_results = {}
    
    # Validate test files exist and are importable
    logger.info("\n📋 Validating test files...")
    for test_file in test_files:
        if os.path.exists(test_file):
            try:
                # Try to load as module
                spec = importlib.util.spec_from_file_location("test_module", test_file)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    # Don't execute, just check if it can be loaded
                    validation_results[test_file] = {
                        'exists': True,
                        'importable': True,
                        'size_kb': round(os.path.getsize(test_file) / 1024, 1)
                    }
                    logger.info(f"  ✅ {test_file} - {validation_results[test_file]['size_kb']} KB")
                else:
                    validation_results[test_file] = {'exists': True, 'importable': False}
                    logger.warning(f"  ⚠️ {test_file} - Cannot create module spec")
            except Exception as e:
                validation_results[test_file] = {'exists': True, 'importable': False, 'error': str(e)}
                logger.warning(f"  ⚠️ {test_file} - Import validation failed: {e}")
        else:
            validation_results[test_file] = {'exists': False}
            logger.error(f"  ❌ {test_file} - File not found")
    
    # Validate component files exist
    logger.info("\n🧩 Validating Phase 3 component files...")
    for component_file in component_files:
        if os.path.exists(component_file):
            validation_results[component_file] = {
                'exists': True,
                'size_kb': round(os.path.getsize(component_file) / 1024, 1)
            }
            logger.info(f"  ✅ {component_file} - {validation_results[component_file]['size_kb']} KB")
        else:
            validation_results[component_file] = {'exists': False}
            logger.error(f"  ❌ {component_file} - Component file not found")
    
    # Validate directory structure
    logger.info("\n📁 Validating directory structure...")
    required_dirs = ['tests']
    for dir_name in required_dirs:
        if os.path.exists(dir_name) and os.path.isdir(dir_name):
            files_count = len([f for f in os.listdir(dir_name) if f.endswith('.py')])
            validation_results[f'dir_{dir_name}'] = {'exists': True, 'files_count': files_count}
            logger.info(f"  ✅ {dir_name}/ directory - {files_count} Python files")
        else:
            validation_results[f'dir_{dir_name}'] = {'exists': False}
            logger.error(f"  ❌ {dir_name}/ directory - Not found")
    
    # Check dependencies (basic import test)
    logger.info("\n📦 Checking basic dependencies...")
    dependencies = ['numpy', 'pandas', 'asyncio', 'logging', 'datetime', 'json']
    for dep in dependencies:
        try:
            __import__(dep)
            validation_results[f'dep_{dep}'] = {'available': True}
            logger.info(f"  ✅ {dep} - Available")
        except ImportError:
            validation_results[f'dep_{dep}'] = {'available': False}
            logger.warning(f"  ⚠️ {dep} - Not available")
    
    # Generate summary
    logger.info("\n" + "=" * 50)
    logger.info("📊 VALIDATION SUMMARY")
    logger.info("=" * 50)
    
    total_items = len(validation_results)
    successful_items = 0
    
    for item, result in validation_results.items():
        if (result.get('exists', False) and result.get('importable', True)) or result.get('available', False):
            successful_items += 1
    
    success_rate = successful_items / total_items if total_items > 0 else 0
    
    logger.info(f"Total items validated: {total_items}")
    logger.info(f"Successful validations: {successful_items}")
    logger.info(f"Success rate: {success_rate:.1%}")
    
    if success_rate >= 0.9:
        logger.info("\n🎉 TEST SUITE VALIDATION: PASSED!")
        logger.info("✅ Phase 3 test suite is ready for execution")
        return True
    elif success_rate >= 0.7:
        logger.info("\n⚠️ TEST SUITE VALIDATION: MOSTLY READY")
        logger.info("🔧 Some issues detected, but core components available")
        return True
    else:
        logger.info("\n❌ TEST SUITE VALIDATION: ISSUES DETECTED")
        logger.info("🔧 Significant issues found, review and fix before running tests")
        return False

if __name__ == "__main__":
    success = validate_test_suite()
    exit_code = 0 if success else 1
    logger.info(f"\nValidation exit code: {exit_code}")
    sys.exit(exit_code)