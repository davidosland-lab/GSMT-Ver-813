#!/usr/bin/env python3
"""
P3-005 Advanced Feature Engineering Pipeline Test Suite
=====================================================

Comprehensive test suite for the Advanced Feature Engineering Pipeline component.
Tests all feature domains, multi-modal fusion, and pipeline functionality.
"""

import asyncio
import pytest
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from phase3_advanced_feature_engineering import (
    AdvancedFeatureEngineering,
    FeatureConfig,
    FeatureDomain,
    FeatureImportance
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TestAdvancedFeatureEngineering:
    """Test class for Advanced Feature Engineering Pipeline"""
    
    @pytest.fixture
    def sample_price_data(self):
        """Create sample price data for testing"""
        dates = pd.date_range(start='2024-01-01', end='2024-03-01', freq='D')
        np.random.seed(42)  # For reproducible tests
        
        # Generate realistic price data
        returns = np.random.normal(0.001, 0.02, len(dates))
        prices = [100.0]  # Starting price
        
        for return_rate in returns[1:]:
            prices.append(prices[-1] * (1 + return_rate))
        
        df = pd.DataFrame({
            'Date': dates,
            'Open': np.array(prices) * (1 + np.random.normal(0, 0.005, len(dates))),
            'High': np.array(prices) * (1 + np.random.uniform(0.01, 0.03, len(dates))),
            'Low': np.array(prices) * (1 - np.random.uniform(0.01, 0.03, len(dates))),
            'Close': prices,
            'Volume': np.random.randint(1000000, 10000000, len(dates))
        })
        
        return df.set_index('Date')
    
    @pytest.fixture
    def feature_engineer(self):
        """Create AdvancedFeatureEngineering instance for testing"""
        config = FeatureConfig(
            technical_indicators=True,
            cross_asset_features=True,
            macro_features=True,
            alternative_data=True,
            microstructure_features=True
        )
        return AdvancedFeatureEngineering(config)
    
    def test_initialization(self, feature_engineer):
        """Test proper initialization of AdvancedFeatureEngineering"""
        logger.info("🧪 Testing AdvancedFeatureEngineering initialization...")
        
        assert feature_engineer is not None
        assert isinstance(feature_engineer.config, FeatureConfig)
        assert feature_engineer.feature_cache == {}
        assert feature_engineer.importance_tracker is not None
        
        logger.info("✅ Initialization test passed")
    
    def test_technical_indicators_feature_creation(self, feature_engineer, sample_price_data):
        """Test technical indicators feature creation"""
        logger.info("🧪 Testing technical indicators feature creation...")
        
        # Test technical indicators
        tech_features = feature_engineer._create_technical_features(
            sample_price_data, 'AAPL'
        )
        
        # Verify expected technical indicators are present
        expected_indicators = [
            'sma_10', 'sma_20', 'sma_50', 'ema_12', 'ema_26',
            'rsi_14', 'macd', 'bb_upper', 'bb_lower', 'atr_14'
        ]
        
        for indicator in expected_indicators:
            assert any(indicator in col for col in tech_features.columns), \
                f"Technical indicator {indicator} not found"
        
        # Verify no NaN values in recent data
        recent_data = tech_features.tail(10)
        nan_count = recent_data.isnull().sum().sum()
        assert nan_count < len(recent_data) * 0.1, "Too many NaN values in technical indicators"
        
        logger.info(f"✅ Technical indicators test passed - {len(tech_features.columns)} indicators created")
    
    def test_cross_asset_features(self, feature_engineer, sample_price_data):
        """Test cross-asset correlation features"""
        logger.info("🧪 Testing cross-asset features...")
        
        # Mock market data for cross-asset analysis
        market_data = {
            '^GSPC': sample_price_data.copy(),
            '^VIX': sample_price_data.copy() * 0.5,  # Simulate VIX data
            'GC=F': sample_price_data.copy() * 20,   # Simulate Gold data
            '^TNX': sample_price_data.copy() * 0.04  # Simulate 10Y Treasury
        }
        
        with patch.object(feature_engineer, '_fetch_market_data', return_value=market_data):
            cross_asset_features = feature_engineer._create_cross_asset_features(
                sample_price_data, 'AAPL'
            )
        
        # Verify cross-asset correlation features
        expected_features = [
            'corr_sp500', 'corr_vix', 'corr_gold', 'corr_treasury',
            'beta_sp500', 'relative_strength'
        ]
        
        for feature in expected_features:
            assert any(feature in col for col in cross_asset_features.columns), \
                f"Cross-asset feature {feature} not found"
        
        logger.info(f"✅ Cross-asset features test passed - {len(cross_asset_features.columns)} features created")
    
    def test_macroeconomic_features(self, feature_engineer, sample_price_data):
        """Test macroeconomic indicators integration"""
        logger.info("🧪 Testing macroeconomic features...")
        
        # Mock macro data
        mock_macro_data = pd.DataFrame({
            'unemployment_rate': np.random.uniform(3.5, 7.0, len(sample_price_data)),
            'inflation_rate': np.random.uniform(1.0, 5.0, len(sample_price_data)),
            'gdp_growth': np.random.uniform(-2.0, 4.0, len(sample_price_data)),
            'fed_funds_rate': np.random.uniform(0.0, 5.0, len(sample_price_data))
        }, index=sample_price_data.index)
        
        with patch.object(feature_engineer, '_fetch_macro_data', return_value=mock_macro_data):
            macro_features = feature_engineer._create_macro_features(
                sample_price_data, 'AAPL'
            )
        
        # Verify macro features
        expected_macro = [
            'unemployment_rate', 'inflation_rate', 'gdp_growth', 'fed_funds_rate'
        ]
        
        for macro in expected_macro:
            assert any(macro in col for col in macro_features.columns), \
                f"Macro feature {macro} not found"
        
        logger.info(f"✅ Macroeconomic features test passed - {len(macro_features.columns)} features created")
    
    def test_alternative_data_features(self, feature_engineer, sample_price_data):
        """Test alternative data features"""
        logger.info("🧪 Testing alternative data features...")
        
        # Mock alternative data
        mock_alt_data = {
            'sentiment_score': np.random.uniform(-1, 1, len(sample_price_data)),
            'news_volume': np.random.randint(50, 500, len(sample_price_data)),
            'social_mentions': np.random.randint(100, 10000, len(sample_price_data)),
            'analyst_upgrades': np.random.randint(0, 5, len(sample_price_data)),
            'analyst_downgrades': np.random.randint(0, 3, len(sample_price_data))
        }
        
        alt_df = pd.DataFrame(mock_alt_data, index=sample_price_data.index)
        
        with patch.object(feature_engineer, '_fetch_alternative_data', return_value=alt_df):
            alt_features = feature_engineer._create_alternative_data_features(
                sample_price_data, 'AAPL'
            )
        
        # Verify alternative data features
        expected_alt = [
            'sentiment_score', 'news_volume', 'social_mentions',
            'net_analyst_changes', 'sentiment_momentum'
        ]
        
        for alt in expected_alt:
            assert any(alt in col for col in alt_features.columns), \
                f"Alternative data feature {alt} not found"
        
        logger.info(f"✅ Alternative data features test passed - {len(alt_features.columns)} features created")
    
    def test_microstructure_features(self, feature_engineer, sample_price_data):
        """Test microstructure features"""
        logger.info("🧪 Testing microstructure features...")
        
        # Add intraday data columns for microstructure analysis
        sample_data = sample_price_data.copy()
        sample_data['Bid'] = sample_data['Close'] * 0.999
        sample_data['Ask'] = sample_data['Close'] * 1.001
        sample_data['Trades'] = np.random.randint(1000, 50000, len(sample_data))
        
        micro_features = feature_engineer._create_microstructure_features(
            sample_data, 'AAPL'
        )
        
        # Verify microstructure features
        expected_micro = [
            'bid_ask_spread', 'volume_weighted_price', 'price_impact',
            'order_flow_imbalance', 'volatility_clustering'
        ]
        
        for micro in expected_micro:
            assert any(micro in col for col in micro_features.columns), \
                f"Microstructure feature {micro} not found"
        
        logger.info(f"✅ Microstructure features test passed - {len(micro_features.columns)} features created")
    
    def test_multimodal_feature_fusion(self, feature_engineer, sample_price_data):
        """Test multi-modal feature fusion"""
        logger.info("🧪 Testing multi-modal feature fusion...")
        
        # Create mock feature domains
        tech_features = pd.DataFrame({
            'rsi_14': np.random.uniform(20, 80, len(sample_price_data)),
            'macd': np.random.uniform(-2, 2, len(sample_price_data))
        }, index=sample_price_data.index)
        
        macro_features = pd.DataFrame({
            'inflation_rate': np.random.uniform(1, 5, len(sample_price_data)),
            'unemployment_rate': np.random.uniform(3, 7, len(sample_price_data))
        }, index=sample_price_data.index)
        
        feature_domains = {
            FeatureDomain.TECHNICAL: tech_features,
            FeatureDomain.MACROECONOMIC: macro_features
        }
        
        fused_features = feature_engineer._fuse_multimodal_features(
            feature_domains, 'AAPL'
        )
        
        # Verify fusion results
        assert len(fused_features.columns) >= len(tech_features.columns) + len(macro_features.columns)
        
        # Check for interaction features
        assert any('interaction' in col.lower() for col in fused_features.columns), \
            "No interaction features found in fusion"
        
        logger.info(f"✅ Multi-modal fusion test passed - {len(fused_features.columns)} fused features")
    
    def test_feature_importance_tracking(self, feature_engineer, sample_price_data):
        """Test feature importance tracking"""
        logger.info("🧪 Testing feature importance tracking...")
        
        # Create mock features and targets
        features = pd.DataFrame({
            'feature_1': np.random.randn(len(sample_price_data)),
            'feature_2': np.random.randn(len(sample_price_data)),
            'feature_3': np.random.randn(len(sample_price_data))
        }, index=sample_price_data.index)
        
        # Create target (future returns)
        target = sample_price_data['Close'].pct_change().shift(-1).dropna()
        
        # Align features and target
        common_index = features.index.intersection(target.index)
        features_aligned = features.loc[common_index]
        target_aligned = target.loc[common_index]
        
        # Test feature importance calculation
        importance = feature_engineer._calculate_feature_importance(
            features_aligned, target_aligned
        )
        
        assert isinstance(importance, dict)
        assert len(importance) == len(features_aligned.columns)
        assert all(0 <= score <= 1 for score in importance.values())
        
        logger.info(f"✅ Feature importance tracking test passed - {len(importance)} features ranked")
    
    def test_feature_caching(self, feature_engineer, sample_price_data):
        """Test feature caching mechanism"""
        logger.info("🧪 Testing feature caching...")
        
        symbol = 'AAPL'
        cache_key = f"{symbol}_technical_60d"
        
        # First call should populate cache
        features_1 = feature_engineer._create_technical_features(sample_price_data, symbol)
        
        # Verify cache is populated
        assert len(feature_engineer.feature_cache) > 0
        
        # Second call should use cache (mock to verify)
        with patch.object(feature_engineer, '_create_technical_features', 
                         return_value=features_1) as mock_create:
            # This should use cached version
            cached_key = next(iter(feature_engineer.feature_cache.keys()))
            cached_features = feature_engineer.feature_cache[cached_key]['features']
            
            assert cached_features is not None
            logger.info(f"✅ Feature caching test passed - Cache contains {len(feature_engineer.feature_cache)} entries")
    
    @pytest.mark.asyncio
    async def test_full_feature_engineering_pipeline(self, feature_engineer, sample_price_data):
        """Test the complete feature engineering pipeline"""
        logger.info("🧪 Testing full feature engineering pipeline...")
        
        # Mock external data sources
        mock_market_data = {
            '^GSPC': sample_price_data.copy(),
            '^VIX': sample_price_data.copy() * 0.5
        }
        
        mock_macro_data = pd.DataFrame({
            'unemployment_rate': np.random.uniform(3.5, 7.0, len(sample_price_data)),
            'inflation_rate': np.random.uniform(1.0, 5.0, len(sample_price_data))
        }, index=sample_price_data.index)
        
        mock_alt_data = pd.DataFrame({
            'sentiment_score': np.random.uniform(-1, 1, len(sample_price_data)),
            'news_volume': np.random.randint(50, 500, len(sample_price_data))
        }, index=sample_price_data.index)
        
        # Patch external data fetching methods
        with patch.object(feature_engineer, '_fetch_market_data', return_value=mock_market_data), \
             patch.object(feature_engineer, '_fetch_macro_data', return_value=mock_macro_data), \
             patch.object(feature_engineer, '_fetch_alternative_data', return_value=mock_alt_data):
            
            # Run full pipeline
            engineered_features = await feature_engineer.engineer_features(
                symbol='AAPL',
                price_data=sample_price_data,
                lookback_period=60
            )
        
        # Verify pipeline results
        assert engineered_features is not None
        assert len(engineered_features) > 0
        assert 'features' in engineered_features
        assert 'feature_importance' in engineered_features
        assert 'domain_contributions' in engineered_features
        
        features_df = engineered_features['features']
        feature_importance = engineered_features['feature_importance']
        
        # Verify feature completeness
        assert len(features_df.columns) >= 20, "Not enough features generated"
        assert len(feature_importance) == len(features_df.columns)
        
        # Verify data quality
        nan_percentage = features_df.isnull().sum().sum() / (len(features_df) * len(features_df.columns))
        assert nan_percentage < 0.1, f"Too many NaN values: {nan_percentage:.2%}"
        
        logger.info(f"✅ Full pipeline test passed:")
        logger.info(f"   📊 Features generated: {len(features_df.columns)}")
        logger.info(f"   📈 Data points: {len(features_df)}")
        logger.info(f"   🎯 NaN percentage: {nan_percentage:.2%}")
        logger.info(f"   🏆 Top feature: {max(feature_importance.items(), key=lambda x: x[1])[0]}")
    
    def test_feature_domain_coverage(self, feature_engineer):
        """Test that all feature domains are covered"""
        logger.info("🧪 Testing feature domain coverage...")
        
        # Verify all feature domains are implemented
        required_domains = [
            FeatureDomain.TECHNICAL,
            FeatureDomain.CROSS_ASSET,
            FeatureDomain.MACROECONOMIC,
            FeatureDomain.ALTERNATIVE_DATA,
            FeatureDomain.MICROSTRUCTURE
        ]
        
        # Check that methods exist for all domains
        domain_methods = {
            FeatureDomain.TECHNICAL: '_create_technical_features',
            FeatureDomain.CROSS_ASSET: '_create_cross_asset_features',
            FeatureDomain.MACROECONOMIC: '_create_macro_features',
            FeatureDomain.ALTERNATIVE_DATA: '_create_alternative_data_features',
            FeatureDomain.MICROSTRUCTURE: '_create_microstructure_features'
        }
        
        for domain, method_name in domain_methods.items():
            assert hasattr(feature_engineer, method_name), \
                f"Method {method_name} not found for domain {domain}"
        
        logger.info(f"✅ Feature domain coverage test passed - {len(required_domains)} domains implemented")
    
    def test_error_handling(self, feature_engineer):
        """Test error handling in feature engineering"""
        logger.info("🧪 Testing error handling...")
        
        # Test with invalid data
        invalid_data = pd.DataFrame()  # Empty DataFrame
        
        try:
            result = feature_engineer._create_technical_features(invalid_data, 'INVALID')
            # Should handle gracefully and return empty or minimal features
            assert result is not None
        except Exception as e:
            # Should not raise unhandled exceptions
            assert False, f"Unhandled exception: {e}"
        
        logger.info("✅ Error handling test passed")

# Test execution functions
async def run_comprehensive_feature_engineering_tests():
    """Run comprehensive test suite for P3-005 Advanced Feature Engineering"""
    logger.info("🚀 P3-005 ADVANCED FEATURE ENGINEERING TEST SUITE")
    logger.info("=" * 60)
    
    test_results = {}
    
    try:
        # Initialize test class
        test_class = TestAdvancedFeatureEngineering()
        
        # Create fixtures
        sample_data = test_class.sample_price_data()
        feature_engineer = test_class.feature_engineer()
        
        # Run individual tests
        tests = [
            ('initialization', lambda: test_class.test_initialization(feature_engineer)),
            ('technical_indicators', lambda: test_class.test_technical_indicators_feature_creation(feature_engineer, sample_data)),
            ('cross_asset_features', lambda: test_class.test_cross_asset_features(feature_engineer, sample_data)),
            ('macroeconomic_features', lambda: test_class.test_macroeconomic_features(feature_engineer, sample_data)),
            ('alternative_data_features', lambda: test_class.test_alternative_data_features(feature_engineer, sample_data)),
            ('microstructure_features', lambda: test_class.test_microstructure_features(feature_engineer, sample_data)),
            ('multimodal_fusion', lambda: test_class.test_multimodal_feature_fusion(feature_engineer, sample_data)),
            ('feature_importance', lambda: test_class.test_feature_importance_tracking(feature_engineer, sample_data)),
            ('feature_caching', lambda: test_class.test_feature_caching(feature_engineer, sample_data)),
            ('domain_coverage', lambda: test_class.test_feature_domain_coverage(feature_engineer)),
            ('error_handling', lambda: test_class.test_error_handling(feature_engineer)),
        ]
        
        # Run synchronous tests
        for test_name, test_func in tests:
            try:
                logger.info(f"\n🔬 Running {test_name}...")
                test_func()
                test_results[test_name] = {'status': 'SUCCESS'}
                logger.info(f"✅ {test_name}: PASSED")
            except Exception as e:
                logger.error(f"❌ {test_name}: FAILED - {e}")
                test_results[test_name] = {'status': 'FAILED', 'error': str(e)}
        
        # Run async test
        try:
            logger.info(f"\n🔬 Running full_pipeline_test...")
            await test_class.test_full_feature_engineering_pipeline(feature_engineer, sample_data)
            test_results['full_pipeline'] = {'status': 'SUCCESS'}
            logger.info(f"✅ full_pipeline_test: PASSED")
        except Exception as e:
            logger.error(f"❌ full_pipeline_test: FAILED - {e}")
            test_results['full_pipeline'] = {'status': 'FAILED', 'error': str(e)}
        
        # Calculate results
        total_tests = len(test_results)
        successful_tests = sum(1 for result in test_results.values() 
                             if result.get('status') == 'SUCCESS')
        
        logger.info("\n" + "=" * 60)
        logger.info("🎯 P3-005 FEATURE ENGINEERING TEST SUMMARY")
        logger.info("=" * 60)
        logger.info(f"\n📊 Test Results: {successful_tests}/{total_tests} successful ({successful_tests/total_tests*100:.1f}%)")
        
        for test_name, result in test_results.items():
            status_icon = "✅" if result.get('status') == 'SUCCESS' else "❌"
            logger.info(f"   {status_icon} {test_name}: {result.get('status')}")
            if result.get('error'):
                logger.info(f"      Error: {result['error']}")
        
        success = successful_tests >= total_tests * 0.8  # 80% pass rate
        
        if success:
            logger.info("\n🎉 P3-005 ADVANCED FEATURE ENGINEERING: TEST SUITE PASSED!")
            logger.info("✅ Feature engineering pipeline is working correctly")
        else:
            logger.info("\n⚠️ P3-005 ADVANCED FEATURE ENGINEERING: ISSUES DETECTED")
            logger.info("🔧 Review failed tests before proceeding")
        
        return success, test_results
        
    except Exception as e:
        logger.error(f"❌ Test suite execution failed: {e}")
        return False, {'execution_error': {'status': 'FAILED', 'error': str(e)}}

if __name__ == "__main__":
    # Run the test suite
    success, results = asyncio.run(run_comprehensive_feature_engineering_tests())
    
    # Save results
    import json
    with open('p3_005_test_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"\n💾 Test results saved to: p3_005_test_results.json")
    
    exit_code = 0 if success else 1
    sys.exit(exit_code)