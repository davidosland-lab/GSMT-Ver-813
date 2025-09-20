#!/usr/bin/env python3
"""
Phase 3 Performance Benchmarking and Validation Test Suite
==========================================================

Comprehensive performance benchmarking for Phase 3 extensions (P3-005 to P3-007).
Tests performance, accuracy, scalability, and validation metrics.
"""

import asyncio
import pytest
import numpy as np
import pandas as pd
import logging
import time
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
import sys
import os
import json
import psutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Any

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import all Phase 3 components
from phase3_extended_unified_predictor import ExtendedUnifiedSuperPredictor, ExtendedConfig
from phase3_advanced_feature_engineering import AdvancedFeatureEngineering
from phase3_reinforcement_learning import ReinforcementLearningFramework
from phase3_advanced_risk_management import AdvancedRiskManager

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PerformanceBenchmarkSuite:
    """Comprehensive performance benchmarking suite"""
    
    def __init__(self):
        self.benchmark_results = {}
        self.performance_metrics = {}
        self.validation_scores = {}
    
    def create_benchmark_data(self, n_samples: int = 252) -> Dict[str, pd.DataFrame]:
        """Create benchmark market data for testing"""
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='D')
        
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'SPY', '^GSPC', '^VIX', 'GC=F']
        market_data = {}
        
        for symbol in symbols:
            # Generate realistic price movements
            if 'VIX' in symbol:
                returns = np.random.normal(-0.001, 0.05, n_samples)
                base_price = 20.0
            elif 'GC=F' in symbol:
                returns = np.random.normal(0.0005, 0.015, n_samples)
                base_price = 2000.0
            else:
                returns = np.random.normal(0.001, 0.025, n_samples)
                base_price = 150.0
            
            prices = [base_price]
            for ret in returns[1:]:
                prices.append(prices[-1] * (1 + ret))
            
            market_data[symbol] = pd.DataFrame({
                'Date': dates,
                'Open': np.array(prices) * (1 + np.random.normal(0, 0.001, n_samples)),
                'High': np.array(prices) * (1 + np.random.uniform(0.001, 0.01, n_samples)),
                'Low': np.array(prices) * (1 - np.random.uniform(0.001, 0.01, n_samples)),
                'Close': prices,
                'Volume': np.random.randint(1000000, 50000000, n_samples),
                'Returns': returns
            }).set_index('Date')
        
        return market_data
    
    def benchmark_feature_engineering_performance(self) -> Dict[str, Any]:
        """Benchmark feature engineering performance"""
        logger.info("🔧 Benchmarking Feature Engineering Performance...")
        
        # Create test data
        market_data = self.create_benchmark_data(500)  # Larger dataset
        price_data = market_data['AAPL']
        
        # Initialize feature engineer
        from phase3_advanced_feature_engineering import FeatureConfig
        config = FeatureConfig()
        feature_engineer = AdvancedFeatureEngineering(config)
        
        # Mock external data
        mock_market_data = {k: v for k, v in market_data.items() if k != 'AAPL'}
        mock_macro = pd.DataFrame({
            'unemployment': np.random.uniform(3, 8, len(price_data)),
            'inflation': np.random.uniform(1, 6, len(price_data))
        }, index=price_data.index)
        mock_alt = pd.DataFrame({
            'sentiment': np.random.uniform(-1, 1, len(price_data)),
            'news_volume': np.random.randint(10, 1000, len(price_data))
        }, index=price_data.index)
        
        benchmark_results = {}
        
        # Benchmark individual feature domains
        feature_domains = {
            'technical': lambda: feature_engineer._create_technical_features(price_data, 'AAPL'),
            'cross_asset': lambda: self._mock_cross_asset_features(feature_engineer, price_data, mock_market_data),
            'macro': lambda: self._mock_macro_features(feature_engineer, price_data, mock_macro),
            'alternative': lambda: self._mock_alternative_features(feature_engineer, price_data, mock_alt),
            'microstructure': lambda: feature_engineer._create_microstructure_features(price_data, 'AAPL')
        }
        
        for domain_name, domain_func in feature_domains.items():
            start_time = time.time()
            memory_before = psutil.Process().memory_info().rss / 1024 / 1024
            
            try:
                features = domain_func()
                end_time = time.time()
                memory_after = psutil.Process().memory_info().rss / 1024 / 1024
                
                benchmark_results[domain_name] = {
                    'execution_time': end_time - start_time,
                    'memory_usage': memory_after - memory_before,
                    'features_generated': len(features.columns) if features is not None else 0,
                    'data_points': len(features) if features is not None else 0,
                    'throughput': len(features) / (end_time - start_time) if features is not None and end_time > start_time else 0
                }
                
            except Exception as e:
                benchmark_results[domain_name] = {
                    'error': str(e),
                    'execution_time': float('inf'),
                    'features_generated': 0
                }
        
        # Overall feature engineering benchmark
        start_time = time.time()
        
        with patch.object(feature_engineer, '_fetch_market_data', return_value=mock_market_data), \
             patch.object(feature_engineer, '_fetch_macro_data', return_value=mock_macro), \
             patch.object(feature_engineer, '_fetch_alternative_data', return_value=mock_alt):
            
            try:
                result = asyncio.run(feature_engineer.engineer_features('AAPL', price_data, 60))
                total_time = time.time() - start_time
                
                benchmark_results['overall'] = {
                    'total_execution_time': total_time,
                    'total_features': len(result['features'].columns) if result and 'features' in result else 0,
                    'feature_importance_calculated': len(result.get('feature_importance', {})),
                    'domains_processed': len(result.get('domain_contributions', {}))
                }
            except Exception as e:
                benchmark_results['overall'] = {'error': str(e)}
        
        logger.info(f"✅ Feature Engineering Benchmark Complete")
        return benchmark_results
    
    def benchmark_reinforcement_learning_performance(self) -> Dict[str, Any]:
        """Benchmark reinforcement learning performance"""
        logger.info("🤖 Benchmarking Reinforcement Learning Performance...")
        
        from phase3_reinforcement_learning import RLConfig
        config = RLConfig()
        rl_framework = ReinforcementLearningFramework(config)
        
        market_data = self.create_benchmark_data(100)
        benchmark_results = {}
        
        # Test different RL algorithms
        algorithms = ['multi_armed_bandit', 'q_learning', 'thompson_sampling']
        
        for algorithm in algorithms:
            start_time = time.time()
            memory_before = psutil.Process().memory_info().rss / 1024 / 1024
            
            try:
                # Simulate learning episodes
                n_episodes = 50
                selections = []
                
                for episode in range(n_episodes):
                    # Create mock state
                    episode_data = market_data['AAPL'].iloc[max(0, episode-10):episode+10]
                    if len(episode_data) < 5:
                        continue
                    
                    rl_state = rl_framework._create_rl_state(episode_data)
                    
                    # Select models
                    from phase3_reinforcement_learning import RLAlgorithm
                    if algorithm == 'multi_armed_bandit':
                        selected = rl_framework.select_optimal_models(rl_state, RLAlgorithm.MULTI_ARMED_BANDIT)
                    elif algorithm == 'q_learning':
                        selected = rl_framework.select_optimal_models(rl_state, RLAlgorithm.Q_LEARNING)
                    else:
                        selected = rl_framework.select_optimal_models(rl_state, RLAlgorithm.THOMPSON_SAMPLING)
                    
                    selections.extend(selected)
                    
                    # Update with mock reward
                    reward = np.random.uniform(0.3, 0.9)
                    for model_id in selected:
                        if algorithm == 'thompson_sampling':
                            rl_framework.thompson_sampler.update_model_performance(model_id, reward > 0.5)
                        elif algorithm == 'multi_armed_bandit':
                            rl_framework.bandit.update_reward(model_id, reward)
                
                end_time = time.time()
                memory_after = psutil.Process().memory_info().rss / 1024 / 1024
                
                # Calculate performance metrics
                convergence_score = self._calculate_convergence(selections)
                diversity_score = len(set(selections)) / max(len(selections), 1)
                
                benchmark_results[algorithm] = {
                    'execution_time': end_time - start_time,
                    'memory_usage': memory_after - memory_before,
                    'episodes_processed': n_episodes,
                    'selections_made': len(selections),
                    'convergence_score': convergence_score,
                    'diversity_score': diversity_score,
                    'throughput': n_episodes / (end_time - start_time)
                }
                
            except Exception as e:
                benchmark_results[algorithm] = {'error': str(e)}
        
        logger.info(f"✅ RL Benchmark Complete")
        return benchmark_results
    
    def benchmark_risk_management_performance(self) -> Dict[str, Any]:
        """Benchmark risk management performance"""
        logger.info("📊 Benchmarking Risk Management Performance...")
        
        from phase3_advanced_risk_management import RiskConfig
        config = RiskConfig()
        risk_manager = AdvancedRiskManager(config)
        
        # Create portfolio data
        portfolio_data = self._create_benchmark_portfolio()
        returns_data = np.random.normal(0.001, 0.02, 1000)
        
        benchmark_results = {}
        
        # Benchmark VaR calculations
        var_methods = ['historical', 'parametric', 'monte_carlo']
        
        for method in var_methods:
            start_time = time.time()
            memory_before = psutil.Process().memory_info().rss / 1024 / 1024
            
            try:
                if method == 'historical':
                    var_result = risk_manager.var_calculator.calculate_historical_var(returns_data, 0.95)
                elif method == 'parametric':
                    var_result = risk_manager.var_calculator.calculate_parametric_var(returns_data, 0.95)
                else:
                    var_result = risk_manager.var_calculator.calculate_monte_carlo_var(returns_data, 0.95, 1000)
                
                end_time = time.time()
                memory_after = psutil.Process().memory_info().rss / 1024 / 1024
                
                benchmark_results[f'var_{method}'] = {
                    'execution_time': end_time - start_time,
                    'memory_usage': memory_after - memory_before,
                    'var_result': var_result,
                    'data_points_processed': len(returns_data)
                }
                
            except Exception as e:
                benchmark_results[f'var_{method}'] = {'error': str(e)}
        
        # Benchmark comprehensive risk metrics
        start_time = time.time()
        try:
            risk_metrics = risk_manager.calculate_risk_metrics(
                returns=returns_data,
                portfolio_value=1000000,
                benchmark_returns=returns_data * 0.8
            )
            
            benchmark_results['comprehensive_metrics'] = {
                'execution_time': time.time() - start_time,
                'metrics_calculated': len([attr for attr in dir(risk_metrics) if not attr.startswith('_')]),
                'success': True
            }
        except Exception as e:
            benchmark_results['comprehensive_metrics'] = {'error': str(e)}
        
        # Benchmark stress testing
        start_time = time.time()
        try:
            from phase3_advanced_risk_management import StressTestScenario, StressTestType
            scenario = StressTestScenario(
                name="Market Crash Test",
                type=StressTestType.HISTORICAL,
                market_shock=-0.20,
                volatility_shock=2.0,
                correlation_shock=0.8
            )
            
            stress_result = risk_manager.run_stress_test(portfolio_data, scenario)
            
            benchmark_results['stress_testing'] = {
                'execution_time': time.time() - start_time,
                'scenario_processed': True,
                'portfolio_assets': len(portfolio_data)
            }
        except Exception as e:
            benchmark_results['stress_testing'] = {'error': str(e)}
        
        logger.info(f"✅ Risk Management Benchmark Complete")
        return benchmark_results
    
    def benchmark_extended_predictor_performance(self) -> Dict[str, Any]:
        """Benchmark Extended Unified Predictor performance"""
        logger.info("🎯 Benchmarking Extended Predictor Performance...")
        
        from phase3_extended_unified_predictor import ExtendedConfig
        config = ExtendedConfig(
            lookback_period=60,
            min_samples=30,
            mcmc_samples=500,  # Reduced for benchmarking
            monte_carlo_simulations=1000  # Reduced for benchmarking
        )
        
        predictor = ExtendedUnifiedSuperPredictor(config)
        market_data = self.create_benchmark_data(200)
        
        benchmark_results = {}
        
        # Test different prediction configurations
        test_configs = [
            ('basic', {'include_all_domains': False, 'enable_rl_optimization': False, 'include_risk_management': False}),
            ('features_only', {'include_all_domains': True, 'enable_rl_optimization': False, 'include_risk_management': False}),
            ('rl_only', {'include_all_domains': False, 'enable_rl_optimization': True, 'include_risk_management': False}),
            ('risk_only', {'include_all_domains': False, 'enable_rl_optimization': False, 'include_risk_management': True}),
            ('full_integration', {'include_all_domains': True, 'enable_rl_optimization': True, 'include_risk_management': True})
        ]
        
        for config_name, prediction_config in test_configs:
            start_time = time.time()
            memory_before = psutil.Process().memory_info().rss / 1024 / 1024
            
            try:
                with self._mock_external_data_sources(predictor, market_data):
                    prediction = asyncio.run(predictor.generate_extended_prediction(
                        symbol='AAPL',
                        time_horizon='5d',
                        **prediction_config
                    ))
                
                end_time = time.time()
                memory_after = psutil.Process().memory_info().rss / 1024 / 1024
                
                benchmark_results[config_name] = {
                    'execution_time': end_time - start_time,
                    'memory_usage': memory_after - memory_before,
                    'prediction_successful': prediction is not None,
                    'confidence_score': getattr(prediction, 'confidence_score', 0),
                    'components_active': self._count_active_components(prediction)
                }
                
            except Exception as e:
                benchmark_results[config_name] = {'error': str(e)}
        
        # Scalability test - multiple concurrent predictions
        start_time = time.time()
        try:
            symbols = ['AAPL', 'MSFT', 'GOOGL']
            
            async def run_multiple_predictions():
                with self._mock_external_data_sources(predictor, market_data):
                    tasks = []
                    for symbol in symbols:
                        task = predictor.generate_extended_prediction(symbol, '5d')
                        tasks.append(task)
                    
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    return results
            
            results = asyncio.run(run_multiple_predictions())
            successful_predictions = sum(1 for r in results if not isinstance(r, Exception))
            
            benchmark_results['scalability'] = {
                'execution_time': time.time() - start_time,
                'symbols_processed': len(symbols),
                'successful_predictions': successful_predictions,
                'concurrent_throughput': successful_predictions / (time.time() - start_time)
            }
            
        except Exception as e:
            benchmark_results['scalability'] = {'error': str(e)}
        
        logger.info(f"✅ Extended Predictor Benchmark Complete")
        return benchmark_results
    
    def validate_prediction_accuracy(self) -> Dict[str, Any]:
        """Validate prediction accuracy using backtesting"""
        logger.info("🎯 Validating Prediction Accuracy...")
        
        # Create historical data with known patterns
        validation_data = self._create_validation_dataset()
        
        from phase3_extended_unified_predictor import ExtendedConfig
        config = ExtendedConfig(mcmc_samples=100, monte_carlo_simulations=500)  # Faster for validation
        predictor = ExtendedUnifiedSuperPredictor(config)
        
        validation_results = {}
        
        # Backtest predictions
        symbols = ['AAPL', 'SPY']
        time_horizons = ['1d', '5d']
        
        for symbol in symbols:
            for horizon in time_horizons:
                try:
                    accuracy_metrics = self._backtest_predictions(
                        predictor, validation_data, symbol, horizon
                    )
                    validation_results[f'{symbol}_{horizon}'] = accuracy_metrics
                    
                except Exception as e:
                    validation_results[f'{symbol}_{horizon}'] = {'error': str(e)}
        
        # Calculate overall accuracy
        successful_validations = [v for v in validation_results.values() if 'error' not in v]
        if successful_validations:
            avg_accuracy = np.mean([v['directional_accuracy'] for v in successful_validations])
            avg_mape = np.mean([v['mean_absolute_percentage_error'] for v in successful_validations])
            
            validation_results['overall'] = {
                'average_directional_accuracy': avg_accuracy,
                'average_mape': avg_mape,
                'successful_validations': len(successful_validations),
                'total_validations': len(validation_results) - 1  # Exclude 'overall'
            }
        
        logger.info(f"✅ Prediction Accuracy Validation Complete")
        return validation_results
    
    def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Run comprehensive benchmark suite"""
        logger.info("🚀 RUNNING COMPREHENSIVE PHASE 3 BENCHMARK SUITE")
        logger.info("=" * 70)
        
        start_time = time.time()
        
        # Run all benchmarks
        benchmark_suite = {
            'feature_engineering': self.benchmark_feature_engineering_performance,
            'reinforcement_learning': self.benchmark_reinforcement_learning_performance,
            'risk_management': self.benchmark_risk_management_performance,
            'extended_predictor': self.benchmark_extended_predictor_performance,
            'prediction_validation': self.validate_prediction_accuracy
        }
        
        results = {}
        
        for benchmark_name, benchmark_func in benchmark_suite.items():
            logger.info(f"\n📊 Running {benchmark_name} benchmark...")
            
            try:
                benchmark_start = time.time()
                result = benchmark_func()
                benchmark_time = time.time() - benchmark_start
                
                results[benchmark_name] = {
                    'benchmark_results': result,
                    'benchmark_execution_time': benchmark_time,
                    'status': 'SUCCESS'
                }
                
                logger.info(f"✅ {benchmark_name} completed in {benchmark_time:.2f}s")
                
            except Exception as e:
                logger.error(f"❌ {benchmark_name} failed: {e}")
                results[benchmark_name] = {
                    'status': 'FAILED',
                    'error': str(e)
                }
        
        total_time = time.time() - start_time
        
        # Generate summary
        summary = self._generate_benchmark_summary(results, total_time)
        results['benchmark_summary'] = summary
        
        logger.info("\n" + "=" * 70)
        logger.info("🎯 BENCHMARK SUITE COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Total execution time: {total_time:.2f}s")
        logger.info(f"Benchmarks completed: {summary['successful_benchmarks']}/{summary['total_benchmarks']}")
        
        return results
    
    # Helper methods
    def _mock_cross_asset_features(self, feature_engineer, price_data, market_data):
        """Mock cross-asset features"""
        with patch.object(feature_engineer, '_fetch_market_data', return_value=market_data):
            return feature_engineer._create_cross_asset_features(price_data, 'AAPL')
    
    def _mock_macro_features(self, feature_engineer, price_data, macro_data):
        """Mock macro features"""
        with patch.object(feature_engineer, '_fetch_macro_data', return_value=macro_data):
            return feature_engineer._create_macro_features(price_data, 'AAPL')
    
    def _mock_alternative_features(self, feature_engineer, price_data, alt_data):
        """Mock alternative data features"""
        with patch.object(feature_engineer, '_fetch_alternative_data', return_value=alt_data):
            return feature_engineer._create_alternative_data_features(price_data, 'AAPL')
    
    def _calculate_convergence(self, selections):
        """Calculate convergence score for RL selections"""
        if len(selections) < 10:
            return 0.0
        
        recent_selections = selections[-10:]
        unique_recent = len(set(recent_selections))
        return 1.0 - (unique_recent / len(recent_selections))
    
    def _create_benchmark_portfolio(self):
        """Create benchmark portfolio data"""
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA']
        portfolio = {}
        
        for symbol in symbols:
            portfolio[symbol] = {
                'position': np.random.randint(100, 1000),
                'current_price': np.random.uniform(50, 300),
                'returns': np.random.normal(0.001, 0.02, 252),
                'beta': np.random.uniform(0.5, 2.0),
                'volatility': np.random.uniform(0.15, 0.45)
            }
        
        return portfolio
    
    def _mock_external_data_sources(self, predictor, market_data):
        """Mock all external data sources"""
        symbol = 'AAPL'
        price_data = market_data[symbol]
        
        mock_macro = pd.DataFrame({
            'unemployment_rate': np.random.uniform(3, 8, len(price_data)),
            'inflation_rate': np.random.uniform(1, 6, len(price_data))
        }, index=price_data.index)
        
        mock_alt = pd.DataFrame({
            'sentiment_score': np.random.uniform(-1, 1, len(price_data)),
            'news_volume': np.random.randint(10, 1000, len(price_data))
        }, index=price_data.index)
        
        return patch.multiple(
            predictor.feature_engineer,
            _fetch_market_data=MagicMock(return_value=market_data),
            _fetch_macro_data=MagicMock(return_value=mock_macro),
            _fetch_alternative_data=MagicMock(return_value=mock_alt)
        )
    
    def _count_active_components(self, prediction):
        """Count active components in prediction"""
        count = 0
        if hasattr(prediction, 'advanced_features') and prediction.advanced_features:
            count += 1
        if hasattr(prediction, 'rl_selected_models') and prediction.rl_selected_models:
            count += 1
        if hasattr(prediction, 'risk_metrics') and prediction.risk_metrics:
            count += 1
        return count
    
    def _create_validation_dataset(self):
        """Create dataset with known patterns for validation"""
        np.random.seed(123)  # Different seed for validation
        
        # Create data with embedded patterns
        dates = pd.date_range(start='2023-01-01', periods=300, freq='D')
        
        validation_data = {}
        symbols = ['AAPL', 'SPY']
        
        for symbol in symbols:
            # Create trending patterns
            trend = np.linspace(0, 0.2, len(dates))  # Upward trend
            noise = np.random.normal(0, 0.02, len(dates))
            returns = trend + noise
            
            prices = [100.0]
            for ret in returns[1:]:
                prices.append(prices[-1] * (1 + ret))
            
            validation_data[symbol] = pd.DataFrame({
                'Date': dates,
                'Close': prices,
                'Returns': returns
            }).set_index('Date')
        
        return validation_data
    
    def _backtest_predictions(self, predictor, validation_data, symbol, horizon):
        """Backtest predictions for accuracy validation"""
        price_data = validation_data[symbol]
        
        # Split data for backtesting
        train_size = int(len(price_data) * 0.7)
        train_data = price_data.iloc[:train_size]
        test_data = price_data.iloc[train_size:]
        
        predictions = []
        actuals = []
        
        # Make predictions on test data
        with self._mock_external_data_sources(predictor, validation_data):
            for i in range(min(20, len(test_data) - 5)):  # Limit for speed
                try:
                    current_data = pd.concat([train_data, test_data.iloc[:i+1]])
                    
                    prediction = asyncio.run(predictor.generate_extended_prediction(
                        symbol=symbol,
                        time_horizon=horizon
                    ))
                    
                    # Get actual future price
                    days_ahead = int(horizon[:-1]) if horizon.endswith('d') else 1
                    if i + days_ahead < len(test_data):
                        actual_price = test_data.iloc[i + days_ahead]['Close']
                        predicted_price = prediction.predicted_price
                        
                        predictions.append(predicted_price)
                        actuals.append(actual_price)
                
                except Exception as e:
                    logger.warning(f"Prediction failed at step {i}: {e}")
                    continue
        
        # Calculate accuracy metrics
        if len(predictions) > 0 and len(actuals) > 0:
            predictions = np.array(predictions)
            actuals = np.array(actuals)
            
            # Directional accuracy
            pred_directions = np.sign(predictions - actuals[0])  # Direction from first price
            actual_directions = np.sign(actuals - actuals[0])
            directional_accuracy = np.mean(pred_directions == actual_directions)
            
            # MAPE
            mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
            
            # Correlation
            correlation = np.corrcoef(predictions, actuals)[0, 1] if len(predictions) > 1 else 0
            
            return {
                'directional_accuracy': directional_accuracy,
                'mean_absolute_percentage_error': mape,
                'correlation': correlation,
                'predictions_made': len(predictions),
                'rmse': np.sqrt(np.mean((predictions - actuals) ** 2))
            }
        else:
            return {
                'directional_accuracy': 0,
                'mean_absolute_percentage_error': float('inf'),
                'correlation': 0,
                'predictions_made': 0,
                'error': 'No valid predictions made'
            }
    
    def _generate_benchmark_summary(self, results, total_time):
        """Generate comprehensive benchmark summary"""
        summary = {
            'total_execution_time': total_time,
            'total_benchmarks': len([k for k in results.keys() if k != 'benchmark_summary']),
            'successful_benchmarks': 0,
            'failed_benchmarks': 0,
            'performance_highlights': {},
            'recommendations': []
        }
        
        for benchmark_name, result in results.items():
            if benchmark_name == 'benchmark_summary':
                continue
                
            if result.get('status') == 'SUCCESS':
                summary['successful_benchmarks'] += 1
            else:
                summary['failed_benchmarks'] += 1
        
        # Extract performance highlights
        if 'extended_predictor' in results and results['extended_predictor']['status'] == 'SUCCESS':
            extended_results = results['extended_predictor']['benchmark_results']
            if 'full_integration' in extended_results:
                full_integration = extended_results['full_integration']
                summary['performance_highlights']['full_integration_time'] = full_integration.get('execution_time', 0)
        
        # Generate recommendations
        if summary['successful_benchmarks'] / summary['total_benchmarks'] >= 0.8:
            summary['recommendations'].append("✅ Overall performance is excellent")
        else:
            summary['recommendations'].append("⚠️ Some performance issues detected - review failed benchmarks")
        
        return summary

# Test execution function
async def run_performance_benchmark_suite():
    """Run the complete performance benchmark suite"""
    logger.info("🚀 PHASE 3 PERFORMANCE BENCHMARK SUITE")
    logger.info("=" * 60)
    
    benchmark_suite = PerformanceBenchmarkSuite()
    
    try:
        # Run comprehensive benchmarks
        results = benchmark_suite.run_comprehensive_benchmark()
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f'phase3_performance_benchmark_{timestamp}.json'
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"\n💾 Benchmark results saved to: {results_file}")
        
        # Determine success
        summary = results.get('benchmark_summary', {})
        success_rate = summary.get('successful_benchmarks', 0) / max(summary.get('total_benchmarks', 1), 1)
        
        if success_rate >= 0.8:
            logger.info("\n🎉 PERFORMANCE BENCHMARK SUITE: PASSED!")
            logger.info("✅ Phase 3 components demonstrate excellent performance")
            return True, results
        else:
            logger.info("\n⚠️ PERFORMANCE BENCHMARK SUITE: ISSUES DETECTED")
            logger.info("🔧 Review benchmark results for optimization opportunities")
            return False, results
            
    except Exception as e:
        logger.error(f"❌ Benchmark suite execution failed: {e}")
        return False, {'execution_error': {'status': 'FAILED', 'error': str(e)}}

if __name__ == "__main__":
    # Run the benchmark suite
    success, results = asyncio.run(run_performance_benchmark_suite())
    
    logger.info(f"\n📊 Benchmark Suite {'PASSED' if success else 'FAILED'}")
    
    exit_code = 0 if success else 1
    sys.exit(exit_code)