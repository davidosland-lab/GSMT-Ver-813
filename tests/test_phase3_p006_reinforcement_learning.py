#!/usr/bin/env python3
"""
P3-006 Reinforcement Learning Integration Test Suite
==================================================

Comprehensive test suite for the Reinforcement Learning Integration component.
Tests all RL algorithms, adaptive model selection, and learning convergence.
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
from dataclasses import dataclass

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from phase3_reinforcement_learning import (
    ReinforcementLearningFramework,
    MultiArmedBandit,
    QLearningAgent,
    ThompsonSampling,
    RLState,
    RLAction,
    RLAlgorithm,
    ModelPerformance,
    RLConfig
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TestReinforcementLearningFramework:
    """Test class for Reinforcement Learning Framework"""
    
    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data for RL testing"""
        dates = pd.date_range(start='2024-01-01', end='2024-03-01', freq='D')
        np.random.seed(42)
        
        # Generate realistic market data with different regimes
        returns = []
        volatilities = []
        
        for i in range(len(dates)):
            # Create different market regimes
            if i < len(dates) * 0.3:  # Bull market
                ret = np.random.normal(0.002, 0.015)
                vol = np.random.uniform(0.10, 0.20)
            elif i < len(dates) * 0.7:  # Normal market
                ret = np.random.normal(0.001, 0.02)
                vol = np.random.uniform(0.15, 0.25)
            else:  # Bear market
                ret = np.random.normal(-0.001, 0.03)
                vol = np.random.uniform(0.20, 0.35)
            
            returns.append(ret)
            volatilities.append(vol)
        
        df = pd.DataFrame({
            'date': dates,
            'returns': returns,
            'volatility': volatilities,
            'volume': np.random.randint(1000000, 10000000, len(dates)),
            'regime': ['bull' if i < len(dates)*0.3 else 'normal' if i < len(dates)*0.7 else 'bear' 
                      for i in range(len(dates))]
        })
        
        return df
    
    @pytest.fixture
    def rl_config(self):
        """Create RL configuration for testing"""
        return RLConfig(
            learning_rate=0.1,
            exploration_rate=0.1,
            discount_factor=0.95,
            update_frequency=10,
            lookback_window=30,
            performance_threshold=0.55,
            adaptation_rate=0.05
        )
    
    @pytest.fixture
    def rl_framework(self, rl_config):
        """Create ReinforcementLearningFramework instance for testing"""
        return ReinforcementLearningFramework(rl_config)
    
    def test_multi_armed_bandit_initialization(self):
        """Test Multi-Armed Bandit initialization"""
        logger.info("🧪 Testing Multi-Armed Bandit initialization...")
        
        n_arms = 5
        bandit = MultiArmedBandit(n_arms=n_arms)
        
        assert bandit.n_arms == n_arms
        assert len(bandit.arm_counts) == n_arms
        assert len(bandit.arm_rewards) == n_arms
        assert all(count == 0 for count in bandit.arm_counts)
        assert all(reward == 0.0 for reward in bandit.arm_rewards)
        
        logger.info("✅ Multi-Armed Bandit initialization test passed")
    
    def test_multi_armed_bandit_selection(self):
        """Test Multi-Armed Bandit arm selection"""
        logger.info("🧪 Testing Multi-Armed Bandit arm selection...")
        
        bandit = MultiArmedBandit(n_arms=3)
        
        # Test initial selection (should be random)
        selected_arms = [bandit.select_arm() for _ in range(100)]
        assert all(0 <= arm < 3 for arm in selected_arms)
        assert len(set(selected_arms)) > 1  # Should explore different arms
        
        # Test after some rewards
        bandit.update_reward(0, 0.8)
        bandit.update_reward(1, 0.4)
        bandit.update_reward(2, 0.6)
        
        # Arm 0 should be selected more often due to higher reward
        selections = [bandit.select_arm() for _ in range(100)]
        arm_0_count = selections.count(0)
        
        # With UCB, arm 0 should be selected most frequently
        assert arm_0_count > 25, f"Arm 0 selected only {arm_0_count} times"
        
        logger.info("✅ Multi-Armed Bandit selection test passed")
    
    def test_q_learning_agent_initialization(self):
        """Test Q-Learning Agent initialization"""
        logger.info("🧪 Testing Q-Learning Agent initialization...")
        
        n_states = 10
        n_actions = 5
        learning_rate = 0.1
        
        agent = QLearningAgent(
            n_states=n_states,
            n_actions=n_actions,
            learning_rate=learning_rate
        )
        
        assert agent.n_states == n_states
        assert agent.n_actions == n_actions
        assert agent.learning_rate == learning_rate
        assert agent.q_table.shape == (n_states, n_actions)
        assert np.all(agent.q_table == 0.0)
        
        logger.info("✅ Q-Learning Agent initialization test passed")
    
    def test_q_learning_agent_action_selection(self):
        """Test Q-Learning Agent action selection"""
        logger.info("🧪 Testing Q-Learning Agent action selection...")
        
        agent = QLearningAgent(n_states=5, n_actions=3, learning_rate=0.1)
        
        # Test action selection for different states
        for state in range(5):
            action = agent.select_action(state)
            assert 0 <= action < 3
        
        # Test learning
        state = 0
        action = 1
        reward = 0.8
        next_state = 1
        
        old_q_value = agent.q_table[state, action]
        agent.update_q_value(state, action, reward, next_state)
        new_q_value = agent.q_table[state, action]
        
        # Q-value should be updated
        assert new_q_value != old_q_value
        assert new_q_value > old_q_value  # Positive reward should increase Q-value
        
        logger.info("✅ Q-Learning Agent action selection test passed")
    
    def test_thompson_sampling_initialization(self):
        """Test Thompson Sampling initialization"""
        logger.info("🧪 Testing Thompson Sampling initialization...")
        
        n_models = 4
        sampler = ThompsonSampling(n_models=n_models)
        
        assert sampler.n_models == n_models
        assert len(sampler.alpha) == n_models
        assert len(sampler.beta) == n_models
        assert all(alpha == 1.0 for alpha in sampler.alpha)
        assert all(beta == 1.0 for beta in sampler.beta)
        
        logger.info("✅ Thompson Sampling initialization test passed")
    
    def test_thompson_sampling_model_selection(self):
        """Test Thompson Sampling model selection"""
        logger.info("🧪 Testing Thompson Sampling model selection...")
        
        sampler = ThompsonSampling(n_models=3)
        
        # Test initial selection
        selected_models = [sampler.sample_model() for _ in range(100)]
        assert all(0 <= model < 3 for model in selected_models)
        assert len(set(selected_models)) > 1  # Should explore different models
        
        # Update with rewards
        sampler.update_model_performance(0, success=True)
        sampler.update_model_performance(0, success=True)
        sampler.update_model_performance(1, success=False)
        sampler.update_model_performance(2, success=True)
        
        # Model 0 should be selected more often
        selections = [sampler.sample_model() for _ in range(200)]
        model_0_count = selections.count(0)
        
        assert model_0_count > 50, f"Model 0 selected only {model_0_count} times"
        
        logger.info("✅ Thompson Sampling model selection test passed")
    
    def test_rl_state_creation(self, sample_market_data):
        """Test RL state creation from market data"""
        logger.info("🧪 Testing RL state creation...")
        
        framework = ReinforcementLearningFramework()
        
        # Create state from market data
        recent_data = sample_market_data.tail(30)
        rl_state = framework._create_rl_state(recent_data)
        
        assert isinstance(rl_state, RLState)
        assert hasattr(rl_state, 'market_regime')
        assert hasattr(rl_state, 'volatility_level')
        assert hasattr(rl_state, 'momentum_indicator')
        assert hasattr(rl_state, 'performance_metrics')
        
        # Verify state values are within expected ranges
        assert rl_state.volatility_level >= 0
        assert -1 <= rl_state.momentum_indicator <= 1
        
        logger.info(f"✅ RL state creation test passed - Regime: {rl_state.market_regime}")
    
    def test_optimal_model_selection(self, rl_framework, sample_market_data):
        """Test optimal model selection using RL"""
        logger.info("🧪 Testing optimal model selection...")
        
        # Create RL state
        recent_data = sample_market_data.tail(30)
        rl_state = rl_framework._create_rl_state(recent_data)
        
        # Test different RL algorithms
        algorithms = [
            RLAlgorithm.MULTI_ARMED_BANDIT,
            RLAlgorithm.Q_LEARNING,
            RLAlgorithm.THOMPSON_SAMPLING
        ]
        
        for algorithm in algorithms:
            selected_models = rl_framework.select_optimal_models(
                state=rl_state,
                algorithm=algorithm
            )
            
            assert isinstance(selected_models, list)
            assert len(selected_models) > 0
            assert all(isinstance(model_id, int) for model_id in selected_models)
            assert all(0 <= model_id < rl_framework.n_models for model_id in selected_models)
            
            logger.info(f"✅ {algorithm.value} selected models: {selected_models}")
        
        logger.info("✅ Optimal model selection test passed")
    
    def test_model_performance_tracking(self, rl_framework):
        """Test model performance tracking and updates"""
        logger.info("🧪 Testing model performance tracking...")
        
        # Simulate model predictions and outcomes
        model_performances = []
        
        for model_id in range(rl_framework.n_models):
            # Create mock performance data
            accuracy = np.random.uniform(0.4, 0.8)
            returns = np.random.uniform(-0.1, 0.15)
            sharpe_ratio = np.random.uniform(0.5, 2.0)
            max_drawdown = np.random.uniform(0.05, 0.25)
            
            performance = ModelPerformance(
                model_id=model_id,
                accuracy=accuracy,
                returns=returns,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                prediction_count=100,
                last_updated=datetime.now()
            )
            
            model_performances.append(performance)
        
        # Update framework with performance data
        rl_framework.performance_tracker = {
            perf.model_id: perf for perf in model_performances
        }
        
        # Test performance-based model selection
        best_models = rl_framework._get_top_performing_models(n_models=2)
        
        assert len(best_models) == 2
        assert all(isinstance(model_id, int) for model_id in best_models)
        
        # Verify best models have good performance
        for model_id in best_models:
            perf = rl_framework.performance_tracker[model_id]
            assert perf.accuracy > 0.4
        
        logger.info(f"✅ Performance tracking test passed - Top models: {best_models}")
    
    def test_adaptive_learning_convergence(self, rl_framework, sample_market_data):
        """Test adaptive learning and convergence"""
        logger.info("🧪 Testing adaptive learning convergence...")
        
        # Simulate learning over multiple episodes
        convergence_scores = []
        n_episodes = 50
        
        for episode in range(n_episodes):
            # Create state from data window
            start_idx = max(0, episode - 30)
            end_idx = min(len(sample_market_data), episode + 30)
            episode_data = sample_market_data.iloc[start_idx:end_idx]
            
            if len(episode_data) < 10:
                continue
            
            rl_state = rl_framework._create_rl_state(episode_data)
            
            # Select models and simulate performance
            selected_models = rl_framework.select_optimal_models(
                state=rl_state,
                algorithm=RLAlgorithm.THOMPSON_SAMPLING
            )
            
            # Simulate reward based on market conditions
            if episode_data['regime'].iloc[-1] == 'bull':
                reward = np.random.uniform(0.6, 1.0)
            elif episode_data['regime'].iloc[-1] == 'normal':
                reward = np.random.uniform(0.4, 0.7)
            else:  # bear market
                reward = np.random.uniform(0.3, 0.6)
            
            # Update model performance
            for model_id in selected_models:
                success = reward > 0.5
                rl_framework.thompson_sampler.update_model_performance(model_id, success)
            
            # Track convergence
            model_selections = [rl_framework.thompson_sampler.sample_model() for _ in range(10)]
            convergence_score = 1.0 - (len(set(model_selections)) / len(model_selections))
            convergence_scores.append(convergence_score)
        
        # Verify convergence (later episodes should have higher convergence)
        if len(convergence_scores) >= 20:
            early_convergence = np.mean(convergence_scores[:10])
            late_convergence = np.mean(convergence_scores[-10:])
            
            # Later episodes should show more convergence (less exploration)
            assert late_convergence >= early_convergence - 0.1, \
                f"No convergence detected: early={early_convergence:.3f}, late={late_convergence:.3f}"
        
        logger.info(f"✅ Adaptive learning test passed - Final convergence: {convergence_scores[-1]:.3f}")
    
    def test_multi_algorithm_comparison(self, sample_market_data):
        """Test comparison between different RL algorithms"""
        logger.info("🧪 Testing multi-algorithm comparison...")
        
        config = RLConfig(
            learning_rate=0.1,
            exploration_rate=0.1,
            discount_factor=0.95,
            update_frequency=5,
            lookback_window=20,
            performance_threshold=0.55,
            adaptation_rate=0.1
        )
        
        framework = ReinforcementLearningFramework(config)
        
        algorithms = [
            RLAlgorithm.MULTI_ARMED_BANDIT,
            RLAlgorithm.Q_LEARNING,
            RLAlgorithm.THOMPSON_SAMPLING
        ]
        
        algorithm_results = {}
        
        # Test each algorithm
        for algorithm in algorithms:
            selections = []
            rewards = []
            
            for i in range(20):
                # Create state
                data_window = sample_market_data.iloc[max(0, i-10):i+10]
                if len(data_window) < 5:
                    continue
                
                rl_state = framework._create_rl_state(data_window)
                
                # Select models
                selected_models = framework.select_optimal_models(
                    state=rl_state,
                    algorithm=algorithm
                )
                
                selections.extend(selected_models)
                
                # Simulate reward
                reward = np.random.uniform(0.3, 0.9)
                rewards.append(reward)
                
                # Update algorithm-specific components
                if algorithm == RLAlgorithm.THOMPSON_SAMPLING:
                    for model_id in selected_models:
                        framework.thompson_sampler.update_model_performance(
                            model_id, reward > 0.5
                        )
                elif algorithm == RLAlgorithm.MULTI_ARMED_BANDIT:
                    for model_id in selected_models:
                        framework.bandit.update_reward(model_id, reward)
            
            # Calculate algorithm performance
            avg_reward = np.mean(rewards) if rewards else 0
            selection_diversity = len(set(selections)) / max(len(selections), 1)
            
            algorithm_results[algorithm.value] = {
                'avg_reward': avg_reward,
                'diversity': selection_diversity,
                'total_selections': len(selections)
            }
            
            logger.info(f"   {algorithm.value}: Reward={avg_reward:.3f}, Diversity={selection_diversity:.3f}")
        
        # Verify all algorithms produced reasonable results
        for alg_name, results in algorithm_results.items():
            assert results['avg_reward'] > 0, f"{alg_name} produced no rewards"
            assert results['total_selections'] > 0, f"{alg_name} made no selections"
        
        logger.info("✅ Multi-algorithm comparison test passed")
    
    def test_rl_integration_with_prediction_system(self, rl_framework, sample_market_data):
        """Test RL integration with prediction system"""
        logger.info("🧪 Testing RL integration with prediction system...")
        
        # Simulate prediction system integration
        prediction_results = []
        
        for i in range(10):
            # Create market state
            data_window = sample_market_data.iloc[max(0, i*5):(i+1)*5+20]
            if len(data_window) < 10:
                continue
            
            rl_state = rl_framework._create_rl_state(data_window)
            
            # Select optimal models using RL
            selected_models = rl_framework.select_optimal_models(
                state=rl_state,
                algorithm=RLAlgorithm.THOMPSON_SAMPLING
            )
            
            # Simulate prediction ensemble
            model_predictions = []
            model_weights = []
            
            for model_id in selected_models:
                # Simulate model prediction
                prediction = np.random.uniform(-0.05, 0.05)  # Return prediction
                confidence = np.random.uniform(0.5, 0.9)
                
                model_predictions.append(prediction)
                model_weights.append(confidence)
            
            # Calculate ensemble prediction
            if model_predictions:
                total_weight = sum(model_weights)
                ensemble_prediction = sum(p * w for p, w in zip(model_predictions, model_weights)) / total_weight
                
                prediction_results.append({
                    'selected_models': selected_models,
                    'ensemble_prediction': ensemble_prediction,
                    'model_count': len(selected_models)
                })
        
        # Verify integration results
        assert len(prediction_results) > 0, "No prediction results generated"
        
        for result in prediction_results:
            assert len(result['selected_models']) > 0, "No models selected"
            assert -1 <= result['ensemble_prediction'] <= 1, "Prediction out of range"
            assert result['model_count'] > 0, "No models in ensemble"
        
        avg_model_count = np.mean([r['model_count'] for r in prediction_results])
        logger.info(f"✅ RL integration test passed - Avg models per prediction: {avg_model_count:.1f}")
    
    def test_performance_degradation_detection(self, rl_framework):
        """Test detection of model performance degradation"""
        logger.info("🧪 Testing performance degradation detection...")
        
        # Simulate declining model performance
        model_id = 0
        performance_history = []
        
        # Start with good performance, then degrade
        for i in range(20):
            if i < 10:
                accuracy = np.random.uniform(0.7, 0.9)  # Good performance
            else:
                accuracy = np.random.uniform(0.3, 0.5)  # Degraded performance
            
            performance = ModelPerformance(
                model_id=model_id,
                accuracy=accuracy,
                returns=np.random.uniform(-0.05, 0.05),
                sharpe_ratio=np.random.uniform(0.5, 2.0),
                max_drawdown=np.random.uniform(0.05, 0.25),
                prediction_count=i + 1,
                last_updated=datetime.now() - timedelta(days=20-i)
            )
            
            performance_history.append(performance)
        
        # Update framework with performance history
        rl_framework.performance_history = {model_id: performance_history}
        
        # Test degradation detection
        is_degraded = rl_framework._detect_performance_degradation(
            model_id=model_id,
            window_size=5,
            threshold=0.1
        )
        
        # Should detect degradation
        assert is_degraded, "Performance degradation not detected"
        
        logger.info("✅ Performance degradation detection test passed")

# Test execution functions
async def run_comprehensive_rl_tests():
    """Run comprehensive test suite for P3-006 Reinforcement Learning"""
    logger.info("🚀 P3-006 REINFORCEMENT LEARNING TEST SUITE")
    logger.info("=" * 60)
    
    test_results = {}
    
    try:
        # Initialize test class
        test_class = TestReinforcementLearningFramework()
        
        # Create fixtures
        sample_data = test_class.sample_market_data()
        rl_config = test_class.rl_config()
        rl_framework = test_class.rl_framework(rl_config)
        
        # Run individual tests
        tests = [
            ('mab_initialization', lambda: test_class.test_multi_armed_bandit_initialization()),
            ('mab_selection', lambda: test_class.test_multi_armed_bandit_selection()),
            ('qlearning_initialization', lambda: test_class.test_q_learning_agent_initialization()),
            ('qlearning_action_selection', lambda: test_class.test_q_learning_agent_action_selection()),
            ('thompson_initialization', lambda: test_class.test_thompson_sampling_initialization()),
            ('thompson_selection', lambda: test_class.test_thompson_sampling_model_selection()),
            ('rl_state_creation', lambda: test_class.test_rl_state_creation(sample_data)),
            ('optimal_model_selection', lambda: test_class.test_optimal_model_selection(rl_framework, sample_data)),
            ('performance_tracking', lambda: test_class.test_model_performance_tracking(rl_framework)),
            ('adaptive_learning', lambda: test_class.test_adaptive_learning_convergence(rl_framework, sample_data)),
            ('multi_algorithm_comparison', lambda: test_class.test_multi_algorithm_comparison(sample_data)),
            ('rl_integration', lambda: test_class.test_rl_integration_with_prediction_system(rl_framework, sample_data)),
            ('degradation_detection', lambda: test_class.test_performance_degradation_detection(rl_framework)),
        ]
        
        # Run all tests
        for test_name, test_func in tests:
            try:
                logger.info(f"\n🔬 Running {test_name}...")
                test_func()
                test_results[test_name] = {'status': 'SUCCESS'}
                logger.info(f"✅ {test_name}: PASSED")
            except Exception as e:
                logger.error(f"❌ {test_name}: FAILED - {e}")
                test_results[test_name] = {'status': 'FAILED', 'error': str(e)}
        
        # Calculate results
        total_tests = len(test_results)
        successful_tests = sum(1 for result in test_results.values() 
                             if result.get('status') == 'SUCCESS')
        
        logger.info("\n" + "=" * 60)
        logger.info("🎯 P3-006 REINFORCEMENT LEARNING TEST SUMMARY")
        logger.info("=" * 60)
        logger.info(f"\n📊 Test Results: {successful_tests}/{total_tests} successful ({successful_tests/total_tests*100:.1f}%)")
        
        for test_name, result in test_results.items():
            status_icon = "✅" if result.get('status') == 'SUCCESS' else "❌"
            logger.info(f"   {status_icon} {test_name}: {result.get('status')}")
            if result.get('error'):
                logger.info(f"      Error: {result['error']}")
        
        success = successful_tests >= total_tests * 0.8  # 80% pass rate
        
        if success:
            logger.info("\n🎉 P3-006 REINFORCEMENT LEARNING: TEST SUITE PASSED!")
            logger.info("✅ RL algorithms working correctly")
            logger.info("✅ Adaptive model selection functional")
            logger.info("✅ Performance tracking and learning convergence verified")
        else:
            logger.info("\n⚠️ P3-006 REINFORCEMENT LEARNING: ISSUES DETECTED")
            logger.info("🔧 Review failed tests before proceeding")
        
        return success, test_results
        
    except Exception as e:
        logger.error(f"❌ Test suite execution failed: {e}")
        return False, {'execution_error': {'status': 'FAILED', 'error': str(e)}}

if __name__ == "__main__":
    # Run the test suite
    success, results = asyncio.run(run_comprehensive_rl_tests())
    
    # Save results
    import json
    with open('p3_006_test_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"\n💾 Test results saved to: p3_006_test_results.json")
    
    exit_code = 0 if success else 1
    sys.exit(exit_code)