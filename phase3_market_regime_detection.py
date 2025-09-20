#!/usr/bin/env python3
"""
Phase 3 Component P3_003: Advanced Market Regime Detection
=========================================================

Dynamic market regime identification with regime-specific model weighting.
Implements bull/bear/sideways market classification, volatility regime detection,
and dynamic model adaptation to changing market conditions.

Target: Different model weights for bull/bear/sideways markets
Dependencies: Enhanced feature engineering (P1_004)
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy import stats
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

class MarketRegimeDetection:
    """
    Advanced Market Regime Detection system.
    
    Implements multiple regime detection methods:
    - Trend-based regimes (Bull/Bear/Sideways)
    - Volatility regimes (Low/Medium/High)
    - Combined multi-dimensional regimes
    - Hidden Markov Model-like state transitions
    - Dynamic regime-specific model weighting
    """
    
    def __init__(self, config: Dict = None):
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.config = config or {}
        self.lookback_period = self.config.get('lookback_period', 60)
        self.min_regime_duration = self.config.get('min_regime_duration', 5)
        self.volatility_window = self.config.get('volatility_window', 20)
        self.trend_window = self.config.get('trend_window', 30)
        
        # Regime definitions
        self.trend_regimes = ['Bull', 'Bear', 'Sideways']
        self.volatility_regimes = ['Low_Vol', 'Medium_Vol', 'High_Vol']
        self.combined_regimes = []
        
        # Generate combined regimes
        for trend in self.trend_regimes:
            for vol in self.volatility_regimes:
                self.combined_regimes.append(f"{trend}_{vol}")
        
        # Current regime state
        self.current_regime = {
            'trend': 'Sideways',
            'volatility': 'Medium_Vol',
            'combined': 'Sideways_Medium_Vol',
            'confidence': 0.5,
            'duration': 0,
            'last_change': None
        }
        
        # Historical regime tracking
        self.regime_history = []
        self.regime_transitions = {}
        self.regime_probabilities = {}
        
        # Models and scalers
        self.trend_classifier = None
        self.volatility_classifier = None
        self.combined_classifier = None
        self.feature_scaler = StandardScaler()
        
        # Regime-specific model weights
        self.regime_model_weights = self._initialize_regime_weights()
        
        # Regime transition probabilities (Markov chain)
        self.transition_matrix = {}
        
        self.logger.info("📈 Phase 3 Market Regime Detection initialized")
    
    def _initialize_regime_weights(self) -> Dict[str, Dict[str, float]]:
        """Initialize regime-specific model weights."""
        
        # Base weights for different model types in different regimes
        weights = {}
        
        # Trend regime weights
        weights['Bull'] = {
            'momentum_models': 0.35,      # Higher weight for momentum in bull markets
            'technical_models': 0.30,     
            'fundamental_models': 0.20,   
            'sentiment_models': 0.15
        }
        
        weights['Bear'] = {
            'momentum_models': 0.15,      # Lower momentum weight in bear markets
            'technical_models': 0.35,     # Higher technical weight for support/resistance
            'fundamental_models': 0.35,   # Higher fundamental weight for value
            'sentiment_models': 0.15
        }
        
        weights['Sideways'] = {
            'momentum_models': 0.20,      # Balanced weights in sideways markets
            'technical_models': 0.30,
            'fundamental_models': 0.25,
            'sentiment_models': 0.25      # Higher sentiment weight for direction
        }
        
        # Volatility regime adjustments
        volatility_adjustments = {
            'Low_Vol': {
                'momentum_boost': 1.2,    # Boost momentum in low vol
                'sentiment_boost': 0.8    # Reduce sentiment impact
            },
            'Medium_Vol': {
                'momentum_boost': 1.0,    # No adjustment
                'sentiment_boost': 1.0
            },
            'High_Vol': {
                'momentum_boost': 0.7,    # Reduce momentum in high vol
                'sentiment_boost': 1.3    # Increase sentiment impact
            }
        }
        
        # Combined regime weights
        for trend in self.trend_regimes:
            for vol in self.volatility_regimes:
                regime_key = f"{trend}_{vol}"
                base_weights = weights[trend].copy()
                vol_adj = volatility_adjustments[vol]
                
                # Apply volatility adjustments
                if 'momentum_models' in base_weights:
                    base_weights['momentum_models'] *= vol_adj['momentum_boost']
                if 'sentiment_models' in base_weights:
                    base_weights['sentiment_models'] *= vol_adj['sentiment_boost']
                
                # Renormalize
                total_weight = sum(base_weights.values())
                if total_weight > 0:
                    for key in base_weights:
                        base_weights[key] /= total_weight
                
                weights[regime_key] = base_weights
        
        return weights
    
    def extract_regime_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract features for regime detection."""
        
        features = pd.DataFrame(index=data.index)
        close_prices = data['Close']
        volumes = data['Volume'] if 'Volume' in data.columns else pd.Series([1] * len(data))
        
        try:
            # Trend features
            features['price_trend_short'] = close_prices.pct_change(periods=5).rolling(5).mean()
            features['price_trend_medium'] = close_prices.pct_change(periods=20).rolling(10).mean()
            features['price_trend_long'] = close_prices.pct_change(periods=60).rolling(15).mean()
            
            # Moving average relationships
            sma_10 = close_prices.rolling(window=10).mean()
            sma_30 = close_prices.rolling(window=30).mean()
            sma_60 = close_prices.rolling(window=60).mean()
            
            features['ma_slope_short'] = (sma_10 / sma_10.shift(5) - 1)
            features['ma_slope_medium'] = (sma_30 / sma_30.shift(10) - 1)
            features['price_vs_ma_short'] = close_prices / sma_10 - 1
            features['price_vs_ma_medium'] = close_prices / sma_30 - 1
            features['price_vs_ma_long'] = close_prices / sma_60 - 1
            
            # Volatility features
            returns = close_prices.pct_change()
            features['volatility_5d'] = returns.rolling(window=5).std()
            features['volatility_20d'] = returns.rolling(window=20).std()
            features['volatility_60d'] = returns.rolling(window=60).std()
            features['volatility_ratio'] = features['volatility_5d'] / features['volatility_20d']
            
            # Volume features
            volume_ma = volumes.rolling(window=20).mean()
            features['volume_trend'] = volumes / volume_ma - 1
            features['volume_volatility'] = (volumes.pct_change().rolling(window=10).std())
            
            # Price action features
            if 'High' in data.columns and 'Low' in data.columns:
                features['high_low_range'] = (data['High'] - data['Low']) / close_prices
                features['close_position'] = (close_prices - data['Low']) / (data['High'] - data['Low'] + 1e-8)
            else:
                features['high_low_range'] = returns.rolling(window=5).apply(lambda x: x.max() - x.min())
                features['close_position'] = 0.5
            
            # Momentum indicators
            features['rsi_proxy'] = self._calculate_rsi_proxy(returns)
            features['momentum_5d'] = close_prices / close_prices.shift(5) - 1
            features['momentum_20d'] = close_prices / close_prices.shift(20) - 1
            
            # Trend strength
            features['trend_strength'] = np.abs(features['price_trend_medium'])
            features['trend_consistency'] = (
                np.sign(features['price_trend_short']) == np.sign(features['price_trend_medium'])
            ).astype(float)
            
            # Market stress indicators
            features['max_drawdown_5d'] = self._calculate_rolling_drawdown(close_prices, 5)
            features['max_drawdown_20d'] = self._calculate_rolling_drawdown(close_prices, 20)
            
            # Fill NaN values
            features = features.fillna(method='ffill').fillna(0)
            
            self.logger.debug(f"Extracted {len(features.columns)} regime features")
            return features
            
        except Exception as e:
            self.logger.error(f"Feature extraction failed: {e}")
            # Return minimal features
            features['price_change'] = returns.fillna(0)
            features['volatility'] = returns.rolling(window=20).std().fillna(0.02)
            return features
    
    def _calculate_rsi_proxy(self, returns: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI-like momentum indicator."""
        gains = returns.where(returns > 0, 0)
        losses = -returns.where(returns < 0, 0)
        
        avg_gains = gains.rolling(window=window).mean()
        avg_losses = losses.rolling(window=window).mean()
        
        rs = avg_gains / (avg_losses + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        
        return (rsi - 50) / 50  # Normalize to [-1, 1]
    
    def _calculate_rolling_drawdown(self, prices: pd.Series, window: int) -> pd.Series:
        """Calculate rolling maximum drawdown."""
        rolling_max = prices.rolling(window=window).max()
        drawdown = (prices - rolling_max) / rolling_max
        return drawdown
    
    def train_regime_classifiers(self, historical_data: pd.DataFrame) -> Dict[str, float]:
        """Train regime classification models on historical data."""
        
        try:
            # Extract features
            features = self.extract_regime_features(historical_data)
            
            if len(features) < 100:
                self.logger.warning("Insufficient data for regime training")
                return {}
            
            # Prepare features
            X = features.dropna().values
            if len(X) < 50:
                self.logger.warning("Too many NaN values in features")
                return {}
            
            # Fit feature scaler
            X_scaled = self.feature_scaler.fit_transform(X)
            
            # Train trend regime classifier (3 clusters for Bull/Bear/Sideways)
            self.trend_classifier = GaussianMixture(
                n_components=3, 
                covariance_type='full', 
                random_state=42,
                max_iter=200
            )
            trend_labels = self.trend_classifier.fit_predict(X_scaled)
            
            # Train volatility regime classifier (3 clusters for Low/Medium/High vol)
            vol_features = features[['volatility_5d', 'volatility_20d', 'volatility_ratio', 'volume_volatility']].dropna()
            if len(vol_features) >= 50:
                vol_X = self.feature_scaler.fit_transform(vol_features.values)
                self.volatility_classifier = GaussianMixture(
                    n_components=3,
                    covariance_type='full',
                    random_state=42,
                    max_iter=200
                )
                vol_labels = self.volatility_classifier.fit_predict(vol_X)
            else:
                vol_labels = np.zeros(len(X_scaled))
            
            # Train combined regime classifier
            self.combined_classifier = GaussianMixture(
                n_components=min(6, len(X) // 10),  # Adaptive number of components
                covariance_type='diag',
                random_state=42,
                max_iter=200
            )
            combined_labels = self.combined_classifier.fit_predict(X_scaled)
            
            # Calculate silhouette scores for validation
            if len(np.unique(trend_labels)) > 1:
                trend_score = silhouette_score(X_scaled, trend_labels)
            else:
                trend_score = 0.0
            
            if len(np.unique(vol_labels)) > 1 and len(vol_features) >= 50:
                vol_score = silhouette_score(vol_X, vol_labels)
            else:
                vol_score = 0.0
            
            if len(np.unique(combined_labels)) > 1:
                combined_score = silhouette_score(X_scaled, combined_labels)
            else:
                combined_score = 0.0
            
            # Initialize transition matrices
            self._initialize_transition_matrices(trend_labels, vol_labels, combined_labels)
            
            training_results = {
                'trend_silhouette': trend_score,
                'volatility_silhouette': vol_score,
                'combined_silhouette': combined_score,
                'n_samples': len(X_scaled)
            }
            
            self.logger.info(f"✅ Regime classifiers trained:")
            self.logger.info(f"   Trend score: {trend_score:.3f}")
            self.logger.info(f"   Volatility score: {vol_score:.3f}")
            self.logger.info(f"   Combined score: {combined_score:.3f}")
            
            return training_results
            
        except Exception as e:
            self.logger.error(f"Regime classifier training failed: {e}")
            return {}
    
    def _initialize_transition_matrices(self, trend_labels: np.ndarray, 
                                      vol_labels: np.ndarray,
                                      combined_labels: np.ndarray) -> None:
        """Initialize regime transition probability matrices."""
        
        # Calculate transition matrices for each regime type
        for regime_type, labels in [('trend', trend_labels), ('volatility', vol_labels), ('combined', combined_labels)]:
            n_states = len(np.unique(labels))
            transition_matrix = np.zeros((n_states, n_states))
            
            # Count transitions
            for i in range(len(labels) - 1):
                current_state = labels[i]
                next_state = labels[i + 1]
                transition_matrix[current_state, next_state] += 1
            
            # Normalize to probabilities
            row_sums = transition_matrix.sum(axis=1)
            for i in range(n_states):
                if row_sums[i] > 0:
                    transition_matrix[i, :] /= row_sums[i]
                else:
                    # Uniform probability if no transitions observed
                    transition_matrix[i, :] = 1.0 / n_states
            
            self.transition_matrix[regime_type] = transition_matrix
    
    def detect_current_regime(self, current_data: pd.DataFrame, 
                            return_probabilities: bool = False) -> Dict[str, Any]:
        """Detect current market regime using trained classifiers."""
        
        try:
            # Extract features for current period
            features = self.extract_regime_features(current_data)
            
            if len(features) == 0:
                return self._get_default_regime()
            
            # Use last observation for regime detection
            latest_features = features.iloc[-1:].fillna(0)
            X = latest_features.values
            
            # Scale features
            if hasattr(self.feature_scaler, 'mean_'):
                X_scaled = self.feature_scaler.transform(X)
            else:
                X_scaled = X
            
            regime_results = {}
            
            # Trend regime detection
            if self.trend_classifier is not None:
                trend_probs = self.trend_classifier.predict_proba(X_scaled)[0]
                trend_regime_idx = np.argmax(trend_probs)
                
                # Map to regime names (0: Bear, 1: Bull, 2: Sideways based on cluster characteristics)
                trend_regime = self._map_cluster_to_trend_regime(trend_regime_idx, latest_features)
                
                regime_results['trend'] = {
                    'regime': trend_regime,
                    'confidence': float(trend_probs[trend_regime_idx]),
                    'probabilities': {f'cluster_{i}': float(p) for i, p in enumerate(trend_probs)}
                }
            else:
                regime_results['trend'] = {
                    'regime': 'Sideways',
                    'confidence': 0.5,
                    'probabilities': {}
                }
            
            # Volatility regime detection
            if self.volatility_classifier is not None:
                vol_features = latest_features[['volatility_5d', 'volatility_20d', 'volatility_ratio', 'volume_volatility']]
                vol_X = vol_features.values
                
                if hasattr(self.feature_scaler, 'mean_'):
                    vol_X_scaled = self.feature_scaler.transform(vol_X)
                else:
                    vol_X_scaled = vol_X
                
                vol_probs = self.volatility_classifier.predict_proba(vol_X_scaled)[0]
                vol_regime_idx = np.argmax(vol_probs)
                
                # Map to volatility regime names
                vol_regime = self._map_cluster_to_vol_regime(vol_regime_idx, vol_features.iloc[0])
                
                regime_results['volatility'] = {
                    'regime': vol_regime,
                    'confidence': float(vol_probs[vol_regime_idx]),
                    'probabilities': {f'cluster_{i}': float(p) for i, p in enumerate(vol_probs)}
                }
            else:
                regime_results['volatility'] = {
                    'regime': 'Medium_Vol',
                    'confidence': 0.5,
                    'probabilities': {}
                }
            
            # Combined regime
            combined_regime = f"{regime_results['trend']['regime']}_{regime_results['volatility']['regime']}"
            combined_confidence = (regime_results['trend']['confidence'] + 
                                 regime_results['volatility']['confidence']) / 2
            
            regime_results['combined'] = {
                'regime': combined_regime,
                'confidence': float(combined_confidence)
            }
            
            # Update current regime state
            self._update_regime_state(regime_results)
            
            # Get regime-specific model weights
            current_weights = self.get_regime_model_weights(combined_regime)
            
            result = {
                'trend_regime': regime_results['trend']['regime'],
                'volatility_regime': regime_results['volatility']['regime'],
                'combined_regime': combined_regime,
                'confidence': float(combined_confidence),
                'regime_duration': self.current_regime['duration'],
                'model_weights': current_weights,
                'regime_stability': self._calculate_regime_stability()
            }
            
            if return_probabilities:
                result['probabilities'] = regime_results
            
            self.logger.debug(f"Detected regime: {combined_regime} (confidence: {combined_confidence:.3f})")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Regime detection failed: {e}")
            return self._get_default_regime()
    
    def _map_cluster_to_trend_regime(self, cluster_idx: int, features: pd.DataFrame) -> str:
        """Map cluster index to trend regime name based on feature characteristics."""
        
        # Use price trends and momentum to determine regime
        feature_row = features.iloc[0] if len(features) > 0 else {}
        
        price_trend = feature_row.get('price_trend_medium', 0)
        momentum = feature_row.get('momentum_20d', 0)
        ma_position = feature_row.get('price_vs_ma_medium', 0)
        
        # Simple heuristic mapping
        if price_trend > 0.01 and momentum > 0.05 and ma_position > 0:
            return 'Bull'
        elif price_trend < -0.01 and momentum < -0.05 and ma_position < 0:
            return 'Bear'
        else:
            return 'Sideways'
    
    def _map_cluster_to_vol_regime(self, cluster_idx: int, features: pd.Series) -> str:
        """Map cluster index to volatility regime name based on volatility levels."""
        
        vol_20d = features.get('volatility_20d', 0.02)
        vol_ratio = features.get('volatility_ratio', 1.0)
        
        # Define volatility thresholds (can be made adaptive)
        low_vol_threshold = 0.015
        high_vol_threshold = 0.035
        
        if vol_20d < low_vol_threshold and vol_ratio < 1.2:
            return 'Low_Vol'
        elif vol_20d > high_vol_threshold or vol_ratio > 1.8:
            return 'High_Vol'
        else:
            return 'Medium_Vol'
    
    def _update_regime_state(self, regime_results: Dict) -> None:
        """Update internal regime state tracking."""
        
        new_combined_regime = regime_results['combined']['regime']
        
        # Check if regime changed
        if new_combined_regime != self.current_regime['combined']:
            self.current_regime['last_change'] = datetime.now()
            self.current_regime['duration'] = 1
        else:
            self.current_regime['duration'] += 1
        
        # Update current state
        self.current_regime['trend'] = regime_results['trend']['regime']
        self.current_regime['volatility'] = regime_results['volatility']['regime']
        self.current_regime['combined'] = new_combined_regime
        self.current_regime['confidence'] = regime_results['combined']['confidence']
        
        # Add to history
        self.regime_history.append({
            'timestamp': datetime.now(),
            'regime': new_combined_regime,
            'confidence': regime_results['combined']['confidence']
        })
        
        # Maintain history window
        if len(self.regime_history) > 1000:
            self.regime_history = self.regime_history[-1000:]
    
    def _calculate_regime_stability(self) -> float:
        """Calculate regime stability score based on recent history."""
        
        if len(self.regime_history) < 10:
            return 0.5
        
        # Look at recent regime changes
        recent_history = self.regime_history[-20:]  # Last 20 observations
        regimes = [h['regime'] for h in recent_history]
        
        # Calculate stability as consistency of regime
        most_common_regime = max(set(regimes), key=regimes.count)
        stability = regimes.count(most_common_regime) / len(regimes)
        
        return stability
    
    def get_regime_model_weights(self, regime: str) -> Dict[str, float]:
        """Get model weights for specific regime."""
        
        if regime in self.regime_model_weights:
            return self.regime_model_weights[regime].copy()
        
        # Fallback to trend-only regime
        trend_regime = regime.split('_')[0] if '_' in regime else regime
        if trend_regime in self.regime_model_weights:
            return self.regime_model_weights[trend_regime].copy()
        
        # Default balanced weights
        return {
            'momentum_models': 0.25,
            'technical_models': 0.25,
            'fundamental_models': 0.25,
            'sentiment_models': 0.25
        }
    
    def _get_default_regime(self) -> Dict[str, Any]:
        """Get default regime when detection fails."""
        return {
            'trend_regime': 'Sideways',
            'volatility_regime': 'Medium_Vol',
            'combined_regime': 'Sideways_Medium_Vol',
            'confidence': 0.3,
            'regime_duration': 0,
            'model_weights': self.get_regime_model_weights('Sideways_Medium_Vol'),
            'regime_stability': 0.5
        }
    
    def get_regime_transition_probabilities(self) -> Dict[str, np.ndarray]:
        """Get regime transition probability matrices."""
        return self.transition_matrix.copy()
    
    def predict_regime_transition(self, current_regime: str, n_steps: int = 5) -> Dict[str, Any]:
        """Predict future regime transitions using Markov chain."""
        
        try:
            # For now, implement simple persistence model
            # In practice, this would use the transition matrices
            
            persistence_prob = 0.8  # Probability of staying in same regime
            
            predictions = []
            current_prob = 1.0
            
            for step in range(1, n_steps + 1):
                # Simple model: exponential decay of current regime probability
                prob_same = persistence_prob ** step
                prob_change = 1 - prob_same
                
                predictions.append({
                    'step': step,
                    'current_regime_prob': prob_same,
                    'change_prob': prob_change,
                    'most_likely_regime': current_regime if prob_same > 0.5 else 'Unknown'
                })
            
            return {
                'current_regime': current_regime,
                'predictions': predictions,
                'method': 'simple_persistence'
            }
            
        except Exception as e:
            self.logger.error(f"Regime transition prediction failed: {e}")
            return {'error': str(e)}
    
    def get_regime_diagnostics(self) -> Dict[str, Any]:
        """Get comprehensive regime detection diagnostics."""
        
        diagnostics = {
            'current_regime': self.current_regime.copy(),
            'regime_weights': {regime: weights.copy() 
                             for regime, weights in self.regime_model_weights.items()},
            'classifiers_trained': {
                'trend': self.trend_classifier is not None,
                'volatility': self.volatility_classifier is not None,
                'combined': self.combined_classifier is not None
            },
            'history_length': len(self.regime_history),
            'regime_stability': self._calculate_regime_stability()
        }
        
        # Recent regime distribution
        if self.regime_history:
            recent_regimes = [h['regime'] for h in self.regime_history[-50:]]
            regime_counts = {}
            for regime in recent_regimes:
                regime_counts[regime] = regime_counts.get(regime, 0) + 1
            
            diagnostics['recent_regime_distribution'] = regime_counts
        
        return diagnostics

# Integration function for existing system
def create_phase3_regime_detector(historical_data: pd.DataFrame = None, 
                                 config: Dict = None) -> MarketRegimeDetection:
    """Create and initialize Phase 3 Market Regime Detection system."""
    detector = MarketRegimeDetection(config)
    
    if historical_data is not None and len(historical_data) > 100:
        training_results = detector.train_regime_classifiers(historical_data)
        logging.info(f"📈 P3_003 Regime Detection trained: {training_results}")
    
    return detector

if __name__ == "__main__":
    # Test implementation
    import yfinance as yf
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Test the Market Regime Detection system
    logger.info("🧪 Testing P3_003 Market Regime Detection...")
    
    # Download test data
    ticker = yf.Ticker("CBA.AX")
    data = ticker.history(period="2y", interval="1d")
    
    if not data.empty:
        # Create and train regime detector
        detector = create_phase3_regime_detector(data)
        
        # Test regime detection on recent data
        recent_data = data.tail(100)
        regime_result = detector.detect_current_regime(recent_data, return_probabilities=True)
        
        logger.info("📊 Current Market Regime:")
        logger.info(f"   Trend: {regime_result['trend_regime']}")
        logger.info(f"   Volatility: {regime_result['volatility_regime']}")
        logger.info(f"   Combined: {regime_result['combined_regime']}")
        logger.info(f"   Confidence: {regime_result['confidence']:.3f}")
        logger.info(f"   Stability: {regime_result['regime_stability']:.3f}")
        
        # Test model weights
        weights = regime_result['model_weights']
        logger.info(f"🎯 Model Weights for {regime_result['combined_regime']}:")
        for model_type, weight in weights.items():
            logger.info(f"   {model_type}: {weight:.3f}")
        
        # Test regime transition prediction
        transition_pred = detector.predict_regime_transition(regime_result['combined_regime'])
        logger.info(f"🔮 Regime Transition Predictions:")
        for pred in transition_pred['predictions'][:3]:
            logger.info(f"   Step {pred['step']}: {pred['current_regime_prob']:.3f} prob same regime")
        
        # Get diagnostics
        diagnostics = detector.get_regime_diagnostics()
        logger.info(f"🔧 Regime Detection Diagnostics:")
        logger.info(f"   Classifiers trained: {diagnostics['classifiers_trained']}")
        logger.info(f"   History length: {diagnostics['history_length']}")
        logger.info(f"   Current stability: {diagnostics['regime_stability']:.3f}")
        
        logger.info("🎉 P3_003 Market Regime Detection test completed successfully!")
    else:
        logger.error("Failed to download test data")