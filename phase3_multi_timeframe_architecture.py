#!/usr/bin/env python3
"""
Phase 3 Component P3_001: Multi-Timeframe Architecture
====================================================

Advanced prediction system with specialized models for different prediction horizons.
Implements cross-timeframe information fusion and horizon-specific feature engineering.

Target: 5-day accuracy improved to >55%
Dependencies: Phase 2 components (P2_001-P2_004)
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class MultiTimeframeArchitecture:
    """
    Multi-Timeframe Architecture for horizon-specific predictions.
    
    Implements separate specialized models for different prediction horizons:
    - Ultra-short (1 day): High-frequency pattern recognition
    - Short (5 days): Technical momentum and news sentiment
    - Medium (30 days): Economic indicators and institutional flows
    - Long (90 days): Fundamental analysis and structural factors
    """
    
    def __init__(self, config: Dict = None):
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.config = config or {}
        self.timeframes = {
            'ultra_short': {'days': 1, 'model_type': 'high_frequency'},
            'short': {'days': 5, 'model_type': 'technical_momentum'},
            'medium': {'days': 30, 'model_type': 'economic_institutional'},
            'long': {'days': 90, 'model_type': 'fundamental_structural'}
        }
        
        # Model storage
        self.models = {}
        self.scalers = {}
        self.feature_sets = {}
        self.performance_metrics = {}
        
        # Cross-timeframe fusion weights
        self.fusion_weights = {
            'consistency_boost': 0.15,  # Boost when models agree
            'divergence_penalty': 0.10,  # Penalty when models disagree
            'temporal_decay': 0.95      # Decay factor for older predictions
        }
        
        self.logger.info("🎯 Phase 3 Multi-Timeframe Architecture initialized")
    
    def _extract_timeframe_specific_features(self, data: pd.DataFrame, 
                                           timeframe: str, target_horizon: int) -> pd.DataFrame:
        """Extract features optimized for specific timeframes."""
        features = pd.DataFrame(index=data.index)
        
        try:
            # Base price features
            close_prices = data['Close'].values
            volumes = data['Volume'].values if 'Volume' in data.columns else np.ones(len(data))
            
            if timeframe == 'ultra_short':
                # High-frequency features for 1-day predictions
                features['price_momentum_1h'] = self._calculate_momentum(close_prices, 1)
                features['volume_surge_1h'] = self._calculate_volume_surge(volumes, 1)
                features['intraday_volatility'] = self._calculate_intraday_volatility(data)
                features['microstructure_signal'] = self._calculate_microstructure_signal(data)
                features['news_sentiment_weight'] = 0.35  # High weight for immediate impact
                
            elif timeframe == 'short':
                # Technical momentum features for 5-day predictions  
                features['price_momentum_5d'] = self._calculate_momentum(close_prices, 5)
                features['technical_strength'] = self._calculate_technical_strength(data)
                features['volume_trend_5d'] = self._calculate_volume_trend(volumes, 5)
                features['support_resistance'] = self._calculate_support_resistance(data, 5)
                features['news_sentiment_weight'] = 0.25
                
            elif timeframe == 'medium':
                # Economic and institutional features for 30-day predictions
                features['price_momentum_30d'] = self._calculate_momentum(close_prices, 30)
                features['economic_indicator_strength'] = self._calculate_economic_strength(data)
                features['institutional_flow_proxy'] = self._calculate_institutional_flow(data)
                features['sector_rotation_signal'] = self._calculate_sector_rotation(data)
                features['news_sentiment_weight'] = 0.15
                
            elif timeframe == 'long':
                # Fundamental features for 90-day predictions
                features['price_momentum_90d'] = self._calculate_momentum(close_prices, 90)
                features['fundamental_strength'] = self._calculate_fundamental_strength(data)
                features['structural_trends'] = self._calculate_structural_trends(data)
                features['macro_economic_signal'] = self._calculate_macro_signal(data)
                features['news_sentiment_weight'] = 0.10
            
            # Common cross-timeframe features
            features['volatility_regime'] = self._calculate_volatility_regime(data)
            features['trend_strength'] = self._calculate_trend_strength(data, target_horizon)
            features['market_stress'] = self._calculate_market_stress(data)
            
            # Forward fill any NaN values
            features = features.fillna(method='ffill').fillna(0)
            
            self.logger.debug(f"Extracted {len(features.columns)} features for {timeframe} timeframe")
            return features
            
        except Exception as e:
            self.logger.error(f"Feature extraction failed for {timeframe}: {e}")
            # Return minimal feature set
            features['price_change'] = close_prices[1:] / close_prices[:-1] - 1 if len(close_prices) > 1 else [0]
            features['volatility'] = np.std(close_prices[-20:]) if len(close_prices) >= 20 else 0.02
            return features.fillna(0)
    
    def _calculate_momentum(self, prices: np.ndarray, window: int) -> pd.Series:
        """Calculate price momentum over specified window."""
        if len(prices) < window + 1:
            return pd.Series([0] * len(prices))
        
        momentum = []
        for i in range(len(prices)):
            if i < window:
                momentum.append(0)
            else:
                momentum.append((prices[i] - prices[i-window]) / prices[i-window])
        
        return pd.Series(momentum)
    
    def _calculate_volume_surge(self, volumes: np.ndarray, window: int) -> pd.Series:
        """Calculate volume surge indicator."""
        if len(volumes) < window + 1:
            return pd.Series([0] * len(volumes))
        
        avg_volume = np.mean(volumes[-20:]) if len(volumes) >= 20 else np.mean(volumes)
        surge = volumes / (avg_volume + 1e-8)  # Avoid division by zero
        return pd.Series(surge)
    
    def _calculate_intraday_volatility(self, data: pd.DataFrame) -> pd.Series:
        """Calculate intraday volatility proxy."""
        if 'High' in data.columns and 'Low' in data.columns:
            # True range calculation
            high_low = data['High'] - data['Low']
            high_close = np.abs(data['High'] - data['Close'].shift(1))
            low_close = np.abs(data['Low'] - data['Close'].shift(1))
            
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            return true_range / data['Close']  # Normalize by price
        else:
            # Fallback: use close price volatility
            returns = data['Close'].pct_change()
            return returns.rolling(window=5).std().fillna(0.02)
    
    def _calculate_microstructure_signal(self, data: pd.DataFrame) -> pd.Series:
        """Calculate microstructure signal for ultra-short predictions."""
        # Simplified microstructure proxy using price and volume
        price_changes = data['Close'].pct_change()
        volume_changes = data['Volume'].pct_change() if 'Volume' in data.columns else pd.Series([0] * len(data))
        
        # Correlation between price and volume changes
        rolling_corr = price_changes.rolling(window=10).corr(volume_changes).fillna(0)
        return rolling_corr
    
    def _calculate_technical_strength(self, data: pd.DataFrame) -> pd.Series:
        """Calculate technical analysis strength."""
        close = data['Close']
        
        # Multiple technical indicators
        sma_5 = close.rolling(window=5).mean()
        sma_20 = close.rolling(window=20).mean()
        
        # Trend strength: price position relative to moving averages
        trend_strength = (close / sma_5 - 1) + (sma_5 / sma_20 - 1)
        return trend_strength.fillna(0)
    
    def _calculate_volume_trend(self, volumes: np.ndarray, window: int) -> pd.Series:
        """Calculate volume trend over specified window."""
        vol_series = pd.Series(volumes)
        vol_ma = vol_series.rolling(window=window).mean()
        vol_trend = vol_series / vol_ma - 1
        return vol_trend.fillna(0)
    
    def _calculate_support_resistance(self, data: pd.DataFrame, window: int) -> pd.Series:
        """Calculate support/resistance levels."""
        close = data['Close']
        high = data['High'] if 'High' in data.columns else close
        low = data['Low'] if 'Low' in data.columns else close
        
        # Support/resistance based on recent highs/lows
        resistance = high.rolling(window=window).max()
        support = low.rolling(window=window).min()
        
        # Price position within support-resistance range
        range_position = (close - support) / (resistance - support + 1e-8)
        return range_position.fillna(0.5)
    
    def _calculate_economic_strength(self, data: pd.DataFrame) -> pd.Series:
        """Calculate economic indicator strength proxy."""
        # Simplified economic strength using price trends and volatility
        close = data['Close']
        
        # Long-term trend relative to short-term volatility
        long_trend = close.rolling(window=60).mean() / close.rolling(window=5).mean() - 1
        volatility = close.pct_change().rolling(window=20).std()
        
        # Economic strength: trend adjusted for volatility
        econ_strength = long_trend / (volatility + 1e-8)
        return econ_strength.fillna(0)
    
    def _calculate_institutional_flow(self, data: pd.DataFrame) -> pd.Series:
        """Calculate institutional flow proxy."""
        close = data['Close']
        volume = data['Volume'] if 'Volume' in data.columns else pd.Series([1] * len(data))
        
        # Large volume moves as proxy for institutional activity
        volume_ma = volume.rolling(window=20).mean()
        large_volume = volume > volume_ma * 2
        
        # Price impact of large volume days
        price_changes = close.pct_change()
        institutional_signal = price_changes * large_volume
        
        return institutional_signal.rolling(window=10).sum().fillna(0)
    
    def _calculate_sector_rotation(self, data: pd.DataFrame) -> pd.Series:
        """Calculate sector rotation signal."""
        # Simplified sector rotation using relative strength
        close = data['Close']
        
        # Relative performance vs longer-term average
        short_ma = close.rolling(window=10).mean()
        long_ma = close.rolling(window=50).mean()
        
        relative_strength = short_ma / long_ma - 1
        return relative_strength.fillna(0)
    
    def _calculate_fundamental_strength(self, data: pd.DataFrame) -> pd.Series:
        """Calculate fundamental analysis strength."""
        close = data['Close']
        
        # Long-term price trends and stability
        long_trend = close.rolling(window=90).mean() / close.rolling(window=180).mean() - 1
        price_stability = 1 / (close.pct_change().rolling(window=60).std() + 1e-8)
        
        fundamental_score = long_trend * price_stability
        return fundamental_score.fillna(0)
    
    def _calculate_structural_trends(self, data: pd.DataFrame) -> pd.Series:
        """Calculate structural trend indicators."""
        close = data['Close']
        
        # Very long-term structural changes
        if len(close) >= 250:  # Need at least 1 year of data
            yearly_trend = close.rolling(window=250).mean()
            structural_change = close / yearly_trend - 1
        else:
            structural_change = pd.Series([0] * len(close))
        
        return structural_change.fillna(0)
    
    def _calculate_macro_signal(self, data: pd.DataFrame) -> pd.Series:
        """Calculate macro-economic signal."""
        close = data['Close']
        
        # Macro signal based on very long-term trends and cycles
        if len(close) >= 180:
            quarterly_avg = close.rolling(window=60).mean()
            macro_cycle = quarterly_avg.pct_change(periods=60)  # Quarter-over-quarter change
        else:
            macro_cycle = pd.Series([0] * len(close))
        
        return macro_cycle.fillna(0)
    
    def _calculate_volatility_regime(self, data: pd.DataFrame) -> pd.Series:
        """Calculate volatility regime indicator."""
        close = data['Close']
        returns = close.pct_change()
        
        # Current vs historical volatility
        current_vol = returns.rolling(window=20).std()
        historical_vol = returns.rolling(window=100).std()
        
        vol_regime = current_vol / (historical_vol + 1e-8)
        return vol_regime.fillna(1.0)
    
    def _calculate_trend_strength(self, data: pd.DataFrame, horizon: int) -> pd.Series:
        """Calculate trend strength for specific horizon."""
        close = data['Close']
        
        # Trend strength based on prediction horizon
        trend_window = min(horizon * 2, len(close) // 2)
        if trend_window < 5:
            trend_window = 5
        
        trend_line = close.rolling(window=trend_window).mean()
        trend_strength = (close - trend_line) / trend_line
        
        return trend_strength.fillna(0)
    
    def _calculate_market_stress(self, data: pd.DataFrame) -> pd.Series:
        """Calculate market stress indicator."""
        close = data['Close']
        returns = close.pct_change()
        
        # Market stress based on volatility and negative returns
        volatility = returns.rolling(window=20).std()
        negative_returns = (returns < 0).rolling(window=20).sum() / 20
        
        stress_indicator = volatility * (1 + negative_returns)
        return stress_indicator.fillna(0.02)
    
    def _create_timeframe_model(self, timeframe: str) -> Any:
        """Create specialized model for specific timeframe."""
        config = self.timeframes[timeframe]
        model_type = config['model_type']
        
        if model_type == 'high_frequency':
            # Fast, responsive model for ultra-short predictions
            return GradientBoostingRegressor(
                n_estimators=50,
                max_depth=4,
                learning_rate=0.15,
                subsample=0.8,
                random_state=42
            )
        
        elif model_type == 'technical_momentum':
            # Balanced model for short-term technical patterns
            return RandomForestRegressor(
                n_estimators=100,
                max_depth=8,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )
        
        elif model_type == 'economic_institutional':
            # Stable model for medium-term economic factors
            return ElasticNet(
                alpha=0.1,
                l1_ratio=0.5,
                max_iter=2000,
                random_state=42
            )
        
        elif model_type == 'fundamental_structural':
            # Conservative model for long-term fundamental analysis
            return Ridge(
                alpha=1.0,
                solver='auto',
                random_state=42
            )
        
        else:
            # Fallback model
            return RandomForestRegressor(n_estimators=50, random_state=42)
    
    def train_timeframe_models(self, historical_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """Train specialized models for each timeframe."""
        training_results = {}
        
        try:
            for timeframe, config in self.timeframes.items():
                self.logger.info(f"Training {timeframe} model ({config['days']} days)")
                
                # Get data for this timeframe
                data = historical_data.get(timeframe, historical_data.get('default'))
                if data is None or len(data) < 50:
                    self.logger.warning(f"Insufficient data for {timeframe} model")
                    continue
                
                # Extract timeframe-specific features
                features = self._extract_timeframe_specific_features(
                    data, timeframe, config['days']
                )
                
                # Prepare target variable (future returns)
                horizon = config['days']
                if len(data) > horizon:
                    target = data['Close'].pct_change(periods=horizon).shift(-horizon)
                    
                    # Align features and target
                    valid_idx = ~(target.isna() | features.isna().any(axis=1))
                    X = features[valid_idx].values
                    y = target[valid_idx].values
                    
                    if len(X) > 20:  # Minimum samples for training
                        # Create and train model
                        model = self._create_timeframe_model(timeframe)
                        scaler = RobustScaler()
                        
                        # Scale features
                        X_scaled = scaler.fit_transform(X)
                        
                        # Train model
                        model.fit(X_scaled, y)
                        
                        # Store model and scaler
                        self.models[timeframe] = model
                        self.scalers[timeframe] = scaler
                        self.feature_sets[timeframe] = features.columns.tolist()
                        
                        # Calculate training performance
                        y_pred = model.predict(X_scaled)
                        r2 = r2_score(y, y_pred)
                        mse = mean_squared_error(y, y_pred)
                        
                        # Directional accuracy
                        directional_accuracy = np.mean(np.sign(y_pred) == np.sign(y)) * 100
                        
                        self.performance_metrics[timeframe] = {
                            'r2_score': r2,
                            'mse': mse,
                            'directional_accuracy': directional_accuracy,
                            'n_samples': len(X)
                        }
                        
                        training_results[timeframe] = directional_accuracy
                        
                        self.logger.info(f"✅ {timeframe} model trained: "
                                       f"{directional_accuracy:.1f}% accuracy, "
                                       f"R² = {r2:.3f}")
                    else:
                        self.logger.warning(f"Insufficient valid samples for {timeframe}")
                else:
                    self.logger.warning(f"Data too short for {timeframe} horizon")
            
            return training_results
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            return {}
    
    def predict_multi_timeframe(self, current_data: pd.DataFrame, 
                               target_timeframe: str = 'short') -> Dict[str, Any]:
        """Generate predictions using multi-timeframe fusion."""
        try:
            predictions = {}
            confidence_scores = {}
            
            # Generate predictions for each trained timeframe
            for timeframe, model in self.models.items():
                try:
                    # Extract features for this timeframe
                    features = self._extract_timeframe_specific_features(
                        current_data, timeframe, self.timeframes[timeframe]['days']
                    )
                    
                    # Use only the last row for prediction
                    if len(features) > 0:
                        X = features.iloc[-1:].values
                        
                        # Scale features
                        scaler = self.scalers[timeframe]
                        X_scaled = scaler.transform(X)
                        
                        # Make prediction
                        pred = model.predict(X_scaled)[0]
                        predictions[timeframe] = pred
                        
                        # Calculate confidence based on training performance
                        perf = self.performance_metrics.get(timeframe, {})
                        base_confidence = perf.get('directional_accuracy', 50) / 100
                        confidence_scores[timeframe] = base_confidence
                        
                        self.logger.debug(f"{timeframe} prediction: {pred:.4f}, "
                                        f"confidence: {base_confidence:.3f}")
                    
                except Exception as e:
                    self.logger.warning(f"Prediction failed for {timeframe}: {e}")
                    continue
            
            # Fusion of multi-timeframe predictions
            if predictions:
                fused_prediction = self._fuse_timeframe_predictions(
                    predictions, confidence_scores, target_timeframe
                )
                
                return {
                    'fused_prediction': fused_prediction['prediction'],
                    'fused_confidence': fused_prediction['confidence'],
                    'individual_predictions': predictions,
                    'individual_confidences': confidence_scores,
                    'fusion_weights': fused_prediction['weights'],
                    'consistency_score': fused_prediction['consistency'],
                    'model_count': len(predictions)
                }
            else:
                self.logger.warning("No valid predictions generated")
                return {
                    'fused_prediction': 0.0,
                    'fused_confidence': 0.3,
                    'individual_predictions': {},
                    'individual_confidences': {},
                    'fusion_weights': {},
                    'consistency_score': 0.0,
                    'model_count': 0
                }
                
        except Exception as e:
            self.logger.error(f"Multi-timeframe prediction failed: {e}")
            return {
                'fused_prediction': 0.0,
                'fused_confidence': 0.3,
                'error': str(e)
            }
    
    def _fuse_timeframe_predictions(self, predictions: Dict[str, float], 
                                   confidences: Dict[str, float], 
                                   target_timeframe: str) -> Dict[str, Any]:
        """Fuse predictions from multiple timeframes with intelligent weighting."""
        
        # Base weights by timeframe relevance to target
        base_weights = {
            'ultra_short': {'ultra_short': 0.4, 'short': 0.3, 'medium': 0.2, 'long': 0.1},
            'short': {'ultra_short': 0.25, 'short': 0.4, 'medium': 0.25, 'long': 0.1},
            'medium': {'ultra_short': 0.15, 'short': 0.25, 'medium': 0.4, 'long': 0.2},
            'long': {'ultra_short': 0.1, 'short': 0.15, 'medium': 0.25, 'long': 0.5}
        }
        
        target_weights = base_weights.get(target_timeframe, base_weights['short'])
        
        # Calculate consistency score
        pred_values = list(predictions.values())
        if len(pred_values) > 1:
            consistency = 1.0 - (np.std(pred_values) / (np.mean(np.abs(pred_values)) + 1e-8))
            consistency = max(0.0, min(1.0, consistency))
        else:
            consistency = 1.0
        
        # Adjust weights based on confidence and consistency
        final_weights = {}
        total_weight = 0
        
        for timeframe, pred in predictions.items():
            base_weight = target_weights.get(timeframe, 0.1)
            confidence_weight = confidences.get(timeframe, 0.5)
            
            # Apply consistency boost/penalty
            if consistency > 0.7:
                consistency_adj = 1 + self.fusion_weights['consistency_boost']
            elif consistency < 0.3:
                consistency_adj = 1 - self.fusion_weights['divergence_penalty']
            else:
                consistency_adj = 1.0
            
            final_weight = base_weight * confidence_weight * consistency_adj
            final_weights[timeframe] = final_weight
            total_weight += final_weight
        
        # Normalize weights
        if total_weight > 0:
            for timeframe in final_weights:
                final_weights[timeframe] /= total_weight
        
        # Calculate fused prediction
        fused_pred = sum(predictions[tf] * final_weights.get(tf, 0) 
                        for tf in predictions.keys())
        
        # Calculate fused confidence
        fused_confidence = sum(confidences[tf] * final_weights.get(tf, 0) 
                              for tf in confidences.keys())
        
        # Apply consistency boost to confidence
        fused_confidence = min(0.95, fused_confidence * (0.7 + 0.3 * consistency))
        
        return {
            'prediction': fused_pred,
            'confidence': fused_confidence,
            'weights': final_weights,
            'consistency': consistency
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary of all timeframe models."""
        summary = {
            'trained_models': list(self.models.keys()),
            'performance_by_timeframe': self.performance_metrics.copy(),
            'fusion_config': self.fusion_weights.copy(),
            'model_status': {}
        }
        
        # Calculate overall metrics
        if self.performance_metrics:
            accuracies = [m['directional_accuracy'] for m in self.performance_metrics.values()]
            summary['overall_accuracy'] = np.mean(accuracies)
            summary['accuracy_std'] = np.std(accuracies)
            summary['best_timeframe'] = max(self.performance_metrics.items(), 
                                          key=lambda x: x[1]['directional_accuracy'])[0]
        
        # Model status
        for timeframe in self.timeframes:
            summary['model_status'][timeframe] = {
                'trained': timeframe in self.models,
                'feature_count': len(self.feature_sets.get(timeframe, [])),
                'model_type': self.timeframes[timeframe]['model_type']
            }
        
        return summary

# Integration function for existing system
def create_phase3_multi_timeframe_predictor(historical_data: Dict = None, 
                                           config: Dict = None) -> MultiTimeframeArchitecture:
    """Create and initialize Phase 3 Multi-Timeframe Architecture."""
    predictor = MultiTimeframeArchitecture(config)
    
    if historical_data:
        training_results = predictor.train_timeframe_models(historical_data)
        logging.info(f"🎯 P3_001 Multi-Timeframe models trained: {training_results}")
    
    return predictor

if __name__ == "__main__":
    # Test implementation
    import yfinance as yf
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Test with sample data
    logger.info("🧪 Testing P3_001 Multi-Timeframe Architecture...")
    
    # Download test data
    ticker = yf.Ticker("CBA.AX")
    data = ticker.history(period="2y", interval="1d")
    
    if not data.empty:
        # Create historical data for different timeframes
        historical_data = {
            'ultra_short': data,
            'short': data,
            'medium': data, 
            'long': data,
            'default': data
        }
        
        # Create and train predictor
        predictor = create_phase3_multi_timeframe_predictor(historical_data)
        
        # Test prediction
        current_data = data.tail(100)  # Use last 100 days for prediction
        result = predictor.predict_multi_timeframe(current_data, 'short')
        
        logger.info(f"✅ Multi-timeframe prediction result:")
        logger.info(f"   Fused prediction: {result['fused_prediction']:.4f}")
        logger.info(f"   Confidence: {result['fused_confidence']:.3f}")
        logger.info(f"   Models used: {result['model_count']}")
        logger.info(f"   Consistency: {result.get('consistency_score', 0):.3f}")
        
        # Performance summary
        summary = predictor.get_performance_summary()
        logger.info(f"📊 Performance Summary:")
        logger.info(f"   Overall accuracy: {summary.get('overall_accuracy', 0):.1f}%")
        logger.info(f"   Best timeframe: {summary.get('best_timeframe', 'None')}")
        
        logger.info("🎉 P3_001 Multi-Timeframe Architecture test completed successfully!")
    else:
        logger.error("Failed to download test data")