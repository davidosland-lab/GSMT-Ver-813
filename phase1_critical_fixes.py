#!/usr/bin/env python3
"""
Phase 1: Critical Bug Fixes - Immediate Implementation
Priority: URGENT - Fix 0% LSTM accuracy and critical ensemble issues
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import logging
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LSTMModelFix:
    """Critical fix for LSTM implementation showing 0% accuracy"""
    
    def __init__(self):
        self.scaler_features = StandardScaler()
        self.scaler_target = StandardScaler()
        self.sequence_length = 60  # 60 days lookback
        self.model = None
        
    def prepare_lstm_data(self, data: pd.DataFrame, target_col: str = 'Close') -> Tuple[np.ndarray, np.ndarray]:
        """
        CRITICAL FIX: Proper LSTM data preparation
        Previous issue: Incorrect sequence windowing and scaling
        """
        
        logger.info("🔧 Fixing LSTM data preparation pipeline...")
        
        # Ensure data is sorted by date
        data = data.sort_index()
        
        # Feature engineering for LSTM
        features = []
        
        # Price-based features
        features.append(data['Close'].pct_change().fillna(0))  # Returns
        features.append(data['Volume'].pct_change().fillna(0))  # Volume changes
        features.append((data['High'] - data['Low']) / data['Close'])  # Daily volatility
        features.append((data['Close'] - data['Open']) / data['Open'])  # Intraday return
        
        # Technical indicators
        if len(data) >= 20:
            features.append(data['Close'].rolling(20).mean() / data['Close'] - 1)  # MA20 ratio
            features.append(data['Close'].rolling(20).std() / data['Close'])  # Volatility ratio
        
        # Combine features
        feature_matrix = np.column_stack(features)
        
        # Remove NaN rows
        feature_matrix = feature_matrix[~np.isnan(feature_matrix).any(axis=1)]
        
        # Prepare target (next day return)
        target = data[target_col].pct_change().shift(-1).fillna(0).values
        target = target[~np.isnan(feature_matrix).any(axis=1)]
        target = target[:len(feature_matrix)]  # Ensure same length
        
        logger.info(f"✅ Prepared LSTM data: {feature_matrix.shape[0]} samples, {feature_matrix.shape[1]} features")
        
        return feature_matrix, target
        
    def create_sequences(self, features: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        CRITICAL FIX: Proper sequence creation for LSTM
        Previous issue: Incorrect windowing causing prediction failures
        """
        
        logger.info(f"🔧 Creating LSTM sequences with length {self.sequence_length}...")
        
        if len(features) < self.sequence_length + 1:
            logger.error(f"❌ Insufficient data: need {self.sequence_length + 1}, got {len(features)}")
            return np.array([]), np.array([])
        
        # Scale features and target
        features_scaled = self.scaler_features.fit_transform(features)
        target_scaled = self.scaler_target.fit_transform(target.reshape(-1, 1)).flatten()
        
        X_sequences = []
        y_sequences = []
        
        for i in range(self.sequence_length, len(features_scaled)):
            # Feature sequence: [i-sequence_length:i]
            X_sequences.append(features_scaled[i-self.sequence_length:i])
            # Target: next day return
            y_sequences.append(target_scaled[i])
            
        X_sequences = np.array(X_sequences)
        y_sequences = np.array(y_sequences)
        
        logger.info(f"✅ Created {len(X_sequences)} LSTM sequences: {X_sequences.shape}")
        
        return X_sequences, y_sequences
    
    def build_improved_lstm_model(self, input_shape: Tuple[int, int]) -> Any:
        """
        CRITICAL FIX: Build properly configured LSTM model
        Previous issue: Poor architecture and configuration
        """
        
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
            from tensorflow.keras.optimizers import Adam
            from tensorflow.keras.regularizers import l2
            
            logger.info("🔧 Building improved LSTM architecture...")
            
            model = Sequential([
                # First LSTM layer with return sequences
                LSTM(50, return_sequences=True, input_shape=input_shape,
                     kernel_regularizer=l2(0.001), recurrent_regularizer=l2(0.001)),
                BatchNormalization(),
                Dropout(0.2),
                
                # Second LSTM layer
                LSTM(25, return_sequences=False,
                     kernel_regularizer=l2(0.001), recurrent_regularizer=l2(0.001)),
                BatchNormalization(), 
                Dropout(0.2),
                
                # Dense layers
                Dense(25, activation='relu', kernel_regularizer=l2(0.001)),
                Dropout(0.1),
                Dense(1, activation='linear')  # Regression output
            ])
            
            # Compile with appropriate optimizer and loss
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='huber',  # More robust than MSE
                metrics=['mae']
            )
            
            logger.info("✅ LSTM model built successfully")
            return model
            
        except ImportError:
            logger.error("❌ TensorFlow not available, using simplified implementation")
            return None
    
    def generate_lstm_prediction(self, recent_data: np.ndarray) -> Tuple[float, float]:
        """
        CRITICAL FIX: Generate properly scaled LSTM prediction
        Previous issue: Incorrect inverse transformation
        """
        
        if self.model is None:
            logger.warning("⚠️ LSTM model not trained, returning neutral prediction")
            return 0.0, 0.5  # Return, uncertainty
        
        try:
            # Prepare input sequence
            if len(recent_data) < self.sequence_length:
                logger.warning("⚠️ Insufficient recent data for LSTM prediction")
                return 0.0, 0.8  # High uncertainty
            
            # Get last sequence_length data points
            input_sequence = recent_data[-self.sequence_length:]
            
            # Scale input (using fitted scaler)
            input_scaled = self.scaler_features.transform(input_sequence)
            
            # Reshape for model input
            input_reshaped = input_scaled.reshape(1, self.sequence_length, -1)
            
            # Generate prediction
            prediction_scaled = self.model.predict(input_reshaped, verbose=0)[0][0]
            
            # CRITICAL FIX: Proper inverse transformation
            prediction = self.scaler_target.inverse_transform([[prediction_scaled]])[0][0]
            
            # Calculate uncertainty (higher uncertainty for extreme predictions)
            uncertainty = min(0.8, abs(prediction) * 2)  # Scale uncertainty with prediction magnitude
            
            logger.info(f"✅ LSTM prediction: {prediction:.4f}, uncertainty: {uncertainty:.3f}")
            
            return float(prediction), float(uncertainty)
            
        except Exception as e:
            logger.error(f"❌ LSTM prediction failed: {e}")
            return 0.0, 0.9  # High uncertainty on error

class PerformanceBasedWeighting:
    """Implementation of performance-based ensemble weighting"""
    
    def __init__(self):
        # Weights based on backtesting results
        self.performance_weights = {
            'quantile_regression': 0.45,  # Best: 29.4% accuracy
            'random_forest': 0.30,        # Moderate: 21.4% accuracy
            'arima': 0.15,               # Diversification
            'lstm': 0.10                 # Worst: 0% accuracy (post-fix)
        }
        
        self.performance_history = {
            'quantile_regression': 0.294,
            'random_forest': 0.214,
            'arima': 0.20,  # Conservative estimate
            'lstm': 0.05    # Very low due to bugs
        }
        
    def calculate_adaptive_weights(self, 
                                  predictions: Dict[str, float],
                                  uncertainties: Dict[str, float],
                                  recent_performance: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        Calculate ensemble weights based on performance and uncertainty
        CRITICAL FIX: Replace simple uncertainty weighting with performance-based approach
        """
        
        logger.info("🔧 Calculating performance-based ensemble weights...")
        
        if not predictions:
            return {}
        
        # Use recent performance if available, otherwise use historical
        performance_scores = recent_performance or self.performance_history
        
        weights = {}
        total_weight = 0
        
        for model_name in predictions.keys():
            # Base performance weight
            perf_weight = self.performance_weights.get(model_name, 0.25)
            
            # Performance reliability factor
            reliability = performance_scores.get(model_name, 0.2)
            
            # Uncertainty penalty (lower uncertainty = higher confidence)
            uncertainty = uncertainties.get(model_name, 0.5)
            uncertainty_factor = 1.0 / (uncertainty + 0.1)
            
            # Combined weight calculation
            combined_weight = perf_weight * (1 + reliability) * uncertainty_factor
            
            weights[model_name] = combined_weight
            total_weight += combined_weight
        
        # Normalize weights
        if total_weight > 0:
            for model_name in weights:
                weights[model_name] /= total_weight
        
        # Log weight distribution
        for model_name, weight in weights.items():
            logger.info(f"  📊 {model_name}: {weight:.1%} weight (perf: {performance_scores.get(model_name, 0):.1%})")
        
        return weights

class ImprovedConfidenceCalibration:
    """Enhanced confidence calibration addressing 35.2% reliability issue"""
    
    def __init__(self):
        self.calibration_history = []
        self.temperature = 1.0  # Temperature scaling parameter
        
    def calibrate_confidence(self, 
                           ensemble_prediction: float,
                           model_predictions: Dict[str, float],
                           model_uncertainties: Dict[str, float],
                           model_weights: Dict[str, float]) -> float:
        """
        CRITICAL FIX: Improved confidence scoring
        Previous issue: Overconfident predictions with poor reliability
        """
        
        # Model agreement factor (higher agreement = higher confidence)
        prediction_values = list(model_predictions.values())
        if len(prediction_values) > 1:
            prediction_std = np.std(prediction_values)
            agreement_factor = 1.0 / (1.0 + prediction_std * 10)  # Penalize disagreement
        else:
            agreement_factor = 0.5
        
        # Weighted uncertainty (lower = higher confidence)
        weighted_uncertainty = sum(
            weight * uncertainty 
            for model, (weight, uncertainty) in zip(
                model_weights.items(),
                [(model_weights.get(model, 0), model_uncertainties.get(model, 0.5)) 
                 for model in model_predictions.keys()]
            )
        )
        
        # Market condition adjustment (higher volatility = lower confidence)
        volatility_penalty = min(0.3, abs(ensemble_prediction) * 5)  # Cap at 30% penalty
        
        # Base confidence from inverse weighted uncertainty
        base_confidence = 1.0 - weighted_uncertainty
        
        # Apply adjustments
        adjusted_confidence = base_confidence * agreement_factor * (1.0 - volatility_penalty)
        
        # Temperature scaling (reduce overconfidence)
        calibrated_confidence = 1.0 / (1.0 + np.exp(-adjusted_confidence / self.temperature))
        
        # Ensure reasonable bounds
        final_confidence = max(0.1, min(0.95, calibrated_confidence))
        
        logger.info(f"🎯 Confidence calibration: base={base_confidence:.3f}, "
                   f"agreement={agreement_factor:.3f}, final={final_confidence:.3f}")
        
        return final_confidence
    
    def update_calibration(self, predicted_confidence: float, actual_outcome: bool):
        """Update calibration based on actual outcomes"""
        
        self.calibration_history.append({
            'predicted_confidence': predicted_confidence,
            'actual_outcome': actual_outcome,
            'timestamp': datetime.now()
        })
        
        # Keep last 100 calibration points
        if len(self.calibration_history) > 100:
            self.calibration_history = self.calibration_history[-100:]
        
        # Adjust temperature based on calibration error
        if len(self.calibration_history) >= 10:
            recent_predictions = [p['predicted_confidence'] for p in self.calibration_history[-10:]]
            recent_outcomes = [p['actual_outcome'] for p in self.calibration_history[-10:]]
            
            # Calculate calibration error
            avg_predicted = np.mean(recent_predictions)
            avg_actual = np.mean(recent_outcomes)
            
            calibration_error = abs(avg_predicted - avg_actual)
            
            # Adjust temperature (increase if overconfident, decrease if underconfident)
            if avg_predicted > avg_actual + 0.1:  # Overconfident
                self.temperature *= 1.1
            elif avg_predicted < avg_actual - 0.1:  # Underconfident
                self.temperature *= 0.9
            
            # Keep temperature in reasonable range
            self.temperature = max(0.5, min(2.0, self.temperature))
            
            logger.info(f"📊 Calibration update: error={calibration_error:.3f}, "
                       f"temperature={self.temperature:.3f}")

def generate_phase1_implementation_guide() -> str:
    """Generate detailed implementation guide for Phase 1"""
    
    return """
🚨 PHASE 1: CRITICAL BUG FIXES - IMPLEMENTATION GUIDE
=====================================================

IMMEDIATE ACTIONS REQUIRED (Week 1-2):

1. LSTM IMPLEMENTATION FIX (P1_001) - 🚨 CRITICAL
   ================================================
   
   Root Cause: LSTM showing 0% accuracy due to:
   • Incorrect sequence windowing
   • Poor feature scaling pipeline  
   • Wrong prediction inverse transformation
   • Inadequate model architecture
   
   Implementation Steps:
   ✅ Use LSTMModelFix class above
   ✅ Fix data preparation pipeline (proper windowing)
   ✅ Implement StandardScaler for features and targets
   ✅ Create proper sequence generation (60-day lookback)
   ✅ Build improved LSTM architecture with regularization
   ✅ Fix inverse transformation for predictions
   
   Testing Requirements:
   • Minimum 40% accuracy on backtesting
   • Proper uncertainty quantification
   • Stable predictions across different market conditions
   
   Success Criteria: LSTM accuracy >40% (vs current 0%)

2. PERFORMANCE-BASED WEIGHTS (P1_002) - 🚨 CRITICAL  
   ==================================================
   
   Current Issue: Random Forest overweighted (53.9%) despite poor performance
   
   Implementation Steps:
   ✅ Use PerformanceBasedWeighting class above
   ✅ Set Quantile Regression weight to 45% (best performer)
   ✅ Reduce Random Forest weight to 30%
   ✅ Set LSTM weight to 10% (post-fix validation)
   ✅ Maintain ARIMA at 15% for diversification
   
   Success Criteria: Proper weight distribution reflecting actual performance

3. CONFIDENCE CALIBRATION FIX (P1_003) - 🔴 HIGH
   ================================================
   
   Current Issue: Only 35.2% confidence reliability (overconfident)
   
   Implementation Steps:
   ✅ Use ImprovedConfidenceCalibration class above
   ✅ Implement model agreement factor
   ✅ Add temperature scaling to reduce overconfidence
   ✅ Include volatility-based confidence adjustment
   ✅ Add calibration feedback loop
   
   Success Criteria: >70% confidence reliability

4. FEATURE ENGINEERING ENHANCEMENT (P1_004) - 🔴 HIGH
   =====================================================
   
   Current Issue: Limited feature set affecting all models
   
   New Features to Add:
   • Technical Indicators: Bollinger Bands, MACD, RSI, Stochastic
   • Market Microstructure: Bid-ask spreads, order flow
   • Volatility Measures: GARCH, realized volatility
   • Momentum Indicators: Various timeframe momentum
   • Market Correlation: AUD/USD, commodity correlation
   
   Success Criteria: Balanced feature importance across indicators

TESTING PROTOCOL:
================

Week 1 Testing:
• Fix LSTM and validate >40% accuracy
• Implement new weights and measure ensemble improvement
• Basic confidence calibration testing

Week 2 Testing:  
• Complete feature engineering integration
• Full ensemble backtesting with all fixes
• Target: >50% overall ensemble accuracy

EXPECTED IMPROVEMENTS:
=====================

Current State → Phase 1 Target:
• Overall Accuracy: 23.7% → 50%+
• LSTM Accuracy: 0% → 40%+
• Confidence Reliability: 35.2% → 70%+
• Quantile Weight: 14.7% → 45%
• Random Forest Weight: 53.9% → 30%

RISK MITIGATION:
===============

• Daily accuracy monitoring during fixes
• Rollback capability for each component
• Isolated testing before ensemble integration
• Performance validation at each step

This Phase 1 implementation will establish the foundation for
subsequent architecture optimization and advanced features.
"""

if __name__ == "__main__":
    print("🚨 Phase 1: Critical Bug Fixes - Implementation Ready")
    print("=" * 60)
    
    # Initialize fix components
    lstm_fix = LSTMModelFix()
    weight_calculator = PerformanceBasedWeighting()
    confidence_calibrator = ImprovedConfidenceCalibration()
    
    print("✅ LSTM Fix module initialized")
    print("✅ Performance-based weighting ready") 
    print("✅ Confidence calibration system ready")
    print("✅ Enhanced feature engineering planned")
    
    print("\n🎯 Ready to begin critical fixes:")
    print("   1. Fix LSTM implementation (0% → 40%+ accuracy)")
    print("   2. Rebalance ensemble weights (performance-based)")
    print("   3. Improve confidence calibration (35% → 70%+ reliability)")
    print("   4. Enhance feature engineering pipeline")
    
    # Generate implementation guide
    guide = generate_phase1_implementation_guide()
    print(guide)
    
    # Save implementation guide
    with open("phase1_implementation_guide.md", "w") as f:
        f.write(guide)
    
    print("\n💾 Phase 1 implementation guide saved to: phase1_implementation_guide.md")
    print("🚀 Ready to begin immediate critical fixes!")