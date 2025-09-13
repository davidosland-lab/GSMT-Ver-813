#!/usr/bin/env python3
"""
Quick improvement patch for ensemble predictor based on backtesting results
This addresses the critical issues identified in the backtest
"""

import numpy as np
from typing import Dict, Tuple
from advanced_ensemble_predictor import PredictionHorizon

def improved_combine_predictions(predictions: Dict[str, float], 
                                uncertainties: Dict[str, float],
                                horizon: PredictionHorizon) -> Tuple[float, Dict[str, float]]:
    """Improved prediction combination based on backtesting results"""
    
    if not predictions:
        return 0.0, {}
    
    # Performance-based weights from backtesting
    performance_weights = {
        'quantile_regression': 0.45,  # Best performer: 29.4% accuracy
        'random_forest': 0.30,        # Moderate performer: 21.4% accuracy  
        'arima': 0.15,               # No data but keep for diversification
        'lstm': 0.10                 # Worst performer: 0% accuracy
    }
    
    # Combine performance weights with uncertainty weighting
    combined_weights = {}
    total_weight = 0
    
    for model_name in predictions.keys():
        # Get base performance weight
        base_weight = performance_weights.get(model_name, 0.25)
        
        # Apply uncertainty penalty (higher uncertainty = lower weight)
        uncertainty_penalty = 1.0 / (uncertainties.get(model_name, 0.5) + 0.01)
        
        # Combine weights
        combined_weight = base_weight * uncertainty_penalty
        combined_weights[model_name] = combined_weight
        total_weight += combined_weight
    
    # Normalize weights
    if total_weight > 0:
        for model_name in combined_weights:
            combined_weights[model_name] /= total_weight
    
    # Calculate ensemble prediction
    ensemble_prediction = 0.0
    for model_name, prediction in predictions.items():
        weight = combined_weights.get(model_name, 0.0)
        ensemble_prediction += weight * prediction
    
    return ensemble_prediction, combined_weights

def improved_uncertainty_scoring(uncertainties: Dict[str, float]) -> float:
    """Improved uncertainty scoring that accounts for model reliability"""
    
    if not uncertainties:
        return 0.5
    
    # Weight uncertainties by model performance
    performance_reliability = {
        'quantile_regression': 0.8,  # Most reliable
        'random_forest': 0.6,        # Moderately reliable
        'arima': 0.5,               # Average reliability
        'lstm': 0.2                 # Least reliable (needs fixing)
    }
    
    weighted_uncertainty = 0.0
    total_reliability = 0.0
    
    for model_name, uncertainty in uncertainties.items():
        reliability = performance_reliability.get(model_name, 0.5)
        weighted_uncertainty += uncertainty * reliability
        total_reliability += reliability
    
    # Return weighted average uncertainty
    if total_reliability > 0:
        return weighted_uncertainty / total_reliability
    else:
        return np.mean(list(uncertainties.values()))

print("🔧 Ensemble Improvement Patch Created")
print("💡 This patch addresses critical issues found in backtesting:")
print("   • Increases Quantile Regression weight (best performer)")
print("   • Reduces LSTM weight (0% accuracy in backtest)")
print("   • Improves uncertainty-based weighting")
print("   • Adds performance-based model reliability scoring")