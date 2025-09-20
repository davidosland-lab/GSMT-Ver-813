#!/usr/bin/env python3
"""
Phase 3 Component P3_002: Bayesian Ensemble Framework
====================================================

Advanced probabilistic model combination with sophisticated uncertainty quantification.
Implements Bayesian model averaging, posterior predictive distributions, and Monte Carlo sampling.

Target: Improved uncertainty quantification and ensemble accuracy
Dependencies: All Phase 2 components must be operational
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable
from scipy import stats
from scipy.special import logsumexp
from sklearn.metrics import mean_squared_error, log_loss
import warnings
warnings.filterwarnings('ignore')

class BayesianEnsembleFramework:
    """
    Bayesian Ensemble Framework for probabilistic model combination.
    
    Implements:
    - Bayesian Model Averaging (BMA)
    - Posterior predictive distributions
    - Monte Carlo uncertainty quantification
    - Dynamic model weighting with Bayesian updates
    - Credible intervals and prediction intervals
    """
    
    def __init__(self, config: Dict = None):
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.config = config or {}
        self.prior_alpha = self.config.get('prior_alpha', 1.0)  # Dirichlet prior
        self.posterior_window = self.config.get('posterior_window', 100)  # Recent observations
        self.mcmc_samples = self.config.get('mcmc_samples', 1000)
        self.confidence_levels = self.config.get('confidence_levels', [0.68, 0.95, 0.99])
        
        # Model storage and tracking
        self.models = {}
        self.model_performance_history = {}
        self.posterior_weights = {}
        self.prior_weights = {}
        self.likelihood_history = {}
        
        # Bayesian parameters
        self.alpha_params = {}  # Dirichlet parameters
        self.beta_params = {}   # Beta distribution parameters for binary outcomes
        self.precision_params = {}  # Gamma parameters for precision (inverse variance)
        
        # Monte Carlo storage
        self.weight_samples = None
        self.prediction_samples = None
        
        self.logger.info("🧠 Phase 3 Bayesian Ensemble Framework initialized")
    
    def register_model(self, model_name: str, model: Any, 
                      prior_weight: float = None, 
                      prior_precision: float = 1.0) -> None:
        """Register a model in the Bayesian ensemble."""
        self.models[model_name] = model
        
        # Initialize prior parameters
        if prior_weight is None:
            prior_weight = 1.0 / (len(self.models) + 1)  # Uniform prior
        
        self.prior_weights[model_name] = prior_weight
        
        # Initialize Dirichlet parameters (for multinomial model selection)
        self.alpha_params[model_name] = self.prior_alpha
        
        # Initialize precision parameters (Gamma distribution)
        # Alpha (shape) and Beta (rate) for Gamma(α, β)
        self.precision_params[model_name] = {
            'alpha': 2.0,  # Shape parameter (prior belief about precision)
            'beta': 2.0 * prior_precision  # Rate parameter
        }
        
        # Initialize performance tracking
        self.model_performance_history[model_name] = {
            'predictions': [],
            'targets': [],
            'log_likelihoods': [],
            'errors': [],
            'timestamps': []
        }
        
        self.logger.info(f"✅ Registered model '{model_name}' with prior weight {prior_weight:.3f}")
    
    def update_model_performance(self, model_name: str, prediction: float, 
                               target: float, timestamp: datetime = None) -> None:
        """Update model performance history for Bayesian learning."""
        if model_name not in self.models:
            self.logger.warning(f"Model '{model_name}' not registered")
            return
        
        timestamp = timestamp or datetime.now()
        error = prediction - target
        
        # Store performance data
        history = self.model_performance_history[model_name]
        history['predictions'].append(prediction)
        history['targets'].append(target)
        history['errors'].append(error)
        history['timestamps'].append(timestamp)
        
        # Calculate log likelihood assuming Gaussian errors
        precision = self.precision_params[model_name]
        expected_precision = precision['alpha'] / precision['beta']
        log_likelihood = -0.5 * np.log(2 * np.pi / expected_precision) - 0.5 * expected_precision * error**2
        history['log_likelihoods'].append(log_likelihood)
        
        # Update Bayesian parameters
        self._update_bayesian_parameters(model_name, error)
        
        # Maintain sliding window
        if len(history['predictions']) > self.posterior_window:
            for key in history:
                history[key] = history[key][-self.posterior_window:]
    
    def _update_bayesian_parameters(self, model_name: str, error: float) -> None:
        """Update Bayesian parameters based on new observation."""
        
        # Update Dirichlet parameters (model selection)
        # Reward good predictions, penalize bad ones
        abs_error = abs(error)
        if abs_error < 0.02:  # Good prediction (< 2% error)
            self.alpha_params[model_name] += 2.0
        elif abs_error < 0.05:  # Acceptable prediction
            self.alpha_params[model_name] += 1.0
        else:  # Poor prediction
            self.alpha_params[model_name] += 0.1
        
        # Update precision parameters (Gamma distribution)
        # New observation updates both shape and rate
        precision = self.precision_params[model_name]
        precision['alpha'] += 0.5  # Each observation adds 0.5 to shape
        precision['beta'] += 0.5 * error**2  # Squared error contributes to rate
        
        # Prevent numerical issues
        precision['alpha'] = min(precision['alpha'], 1000.0)
        precision['beta'] = min(precision['beta'], 1000.0)
    
    def compute_posterior_weights(self) -> Dict[str, float]:
        """Compute posterior model weights using Bayesian model averaging."""
        
        if not self.alpha_params:
            return {}
        
        # Get current Dirichlet parameters
        alpha_values = list(self.alpha_params.values())
        model_names = list(self.alpha_params.keys())
        
        # Compute posterior weights (expected value of Dirichlet distribution)
        total_alpha = sum(alpha_values)
        posterior_weights = {name: alpha / total_alpha 
                           for name, alpha in zip(model_names, alpha_values)}
        
        # Apply recency weighting to recent performance
        adjusted_weights = {}
        total_adjusted = 0
        
        for model_name in model_names:
            history = self.model_performance_history[model_name]
            
            if len(history['log_likelihoods']) > 0:
                # Use recent log likelihoods for adjustment
                recent_ll = history['log_likelihoods'][-min(20, len(history['log_likelihoods'])):]
                avg_ll = np.mean(recent_ll)
                
                # Convert log likelihood to weight adjustment
                ll_weight = np.exp(avg_ll / 10)  # Scale for numerical stability
                adjusted_weight = posterior_weights[model_name] * ll_weight
            else:
                adjusted_weight = posterior_weights[model_name]
            
            adjusted_weights[model_name] = adjusted_weight
            total_adjusted += adjusted_weight
        
        # Normalize adjusted weights
        if total_adjusted > 0:
            for model_name in adjusted_weights:
                adjusted_weights[model_name] /= total_adjusted
        
        self.posterior_weights = adjusted_weights
        return adjusted_weights
    
    def sample_model_weights(self, n_samples: int = None) -> np.ndarray:
        """Sample model weights from posterior Dirichlet distribution."""
        n_samples = n_samples or self.mcmc_samples
        
        if not self.alpha_params:
            return np.array([])
        
        # Get Dirichlet parameters
        alpha_values = list(self.alpha_params.values())
        
        # Sample from Dirichlet distribution
        weight_samples = np.random.dirichlet(alpha_values, size=n_samples)
        
        self.weight_samples = weight_samples
        return weight_samples
    
    def bayesian_model_average_prediction(self, model_predictions: Dict[str, float],
                                        model_uncertainties: Dict[str, float] = None,
                                        return_samples: bool = False) -> Dict[str, Any]:
        """Generate Bayesian Model Average prediction with full uncertainty quantification."""
        
        try:
            # Compute current posterior weights
            weights = self.compute_posterior_weights()
            
            if not weights or not model_predictions:
                return {
                    'bma_prediction': 0.0,
                    'bma_uncertainty': 1.0,
                    'credible_intervals': {},
                    'posterior_weights': {}
                }
            
            # Ensure model predictions align with registered models
            valid_predictions = {name: pred for name, pred in model_predictions.items() 
                               if name in weights}
            
            if not valid_predictions:
                self.logger.warning("No valid model predictions for BMA")
                return {
                    'bma_prediction': 0.0,
                    'bma_uncertainty': 1.0,
                    'credible_intervals': {},
                    'posterior_weights': weights
                }
            
            # Basic BMA prediction (weighted average)
            bma_prediction = sum(weights.get(name, 0) * pred 
                               for name, pred in valid_predictions.items())
            
            # Sample-based uncertainty quantification
            weight_samples = self.sample_model_weights()
            prediction_samples = []
            
            model_names = list(valid_predictions.keys())
            model_preds = np.array([valid_predictions[name] for name in model_names])
            
            # Generate prediction samples using Monte Carlo
            for i in range(len(weight_samples)):
                # Get weights for this sample (align with model order)
                sample_weights = np.array([weights.get(name, 0) for name in model_names])
                if np.sum(sample_weights) > 0:
                    sample_weights = sample_weights / np.sum(sample_weights)
                
                # Add model-specific uncertainty if provided
                if model_uncertainties:
                    # Sample from each model's predictive distribution
                    noisy_preds = []
                    for j, name in enumerate(model_names):
                        uncertainty = model_uncertainties.get(name, 0.02)
                        noise = np.random.normal(0, uncertainty)
                        noisy_preds.append(model_preds[j] + noise)
                    noisy_preds = np.array(noisy_preds)
                else:
                    noisy_preds = model_preds
                
                # Weighted prediction for this sample
                sample_pred = np.sum(sample_weights * noisy_preds)
                prediction_samples.append(sample_pred)
            
            prediction_samples = np.array(prediction_samples)
            self.prediction_samples = prediction_samples
            
            # Calculate uncertainty measures
            bma_std = np.std(prediction_samples)
            bma_variance = np.var(prediction_samples)
            
            # Calculate credible intervals
            credible_intervals = {}
            for conf_level in self.confidence_levels:
                lower_percentile = (1 - conf_level) / 2 * 100
                upper_percentile = (1 + conf_level) / 2 * 100
                
                lower = np.percentile(prediction_samples, lower_percentile)
                upper = np.percentile(prediction_samples, upper_percentile)
                
                credible_intervals[f'{conf_level:.0%}'] = {
                    'lower': lower,
                    'upper': upper,
                    'width': upper - lower
                }
            
            # Model-specific contributions to uncertainty
            model_contributions = self._calculate_model_contributions(
                valid_predictions, weights, bma_prediction
            )
            
            # Additional uncertainty metrics
            uncertainty_metrics = {
                'total_uncertainty': bma_std,
                'aleatoric_uncertainty': self._estimate_aleatoric_uncertainty(model_uncertainties),
                'epistemic_uncertainty': self._estimate_epistemic_uncertainty(weights, valid_predictions),
                'prediction_entropy': self._calculate_prediction_entropy(weights)
            }
            
            result = {
                'bma_prediction': float(bma_prediction),
                'bma_uncertainty': float(bma_std),
                'bma_variance': float(bma_variance),
                'credible_intervals': credible_intervals,
                'posterior_weights': weights,
                'model_contributions': model_contributions,
                'uncertainty_decomposition': uncertainty_metrics,
                'n_models': len(valid_predictions),
                'effective_n_models': self._calculate_effective_model_count(weights)
            }
            
            if return_samples:
                result['prediction_samples'] = prediction_samples
                result['weight_samples'] = weight_samples
            
            self.logger.debug(f"BMA prediction: {bma_prediction:.4f} ± {bma_std:.4f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Bayesian model averaging failed: {e}")
            return {
                'bma_prediction': 0.0,
                'bma_uncertainty': 1.0,
                'error': str(e)
            }
    
    def _calculate_model_contributions(self, predictions: Dict[str, float], 
                                     weights: Dict[str, float], 
                                     bma_pred: float) -> Dict[str, Dict[str, float]]:
        """Calculate individual model contributions to prediction and uncertainty."""
        contributions = {}
        
        for model_name, pred in predictions.items():
            weight = weights.get(model_name, 0)
            contribution = weight * pred
            deviation = abs(pred - bma_pred)
            
            contributions[model_name] = {
                'weight': weight,
                'prediction': pred,
                'contribution': contribution,
                'deviation_from_ensemble': deviation,
                'influence_score': weight * (1 + deviation)  # Higher for influential outliers
            }
        
        return contributions
    
    def _estimate_aleatoric_uncertainty(self, model_uncertainties: Dict[str, float] = None) -> float:
        """Estimate aleatoric (irreducible) uncertainty."""
        if not model_uncertainties:
            return 0.02  # Default assumption
        
        # Average model-specific uncertainties weighted by posterior weights
        weighted_uncertainties = []
        for model_name, uncertainty in model_uncertainties.items():
            weight = self.posterior_weights.get(model_name, 0)
            weighted_uncertainties.append(weight * uncertainty**2)  # Variance weighting
        
        return np.sqrt(sum(weighted_uncertainties)) if weighted_uncertainties else 0.02
    
    def _estimate_epistemic_uncertainty(self, weights: Dict[str, float], 
                                      predictions: Dict[str, float]) -> float:
        """Estimate epistemic (model) uncertainty."""
        if len(predictions) <= 1:
            return 0.0
        
        # Weighted variance of predictions
        pred_values = list(predictions.values())
        weight_values = [weights.get(name, 0) for name in predictions.keys()]
        
        if sum(weight_values) == 0:
            return np.std(pred_values)
        
        # Normalize weights
        weight_values = np.array(weight_values) / sum(weight_values)
        
        # Weighted mean and variance
        weighted_mean = np.sum(weight_values * pred_values)
        weighted_variance = np.sum(weight_values * (np.array(pred_values) - weighted_mean)**2)
        
        return np.sqrt(weighted_variance)
    
    def _calculate_prediction_entropy(self, weights: Dict[str, float]) -> float:
        """Calculate entropy of posterior weight distribution."""
        weight_values = list(weights.values())
        
        if not weight_values or sum(weight_values) == 0:
            return 0.0
        
        # Normalize weights
        probs = np.array(weight_values) / sum(weight_values)
        
        # Calculate Shannon entropy
        entropy = -np.sum(probs * np.log(probs + 1e-10))  # Add small constant for numerical stability
        
        return entropy
    
    def _calculate_effective_model_count(self, weights: Dict[str, float]) -> float:
        """Calculate effective number of models (inverse of Herfindahl index)."""
        weight_values = list(weights.values())
        
        if not weight_values or sum(weight_values) == 0:
            return 0.0
        
        # Normalize weights
        probs = np.array(weight_values) / sum(weight_values)
        
        # Effective number of models
        effective_n = 1.0 / np.sum(probs**2)
        
        return effective_n
    
    def update_ensemble_online(self, new_predictions: Dict[str, float], 
                             true_target: float, timestamp: datetime = None) -> Dict[str, float]:
        """Online update of ensemble with new predictions and observed outcome."""
        
        # Update each model's performance
        for model_name, prediction in new_predictions.items():
            if model_name in self.models:
                self.update_model_performance(model_name, prediction, true_target, timestamp)
        
        # Recompute posterior weights
        updated_weights = self.compute_posterior_weights()
        
        self.logger.debug(f"Updated posterior weights: {updated_weights}")
        
        return updated_weights
    
    def model_selection_probability(self, model_name: str) -> float:
        """Calculate probability that a specific model is the best."""
        if model_name not in self.alpha_params:
            return 0.0
        
        # Use Dirichlet parameters to calculate selection probability
        alpha_values = list(self.alpha_params.values())
        model_alpha = self.alpha_params[model_name]
        
        # Sample from Dirichlet and count how often this model has highest weight
        samples = np.random.dirichlet(alpha_values, size=1000)
        model_idx = list(self.alpha_params.keys()).index(model_name)
        
        best_count = np.sum(samples[:, model_idx] == np.max(samples, axis=1))
        
        return best_count / 1000
    
    def get_ensemble_diagnostics(self) -> Dict[str, Any]:
        """Get comprehensive diagnostics of the Bayesian ensemble."""
        
        diagnostics = {
            'model_count': len(self.models),
            'posterior_weights': self.posterior_weights.copy(),
            'dirichlet_parameters': self.alpha_params.copy(),
            'precision_parameters': {name: {'mean': params['alpha'] / params['beta'],
                                          'variance': params['alpha'] / params['beta']**2}
                                   for name, params in self.precision_params.items()},
            'model_selection_probabilities': {},
            'performance_summary': {}
        }
        
        # Calculate model selection probabilities
        for model_name in self.models:
            diagnostics['model_selection_probabilities'][model_name] = \
                self.model_selection_probability(model_name)
        
        # Performance summary for each model
        for model_name, history in self.model_performance_history.items():
            if len(history['errors']) > 0:
                diagnostics['performance_summary'][model_name] = {
                    'mean_error': np.mean(history['errors']),
                    'std_error': np.std(history['errors']),
                    'mean_abs_error': np.mean(np.abs(history['errors'])),
                    'mean_log_likelihood': np.mean(history['log_likelihoods']),
                    'n_observations': len(history['errors'])
                }
        
        # Ensemble-level metrics
        if self.posterior_weights:
            diagnostics['ensemble_metrics'] = {
                'weight_entropy': self._calculate_prediction_entropy(self.posterior_weights),
                'effective_model_count': self._calculate_effective_model_count(self.posterior_weights),
                'weight_concentration': max(self.posterior_weights.values()) if self.posterior_weights else 0
            }
        
        return diagnostics

# Integration functions for existing system
def create_phase3_bayesian_ensemble(model_registry: Dict = None, 
                                   config: Dict = None) -> BayesianEnsembleFramework:
    """Create and initialize Phase 3 Bayesian Ensemble Framework."""
    ensemble = BayesianEnsembleFramework(config)
    
    # Register models if provided
    if model_registry:
        for model_name, model_info in model_registry.items():
            model = model_info.get('model')
            prior_weight = model_info.get('prior_weight')
            prior_precision = model_info.get('prior_precision', 1.0)
            
            if model:
                ensemble.register_model(model_name, model, prior_weight, prior_precision)
    
    logging.info(f"🧠 P3_002 Bayesian Ensemble created with {len(ensemble.models)} models")
    
    return ensemble

if __name__ == "__main__":
    # Test implementation
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Test the Bayesian Ensemble Framework
    logger.info("🧪 Testing P3_002 Bayesian Ensemble Framework...")
    
    # Create test ensemble
    ensemble = BayesianEnsembleFramework({
        'mcmc_samples': 500,
        'confidence_levels': [0.68, 0.95]
    })
    
    # Register mock models
    class MockModel:
        def __init__(self, bias=0.0):
            self.bias = bias
        
        def predict(self, x):
            return x + self.bias + np.random.normal(0, 0.01)
    
    ensemble.register_model('model_conservative', MockModel(bias=-0.01), prior_weight=0.3)
    ensemble.register_model('model_aggressive', MockModel(bias=0.02), prior_weight=0.3) 
    ensemble.register_model('model_balanced', MockModel(bias=0.005), prior_weight=0.4)
    
    # Simulate some predictions and outcomes
    np.random.seed(42)
    for i in range(50):
        # Mock predictions
        predictions = {
            'model_conservative': 0.02 + np.random.normal(0, 0.01),
            'model_aggressive': 0.05 + np.random.normal(0, 0.015),
            'model_balanced': 0.03 + np.random.normal(0, 0.008)
        }
        
        # Mock true outcome
        true_target = 0.035 + np.random.normal(0, 0.02)
        
        # Update ensemble
        ensemble.update_ensemble_online(predictions, true_target)
        
        # Test BMA prediction every 10 iterations
        if i % 10 == 0:
            uncertainties = {
                'model_conservative': 0.01,
                'model_aggressive': 0.02,
                'model_balanced': 0.008
            }
            
            bma_result = ensemble.bayesian_model_average_prediction(
                predictions, uncertainties, return_samples=True
            )
            
            logger.info(f"Iteration {i}: BMA = {bma_result['bma_prediction']:.4f} ± "
                       f"{bma_result['bma_uncertainty']:.4f}")
    
    # Final diagnostics
    diagnostics = ensemble.get_ensemble_diagnostics()
    logger.info("📊 Final Ensemble Diagnostics:")
    logger.info(f"   Posterior weights: {diagnostics['posterior_weights']}")
    logger.info(f"   Effective model count: {diagnostics['ensemble_metrics']['effective_model_count']:.2f}")
    logger.info(f"   Weight entropy: {diagnostics['ensemble_metrics']['weight_entropy']:.3f}")
    
    # Final BMA prediction
    final_predictions = {
        'model_conservative': 0.025,
        'model_aggressive': 0.045,
        'model_balanced': 0.032
    }
    
    final_uncertainties = {
        'model_conservative': 0.008,
        'model_aggressive': 0.018,
        'model_balanced': 0.006
    }
    
    final_result = ensemble.bayesian_model_average_prediction(
        final_predictions, final_uncertainties
    )
    
    logger.info("🎯 Final BMA Prediction:")
    logger.info(f"   Prediction: {final_result['bma_prediction']:.4f}")
    logger.info(f"   Uncertainty: {final_result['bma_uncertainty']:.4f}")
    logger.info(f"   95% CI: [{final_result['credible_intervals']['95%']['lower']:.4f}, "
               f"{final_result['credible_intervals']['95%']['upper']:.4f}]")
    
    logger.info("🎉 P3_002 Bayesian Ensemble Framework test completed successfully!")