#!/usr/bin/env python3
"""
Phase 3 Component P3_007: Advanced Risk Management Framework
==========================================================

Comprehensive risk management system with sophisticated metrics and controls.
Implements advanced risk quantification, dynamic position sizing, and
real-time risk monitoring for prediction-based trading strategies.

Features:
- Value at Risk (VaR) modeling with multiple methods
- Expected Shortfall (Conditional VaR) calculations  
- Dynamic position sizing algorithms
- Drawdown protection mechanisms
- Stress testing and scenario analysis
- Real-time risk monitoring and alerts

Target: Risk-adjusted performance optimization with 20-30% better Sharpe ratios
Dependencies: P3-001 to P3-006 operational
"""

import numpy as np
import pandas as pd
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict, deque
from scipy import stats
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf, EmpiricalCovariance
import warnings
warnings.filterwarnings('ignore')

class RiskMeasure(Enum):
    """Risk measurement types."""
    VAR_HISTORICAL = "var_historical"
    VAR_PARAMETRIC = "var_parametric"
    VAR_MONTE_CARLO = "var_monte_carlo"
    EXPECTED_SHORTFALL = "expected_shortfall"
    MAX_DRAWDOWN = "max_drawdown"
    VOLATILITY = "volatility"
    BETA = "beta"
    TRACKING_ERROR = "tracking_error"

class PositionSizingMethod(Enum):
    """Position sizing methods."""
    FIXED_FRACTIONAL = "fixed_fractional"
    KELLY_CRITERION = "kelly_criterion"
    RISK_PARITY = "risk_parity"
    MAX_DIVERSIFICATION = "max_diversification"
    MINIMUM_VARIANCE = "minimum_variance"
    VOLATILITY_TARGETING = "volatility_targeting"

@dataclass
class RiskMetrics:
    """Comprehensive risk metrics container."""
    var_95: float  # 95% Value at Risk
    var_99: float  # 99% Value at Risk
    expected_shortfall_95: float  # Expected Shortfall at 95%
    expected_shortfall_99: float  # Expected Shortfall at 99%
    max_drawdown: float
    current_drawdown: float
    volatility_annual: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    beta: Optional[float] = None
    tracking_error: Optional[float] = None
    information_ratio: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class PositionSizing:
    """Position sizing recommendations."""
    recommended_weights: Dict[str, float]
    risk_contribution: Dict[str, float]
    expected_portfolio_volatility: float
    expected_portfolio_return: float
    max_individual_weight: float
    sizing_method: str
    confidence_level: float
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class RiskAlert:
    """Risk alert data structure."""
    alert_id: str
    alert_type: str  # 'VAR_BREACH', 'DRAWDOWN_LIMIT', 'VOLATILITY_SPIKE', etc.
    severity: str  # 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'
    message: str
    current_value: float
    threshold_value: float
    recommended_action: str
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class StressTestScenario:
    """Stress test scenario definition."""
    scenario_name: str
    market_shocks: Dict[str, float]  # Asset -> shock magnitude
    correlation_changes: Dict[Tuple[str, str], float]  # (Asset1, Asset2) -> new correlation
    volatility_multipliers: Dict[str, float]  # Asset -> volatility multiplier
    description: str

class VaRCalculator:
    """
    Value at Risk calculator with multiple methodologies.
    
    Implements Historical, Parametric, and Monte Carlo VaR calculations
    along with Expected Shortfall (Conditional VaR) computations.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def calculate_historical_var(self, 
                               returns: np.ndarray, 
                               confidence_level: float = 0.95) -> Tuple[float, float]:
        """Calculate Historical Value at Risk and Expected Shortfall."""
        
        if len(returns) == 0:
            return 0.0, 0.0
        
        # Sort returns in ascending order
        sorted_returns = np.sort(returns)
        
        # Calculate VaR as percentile
        var_index = int((1 - confidence_level) * len(sorted_returns))
        var = -sorted_returns[var_index] if var_index < len(sorted_returns) else 0.0
        
        # Calculate Expected Shortfall (average of tail losses)
        tail_losses = sorted_returns[:var_index + 1]
        expected_shortfall = -np.mean(tail_losses) if len(tail_losses) > 0 else 0.0
        
        return var, expected_shortfall
    
    def calculate_parametric_var(self, 
                               returns: np.ndarray,
                               confidence_level: float = 0.95,
                               distribution: str = 'normal') -> Tuple[float, float]:
        """Calculate Parametric VaR assuming specified distribution."""
        
        if len(returns) == 0:
            return 0.0, 0.0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if distribution == 'normal':
            # Normal distribution VaR
            z_score = stats.norm.ppf(1 - confidence_level)
            var = -(mean_return + z_score * std_return)
            
            # Expected Shortfall for normal distribution
            expected_shortfall = -(mean_return - std_return * stats.norm.pdf(z_score) / (1 - confidence_level))
            
        elif distribution == 't':
            # Student's t-distribution (more conservative for fat tails)
            df = len(returns) - 1  # Degrees of freedom
            t_score = stats.t.ppf(1 - confidence_level, df)
            var = -(mean_return + t_score * std_return)
            
            # Expected Shortfall for t-distribution (approximation)
            expected_shortfall = var * 1.2  # Conservative multiplier
            
        else:
            # Fallback to normal
            return self.calculate_parametric_var(returns, confidence_level, 'normal')
        
        return max(0, var), max(0, expected_shortfall)
    
    def calculate_monte_carlo_var(self, 
                                returns: np.ndarray,
                                confidence_level: float = 0.95,
                                n_simulations: int = 10000) -> Tuple[float, float]:
        """Calculate Monte Carlo VaR using simulated returns."""
        
        if len(returns) == 0:
            return 0.0, 0.0
        
        # Fit distribution to historical returns
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        # Generate random scenarios
        np.random.seed(42)  # For reproducible results
        simulated_returns = np.random.normal(mean_return, std_return, n_simulations)
        
        # Calculate VaR and ES from simulated returns
        return self.calculate_historical_var(simulated_returns, confidence_level)

class AdvancedRiskManager:
    """
    Advanced Risk Management Framework for prediction-based strategies.
    
    Provides comprehensive risk measurement, monitoring, and control
    capabilities including VaR, position sizing, and stress testing.
    """
    
    def __init__(self, config: Dict = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Risk calculation components
        self.var_calculator = VaRCalculator()
        
        # Risk monitoring
        self.risk_history = deque(maxlen=1000)
        self.position_history = deque(maxlen=1000)
        self.alert_history = deque(maxlen=500)
        
        # Risk limits and thresholds
        self.risk_limits = {
            'max_portfolio_var_95': self.config.get('max_var_95', 0.05),  # 5% daily VaR
            'max_portfolio_var_99': self.config.get('max_var_99', 0.08),  # 8% daily VaR
            'max_drawdown_limit': self.config.get('max_drawdown', 0.15),  # 15% max drawdown
            'max_position_size': self.config.get('max_position', 0.25),   # 25% max position
            'volatility_threshold': self.config.get('vol_threshold', 0.30), # 30% annual vol
            'correlation_threshold': self.config.get('corr_threshold', 0.80) # 80% max correlation
        }
        
        # Stress test scenarios
        self.stress_scenarios = self._create_stress_scenarios()
        
        self.logger.info("🛡️ Advanced Risk Management Framework initialized")
    
    def calculate_portfolio_risk_metrics(self, 
                                       returns: pd.DataFrame,
                                       weights: Dict[str, float],
                                       benchmark_returns: pd.Series = None) -> RiskMetrics:
        """Calculate comprehensive portfolio risk metrics."""
        
        try:
            # Calculate portfolio returns
            portfolio_returns = self._calculate_portfolio_returns(returns, weights)
            
            if len(portfolio_returns) == 0:
                return self._empty_risk_metrics()
            
            # VaR calculations
            var_95, es_95 = self.var_calculator.calculate_historical_var(portfolio_returns, 0.95)
            var_99, es_99 = self.var_calculator.calculate_historical_var(portfolio_returns, 0.99)
            
            # Drawdown calculations
            max_dd, current_dd = self._calculate_drawdowns(portfolio_returns)
            
            # Performance metrics
            annual_vol = np.std(portfolio_returns) * np.sqrt(252)
            annual_return = np.mean(portfolio_returns) * 252
            
            # Risk-adjusted ratios
            sharpe_ratio = annual_return / annual_vol if annual_vol > 0 else 0
            sortino_ratio = self._calculate_sortino_ratio(portfolio_returns)
            calmar_ratio = annual_return / abs(max_dd) if max_dd != 0 else 0
            
            # Benchmark-relative metrics
            beta, tracking_error, info_ratio = None, None, None
            if benchmark_returns is not None:
                beta = self._calculate_beta(portfolio_returns, benchmark_returns)
                tracking_error = self._calculate_tracking_error(portfolio_returns, benchmark_returns)
                info_ratio = self._calculate_information_ratio(portfolio_returns, benchmark_returns)
            
            metrics = RiskMetrics(
                var_95=var_95,
                var_99=var_99,
                expected_shortfall_95=es_95,
                expected_shortfall_99=es_99,
                max_drawdown=max_dd,
                current_drawdown=current_dd,
                volatility_annual=annual_vol,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                calmar_ratio=calmar_ratio,
                beta=beta,
                tracking_error=tracking_error,
                information_ratio=info_ratio
            )
            
            # Store in history
            self.risk_history.append(metrics)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Risk metric calculation failed: {e}")
            return self._empty_risk_metrics()
    
    def optimize_position_sizing(self, 
                               expected_returns: Dict[str, float],
                               risk_estimates: Dict[str, float],
                               correlation_matrix: pd.DataFrame,
                               method: PositionSizingMethod = PositionSizingMethod.RISK_PARITY) -> PositionSizing:
        """Optimize position sizing using specified method."""
        
        try:
            assets = list(expected_returns.keys())
            n_assets = len(assets)
            
            if n_assets == 0:
                return self._empty_position_sizing()
            
            # Convert to arrays for optimization
            mu = np.array([expected_returns[asset] for asset in assets])
            sigma = np.array([risk_estimates[asset] for asset in assets])
            
            # Ensure correlation matrix alignment
            corr_matrix = correlation_matrix.reindex(index=assets, columns=assets).fillna(0)
            np.fill_diagonal(corr_matrix.values, 1.0)
            
            # Calculate covariance matrix
            cov_matrix = np.outer(sigma, sigma) * corr_matrix.values
            
            # Optimize weights based on method
            if method == PositionSizingMethod.RISK_PARITY:
                weights = self._risk_parity_optimization(sigma, cov_matrix)
            elif method == PositionSizingMethod.MINIMUM_VARIANCE:
                weights = self._minimum_variance_optimization(cov_matrix)
            elif method == PositionSizingMethod.MAX_DIVERSIFICATION:
                weights = self._maximum_diversification_optimization(sigma, cov_matrix)
            elif method == PositionSizingMethod.VOLATILITY_TARGETING:
                weights = self._volatility_targeting_optimization(sigma, target_vol=0.15)
            elif method == PositionSizingMethod.KELLY_CRITERION:
                weights = self._kelly_criterion_optimization(mu, cov_matrix)
            else:
                # Default to equal weights
                weights = np.ones(n_assets) / n_assets
            
            # Apply position limits
            weights = self._apply_position_limits(weights)
            
            # Calculate portfolio metrics
            portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
            portfolio_return = np.dot(weights, mu)
            
            # Calculate risk contributions
            risk_contrib = self._calculate_risk_contributions(weights, cov_matrix)
            
            # Create result
            weight_dict = {asset: weight for asset, weight in zip(assets, weights)}
            risk_contrib_dict = {asset: contrib for asset, contrib in zip(assets, risk_contrib)}
            
            sizing = PositionSizing(
                recommended_weights=weight_dict,
                risk_contribution=risk_contrib_dict,
                expected_portfolio_volatility=portfolio_vol,
                expected_portfolio_return=portfolio_return,
                max_individual_weight=np.max(weights),
                sizing_method=method.value,
                confidence_level=0.95
            )
            
            self.position_history.append(sizing)
            return sizing
            
        except Exception as e:
            self.logger.error(f"Position sizing optimization failed: {e}")
            return self._empty_position_sizing()
    
    def _risk_parity_optimization(self, sigma: np.ndarray, cov_matrix: np.ndarray) -> np.ndarray:
        """Optimize for risk parity (equal risk contribution)."""
        
        n_assets = len(sigma)
        
        def risk_parity_objective(weights):
            # Portfolio variance
            port_var = np.dot(weights, np.dot(cov_matrix, weights))
            
            # Marginal risk contributions
            marginal_contrib = np.dot(cov_matrix, weights)
            
            # Risk contributions
            risk_contrib = weights * marginal_contrib / port_var
            
            # Target equal risk contribution
            target_contrib = 1.0 / n_assets
            
            # Minimize sum of squared deviations from target
            return np.sum((risk_contrib - target_contrib) ** 2)
        
        # Constraints
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
        bounds = tuple((0.0, self.risk_limits['max_position_size']) for _ in range(n_assets))
        
        # Initial guess
        x0 = np.ones(n_assets) / n_assets
        
        # Optimize
        result = minimize(risk_parity_objective, x0, method='SLSQP', 
                         bounds=bounds, constraints=constraints)
        
        return result.x if result.success else x0
    
    def _minimum_variance_optimization(self, cov_matrix: np.ndarray) -> np.ndarray:
        """Optimize for minimum variance portfolio."""
        
        n_assets = cov_matrix.shape[0]
        
        def min_var_objective(weights):
            return np.dot(weights, np.dot(cov_matrix, weights))
        
        # Constraints
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
        bounds = tuple((0.0, self.risk_limits['max_position_size']) for _ in range(n_assets))
        
        # Initial guess
        x0 = np.ones(n_assets) / n_assets
        
        # Optimize
        result = minimize(min_var_objective, x0, method='SLSQP',
                         bounds=bounds, constraints=constraints)
        
        return result.x if result.success else x0
    
    def _maximum_diversification_optimization(self, sigma: np.ndarray, cov_matrix: np.ndarray) -> np.ndarray:
        """Optimize for maximum diversification ratio."""
        
        n_assets = len(sigma)
        
        def max_div_objective(weights):
            # Weighted average volatility
            weighted_vol = np.dot(weights, sigma)
            
            # Portfolio volatility
            port_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
            
            # Maximize diversification ratio (minimize negative ratio)
            return -weighted_vol / port_vol if port_vol > 0 else 0
        
        # Constraints
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
        bounds = tuple((0.0, self.risk_limits['max_position_size']) for _ in range(n_assets))
        
        # Initial guess
        x0 = np.ones(n_assets) / n_assets
        
        # Optimize
        result = minimize(max_div_objective, x0, method='SLSQP',
                         bounds=bounds, constraints=constraints)
        
        return result.x if result.success else x0
    
    def _volatility_targeting_optimization(self, sigma: np.ndarray, target_vol: float = 0.15) -> np.ndarray:
        """Optimize weights to target specific portfolio volatility."""
        
        # Simple volatility targeting: inverse volatility weighting scaled to target
        inv_vol_weights = 1.0 / sigma
        inv_vol_weights /= np.sum(inv_vol_weights)  # Normalize
        
        # Scale to target volatility (simplified approach)
        portfolio_vol = np.sqrt(np.sum((inv_vol_weights * sigma) ** 2))
        scaling_factor = target_vol / portfolio_vol if portfolio_vol > 0 else 1.0
        
        # Apply scaling and position limits
        scaled_weights = inv_vol_weights * min(scaling_factor, 1.0)
        
        return self._apply_position_limits(scaled_weights)
    
    def _kelly_criterion_optimization(self, mu: np.ndarray, cov_matrix: np.ndarray) -> np.ndarray:
        """Optimize using Kelly Criterion for growth maximization."""
        
        try:
            # Kelly formula: f* = Σ^(-1) * μ
            # Where f* is optimal fraction, Σ is covariance matrix, μ is expected returns
            
            inv_cov = np.linalg.inv(cov_matrix)
            kelly_weights = np.dot(inv_cov, mu)
            
            # Normalize to sum to 1 and apply limits
            kelly_weights = np.maximum(kelly_weights, 0)  # No short selling
            if np.sum(kelly_weights) > 0:
                kelly_weights /= np.sum(kelly_weights)
            
            return self._apply_position_limits(kelly_weights)
            
        except np.linalg.LinAlgError:
            # Fallback to equal weights if covariance matrix is singular
            return np.ones(len(mu)) / len(mu)
    
    def _apply_position_limits(self, weights: np.ndarray) -> np.ndarray:
        """Apply position size limits to weights."""
        
        max_weight = self.risk_limits['max_position_size']
        
        # Cap individual weights
        weights = np.minimum(weights, max_weight)
        
        # Renormalize
        if np.sum(weights) > 0:
            weights /= np.sum(weights)
        
        return weights
    
    def _calculate_risk_contributions(self, weights: np.ndarray, cov_matrix: np.ndarray) -> np.ndarray:
        """Calculate individual asset risk contributions to portfolio risk."""
        
        portfolio_var = np.dot(weights, np.dot(cov_matrix, weights))
        
        if portfolio_var > 0:
            marginal_contrib = np.dot(cov_matrix, weights)
            risk_contrib = weights * marginal_contrib / portfolio_var
        else:
            risk_contrib = np.zeros_like(weights)
        
        return risk_contrib
    
    def monitor_risk_limits(self, current_metrics: RiskMetrics) -> List[RiskAlert]:
        """Monitor risk metrics against defined limits and generate alerts."""
        
        alerts = []
        
        # VaR limit checks
        if current_metrics.var_95 > self.risk_limits['max_portfolio_var_95']:
            alerts.append(RiskAlert(
                alert_id=f"VAR95_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                alert_type="VAR_BREACH",
                severity="HIGH",
                message=f"Portfolio VaR 95% ({current_metrics.var_95:.2%}) exceeds limit ({self.risk_limits['max_portfolio_var_95']:.2%})",
                current_value=current_metrics.var_95,
                threshold_value=self.risk_limits['max_portfolio_var_95'],
                recommended_action="Reduce position sizes or hedge exposure"
            ))
        
        if current_metrics.var_99 > self.risk_limits['max_portfolio_var_99']:
            alerts.append(RiskAlert(
                alert_id=f"VAR99_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                alert_type="VAR_BREACH",
                severity="CRITICAL",
                message=f"Portfolio VaR 99% ({current_metrics.var_99:.2%}) exceeds limit ({self.risk_limits['max_portfolio_var_99']:.2%})",
                current_value=current_metrics.var_99,
                threshold_value=self.risk_limits['max_portfolio_var_99'],
                recommended_action="Immediate risk reduction required"
            ))
        
        # Drawdown limit check
        if abs(current_metrics.current_drawdown) > self.risk_limits['max_drawdown_limit']:
            alerts.append(RiskAlert(
                alert_id=f"DD_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                alert_type="DRAWDOWN_LIMIT",
                severity="HIGH",
                message=f"Current drawdown ({abs(current_metrics.current_drawdown):.2%}) exceeds limit ({self.risk_limits['max_drawdown_limit']:.2%})",
                current_value=abs(current_metrics.current_drawdown),
                threshold_value=self.risk_limits['max_drawdown_limit'],
                recommended_action="Consider position reduction or trading halt"
            ))
        
        # Volatility spike check
        if current_metrics.volatility_annual > self.risk_limits['volatility_threshold']:
            alerts.append(RiskAlert(
                alert_id=f"VOL_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                alert_type="VOLATILITY_SPIKE",
                severity="MEDIUM",
                message=f"Portfolio volatility ({current_metrics.volatility_annual:.2%}) exceeds threshold ({self.risk_limits['volatility_threshold']:.2%})",
                current_value=current_metrics.volatility_annual,
                threshold_value=self.risk_limits['volatility_threshold'],
                recommended_action="Review position sizing and consider volatility targeting"
            ))
        
        # Store alerts
        self.alert_history.extend(alerts)
        
        return alerts
    
    def run_stress_tests(self, 
                        current_positions: Dict[str, float],
                        asset_prices: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """Run comprehensive stress tests on current portfolio."""
        
        stress_results = {}
        
        for scenario in self.stress_scenarios:
            scenario_result = self._apply_stress_scenario(
                scenario, current_positions, asset_prices
            )
            stress_results[scenario.scenario_name] = scenario_result
        
        return stress_results
    
    def _apply_stress_scenario(self, 
                             scenario: StressTestScenario,
                             positions: Dict[str, float],
                             prices: Dict[str, float]) -> Dict[str, float]:
        """Apply single stress scenario to portfolio."""
        
        results = {
            'portfolio_pnl': 0.0,
            'portfolio_pnl_pct': 0.0,
            'worst_asset_pnl': 0.0,
            'assets_affected': 0
        }
        
        total_portfolio_value = sum(positions.get(asset, 0) * prices.get(asset, 0) 
                                  for asset in positions)
        
        if total_portfolio_value == 0:
            return results
        
        total_pnl = 0.0
        worst_asset_pnl = 0.0
        affected_assets = 0
        
        for asset, position_value in positions.items():
            if asset in scenario.market_shocks:
                shock = scenario.market_shocks[asset]
                asset_pnl = position_value * shock
                total_pnl += asset_pnl
                
                if asset_pnl < worst_asset_pnl:
                    worst_asset_pnl = asset_pnl
                
                affected_assets += 1
        
        results.update({
            'portfolio_pnl': total_pnl,
            'portfolio_pnl_pct': total_pnl / total_portfolio_value,
            'worst_asset_pnl': worst_asset_pnl,
            'assets_affected': affected_assets
        })
        
        return results
    
    def _create_stress_scenarios(self) -> List[StressTestScenario]:
        """Create predefined stress test scenarios."""
        
        scenarios = []
        
        # Market crash scenario (2008-style)
        scenarios.append(StressTestScenario(
            scenario_name="Market_Crash_2008",
            market_shocks={
                '^GSPC': -0.20,    # S&P 500 down 20%
                '^FTSE': -0.18,    # FTSE down 18%
                '^N225': -0.15,    # Nikkei down 15%
                'TLT': 0.10,       # Bonds up 10%
                'GLD': 0.05,       # Gold up 5%
                'USO': -0.35       # Oil down 35%
            },
            correlation_changes={},
            volatility_multipliers={'default': 3.0},
            description="Severe market crash with flight to quality"
        ))
        
        # COVID-style shock
        scenarios.append(StressTestScenario(
            scenario_name="Pandemic_Shock_2020",
            market_shocks={
                '^GSPC': -0.30,    # Initial COVID crash
                'XLE': -0.50,      # Energy sector collapse
                'XLF': -0.35,      # Financial sector stress
                'TLT': 0.15,       # Bond rally
                'GLD': 0.08        # Gold rally
            },
            correlation_changes={},
            volatility_multipliers={'default': 4.0},
            description="Pandemic-style systematic shock"
        ))
        
        # Interest rate shock
        scenarios.append(StressTestScenario(
            scenario_name="Interest_Rate_Shock",
            market_shocks={
                'TLT': -0.25,      # Long bonds crash
                'IEF': -0.15,      # Medium bonds down
                'XLF': 0.10,       # Banks benefit
                '^GSPC': -0.08,    # Equities down moderately
                'GLD': -0.12       # Gold down
            },
            correlation_changes={},
            volatility_multipliers={'bonds': 2.0},
            description="Sudden interest rate spike"
        ))
        
        # Currency crisis
        scenarios.append(StressTestScenario(
            scenario_name="Currency_Crisis",
            market_shocks={
                'EURUSD=X': -0.15,  # EUR weakness
                'GBPUSD=X': -0.20,  # GBP weakness  
                'JPYUSD=X': 0.10,   # JPY strength (safe haven)
                '^GSPC': -0.05,     # Modest equity decline
                'GLD': 0.12         # Gold strength
            },
            correlation_changes={},
            volatility_multipliers={'forex': 2.5},
            description="Major currency crisis with safe haven flows"
        ))
        
        return scenarios
    
    def _calculate_portfolio_returns(self, 
                                   returns: pd.DataFrame,
                                   weights: Dict[str, float]) -> np.ndarray:
        """Calculate portfolio returns from asset returns and weights."""
        
        try:
            # Align weights with available returns
            available_assets = [asset for asset in weights.keys() if asset in returns.columns]
            
            if not available_assets:
                return np.array([])
            
            # Normalize weights for available assets
            total_weight = sum(weights[asset] for asset in available_assets)
            if total_weight == 0:
                return np.array([])
            
            normalized_weights = {asset: weights[asset] / total_weight for asset in available_assets}
            
            # Calculate weighted returns
            portfolio_returns = np.zeros(len(returns))
            
            for asset in available_assets:
                asset_returns = returns[asset].fillna(0).values
                portfolio_returns += normalized_weights[asset] * asset_returns
            
            return portfolio_returns
            
        except Exception as e:
            self.logger.error(f"Portfolio return calculation failed: {e}")
            return np.array([])
    
    def _calculate_drawdowns(self, returns: np.ndarray) -> Tuple[float, float]:
        """Calculate maximum drawdown and current drawdown."""
        
        if len(returns) == 0:
            return 0.0, 0.0
        
        # Calculate cumulative returns
        cumulative = np.cumprod(1 + returns)
        
        # Calculate running maximum
        running_max = np.maximum.accumulate(cumulative)
        
        # Calculate drawdowns
        drawdowns = (cumulative - running_max) / running_max
        
        max_drawdown = np.min(drawdowns)
        current_drawdown = drawdowns[-1]
        
        return max_drawdown, current_drawdown
    
    def _calculate_sortino_ratio(self, returns: np.ndarray) -> float:
        """Calculate Sortino ratio (return/downside deviation)."""
        
        if len(returns) == 0:
            return 0.0
        
        annual_return = np.mean(returns) * 252
        
        # Downside deviation (negative returns only)
        negative_returns = returns[returns < 0]
        if len(negative_returns) > 0:
            downside_deviation = np.std(negative_returns) * np.sqrt(252)
        else:
            downside_deviation = 0.0
        
        return annual_return / downside_deviation if downside_deviation > 0 else 0.0
    
    def _calculate_beta(self, portfolio_returns: np.ndarray, benchmark_returns: pd.Series) -> float:
        """Calculate portfolio beta relative to benchmark."""
        
        try:
            # Align data
            aligned_benchmark = benchmark_returns.reindex(
                range(len(portfolio_returns)), method='ffill'
            ).fillna(0).values
            
            # Calculate covariance and variance
            covariance = np.cov(portfolio_returns, aligned_benchmark)[0, 1]
            benchmark_variance = np.var(aligned_benchmark)
            
            return covariance / benchmark_variance if benchmark_variance > 0 else 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_tracking_error(self, portfolio_returns: np.ndarray, benchmark_returns: pd.Series) -> float:
        """Calculate tracking error (standard deviation of excess returns)."""
        
        try:
            # Align data
            aligned_benchmark = benchmark_returns.reindex(
                range(len(portfolio_returns)), method='ffill'
            ).fillna(0).values
            
            # Calculate excess returns
            excess_returns = portfolio_returns - aligned_benchmark
            
            # Annualized tracking error
            return np.std(excess_returns) * np.sqrt(252)
            
        except Exception:
            return 0.0
    
    def _calculate_information_ratio(self, portfolio_returns: np.ndarray, benchmark_returns: pd.Series) -> float:
        """Calculate information ratio (excess return / tracking error)."""
        
        try:
            tracking_error = self._calculate_tracking_error(portfolio_returns, benchmark_returns)
            
            if tracking_error == 0:
                return 0.0
            
            # Align data for excess return calculation
            aligned_benchmark = benchmark_returns.reindex(
                range(len(portfolio_returns)), method='ffill'
            ).fillna(0).values
            
            excess_returns = portfolio_returns - aligned_benchmark
            annualized_excess_return = np.mean(excess_returns) * 252
            
            return annualized_excess_return / tracking_error
            
        except Exception:
            return 0.0
    
    def _empty_risk_metrics(self) -> RiskMetrics:
        """Return empty risk metrics for error cases."""
        return RiskMetrics(
            var_95=0.0, var_99=0.0, expected_shortfall_95=0.0, expected_shortfall_99=0.0,
            max_drawdown=0.0, current_drawdown=0.0, volatility_annual=0.0,
            sharpe_ratio=0.0, sortino_ratio=0.0, calmar_ratio=0.0
        )
    
    def _empty_position_sizing(self) -> PositionSizing:
        """Return empty position sizing for error cases."""
        return PositionSizing(
            recommended_weights={}, risk_contribution={},
            expected_portfolio_volatility=0.0, expected_portfolio_return=0.0,
            max_individual_weight=0.0, sizing_method="none", confidence_level=0.0
        )
    
    def get_risk_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive risk dashboard data."""
        
        current_metrics = self.risk_history[-1] if self.risk_history else self._empty_risk_metrics()
        recent_alerts = list(self.alert_history)[-10:]  # Last 10 alerts
        
        return {
            'current_metrics': asdict(current_metrics),
            'recent_alerts': [asdict(alert) for alert in recent_alerts],
            'risk_limits': self.risk_limits,
            'stress_scenarios': [scenario.scenario_name for scenario in self.stress_scenarios],
            'historical_var_95': [m.var_95 for m in list(self.risk_history)[-50:]],
            'historical_drawdown': [m.current_drawdown for m in list(self.risk_history)[-50:]],
            'risk_trend_analysis': self._analyze_risk_trends()
        }
    
    def _analyze_risk_trends(self) -> Dict[str, str]:
        """Analyze trends in risk metrics."""
        
        if len(self.risk_history) < 10:
            return {'status': 'insufficient_data'}
        
        recent_metrics = list(self.risk_history)[-10:]
        
        # Analyze VaR trend
        var_values = [m.var_95 for m in recent_metrics]
        var_trend = "increasing" if var_values[-1] > var_values[0] else "decreasing"
        
        # Analyze volatility trend
        vol_values = [m.volatility_annual for m in recent_metrics]
        vol_trend = "increasing" if vol_values[-1] > vol_values[0] else "decreasing"
        
        # Analyze Sharpe ratio trend
        sharpe_values = [m.sharpe_ratio for m in recent_metrics]
        sharpe_trend = "improving" if sharpe_values[-1] > sharpe_values[0] else "declining"
        
        return {
            'var_trend': var_trend,
            'volatility_trend': vol_trend,
            'sharpe_trend': sharpe_trend,
            'overall_assessment': self._assess_overall_risk_status(recent_metrics)
        }
    
    def _assess_overall_risk_status(self, recent_metrics: List[RiskMetrics]) -> str:
        """Assess overall risk status."""
        
        latest = recent_metrics[-1]
        
        # Count risk limit breaches
        breaches = 0
        if latest.var_95 > self.risk_limits['max_portfolio_var_95']:
            breaches += 1
        if abs(latest.current_drawdown) > self.risk_limits['max_drawdown_limit']:
            breaches += 2  # Drawdown gets higher weight
        if latest.volatility_annual > self.risk_limits['volatility_threshold']:
            breaches += 1
        
        if breaches >= 3:
            return "high_risk"
        elif breaches >= 1:
            return "moderate_risk"
        else:
            return "low_risk"

# Global instance for integration
advanced_risk_manager = AdvancedRiskManager()