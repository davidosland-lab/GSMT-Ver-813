#!/usr/bin/env python3
"""
Enhanced ASX SPI Market Prediction System with Backtesting
Integrates ASX SPI futures data as a key parameter for improved prediction accuracy
Includes comprehensive backtesting framework to validate model efficiency

Key Features:
- ASX SPI 200 futures data integration
- Multi-factor prediction models  
- Backtesting framework with performance metrics
- Walk-forward analysis for model validation
- Risk-adjusted returns calculation
- Sharpe ratio and other efficiency metrics
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import asyncio
import json
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PredictionHorizon(Enum):
    """Prediction horizons for ASX SPI analysis"""
    INTRADAY = "1d"      # Next day prediction
    SHORT_TERM = "5d"    # 5-day prediction
    MEDIUM_TERM = "15d"  # 15-day prediction
    LONG_TERM = "30d"    # 30-day prediction

class BacktestMetric(Enum):
    """Backtesting performance metrics"""
    RETURNS = "total_returns"
    SHARPE_RATIO = "sharpe_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    WIN_RATE = "win_rate"
    RMSE = "rmse"
    MAE = "mae"
    R_SQUARED = "r_squared"
    VOLATILITY = "volatility"

@dataclass
class ASXSPIDataPoint:
    """ASX SPI data point with related market factors"""
    timestamp: datetime
    spi_price: float
    spi_volume: int
    asx200_price: float
    aord_price: float
    volatility_index: float
    futures_basis: float  # SPI - ASX200 spread
    market_sentiment: str
    
@dataclass
class PredictionResult:
    """Enhanced prediction result with ASX SPI integration"""
    symbol: str
    target_date: datetime
    predicted_price: float
    confidence_interval: Tuple[float, float]  # (lower, upper)
    probability_up: float
    probability_down: float
    spi_influence: float  # How much SPI data influenced the prediction
    risk_score: float
    model_used: str
    features_used: List[str]
    
@dataclass
class BacktestResult:
    """Comprehensive backtesting results"""
    start_date: datetime
    end_date: datetime
    total_predictions: int
    correct_predictions: int
    accuracy: float
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    avg_prediction_error: float
    rmse: float
    r_squared: float
    volatility: float
    metrics: Dict[str, float] = field(default_factory=dict)
    prediction_history: List[Dict] = field(default_factory=list)

class ASXSPIPredictionSystem:
    """Advanced prediction system with ASX SPI integration and backtesting"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        self.spi_symbol = "^AXJO"  # Use ASX 200 as proxy for SPI (real SPI futures may not be available via yfinance)
        self.asx200_symbol = "^AXJO"  # ASX 200 index
        self.aord_symbol = "^AORD"   # All Ordinaries
        
        # Initialize models for different horizons
        self._initialize_models()
        logger.info("🚀 ASX SPI Prediction System initialized with backtesting capabilities")
    
    def _initialize_models(self):
        """Initialize prediction models for different time horizons"""
        for horizon in PredictionHorizon:
            # Use different models for different horizons
            if horizon == PredictionHorizon.INTRADAY:
                self.models[horizon.value] = GradientBoostingRegressor(
                    n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42
                )
            elif horizon in [PredictionHorizon.SHORT_TERM, PredictionHorizon.MEDIUM_TERM]:
                self.models[horizon.value] = RandomForestRegressor(
                    n_estimators=300, max_depth=8, random_state=42, n_jobs=-1
                )
            else:  # LONG_TERM
                self.models[horizon.value] = LinearRegression()
            
            self.scalers[horizon.value] = RobustScaler()
    
    async def collect_asx_spi_data(self, 
                                  symbol: str, 
                                  days_back: int = 252) -> pd.DataFrame:
        """Collect ASX SPI and related market data"""
        try:
            logger.info(f"📊 Collecting ASX SPI data for {symbol} ({days_back} days)")
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            # Collect data for all related symbols
            symbols = [self.spi_symbol, self.asx200_symbol, self.aord_symbol, symbol]
            data_frames = {}
            
            for sym in symbols:
                try:
                    ticker = yf.Ticker(sym)
                    hist = ticker.history(
                        start=start_date.date(),
                        end=end_date.date(),
                        interval="1d"
                    )
                    if not hist.empty:
                        data_frames[sym] = hist
                        logger.info(f"✅ Retrieved {len(hist)} data points for {sym}")
                    else:
                        logger.warning(f"⚠️ No data available for {sym}")
                except Exception as e:
                    logger.error(f"❌ Error fetching data for {sym}: {e}")
            
            if len(data_frames) < 2:
                raise ValueError("Insufficient data collected for ASX SPI analysis")
            
            # Combine and engineer features
            combined_data = self._engineer_asx_spi_features(data_frames, symbol)
            logger.info(f"🔧 Engineered {len(combined_data.columns)} features including ASX SPI data")
            
            return combined_data
            
        except Exception as e:
            logger.error(f"❌ Error collecting ASX SPI data: {e}")
            raise
    
    def _engineer_asx_spi_features(self, 
                                  data_frames: Dict[str, pd.DataFrame], 
                                  target_symbol: str) -> pd.DataFrame:
        """Engineer features including ASX SPI relationships"""
        
        # Start with the target symbol data
        if target_symbol not in data_frames:
            raise ValueError(f"Target symbol {target_symbol} not found in data")
        
        df = data_frames[target_symbol].copy()
        df['target_close'] = df['Close']
        df['target_volume'] = df['Volume']
        
        # Add ASX SPI features if available
        if self.spi_symbol in data_frames:
            spi_data = data_frames[self.spi_symbol]
            df['spi_close'] = spi_data['Close'].reindex(df.index, method='ffill')
            df['spi_volume'] = spi_data['Volume'].reindex(df.index, method='ffill')
            
            # Calculate SPI-specific features
            df['spi_returns'] = df['spi_close'].pct_change()
            df['spi_volatility'] = df['spi_returns'].rolling(window=20).std()
            
            # Calculate futures basis (SPI vs ASX200)
            if self.asx200_symbol in data_frames:
                asx200_data = data_frames[self.asx200_symbol]
                df['asx200_close'] = asx200_data['Close'].reindex(df.index, method='ffill')
                df['futures_basis'] = df['spi_close'] - df['asx200_close']
                df['basis_ratio'] = df['futures_basis'] / df['asx200_close']
                
        # Add ASX 200 features
        if self.asx200_symbol in data_frames:
            asx200_data = data_frames[self.asx200_symbol]
            df['asx200_close'] = asx200_data['Close'].reindex(df.index, method='ffill')
            df['asx200_returns'] = df['asx200_close'].pct_change()
            df['asx200_volatility'] = df['asx200_returns'].rolling(window=20).std()
        
        # Add All Ordinaries features
        if self.aord_symbol in data_frames:
            aord_data = data_frames[self.aord_symbol]
            df['aord_close'] = aord_data['Close'].reindex(df.index, method='ffill')
            df['aord_returns'] = df['aord_close'].pct_change()
        
        # Technical indicators for target
        df['target_returns'] = df['target_close'].pct_change()
        df['target_sma_5'] = df['target_close'].rolling(window=5).mean()
        df['target_sma_20'] = df['target_close'].rolling(window=20).mean()
        df['target_rsi'] = self._calculate_rsi(df['target_close'])
        df['target_volatility'] = df['target_returns'].rolling(window=20).std()
        
        # Cross-correlations with ASX SPI
        if 'spi_returns' in df.columns:
            df['target_spi_correlation'] = df['target_returns'].rolling(window=30).corr(df['spi_returns'])
            df['target_spi_beta'] = self._calculate_rolling_beta(
                df['target_returns'], df['spi_returns'], window=30
            )
        
        # Market regime indicators
        df['market_trend'] = np.where(df['target_sma_5'] > df['target_sma_20'], 1, 0)
        df['volatility_regime'] = np.where(df['target_volatility'] > df['target_volatility'].rolling(60).median(), 1, 0)
        
        # Clean data - fill forward for missing values and only drop rows with missing essential data
        df = df.ffill().bfill()
        df = df.dropna(subset=['target_close'])  # Only drop if target column is NaN
        
        # Store feature columns for later use
        self.feature_columns = [col for col in df.columns if col not in ['target_close', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_rolling_beta(self, 
                               returns1: pd.Series, 
                               returns2: pd.Series, 
                               window: int = 30) -> pd.Series:
        """Calculate rolling beta between two return series"""
        covariance = returns1.rolling(window).cov(returns2)
        variance = returns2.rolling(window).var()
        return covariance / variance
    
    async def train_model(self, 
                         symbol: str, 
                         horizon: PredictionHorizon = PredictionHorizon.SHORT_TERM) -> Dict[str, Any]:
        """Train prediction model with ASX SPI integration"""
        try:
            logger.info(f"🤖 Training ASX SPI-enhanced model for {symbol} ({horizon.value})")
            
            # Collect training data
            data = await self.collect_asx_spi_data(symbol, days_back=365)
            
            if len(data) < 60:
                raise ValueError("Insufficient data for model training")
            
            # Prepare features and targets
            features = data[self.feature_columns].fillna(0)
            
            # Create target based on horizon
            horizon_days = int(horizon.value.replace('d', ''))
            target = data['target_close'].shift(-horizon_days).dropna()
            
            # Align features and targets
            min_length = min(len(features), len(target))
            features = features.iloc[:min_length]
            target = target.iloc[:min_length]
            
            # Split data (80% train, 20% validation)
            split_idx = int(len(features) * 0.8)
            X_train, X_val = features.iloc[:split_idx], features.iloc[split_idx:]
            y_train, y_val = target.iloc[:split_idx], target.iloc[split_idx:]
            
            # Scale features
            scaler = self.scalers[horizon.value]
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            # Train model
            model = self.models[horizon.value]
            model.fit(X_train_scaled, y_train)
            
            # Validate model
            val_predictions = model.predict(X_val_scaled)
            val_rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
            val_r2 = r2_score(y_val, val_predictions)
            
            # Calculate feature importance (if available)
            feature_importance = {}
            if hasattr(model, 'feature_importances_'):
                for i, importance in enumerate(model.feature_importances_):
                    feature_importance[self.feature_columns[i]] = importance
            
            training_result = {
                'symbol': symbol,
                'horizon': horizon.value,
                'training_samples': len(X_train),
                'validation_samples': len(X_val),
                'validation_rmse': val_rmse,
                'validation_r2': val_r2,
                'features_count': len(self.feature_columns),
                'feature_importance': feature_importance,
                'model_type': type(model).__name__
            }
            
            logger.info(f"✅ Model trained successfully - RMSE: {val_rmse:.4f}, R²: {val_r2:.4f}")
            return training_result
            
        except Exception as e:
            logger.error(f"❌ Error training model: {e}")
            raise
    
    async def predict(self, 
                     symbol: str, 
                     horizon: PredictionHorizon = PredictionHorizon.SHORT_TERM) -> PredictionResult:
        """Make prediction using ASX SPI-enhanced model"""
        try:
            logger.info(f"🔮 Making ASX SPI-enhanced prediction for {symbol} ({horizon.value})")
            
            # Collect recent data
            data = await self.collect_asx_spi_data(symbol, days_back=120)
            
            if len(data) < 20:
                raise ValueError("Insufficient recent data for prediction")
            
            # Prepare features from the latest data
            latest_features = data[self.feature_columns].iloc[-1:].fillna(0)
            
            # Scale features
            scaler = self.scalers[horizon.value]
            features_scaled = scaler.transform(latest_features)
            
            # Make prediction
            model = self.models[horizon.value]
            predicted_price = model.predict(features_scaled)[0]
            
            # Calculate confidence interval (using historical prediction errors)
            prediction_std = np.std(data['target_close'].pct_change().dropna()) * np.sqrt(int(horizon.value.replace('d', '')))
            current_price = data['target_close'].iloc[-1]
            
            lower_bound = predicted_price - (1.96 * prediction_std * current_price)
            upper_bound = predicted_price + (1.96 * prediction_std * current_price)
            
            # Calculate probabilities
            prob_up = 0.5 + (predicted_price - current_price) / (2 * prediction_std * current_price)
            prob_up = max(0.0, min(1.0, prob_up))
            prob_down = 1.0 - prob_up
            
            # Calculate ASX SPI influence (if SPI features exist)
            spi_influence = 0.0
            if hasattr(model, 'feature_importances_'):
                spi_features = [i for i, col in enumerate(self.feature_columns) 
                              if 'spi' in col.lower() or 'basis' in col.lower()]
                if spi_features:
                    spi_influence = sum(model.feature_importances_[i] for i in spi_features)
            
            # Calculate risk score based on volatility and uncertainty
            recent_volatility = data['target_volatility'].iloc[-1] if 'target_volatility' in data.columns else 0.02
            risk_score = min(1.0, recent_volatility * 10)  # Scale to 0-1
            
            result = PredictionResult(
                symbol=symbol,
                target_date=datetime.now() + timedelta(days=int(horizon.value.replace('d', ''))),
                predicted_price=predicted_price,
                confidence_interval=(lower_bound, upper_bound),
                probability_up=prob_up,
                probability_down=prob_down,
                spi_influence=spi_influence,
                risk_score=risk_score,
                model_used=type(model).__name__,
                features_used=self.feature_columns
            )
            
            logger.info(f"🎯 Prediction: ${predicted_price:.2f} (SPI influence: {spi_influence:.3f})")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error making prediction: {e}")
            raise
    
    async def run_backtest(self, 
                          symbol: str, 
                          start_date: datetime, 
                          end_date: datetime, 
                          horizon: PredictionHorizon = PredictionHorizon.SHORT_TERM,
                          rebalance_frequency: int = 5) -> BacktestResult:
        """Comprehensive backtesting with ASX SPI integration"""
        try:
            logger.info(f"🔄 Starting backtest for {symbol} from {start_date.date()} to {end_date.date()}")
            
            # Collect historical data for backtesting period
            total_days = (end_date - start_date).days + 60  # Extra buffer for training
            backtest_start = start_date - timedelta(days=60)
            data = await self.collect_asx_spi_data(symbol, days_back=total_days)
            
            # Filter data to backtest period
            data = data[data.index >= pd.Timestamp(backtest_start)]
            
            if len(data) < 100:
                raise ValueError("Insufficient data for backtesting")
            
            horizon_days = int(horizon.value.replace('d', ''))
            predictions = []
            actual_prices = []
            prediction_dates = []
            returns = []
            
            # Walk-forward backtesting
            current_date = start_date
            train_window = 120  # Days of training data
            
            while current_date <= end_date - timedelta(days=horizon_days):
                try:
                    # Get training data window
                    train_end = current_date
                    train_start = train_end - timedelta(days=train_window)
                    
                    train_data = data[(data.index >= pd.Timestamp(train_start)) & 
                                    (data.index <= pd.Timestamp(train_end))]
                    
                    if len(train_data) < 60:
                        current_date += timedelta(days=rebalance_frequency)
                        continue
                    
                    # Prepare training features and targets
                    features = train_data[self.feature_columns].fillna(0)
                    target = train_data['target_close'].shift(-horizon_days).dropna()
                    
                    # Align features and targets
                    min_length = min(len(features), len(target))
                    X_train = features.iloc[:min_length]
                    y_train = target.iloc[:min_length]
                    
                    if len(X_train) < 30:
                        current_date += timedelta(days=rebalance_frequency)
                        continue
                    
                    # Train model
                    scaler = RobustScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    
                    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                    model.fit(X_train_scaled, y_train)
                    
                    # Make prediction for current date
                    current_features = train_data[self.feature_columns].iloc[-1:].fillna(0)
                    current_scaled = scaler.transform(current_features)
                    prediction = model.predict(current_scaled)[0]
                    
                    # Get actual future price
                    future_date = current_date + timedelta(days=horizon_days)
                    future_data = data[data.index >= pd.Timestamp(future_date)]
                    
                    if len(future_data) > 0:
                        actual_price = future_data['target_close'].iloc[0]
                        current_price = train_data['target_close'].iloc[-1]
                        
                        predictions.append(prediction)
                        actual_prices.append(actual_price)
                        prediction_dates.append(current_date)
                        
                        # Calculate returns
                        predicted_return = (prediction - current_price) / current_price
                        actual_return = (actual_price - current_price) / current_price
                        returns.append({'predicted': predicted_return, 'actual': actual_return})
                    
                    current_date += timedelta(days=rebalance_frequency)
                    
                except Exception as e:
                    logger.warning(f"⚠️ Error in backtest iteration for {current_date}: {e}")
                    current_date += timedelta(days=rebalance_frequency)
                    continue
            
            if len(predictions) < 10:
                raise ValueError("Insufficient predictions generated for meaningful backtesting")
            
            # Calculate performance metrics
            predictions_array = np.array(predictions)
            actual_array = np.array(actual_prices)
            
            # Prediction accuracy metrics
            rmse = np.sqrt(mean_squared_error(actual_array, predictions_array))
            mae = mean_absolute_error(actual_array, predictions_array)
            r_squared = r2_score(actual_array, predictions_array)
            
            # Direction accuracy
            predicted_directions = np.diff(predictions_array) > 0
            actual_directions = np.diff(actual_array) > 0
            direction_accuracy = np.mean(predicted_directions == actual_directions) if len(predicted_directions) > 0 else 0.0
            
            # Trading performance metrics
            actual_returns = [r['actual'] for r in returns]
            predicted_returns = [r['predicted'] for r in returns]
            
            # Calculate strategy returns (assuming we trade based on predictions)
            strategy_returns = []
            for pred_ret, actual_ret in zip(predicted_returns, actual_returns):
                # Simple strategy: go long if predicted positive return, otherwise stay flat
                if pred_ret > 0:
                    strategy_returns.append(actual_ret)
                else:
                    strategy_returns.append(0.0)
            
            total_return = np.prod([1 + r for r in strategy_returns]) - 1
            annualized_return = (1 + total_return) ** (252 / len(strategy_returns)) - 1 if strategy_returns else 0.0
            
            # Risk metrics
            returns_std = np.std(strategy_returns) if strategy_returns else 0.0
            volatility = returns_std * np.sqrt(252)  # Annualized
            sharpe_ratio = (annualized_return / volatility) if volatility > 0 else 0.0
            
            # Maximum drawdown
            cumulative_returns = np.cumprod([1 + r for r in strategy_returns])
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = (cumulative_returns - running_max) / running_max
            max_drawdown = np.min(drawdowns) if len(drawdowns) > 0 else 0.0
            
            # Win rate
            winning_trades = [r for r in strategy_returns if r > 0]
            win_rate = len(winning_trades) / len(strategy_returns) if strategy_returns else 0.0
            
            # Create detailed prediction history
            prediction_history = []
            for i, (date, pred, actual) in enumerate(zip(prediction_dates, predictions, actual_prices)):
                prediction_history.append({
                    'date': date.isoformat(),
                    'predicted_price': float(pred),
                    'actual_price': float(actual),
                    'error': float(abs(pred - actual)),
                    'error_pct': float(abs(pred - actual) / actual * 100),
                    'predicted_return': float(predicted_returns[i]) if i < len(predicted_returns) else 0.0,
                    'actual_return': float(actual_returns[i]) if i < len(actual_returns) else 0.0,
                    'strategy_return': float(strategy_returns[i]) if i < len(strategy_returns) else 0.0
                })
            
            result = BacktestResult(
                start_date=start_date,
                end_date=end_date,
                total_predictions=len(predictions),
                correct_predictions=int(direction_accuracy * len(predictions)),
                accuracy=direction_accuracy,
                total_return=total_return,
                annualized_return=annualized_return,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                win_rate=win_rate,
                avg_prediction_error=mae,
                rmse=rmse,
                r_squared=r_squared,
                volatility=volatility,
                metrics={
                    'mae': mae,
                    'direction_accuracy': direction_accuracy,
                    'total_predictions': len(predictions),
                    'avg_return_per_trade': np.mean(strategy_returns) if strategy_returns else 0.0,
                    'best_prediction': np.min([abs(p - a) for p, a in zip(predictions, actual_prices)]),
                    'worst_prediction': np.max([abs(p - a) for p, a in zip(predictions, actual_prices)]),
                },
                prediction_history=prediction_history
            )
            
            logger.info(f"✅ Backtest completed: {len(predictions)} predictions, "
                       f"Accuracy: {direction_accuracy:.2%}, Sharpe: {sharpe_ratio:.3f}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error in backtesting: {e}")
            raise

# Initialize global instance
asx_spi_predictor = ASXSPIPredictionSystem()

async def main():
    """Test the ASX SPI prediction system"""
    try:
        symbol = "^AORD"  # All Ordinaries
        
        # Train model
        training_result = await asx_spi_predictor.train_model(symbol, PredictionHorizon.SHORT_TERM)
        print(f"Training Result: {json.dumps(training_result, indent=2)}")
        
        # Make prediction
        prediction = await asx_spi_predictor.predict(symbol, PredictionHorizon.SHORT_TERM)
        print(f"Prediction: ${prediction.predicted_price:.2f} (SPI influence: {prediction.spi_influence:.3f})")
        
        # Run backtest
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)
        backtest = await asx_spi_predictor.run_backtest(symbol, start_date, end_date)
        print(f"Backtest Accuracy: {backtest.accuracy:.2%}, Sharpe: {backtest.sharpe_ratio:.3f}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())