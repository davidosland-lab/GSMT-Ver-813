#!/usr/bin/env python3
"""
🚀 Phase 4 - P4-001: Temporal Fusion Transformer (TFT) Implementation
========================================================================

State-of-the-art attention-based time series forecasting architecture combining:
- Variable Selection Networks for automatic feature importance
- Gated Residual Networks for non-linear processing  
- Multi-Head Attention for temporal relationship modeling
- Interpretable outputs with attention visualization
- Multi-horizon forecasting with uncertainty quantification

Target: +8-12% accuracy improvement over Phase 3 (85% -> 90-92%)
Reference: "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import warnings
import math
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import yfinance as yf

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TFTConfig:
    """Configuration for Temporal Fusion Transformer."""
    # Model architecture
    hidden_size: int = 256
    num_attention_heads: int = 4
    num_encoder_layers: int = 2
    num_decoder_layers: int = 2
    dropout_rate: float = 0.1
    
    # Input processing
    sequence_length: int = 60  # Historical sequence length
    prediction_horizons: List[str] = field(default_factory=lambda: ['1d', '5d', '30d', '90d'])
    
    # Variable selection
    num_static_vars: int = 10  # Market regime, sector, etc.
    num_historical_vars: int = 20  # OHLCV, technical indicators
    num_future_vars: int = 5   # Known future inputs (calendar features)
    
    # Training parameters
    learning_rate: float = 0.001
    batch_size: int = 64
    max_epochs: int = 100
    early_stopping_patience: int = 10
    
    # Quantile regression for uncertainty
    quantiles: List[float] = field(default_factory=lambda: [0.1, 0.25, 0.5, 0.75, 0.9])
    
    # Device configuration
    device: str = 'cpu'  # Will auto-detect GPU if available

class GatedLinearUnit(nn.Module):
    """Gated Linear Unit for controlling information flow."""
    
    def __init__(self, input_size: int, hidden_size: Optional[int] = None):
        super().__init__()
        if hidden_size is None:
            hidden_size = input_size
        
        self.hidden_size = hidden_size
        self.fc = nn.Linear(input_size, hidden_size * 2)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        h = self.fc(x)
        return h[:, :self.hidden_size] * self.sigmoid(h[:, self.hidden_size:])

class GatedResidualNetwork(nn.Module):
    """Gated Residual Network for non-linear processing with skip connections."""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        dropout_rate: float = 0.1,
        use_time_distributed: bool = True
    ):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_time_distributed = use_time_distributed
        
        # Main processing layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.gate = GatedLinearUnit(hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # Skip connection projection if dimensions don't match
        self.skip_projection = None
        if input_size != hidden_size:
            self.skip_projection = nn.Linear(input_size, hidden_size)
            
    def forward(self, x):
        # Main path
        h = self.fc1(x)
        h = self.elu(h)
        h = self.fc2(h)
        h = self.dropout(h)
        h = self.gate(h)
        
        # Skip connection
        if self.skip_projection is not None:
            x = self.skip_projection(x)
        
        # Add and normalize
        return self.layer_norm(h + x)

class VariableSelectionNetwork(nn.Module):
    """Variable Selection Network for automatic feature importance detection."""
    
    def __init__(
        self,
        input_size: int,
        num_inputs: int,
        hidden_size: int,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        
        self.input_size = input_size
        self.num_inputs = num_inputs
        self.hidden_size = hidden_size
        
        # Flattened input processing
        self.flattened_grn = GatedResidualNetwork(
            input_size=num_inputs * input_size,
            hidden_size=hidden_size,
            dropout_rate=dropout_rate
        )
        
        # Variable selection weights
        self.variable_selection = nn.Sequential(
            nn.Linear(hidden_size, num_inputs),
            nn.Softmax(dim=-1)
        )
        
        # Individual variable processing
        self.single_variable_grns = nn.ModuleList([
            GatedResidualNetwork(
                input_size=input_size,
                hidden_size=hidden_size,
                dropout_rate=dropout_rate
            ) for _ in range(num_inputs)
        ])
        
    def forward(self, flattened_inputs):
        # flattened_inputs shape: (batch_size, num_inputs * input_size)
        batch_size = flattened_inputs.size(0)
        
        # Get variable selection weights
        mlp_outputs = self.flattened_grn(flattened_inputs)
        sparse_weights = self.variable_selection(mlp_outputs)
        
        # Reshape inputs for individual processing
        inputs_for_selection = flattened_inputs.view(
            batch_size, self.num_inputs, self.input_size
        )
        
        # Process each variable individually
        var_outputs = []
        for i in range(self.num_inputs):
            var_input = inputs_for_selection[:, i, :]  # (batch_size, input_size)
            var_output = self.single_variable_grns[i](var_input)
            var_outputs.append(var_output)
        
        # Stack and apply selection weights
        var_outputs = torch.stack(var_outputs, dim=1)  # (batch_size, num_inputs, hidden_size)
        sparse_weights = sparse_weights.unsqueeze(-1)  # (batch_size, num_inputs, 1)
        
        # Weighted combination
        combined = torch.sum(var_outputs * sparse_weights, dim=1)  # (batch_size, hidden_size)
        
        return combined, sparse_weights.squeeze(-1)

class InterpretableMultiHeadAttention(nn.Module):
    """Multi-head attention with interpretability features."""
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout_rate)
        
    def scaled_dot_product_attention(self, q, k, v, mask=None):
        """Scaled dot-product attention with optional masking."""
        # q, k, v: (batch_size, num_heads, seq_len, d_k)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, v)
        
        return output, attention_weights
    
    def forward(self, query, key, value, mask=None):
        batch_size, seq_len, _ = query.size()
        
        # Linear transformations
        q = self.w_q(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(key).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(value).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        attn_output, attention_weights = self.scaled_dot_product_attention(q, k, v, mask)
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        # Final linear layer
        output = self.w_o(attn_output)
        
        # Average attention weights across heads for interpretability
        avg_attention_weights = attention_weights.mean(dim=1)  # (batch_size, seq_len, seq_len)
        
        return output, avg_attention_weights

class TemporalFusionTransformer(nn.Module):
    """
    Complete Temporal Fusion Transformer implementation.
    
    Architecture:
    1. Variable Selection Networks for static, historical, and future inputs
    2. Locality Enhancement with ConvNets
    3. Static Covariate Encoders
    4. Temporal Self-Attention
    5. Position-wise Feed Forward
    6. Quantile Regression Outputs
    """
    
    def __init__(self, config: TFTConfig):
        super().__init__()
        
        self.config = config
        self.hidden_size = config.hidden_size
        
        # Variable Selection Networks
        self.static_vsn = VariableSelectionNetwork(
            input_size=1,
            num_inputs=config.num_static_vars,
            hidden_size=config.hidden_size,
            dropout_rate=config.dropout_rate
        )
        
        self.historical_vsn = VariableSelectionNetwork(
            input_size=config.sequence_length,
            num_inputs=config.num_historical_vars,
            hidden_size=config.hidden_size,
            dropout_rate=config.dropout_rate
        )
        
        self.future_vsn = VariableSelectionNetwork(
            input_size=len(config.prediction_horizons),
            num_inputs=config.num_future_vars,
            hidden_size=config.hidden_size,
            dropout_rate=config.dropout_rate
        )
        
        # Static covariate encoders
        self.static_encoder_selection = GatedResidualNetwork(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            dropout_rate=config.dropout_rate
        )
        
        self.static_encoder_enrichment = GatedResidualNetwork(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            dropout_rate=config.dropout_rate
        )
        
        self.static_encoder_state_h = GatedResidualNetwork(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            dropout_rate=config.dropout_rate
        )
        
        self.static_encoder_state_c = GatedResidualNetwork(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            dropout_rate=config.dropout_rate
        )
        
        # Locality enhancement (temporal convolution)
        self.locality_enhancement = nn.Conv1d(
            in_channels=config.hidden_size,
            out_channels=config.hidden_size,
            kernel_size=3,
            padding=1
        )
        
        # LSTM for sequential processing
        self.lstm = nn.LSTM(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            num_layers=2,
            dropout=config.dropout_rate,
            batch_first=True
        )
        
        # Self-attention mechanism
        self.self_attention = InterpretableMultiHeadAttention(
            d_model=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout_rate=config.dropout_rate
        )
        
        # Position-wise feed forward
        self.feed_forward = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 4),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_size * 4, config.hidden_size)
        )
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(config.hidden_size)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size)
        
        # Quantile regression heads for multi-horizon prediction
        self.quantile_heads = nn.ModuleDict()
        for horizon in config.prediction_horizons:
            self.quantile_heads[horizon] = nn.ModuleDict({
                f'q_{int(q*100)}': nn.Linear(config.hidden_size, 1)
                for q in config.quantiles
            })
        
        # Attention interpretation head
        self.attention_interpretation = nn.Linear(config.hidden_size, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
    
    def forward(
        self,
        static_inputs: torch.Tensor,
        historical_inputs: torch.Tensor,
        future_inputs: torch.Tensor
    ):
        """
        Forward pass of the TFT model.
        
        Args:
            static_inputs: (batch_size, num_static_vars)
            historical_inputs: (batch_size, sequence_length, num_historical_vars)
            future_inputs: (batch_size, num_horizons, num_future_vars)
        
        Returns:
            Dict containing predictions, attention weights, and variable importances
        """
        batch_size = static_inputs.size(0)
        
        # Variable selection for static inputs
        static_flattened = static_inputs.view(batch_size, -1)
        static_selected, static_weights = self.static_vsn(static_flattened)
        
        # Variable selection for historical inputs
        historical_flattened = historical_inputs.view(batch_size, -1)
        historical_selected, historical_weights = self.historical_vsn(historical_flattened)
        
        # Variable selection for future inputs
        future_flattened = future_inputs.view(batch_size, -1)
        future_selected, future_weights = self.future_vsn(future_flattened)
        
        # Static covariate processing
        static_selection = self.static_encoder_selection(static_selected)
        static_enrichment = self.static_encoder_enrichment(static_selected)
        
        # LSTM initial states from static covariates
        lstm_h = self.static_encoder_state_h(static_selected).unsqueeze(0)
        lstm_c = self.static_encoder_state_c(static_selected).unsqueeze(0)
        lstm_state = (lstm_h, lstm_c)
        
        # Combine historical and future information
        # For simplicity, we'll process them sequentially
        seq_len = self.config.sequence_length + len(self.config.prediction_horizons)
        
        # Create combined sequence (historical + future placeholders)
        combined_sequence = torch.zeros(batch_size, seq_len, self.config.hidden_size).to(static_inputs.device)
        combined_sequence[:, :self.config.sequence_length, :] = historical_selected.unsqueeze(1).repeat(1, self.config.sequence_length, 1)
        
        # Add future information
        for i, _ in enumerate(self.config.prediction_horizons):
            combined_sequence[:, self.config.sequence_length + i, :] = future_selected
        
        # Locality enhancement (temporal convolution)
        conv_input = combined_sequence.transpose(1, 2)  # (batch_size, hidden_size, seq_len)
        conv_output = self.locality_enhancement(conv_input)
        conv_output = conv_output.transpose(1, 2)  # (batch_size, seq_len, hidden_size)
        
        # LSTM processing
        lstm_output, _ = self.lstm(conv_output, lstm_state)
        
        # Self-attention
        attn_output, attention_weights = self.self_attention(lstm_output, lstm_output, lstm_output)
        
        # Residual connection and layer norm
        attn_output = self.layer_norm1(lstm_output + attn_output)
        
        # Feed forward
        ff_output = self.feed_forward(attn_output)
        
        # Final residual connection and layer norm
        final_output = self.layer_norm2(attn_output + ff_output)
        
        # Generate predictions for each horizon using quantile regression
        predictions = {}
        for horizon in self.config.prediction_horizons:
            # Use the last timestep for prediction
            horizon_input = final_output[:, -1, :]
            
            horizon_predictions = {}
            for quantile_name, quantile_head in self.quantile_heads[horizon].items():
                quantile_pred = quantile_head(horizon_input).squeeze(-1)
                horizon_predictions[quantile_name] = quantile_pred
            
            predictions[horizon] = horizon_predictions
        
        # Attention interpretation scores
        attention_scores = self.attention_interpretation(final_output).squeeze(-1)
        
        return {
            'predictions': predictions,
            'attention_weights': attention_weights,
            'attention_scores': attention_scores,
            'variable_importances': {
                'static': static_weights,
                'historical': historical_weights,
                'future': future_weights
            },
            'static_selection': static_selection,
            'enrichment': static_enrichment
        }

class QuantileLoss(nn.Module):
    """Quantile loss for probabilistic forecasting."""
    
    def __init__(self, quantiles: List[float]):
        super().__init__()
        self.quantiles = quantiles
    
    def forward(self, predictions: Dict, targets: torch.Tensor, horizon: str):
        """
        Calculate quantile loss for a specific horizon.
        
        Args:
            predictions: Dict of quantile predictions
            targets: Ground truth values
            horizon: Prediction horizon
        """
        total_loss = 0.0
        
        for i, quantile in enumerate(self.quantiles):
            quantile_name = f'q_{int(quantile*100)}'
            if quantile_name in predictions[horizon]:
                pred = predictions[horizon][quantile_name]
                diff = targets - pred
                
                # Quantile loss: q * max(diff, 0) + (1-q) * max(-diff, 0)
                loss = torch.max(quantile * diff, (quantile - 1) * diff)
                total_loss += loss.mean()
        
        return total_loss / len(self.quantiles)

@dataclass
class TFTPredictionResult:
    """Result structure for TFT predictions."""
    symbol: str
    prediction_timestamp: datetime
    horizon_predictions: Dict[str, Dict[str, float]]  # horizon -> quantile -> value
    point_predictions: Dict[str, float]  # horizon -> median prediction
    confidence_intervals: Dict[str, Tuple[float, float]]  # horizon -> (lower, upper)
    attention_weights: np.ndarray
    variable_importances: Dict[str, np.ndarray]
    attention_scores: np.ndarray
    uncertainty_scores: Dict[str, float]
    model_confidence: float

class TemporalFusionPredictor:
    """
    High-level interface for TFT-based prediction system.
    Integrates with existing Phase 3 infrastructure.
    """
    
    def __init__(self, config: TFTConfig = None):
        self.config = config or TFTConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize model
        self.model = TemporalFusionTransformer(self.config)
        self.model.to(self.config.device)
        
        # Loss function and optimizer
        self.criterion = QuantileLoss(self.config.quantiles)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate
        )
        
        # Data preprocessing
        self.scalers = {
            'static': StandardScaler(),
            'historical': StandardScaler(),
            'future': StandardScaler(),
            'targets': StandardScaler()
        }
        
        # Training state
        self.is_trained = False
        self.training_history = []
        
    def prepare_features(self, symbol: str, lookback_days: int = None) -> Dict[str, np.ndarray]:
        """
        Prepare features for TFT model from market data.
        
        Returns:
            Dict containing static, historical, and future features
        """
        try:
            lookback_days = lookback_days or (self.config.sequence_length + 30)
            
            # Fetch market data
            ticker = yf.Ticker(symbol)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days)
            
            df = ticker.history(start=start_date, end=end_date, interval='1d')
            
            if df.empty:
                raise ValueError(f"No data available for {symbol}")
            
            # Static features (market regime, sector characteristics)
            static_features = self._extract_static_features(symbol, df)
            
            # Historical features (OHLCV + technical indicators)
            historical_features = self._extract_historical_features(df)
            
            # Future features (calendar features, known future inputs)
            future_features = self._extract_future_features(df)
            
            return {
                'static': static_features,
                'historical': historical_features,
                'future': future_features,
                'targets': df['Close'].values[-len(historical_features):],
                'dates': df.index[-len(historical_features):]
            }
            
        except Exception as e:
            self.logger.error(f"Error preparing features for {symbol}: {e}")
            raise
    
    def _extract_static_features(self, symbol: str, df: pd.DataFrame) -> np.ndarray:
        """Extract static market features."""
        # Market characteristics that don't change over time
        features = []
        
        # Market capitalization category (estimated from symbol)
        market_cap_category = self._estimate_market_cap_category(symbol)
        features.append(market_cap_category)
        
        # Sector classification (simplified)
        sector_class = self._estimate_sector_class(symbol)
        features.append(sector_class)
        
        # Average volume (normalized)
        avg_volume = df['Volume'].mean()
        volume_normalized = min(avg_volume / 1e6, 10.0)  # Cap at 10M
        features.append(volume_normalized)
        
        # Historical volatility
        returns = df['Close'].pct_change().dropna()
        historical_vol = returns.std() * np.sqrt(252)  # Annualized
        features.append(historical_vol)
        
        # Market correlation (simplified)
        market_correlation = 0.5  # Default correlation
        features.append(market_correlation)
        
        # Padding to reach num_static_vars
        while len(features) < self.config.num_static_vars:
            features.append(0.0)
        
        return np.array(features[:self.config.num_static_vars])
    
    def _extract_historical_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract time-varying historical features."""
        # Technical indicators and price features
        features_list = []
        
        # Price features
        features_list.append(df['Close'].values)
        features_list.append(df['Open'].values)
        features_list.append(df['High'].values)
        features_list.append(df['Low'].values)
        features_list.append(df['Volume'].values)
        
        # Returns
        returns = df['Close'].pct_change().fillna(0).values
        features_list.append(returns)
        
        # Moving averages
        for window in [5, 10, 20, 50]:
            ma = df['Close'].rolling(window=window).mean().fillna(df['Close']).values
            features_list.append(ma)
        
        # RSI
        rsi = self._calculate_rsi(df['Close']).values
        features_list.append(rsi)
        
        # Bollinger Bands
        bb_upper, bb_lower = self._calculate_bollinger_bands(df['Close'])
        features_list.append(bb_upper.values)
        features_list.append(bb_lower.values)
        
        # MACD
        macd, signal = self._calculate_macd(df['Close'])
        features_list.append(macd.values)
        features_list.append(signal.values)
        
        # Volatility
        volatility = returns.rolling(window=20).std().fillna(0).values
        features_list.append(volatility)
        
        # Padding or truncating to match num_historical_vars
        while len(features_list) < self.config.num_historical_vars:
            features_list.append(np.zeros_like(features_list[0]))
        
        # Stack and transpose to (time_steps, features)
        historical_features = np.stack(features_list[:self.config.num_historical_vars], axis=1)
        
        return historical_features[-self.config.sequence_length:]
    
    def _extract_future_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract known future features (calendar features)."""
        # Calendar and cyclical features that are known in advance
        current_date = df.index[-1]
        
        features = []
        
        # Day of week
        day_of_week = current_date.dayofweek / 6.0  # Normalized to [0, 1]
        features.append(day_of_week)
        
        # Month
        month = current_date.month / 12.0  # Normalized to [0, 1]
        features.append(month)
        
        # Quarter
        quarter = ((current_date.month - 1) // 3) / 3.0  # Normalized to [0, 1]
        features.append(quarter)
        
        # Year (trend)
        year_trend = (current_date.year - 2020) / 10.0  # Relative to 2020
        features.append(year_trend)
        
        # Market session (simplified)
        market_session = 1.0 if current_date.weekday() < 5 else 0.0  # Trading day
        features.append(market_session)
        
        # Padding to reach num_future_vars
        while len(features) < self.config.num_future_vars:
            features.append(0.0)
        
        # Replicate for each prediction horizon
        future_features = np.tile(
            np.array(features[:self.config.num_future_vars]),
            (len(self.config.prediction_horizons), 1)
        )
        
        return future_features
    
    def _estimate_market_cap_category(self, symbol: str) -> float:
        """Estimate market cap category (0=small, 0.5=mid, 1=large)."""
        # Simplified classification based on symbol patterns
        if symbol in ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'CBA.AX', 'BHP.AX']:
            return 1.0  # Large cap
        elif '.AX' in symbol or len(symbol) <= 4:
            return 0.5  # Mid cap (most ASX stocks)
        else:
            return 0.0  # Small cap
    
    def _estimate_sector_class(self, symbol: str) -> float:
        """Estimate sector classification."""
        # Simplified sector mapping
        if 'CBA' in symbol or 'WBC' in symbol or 'ANZ' in symbol:
            return 0.8  # Banking
        elif 'BHP' in symbol or 'RIO' in symbol:
            return 0.6  # Mining
        elif 'AAPL' in symbol or 'MSFT' in symbol or 'GOOGL' in symbol:
            return 0.4  # Technology
        else:
            return 0.2  # Other
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)  # Neutral RSI for initial values
    
    def _calculate_bollinger_bands(self, prices: pd.Series, window: int = 20) -> Tuple[pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        ma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper = ma + (2 * std)
        lower = ma - (2 * std)
        return upper.fillna(prices), lower.fillna(prices)
    
    def _calculate_macd(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD indicator."""
        ema_12 = prices.ewm(span=12).mean()
        ema_26 = prices.ewm(span=26).mean()
        macd = ema_12 - ema_26
        signal = macd.ewm(span=9).mean()
        return macd.fillna(0), signal.fillna(0)
    
    async def generate_tft_prediction(
        self,
        symbol: str,
        time_horizons: List[str] = None
    ) -> TFTPredictionResult:
        """
        Generate TFT-based prediction for a symbol.
        
        Args:
            symbol: Stock symbol to predict
            time_horizons: List of prediction horizons (default: all configured)
        
        Returns:
            TFTPredictionResult with multi-horizon predictions and interpretability
        """
        try:
            self.logger.info(f"Generating TFT prediction for {symbol}")
            
            # Use configured horizons if none specified
            if time_horizons is None:
                time_horizons = self.config.prediction_horizons
            
            # Prepare features
            feature_data = self.prepare_features(symbol)
            
            # Convert to tensors
            static_tensor = torch.FloatTensor(feature_data['static']).unsqueeze(0).to(self.config.device)
            historical_tensor = torch.FloatTensor(feature_data['historical']).unsqueeze(0).to(self.config.device)
            future_tensor = torch.FloatTensor(feature_data['future']).unsqueeze(0).to(self.config.device)
            
            # Model prediction
            self.model.eval()
            with torch.no_grad():
                output = self.model(static_tensor, historical_tensor, future_tensor)
            
            # Process predictions
            horizon_predictions = {}
            point_predictions = {}
            confidence_intervals = {}
            uncertainty_scores = {}
            
            current_price = feature_data['targets'][-1]
            
            for horizon in time_horizons:
                if horizon in output['predictions']:
                    # Extract quantile predictions
                    quantile_preds = {}
                    for q_name, pred_tensor in output['predictions'][horizon].items():
                        quantile_preds[q_name] = float(pred_tensor.cpu().numpy()[0])
                    
                    horizon_predictions[horizon] = quantile_preds
                    
                    # Point prediction (median)
                    if 'q_50' in quantile_preds:
                        point_predictions[horizon] = quantile_preds['q_50']
                    else:
                        # Fallback to mean of available quantiles
                        point_predictions[horizon] = np.mean(list(quantile_preds.values()))
                    
                    # Confidence intervals (10th to 90th percentile)
                    lower = quantile_preds.get('q_10', point_predictions[horizon] * 0.95)
                    upper = quantile_preds.get('q_90', point_predictions[horizon] * 1.05)
                    confidence_intervals[horizon] = (lower, upper)
                    
                    # Uncertainty score
                    uncertainty = (upper - lower) / point_predictions[horizon]
                    uncertainty_scores[horizon] = uncertainty
            
            # Process attention and variable importances
            attention_weights = output['attention_weights'].cpu().numpy()[0]
            variable_importances = {
                key: weights.cpu().numpy()[0]
                for key, weights in output['variable_importances'].items()
            }
            attention_scores = output['attention_scores'].cpu().numpy()[0]
            
            # Overall model confidence
            avg_uncertainty = np.mean(list(uncertainty_scores.values()))
            model_confidence = max(0.0, min(1.0, 1.0 - avg_uncertainty))
            
            result = TFTPredictionResult(
                symbol=symbol,
                prediction_timestamp=datetime.now(),
                horizon_predictions=horizon_predictions,
                point_predictions=point_predictions,
                confidence_intervals=confidence_intervals,
                attention_weights=attention_weights,
                variable_importances=variable_importances,
                attention_scores=attention_scores,
                uncertainty_scores=uncertainty_scores,
                model_confidence=model_confidence
            )
            
            self.logger.info(f"TFT prediction completed for {symbol}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error generating TFT prediction for {symbol}: {e}")
            raise

# Global TFT predictor instance
tft_predictor = TemporalFusionPredictor()

if __name__ == "__main__":
    # Example usage
    import asyncio
    
    async def test_tft():
        try:
            # Test prediction
            result = await tft_predictor.generate_tft_prediction('AAPL')
            
            print(f"TFT Prediction for {result.symbol}")
            print(f"Timestamp: {result.prediction_timestamp}")
            print(f"Model Confidence: {result.model_confidence:.3f}")
            
            for horizon, pred in result.point_predictions.items():
                ci_lower, ci_upper = result.confidence_intervals[horizon]
                uncertainty = result.uncertainty_scores[horizon]
                
                print(f"\n{horizon} Prediction:")
                print(f"  Point: ${pred:.2f}")
                print(f"  CI: ${ci_lower:.2f} - ${ci_upper:.2f}")
                print(f"  Uncertainty: {uncertainty:.3f}")
            
            print(f"\nAttention Weights Shape: {result.attention_weights.shape}")
            print(f"Variable Importances: {list(result.variable_importances.keys())}")
            
        except Exception as e:
            print(f"Test failed: {e}")
    
    # Run test if executed directly
    asyncio.run(test_tft())