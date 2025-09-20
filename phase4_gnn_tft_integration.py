#!/usr/bin/env python3
"""
🚀 Phase 4 - GNN + TFT Integration: Advanced Multi-Modal Prediction System
=========================================================================

Integrates Graph Neural Networks (P4-002) with Temporal Fusion Transformers (P4-001)
to create a comprehensive prediction system that leverages both:

TFT Capabilities:
- Attention-based temporal modeling
- Variable selection networks
- Multi-horizon forecasting
- Interpretable AI outputs

GNN Capabilities:
- Market relationship modeling
- Cross-asset intelligence
- Systemic risk assessment
- Information propagation analysis

Integration Benefits:
- Enhanced prediction accuracy through multi-modal intelligence
- Comprehensive market understanding (temporal + relational)
- Advanced interpretability with both attention and graph insights
- Robust ensemble with multiple complementary approaches

Target: 92-94% prediction accuracy (+7-9% over Phase 3 baseline)
"""

import numpy as np
import pandas as pd
import logging
import asyncio
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from enum import Enum
import json
import warnings

# Import Phase 4 components
try:
    from phase4_temporal_fusion_transformer import (
        TemporalFusionPredictor, 
        TFTPredictionResult,
        TFTConfig
    )
    TFT_AVAILABLE = True
except ImportError:
    TFT_AVAILABLE = False

try:
    from phase4_graph_neural_networks import (
        GNNEnhancedPredictor,
        GNNPredictionResult, 
        GNNConfig
    )
    GNN_AVAILABLE = True
except ImportError:
    GNN_AVAILABLE = False

# Import Phase 3 fallback
try:
    from phase4_tft_integration import (
        Phase4TFTIntegratedPredictor,
        Phase4Prediction,
        Phase4Config
    )
    PHASE4_TFT_AVAILABLE = True
except ImportError:
    PHASE4_TFT_AVAILABLE = False

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FusionMethod(Enum):
    """Methods for fusing TFT and GNN predictions."""
    WEIGHTED_AVERAGE = "weighted_average"
    ATTENTION_FUSION = "attention_fusion"  
    CONFIDENCE_BASED = "confidence_based"
    ADAPTIVE = "adaptive"
    HIERARCHICAL = "hierarchical"

@dataclass
class GNNTFTConfig:
    """Configuration for GNN + TFT integrated system."""
    # Component configs
    tft_config: Any = field(default_factory=dict)  # Will be TFTConfig if available
    gnn_config: 'GNNConfig' = field(default_factory=lambda: GNNConfig() if GNN_AVAILABLE else {})
    
    # Integration settings
    fusion_method: FusionMethod = FusionMethod.CONFIDENCE_BASED
    tft_weight: float = 0.6  # Default weight for TFT in fusion
    gnn_weight: float = 0.4  # Default weight for GNN in fusion
    
    # Adaptive weighting
    enable_adaptive_weighting: bool = True
    confidence_threshold: float = 0.7
    relationship_threshold: float = 0.3  # Minimum relationship strength to use GNN
    
    # Performance settings
    enable_ensemble: bool = True
    fallback_enabled: bool = True
    max_prediction_time: float = 15.0  # seconds
    
    # Graph settings for prediction
    max_related_symbols: int = 15
    include_sector_analysis: bool = True
    include_market_analysis: bool = True

@dataclass 
class MultiModalPrediction:
    """Enhanced prediction result combining TFT temporal analysis and GNN relationship intelligence."""
    
    # Core prediction
    symbol: str
    time_horizon: str
    prediction_timestamp: datetime
    predicted_price: float
    current_price: float
    expected_return: float
    direction: str
    confidence_score: float
    uncertainty_score: float
    confidence_interval: Tuple[float, float]
    probability_up: float
    
    # TFT Analysis
    tft_result: Optional[Any] = None  # TFTPredictionResult when available
    tft_confidence: float = 0.0
    tft_attention_insights: Dict[str, Any] = field(default_factory=dict)
    
    # GNN Analysis  
    gnn_result: Optional[Any] = None  # GNNPredictionResult when available
    gnn_confidence: float = 0.0
    relationship_insights: Dict[str, Any] = field(default_factory=dict)
    
    # Fusion Analysis
    fusion_method: str = "confidence_based"
    component_weights: Dict[str, float] = field(default_factory=dict)
    model_agreement: float = 0.0
    
    # Enhanced Interpretability
    temporal_factors: List[str] = field(default_factory=list)  # From TFT attention
    relationship_factors: List[str] = field(default_factory=list)  # From GNN analysis
    cross_modal_insights: List[str] = field(default_factory=list)  # Combined insights
    
    # Risk and Market Analysis
    systemic_risk_score: float = 0.0
    sector_influence: float = 0.0
    market_influence: float = 0.0
    contagion_risk: float = 0.0
    
    # Performance Metrics
    prediction_time: float = 0.0
    components_used: List[str] = field(default_factory=list)
    model_version: str = "Phase4_GNN_TFT_v1.0"

class GNNTFTIntegratedPredictor:
    """
    Advanced integrated predictor combining Graph Neural Networks and Temporal Fusion Transformers.
    
    This system provides comprehensive market prediction by leveraging:
    1. TFT for temporal pattern recognition and attention-based forecasting
    2. GNN for market relationship modeling and cross-asset intelligence
    3. Intelligent fusion mechanisms for optimal prediction combination
    4. Enhanced interpretability through multi-modal analysis
    """
    
    def __init__(self, config: Union[GNNTFTConfig, Dict] = None):
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        if isinstance(config, GNNTFTConfig):
            self.config = config
        else:
            config_dict = config or {}
            self.config = GNNTFTConfig(**config_dict)
        
        # Initialize components
        self.tft_predictor = None
        self.gnn_predictor = None
        self.phase4_fallback = None
        
        # Initialize TFT predictor
        if TFT_AVAILABLE:
            try:
                from phase4_temporal_fusion_transformer import TFTConfig
                tft_config = TFTConfig() if isinstance(self.config.tft_config, dict) else self.config.tft_config
                self.tft_predictor = TemporalFusionPredictor(tft_config)
                self.logger.info("✅ TFT predictor initialized")
            except Exception as e:
                self.logger.warning(f"TFT predictor initialization failed: {e}")
        else:
            self.logger.info("⚠️ TFT predictor not available (missing PyTorch dependencies)")
        
        # Initialize GNN predictor
        if GNN_AVAILABLE:
            try:
                gnn_config = self.config.gnn_config if hasattr(self.config, 'gnn_config') and self.config.gnn_config else GNNConfig()
                self.gnn_predictor = GNNEnhancedPredictor(gnn_config)
                self.logger.info("✅ GNN predictor initialized")
            except Exception as e:
                self.logger.warning(f"GNN predictor initialization failed: {e}")
        else:
            self.logger.warning("⚠️ GNN predictor not available")
        
        # Initialize Phase 4 TFT fallback
        if PHASE4_TFT_AVAILABLE:
            try:
                self.phase4_fallback = Phase4TFTIntegratedPredictor()
                self.logger.info("✅ Phase 4 TFT fallback available")
            except Exception as e:
                self.logger.warning(f"Phase 4 TFT fallback initialization failed: {e}")
        
        # Performance tracking
        self.prediction_history = []
        self.performance_stats = {
            'tft_usage_rate': 0.0,
            'gnn_usage_rate': 0.0,
            'fusion_accuracy': 0.0,
            'average_prediction_time': 0.0
        }
        
        self.logger.info("🚀 GNN + TFT Integrated Predictor initialized")
    
    async def generate_multimodal_prediction(
        self,
        symbol: str,
        time_horizon: str = '5d',
        related_symbols: List[str] = None,
        include_detailed_analysis: bool = True
    ) -> MultiModalPrediction:
        """
        Generate comprehensive multi-modal prediction using both TFT and GNN.
        
        Args:
            symbol: Target symbol for prediction
            time_horizon: Prediction horizon ('1d', '5d', '30d', '90d')
            related_symbols: Symbols to include in GNN graph analysis
            include_detailed_analysis: Whether to include detailed interpretability analysis
        
        Returns:
            MultiModalPrediction with comprehensive temporal and relational insights
        """
        start_time = datetime.now()
        
        try:
            self.logger.info(f"🚀 Generating multi-modal prediction for {symbol} ({time_horizon})")
            
            # Get current price
            current_price = await self._get_current_price(symbol)
            
            # Initialize prediction containers
            tft_result = None
            gnn_result = None
            tft_confidence = 0.0
            gnn_confidence = 0.0
            
            # Run TFT and GNN predictions concurrently
            prediction_tasks = []
            
            # TFT prediction task
            if self.tft_predictor:
                tft_task = asyncio.create_task(
                    self._generate_tft_prediction(symbol, time_horizon)
                )
                prediction_tasks.append(('tft', tft_task))
            
            # GNN prediction task
            if self.gnn_predictor:
                gnn_task = asyncio.create_task(
                    self._generate_gnn_prediction(symbol, related_symbols)
                )
                prediction_tasks.append(('gnn', gnn_task))
            
            # Execute predictions with timeout
            completed_predictions = {}
            
            for pred_type, task in prediction_tasks:
                try:
                    result = await asyncio.wait_for(task, timeout=self.config.max_prediction_time)
                    completed_predictions[pred_type] = result
                except asyncio.TimeoutError:
                    self.logger.warning(f"{pred_type.upper()} prediction timed out")
                except Exception as e:
                    self.logger.warning(f"{pred_type.upper()} prediction failed: {e}")
            
            # Extract results
            if 'tft' in completed_predictions:
                tft_result = completed_predictions['tft']
                tft_confidence = tft_result.model_confidence if tft_result else 0.0
            
            if 'gnn' in completed_predictions:
                gnn_result = completed_predictions['gnn'] 
                gnn_confidence = gnn_result.confidence_score if gnn_result else 0.0
            
            # Determine fusion strategy
            fusion_method, component_weights = self._determine_fusion_strategy(
                tft_result, gnn_result, tft_confidence, gnn_confidence
            )
            
            # Fuse predictions
            fused_prediction = self._fuse_predictions(
                symbol=symbol,
                time_horizon=time_horizon,
                current_price=current_price,
                tft_result=tft_result,
                gnn_result=gnn_result,
                fusion_method=fusion_method,
                weights=component_weights,
                start_time=start_time
            )
            
            # Enhance with interpretability analysis
            if include_detailed_analysis:
                fused_prediction = await self._enhance_interpretability(fused_prediction)
            
            # Update performance tracking
            self._update_performance_stats(fused_prediction)
            
            prediction_time = (datetime.now() - start_time).total_seconds()
            fused_prediction.prediction_time = prediction_time
            
            self.logger.info(
                f"✅ Multi-modal prediction completed for {symbol}: "
                f"${fused_prediction.predicted_price:.2f} "
                f"({fused_prediction.direction}, {fused_prediction.confidence_score:.3f} conf, "
                f"{prediction_time:.2f}s, {fusion_method})"
            )
            
            return fused_prediction
            
        except Exception as e:
            self.logger.error(f"❌ Multi-modal prediction failed for {symbol}: {e}")
            
            # Fallback to Phase 4 TFT if available
            if self.phase4_fallback:
                try:
                    fallback_result = await self.phase4_fallback.generate_phase4_prediction(
                        symbol, time_horizon
                    )
                    return self._convert_phase4_to_multimodal(fallback_result, start_time)
                except Exception as fallback_error:
                    self.logger.error(f"Fallback also failed: {fallback_error}")
            
            raise RuntimeError(f"All prediction methods failed for {symbol}: {e}")
    
    async def _generate_tft_prediction(self, symbol: str, time_horizon: str) -> Any:
        """Generate TFT prediction with error handling."""
        try:
            return await self.tft_predictor.generate_tft_prediction(symbol, [time_horizon])
        except Exception as e:
            self.logger.warning(f"TFT prediction failed for {symbol}: {e}")
            return None
    
    async def _generate_gnn_prediction(self, symbol: str, related_symbols: List[str] = None) -> Any:
        """Generate GNN prediction with error handling."""
        try:
            # Limit related symbols to avoid large graphs
            if related_symbols:
                related_symbols = related_symbols[:self.config.max_related_symbols]
            
            return await self.gnn_predictor.generate_gnn_enhanced_prediction(
                symbol, related_symbols
            )
        except Exception as e:
            self.logger.warning(f"GNN prediction failed for {symbol}: {e}")
            return None
    
    def _determine_fusion_strategy(
        self,
        tft_result: Optional[Any],
        gnn_result: Optional[Any],
        tft_confidence: float,
        gnn_confidence: float
    ) -> Tuple[str, Dict[str, float]]:
        """Determine optimal fusion strategy based on available predictions and confidence."""
        
        tft_available = tft_result is not None and tft_confidence >= self.config.confidence_threshold
        gnn_available = gnn_result is not None and gnn_confidence >= self.config.confidence_threshold
        
        if self.config.fusion_method == FusionMethod.CONFIDENCE_BASED:
            if tft_available and gnn_available:
                # Weight by confidence
                total_conf = tft_confidence + gnn_confidence
                if total_conf > 0:
                    tft_weight = tft_confidence / total_conf
                    gnn_weight = gnn_confidence / total_conf
                else:
                    tft_weight = self.config.tft_weight
                    gnn_weight = self.config.gnn_weight
                
                return "confidence_fusion", {"tft": tft_weight, "gnn": gnn_weight}
            
            elif tft_available:
                return "tft_primary", {"tft": 1.0, "gnn": 0.0}
            
            elif gnn_available:
                return "gnn_primary", {"tft": 0.0, "gnn": 1.0}
            
            else:
                return "none_available", {"tft": 0.0, "gnn": 0.0}
        
        elif self.config.fusion_method == FusionMethod.WEIGHTED_AVERAGE:
            if tft_available and gnn_available:
                return "weighted_average", {
                    "tft": self.config.tft_weight, 
                    "gnn": self.config.gnn_weight
                }
            elif tft_available:
                return "tft_primary", {"tft": 1.0, "gnn": 0.0}
            elif gnn_available:
                return "gnn_primary", {"tft": 0.0, "gnn": 1.0}
            else:
                return "none_available", {"tft": 0.0, "gnn": 0.0}
        
        elif self.config.fusion_method == FusionMethod.ADAPTIVE:
            # Adaptive based on market conditions and relationship strength
            if tft_available and gnn_available:
                # Favor GNN if strong relationships exist
                relationship_strength = 0.0
                if gnn_result and gnn_result.neighbor_influence:
                    relationship_strength = max(gnn_result.neighbor_influence.values())
                
                if relationship_strength > self.config.relationship_threshold:
                    # Strong relationships - favor GNN
                    tft_weight = 0.3
                    gnn_weight = 0.7
                else:
                    # Weak relationships - favor TFT
                    tft_weight = 0.7
                    gnn_weight = 0.3
                
                return "adaptive_fusion", {"tft": tft_weight, "gnn": gnn_weight}
            
            elif tft_available:
                return "tft_primary", {"tft": 1.0, "gnn": 0.0}
            elif gnn_available:
                return "gnn_primary", {"tft": 0.0, "gnn": 1.0}
            else:
                return "none_available", {"tft": 0.0, "gnn": 0.0}
        
        # Default to confidence-based
        return self._determine_fusion_strategy(tft_result, gnn_result, tft_confidence, gnn_confidence)
    
    def _fuse_predictions(
        self,
        symbol: str,
        time_horizon: str,
        current_price: float,
        tft_result: Optional[Any],
        gnn_result: Optional[Any],
        fusion_method: str,
        weights: Dict[str, float],
        start_time: datetime
    ) -> MultiModalPrediction:
        """Fuse TFT and GNN predictions into unified result."""
        
        # Extract individual predictions
        tft_price = None
        gnn_price = None
        
        if tft_result and time_horizon in tft_result.point_predictions:
            tft_price = tft_result.point_predictions[time_horizon]
        
        if gnn_result:
            gnn_price = gnn_result.predicted_price
        
        # Fuse predictions
        if tft_price and gnn_price and weights["tft"] > 0 and weights["gnn"] > 0:
            # Weighted fusion
            predicted_price = (
                tft_price * weights["tft"] + 
                gnn_price * weights["gnn"]
            )
            
            # Fuse confidence
            tft_conf = tft_result.model_confidence if tft_result else 0.0
            gnn_conf = gnn_result.confidence_score if gnn_result else 0.0
            confidence_score = (
                tft_conf * weights["tft"] +
                gnn_conf * weights["gnn"]
            )
            
            components_used = ["TFT", "GNN"]
            
        elif tft_price and weights["tft"] > 0:
            # TFT only
            predicted_price = tft_price
            confidence_score = tft_result.model_confidence
            components_used = ["TFT"]
            
        elif gnn_price and weights["gnn"] > 0:
            # GNN only
            predicted_price = gnn_price
            confidence_score = gnn_result.confidence_score
            components_used = ["GNN"]
            
        else:
            # Fallback - simple prediction
            predicted_price = current_price * (1 + np.random.normal(0, 0.02))
            confidence_score = 0.1
            components_used = ["FALLBACK"]
        
        # Calculate derived metrics
        expected_return = (predicted_price - current_price) / current_price
        direction = "UP" if expected_return > 0 else ("DOWN" if expected_return < 0 else "FLAT")
        probability_up = 0.5 + (expected_return * 2)
        probability_up = max(0.0, min(1.0, probability_up))
        
        # Calculate uncertainty and confidence interval
        uncertainty_score = 0.05  # Default
        confidence_interval = (predicted_price * 0.95, predicted_price * 1.05)
        
        if tft_result and time_horizon in tft_result.uncertainty_scores:
            uncertainty_score = tft_result.uncertainty_scores[time_horizon]
            if time_horizon in tft_result.confidence_intervals:
                confidence_interval = tft_result.confidence_intervals[time_horizon]
        
        # Model agreement
        model_agreement = 0.0
        if tft_price and gnn_price:
            price_diff = abs(tft_price - gnn_price) / max(tft_price, gnn_price)
            model_agreement = max(0.0, 1.0 - price_diff * 5)
        
        # Risk metrics from GNN
        systemic_risk = gnn_result.systemic_risk_score if gnn_result else 0.0
        sector_influence = gnn_result.sector_influence if gnn_result else 0.0
        market_influence = gnn_result.market_influence if gnn_result else 0.0
        contagion_risk = gnn_result.contagion_potential if gnn_result else 0.0
        
        return MultiModalPrediction(
            symbol=symbol,
            time_horizon=time_horizon,
            prediction_timestamp=datetime.now(timezone.utc),
            predicted_price=predicted_price,
            current_price=current_price,
            expected_return=expected_return,
            direction=direction,
            confidence_score=confidence_score,
            uncertainty_score=uncertainty_score,
            confidence_interval=confidence_interval,
            probability_up=probability_up,
            tft_result=tft_result,
            tft_confidence=tft_result.model_confidence if tft_result else 0.0,
            gnn_result=gnn_result,
            gnn_confidence=gnn_result.confidence_score if gnn_result else 0.0,
            fusion_method=fusion_method,
            component_weights=weights,
            model_agreement=model_agreement,
            systemic_risk_score=systemic_risk,
            sector_influence=sector_influence,
            market_influence=market_influence,
            contagion_risk=contagion_risk,
            components_used=components_used
        )
    
    async def _enhance_interpretability(self, prediction: MultiModalPrediction) -> MultiModalPrediction:
        """Enhance prediction with interpretability insights."""
        
        # Extract TFT temporal factors
        temporal_factors = []
        if prediction.tft_result and prediction.tft_result.variable_importances:
            for var_type, importance in prediction.tft_result.variable_importances.items():
                if len(importance) > 0:
                    top_idx = np.argmax(importance)
                    temporal_factors.append(f"{var_type}_feature_{top_idx}")
        
        # Extract GNN relationship factors
        relationship_factors = []
        if prediction.gnn_result and prediction.gnn_result.key_relationships:
            for symbol, rel_type, strength in prediction.gnn_result.key_relationships[:3]:
                relationship_factors.append(f"{symbol}_{rel_type}")
        
        # Generate cross-modal insights
        cross_modal_insights = []
        
        if prediction.model_agreement > 0.8:
            cross_modal_insights.append("High temporal-relational agreement")
        elif prediction.model_agreement < 0.5:
            cross_modal_insights.append("Divergent temporal vs relational signals")
        
        if prediction.systemic_risk_score > 0.7:
            cross_modal_insights.append("High systemic risk detected")
        
        if prediction.sector_influence > 0.6:
            cross_modal_insights.append("Strong sector influence")
        
        # TFT attention insights
        tft_insights = {}
        if prediction.tft_result:
            tft_insights = {
                "multi_horizon_available": len(prediction.tft_result.point_predictions) > 1,
                "uncertainty_quantified": len(prediction.tft_result.uncertainty_scores) > 0,
                "attention_available": prediction.tft_result.attention_weights is not None
            }
        
        # GNN relationship insights
        relationship_insights = {}
        if prediction.gnn_result:
            relationship_insights = {
                "node_importance": prediction.gnn_result.node_importance,
                "neighbor_count": len(prediction.gnn_result.neighbor_influence),
                "centrality_score": prediction.gnn_result.graph_centrality,
                "key_relationships_count": len(prediction.gnn_result.key_relationships)
            }
        
        # Update prediction with insights
        prediction.temporal_factors = temporal_factors
        prediction.relationship_factors = relationship_factors
        prediction.cross_modal_insights = cross_modal_insights
        prediction.tft_attention_insights = tft_insights
        prediction.relationship_insights = relationship_insights
        
        return prediction
    
    async def _get_current_price(self, symbol: str) -> float:
        """Get current market price for the symbol."""
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="1d", interval="1m")
            
            if not data.empty:
                return float(data['Close'].iloc[-1])
            else:
                # Fallback to daily data
                data = ticker.history(period="5d")
                if not data.empty:
                    return float(data['Close'].iloc[-1])
                else:
                    raise ValueError(f"No price data available for {symbol}")
        except Exception as e:
            self.logger.error(f"Error getting current price for {symbol}: {e}")
            raise
    
    def _convert_phase4_to_multimodal(
        self, 
        phase4_result: 'Phase4Prediction', 
        start_time: datetime
    ) -> MultiModalPrediction:
        """Convert Phase4 prediction to MultiModal format as fallback."""
        
        prediction_time = (datetime.now() - start_time).total_seconds()
        
        return MultiModalPrediction(
            symbol=phase4_result.symbol,
            time_horizon=phase4_result.time_horizon,
            prediction_timestamp=phase4_result.prediction_timestamp,
            predicted_price=phase4_result.predicted_price,
            current_price=phase4_result.current_price,
            expected_return=phase4_result.expected_return,
            direction=phase4_result.direction,
            confidence_score=phase4_result.confidence_score,
            uncertainty_score=phase4_result.uncertainty_score,
            confidence_interval=phase4_result.confidence_interval,
            probability_up=phase4_result.probability_up,
            fusion_method="phase4_fallback",
            component_weights={"phase4": 1.0},
            components_used=["PHASE4_FALLBACK"],
            prediction_time=prediction_time
        )
    
    def _update_performance_stats(self, prediction: MultiModalPrediction):
        """Update performance statistics."""
        
        # Update usage rates
        if "TFT" in prediction.components_used:
            self.performance_stats['tft_usage_rate'] += 1
        
        if "GNN" in prediction.components_used:
            self.performance_stats['gnn_usage_rate'] += 1
        
        # Add to history
        self.prediction_history.append({
            'timestamp': prediction.prediction_timestamp,
            'symbol': prediction.symbol,
            'fusion_method': prediction.fusion_method,
            'components_used': prediction.components_used,
            'confidence': prediction.confidence_score,
            'model_agreement': prediction.model_agreement
        })
        
        # Keep history limited
        if len(self.prediction_history) > 100:
            self.prediction_history = self.prediction_history[-100:]
        
        # Update averages
        if len(self.prediction_history) > 0:
            total_predictions = len(self.prediction_history)
            self.performance_stats['tft_usage_rate'] = self.performance_stats['tft_usage_rate'] / total_predictions
            self.performance_stats['gnn_usage_rate'] = self.performance_stats['gnn_usage_rate'] / total_predictions
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        
        return {
            'multimodal_integration': {
                'tft_available': self.tft_predictor is not None,
                'gnn_available': self.gnn_predictor is not None,
                'phase4_fallback_available': self.phase4_fallback is not None,
                'fusion_method': self.config.fusion_method.value,
                'ensemble_enabled': self.config.enable_ensemble
            },
            'component_status': {
                'tft_system': self.tft_predictor is not None,
                'gnn_system': self.gnn_predictor is not None
            },
            'performance_stats': self.performance_stats,
            'configuration': {
                'tft_weight': self.config.tft_weight,
                'gnn_weight': self.config.gnn_weight,
                'adaptive_weighting': self.config.enable_adaptive_weighting,
                'max_related_symbols': self.config.max_related_symbols
            },
            'recent_predictions': len(self.prediction_history),
            'version': "Phase4_GNN_TFT_Integration_v1.0"
        }

# Global integrated predictor instance
gnn_tft_predictor = GNNTFTIntegratedPredictor()

if __name__ == "__main__":
    # Example usage and testing
    import asyncio
    
    async def test_multimodal_prediction():
        """Test multi-modal GNN + TFT prediction."""
        try:
            # Test symbols
            test_symbols = ['AAPL', 'CBA.AX']
            
            for symbol in test_symbols:
                print(f"\n=== Multi-Modal Prediction Test for {symbol} ===")
                
                result = await gnn_tft_predictor.generate_multimodal_prediction(
                    symbol=symbol,
                    time_horizon='5d',
                    related_symbols=['MSFT', 'GOOGL', 'WBC.AX', 'ANZ.AX'][:5]
                )
                
                print(f"Symbol: {result.symbol}")
                print(f"Predicted Price: ${result.predicted_price:.2f}")
                print(f"Current Price: ${result.current_price:.2f}")
                print(f"Expected Return: {result.expected_return:.2%}")
                print(f"Direction: {result.direction}")
                print(f"Confidence: {result.confidence_score:.3f}")
                print(f"Fusion Method: {result.fusion_method}")
                print(f"Component Weights: {result.component_weights}")
                print(f"Model Agreement: {result.model_agreement:.3f}")
                print(f"Components Used: {result.components_used}")
                print(f"Prediction Time: {result.prediction_time:.2f}s")
                
                if result.temporal_factors:
                    print(f"Temporal Factors: {result.temporal_factors}")
                
                if result.relationship_factors:
                    print(f"Relationship Factors: {result.relationship_factors}")
                
                if result.cross_modal_insights:
                    print(f"Cross-Modal Insights: {result.cross_modal_insights}")
                
                print(f"Systemic Risk: {result.systemic_risk_score:.3f}")
                
        except Exception as e:
            print(f"Test failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Run test
    asyncio.run(test_multimodal_prediction())