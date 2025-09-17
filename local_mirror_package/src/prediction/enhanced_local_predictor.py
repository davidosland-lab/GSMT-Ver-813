"""
Enhanced Local Prediction Model - Stock Market Tracker
Uses comprehensive document analysis and technical data for superior predictions
"""

import sqlite3
import json
import logging
import asyncio
import numpy as np
import pandas as pd
import yfinance as yf
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from pathlib import Path

class EnhancedLocalPredictor:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
            
        self.db_path = self.config['database']['path']
        self.model_version = self.config['prediction']['model_version']
        
        # Model weights
        self.document_weight = self.config['prediction']['document_weight']
        self.technical_weight = self.config['prediction']['technical_weight']
        self.market_weight = self.config['prediction']['market_weight']
        self.min_documents = self.config['prediction']['min_documents_for_analysis']
        
        # Model storage
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize models
        self.price_model = None
        self.direction_model = None
        self.scaler = StandardScaler()
        
        # Load existing models if available
        self._load_models()
        
    async def predict_stock_price(self, symbol: str, timeframe: str = '5d') -> Dict:
        """Generate enhanced prediction using document analysis and technical data"""
        self.logger.info(f"🎯 Generating enhanced prediction for {symbol} ({timeframe})")
        
        try:
            # Get current price data
            current_price = await self._get_current_price(symbol)
            if not current_price:
                return {'success': False, 'error': 'Could not fetch current price'}
                
            # Extract features for prediction
            features = await self._extract_prediction_features(symbol)
            if not features:
                return {'success': False, 'error': 'Could not extract features'}
                
            # Make prediction
            prediction_result = await self._make_prediction(symbol, features, timeframe)
            
            # Calculate confidence and additional metrics
            confidence_metrics = await self._calculate_confidence_metrics(symbol, features, prediction_result)
            
            # Get supporting evidence from documents
            document_evidence = await self._get_document_evidence(symbol)
            
            # Prepare final result
            result = {
                'success': True,
                'symbol': symbol,
                'timeframe': timeframe,
                'current_price': current_price,
                'prediction': {
                    'predicted_price': prediction_result['predicted_price'],
                    'direction': prediction_result['direction'],
                    'confidence_score': confidence_metrics['confidence'],
                    'expected_change_percent': ((prediction_result['predicted_price'] - current_price) / current_price) * 100,
                    'confidence_interval': {
                        'lower': prediction_result['predicted_price'] - confidence_metrics['uncertainty'],
                        'upper': prediction_result['predicted_price'] + confidence_metrics['uncertainty']
                    },
                    'probability_up': prediction_result['probability_up'],
                    'probability_down': 1 - prediction_result['probability_up']
                },
                'analysis_breakdown': {
                    'technical_analysis': features.get('technical_score', 0.5),
                    'document_sentiment': features.get('document_sentiment', 0.5),
                    'market_conditions': features.get('market_score', 0.5),
                    'document_count': features.get('document_count', 0)
                },
                'supporting_evidence': document_evidence,
                'model_version': self.model_version,
                'prediction_timestamp': datetime.now().isoformat()
            }
            
            # Save prediction to database
            await self._save_prediction(symbol, result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error predicting {symbol}: {e}")
            return {'success': False, 'error': str(e)}
            
    async def _extract_prediction_features(self, symbol: str) -> Optional[Dict]:
        """Extract comprehensive features for prediction"""
        features = {}
        
        try:
            # 1. Technical Analysis Features
            technical_features = await self._extract_technical_features(symbol)
            features.update(technical_features)
            
            # 2. Document Analysis Features
            document_features = await self._extract_document_features(symbol)
            features.update(document_features)
            
            # 3. Market Context Features
            market_features = await self._extract_market_features()
            features.update(market_features)
            
            # 4. Historical Performance Features
            historical_features = await self._extract_historical_features(symbol)
            features.update(historical_features)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting features for {symbol}: {e}")
            return None
            
    async def _extract_technical_features(self, symbol: str) -> Dict:
        """Extract technical analysis features"""
        features = {}
        
        try:
            # Get historical data
            ticker = yf.Ticker(symbol)
            hist_data = ticker.history(period="3mo")
            
            if hist_data.empty:
                return {'technical_score': 0.5}
                
            # Calculate technical indicators
            close_prices = hist_data['Close']
            
            # Moving averages
            ma_5 = close_prices.rolling(window=5).mean()
            ma_20 = close_prices.rolling(window=20).mean()
            ma_50 = close_prices.rolling(window=50).mean()
            
            # Current position relative to MAs
            current_price = close_prices.iloc[-1]
            features['price_vs_ma5'] = (current_price - ma_5.iloc[-1]) / ma_5.iloc[-1] if not pd.isna(ma_5.iloc[-1]) else 0
            features['price_vs_ma20'] = (current_price - ma_20.iloc[-1]) / ma_20.iloc[-1] if not pd.isna(ma_20.iloc[-1]) else 0
            features['price_vs_ma50'] = (current_price - ma_50.iloc[-1]) / ma_50.iloc[-1] if not pd.isna(ma_50.iloc[-1]) else 0
            
            # RSI
            rsi = self._calculate_rsi(close_prices)
            features['rsi'] = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
            features['rsi_normalized'] = (features['rsi'] - 50) / 50  # Normalize to -1 to 1
            
            # Volatility
            returns = close_prices.pct_change().dropna()
            features['volatility'] = returns.std() * np.sqrt(252)  # Annualized volatility
            
            # Volume analysis
            if 'Volume' in hist_data.columns:
                volume_ma = hist_data['Volume'].rolling(window=20).mean()
                current_volume = hist_data['Volume'].iloc[-1]
                features['volume_ratio'] = current_volume / volume_ma.iloc[-1] if not pd.isna(volume_ma.iloc[-1]) else 1
            else:
                features['volume_ratio'] = 1
                
            # Price momentum
            price_1d = close_prices.iloc[-2] if len(close_prices) > 1 else current_price
            price_5d = close_prices.iloc[-6] if len(close_prices) > 5 else current_price
            price_20d = close_prices.iloc[-21] if len(close_prices) > 20 else current_price
            
            features['momentum_1d'] = (current_price - price_1d) / price_1d
            features['momentum_5d'] = (current_price - price_5d) / price_5d
            features['momentum_20d'] = (current_price - price_20d) / price_20d
            
            # Technical score (0-1 scale)
            technical_indicators = [
                1 if features['price_vs_ma5'] > 0 else 0,
                1 if features['price_vs_ma20'] > 0 else 0,
                1 if features['rsi'] > 30 and features['rsi'] < 70 else 0.5,
                1 if features['momentum_5d'] > 0 else 0,
                1 if features['volume_ratio'] > 1 else 0.5
            ]
            
            features['technical_score'] = np.mean(technical_indicators)
            
        except Exception as e:
            self.logger.error(f"Error extracting technical features: {e}")
            features['technical_score'] = 0.5
            
        return features
        
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
        
    async def _extract_document_features(self, symbol: str) -> Dict:
        """Extract features from document analysis"""
        features = {}
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get recent document analyses
            cursor.execute('''
            SELECT da.analysis_type, da.analysis_result, da.confidence_score
            FROM document_analysis da
            JOIN documents d ON da.document_id = d.id
            WHERE d.symbol = ? 
            AND da.created_date > date('now', '-90 days')
            ORDER BY da.created_date DESC
            ''', (symbol,))
            
            analyses = cursor.fetchall()
            conn.close()
            
            if not analyses:
                features.update({
                    'document_sentiment': 0.5,
                    'document_count': 0,
                    'sentiment_confidence': 0.3,
                    'risk_score': 0.5,
                    'business_outlook': 0.5
                })
                return features
                
            # Process sentiment analyses
            sentiment_scores = []
            risk_scores = []
            business_scores = []
            
            for analysis_type, analysis_result, confidence in analyses:
                try:
                    result = json.loads(analysis_result)
                    
                    if analysis_type == 'sentiment':
                        if 'overall_sentiment' in result:
                            sentiment = result['overall_sentiment']
                            if sentiment == 'positive':
                                sentiment_scores.append(0.8)
                            elif sentiment == 'negative':
                                sentiment_scores.append(0.2)
                            else:
                                sentiment_scores.append(0.5)
                                
                    elif analysis_type == 'risk_analysis':
                        if 'risk_score' in result:
                            risk_scores.append(result['risk_score'])
                            
                    elif analysis_type == 'business_insights':
                        if 'strategic_mentions' in result:
                            # Score based on positive business indicators
                            positive_indicators = ['growth', 'expansion', 'innovation', 'strategy']
                            total_mentions = sum(result['strategic_mentions'].values())
                            positive_mentions = sum(
                                result['strategic_mentions'].get(indicator, 0) 
                                for indicator in positive_indicators
                            )
                            if total_mentions > 0:
                                business_scores.append(positive_mentions / total_mentions)
                                
                except Exception as e:
                    self.logger.warning(f"Error processing analysis result: {e}")
                    continue
                    
            # Aggregate scores
            features['document_sentiment'] = np.mean(sentiment_scores) if sentiment_scores else 0.5
            features['risk_score'] = np.mean(risk_scores) if risk_scores else 0.5
            features['business_outlook'] = np.mean(business_scores) if business_scores else 0.5
            features['document_count'] = len(analyses)
            features['sentiment_confidence'] = 0.8 if len(sentiment_scores) >= self.min_documents else 0.3
            
            # Weighted document score
            if features['document_count'] >= self.min_documents:
                document_weights = [0.4, 0.3, 0.3]  # sentiment, risk, business
                document_components = [
                    features['document_sentiment'],
                    1 - features['risk_score'],  # Lower risk = better score
                    features['business_outlook']
                ]
                features['document_composite'] = np.average(document_components, weights=document_weights)
            else:
                features['document_composite'] = 0.5
                
        except Exception as e:
            self.logger.error(f"Error extracting document features: {e}")
            features.update({
                'document_sentiment': 0.5,
                'document_count': 0,
                'sentiment_confidence': 0.3,
                'risk_score': 0.5,
                'business_outlook': 0.5,
                'document_composite': 0.5
            })
            
        return features
        
    async def _extract_market_features(self) -> Dict:
        """Extract broader market context features"""
        features = {}
        
        try:
            # Get ASX 200 index data for market context
            asx_ticker = yf.Ticker("^AXJO")
            asx_data = asx_ticker.history(period="1mo")
            
            if not asx_data.empty:
                asx_returns = asx_data['Close'].pct_change().dropna()
                current_asx = asx_data['Close'].iloc[-1]
                asx_20d_ago = asx_data['Close'].iloc[-20] if len(asx_data) > 20 else current_asx
                
                features['market_momentum'] = (current_asx - asx_20d_ago) / asx_20d_ago
                features['market_volatility'] = asx_returns.std() * np.sqrt(252)
                features['market_trend'] = 1 if features['market_momentum'] > 0 else 0
            else:
                features.update({
                    'market_momentum': 0,
                    'market_volatility': 0.2,
                    'market_trend': 0.5
                })
                
            # Market score
            market_indicators = [
                1 if features['market_momentum'] > 0 else 0,
                1 if features['market_volatility'] < 0.3 else 0,  # Low volatility is good
            ]
            features['market_score'] = np.mean(market_indicators)
            
        except Exception as e:
            self.logger.error(f"Error extracting market features: {e}")
            features.update({
                'market_momentum': 0,
                'market_volatility': 0.2,
                'market_trend': 0.5,
                'market_score': 0.5
            })
            
        return features
        
    async def _extract_historical_features(self, symbol: str) -> Dict:
        """Extract historical performance features"""
        features = {}
        
        try:
            ticker = yf.Ticker(symbol)
            hist_data = ticker.history(period="1y")
            
            if not hist_data.empty:
                close_prices = hist_data['Close']
                
                # Performance metrics
                current_price = close_prices.iloc[-1]
                ytd_start = close_prices.iloc[0]
                
                features['ytd_return'] = (current_price - ytd_start) / ytd_start
                features['max_drawdown'] = self._calculate_max_drawdown(close_prices)
                
                # Seasonal patterns (month of year effect)
                current_month = datetime.now().month
                features['month_effect'] = self._get_seasonal_effect(current_month)
                
            else:
                features.update({
                    'ytd_return': 0,
                    'max_drawdown': 0,
                    'month_effect': 0
                })
                
        except Exception as e:
            self.logger.error(f"Error extracting historical features: {e}")
            features.update({
                'ytd_return': 0,
                'max_drawdown': 0,
                'month_effect': 0
            })
            
        return features
        
    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cumulative = (1 + prices.pct_change()).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
        
    def _get_seasonal_effect(self, month: int) -> float:
        """Get seasonal effect for given month (simplified)"""
        # Simplified seasonal effects (can be enhanced with historical data)
        seasonal_effects = {
            1: 0.02, 2: 0.01, 3: 0.03, 4: 0.02, 5: -0.01, 6: -0.02,
            7: 0.01, 8: -0.01, 9: -0.02, 10: 0.02, 11: 0.03, 12: 0.01
        }
        return seasonal_effects.get(month, 0)
        
    async def _make_prediction(self, symbol: str, features: Dict, timeframe: str) -> Dict:
        """Make price prediction using extracted features"""
        try:
            # Convert timeframe to days
            timeframe_days = {'1d': 1, '5d': 5, '30d': 30, '90d': 90}.get(timeframe, 5)
            
            # Get current price
            current_price = await self._get_current_price(symbol)
            
            # Prepare feature vector
            feature_vector = self._prepare_feature_vector(features)
            
            # Use ensemble approach
            predictions = []
            
            # 1. Document-informed prediction
            if features.get('document_count', 0) >= self.min_documents:
                doc_prediction = self._document_based_prediction(current_price, features, timeframe_days)
                predictions.append(('document', doc_prediction, self.document_weight))
                
            # 2. Technical analysis prediction
            technical_prediction = self._technical_prediction(current_price, features, timeframe_days)
            predictions.append(('technical', technical_prediction, self.technical_weight))
            
            # 3. Market context prediction
            market_prediction = self._market_based_prediction(current_price, features, timeframe_days)
            predictions.append(('market', market_prediction, self.market_weight))
            
            # Weighted ensemble
            weighted_price = 0
            total_weight = 0
            
            for pred_type, pred_price, weight in predictions:
                weighted_price += pred_price * weight
                total_weight += weight
                
            final_prediction = weighted_price / total_weight if total_weight > 0 else current_price
            
            # Direction and probability
            direction = 'up' if final_prediction > current_price else 'down'
            change_magnitude = abs((final_prediction - current_price) / current_price)
            
            # Probability based on confidence in direction
            base_probability = 0.5
            confidence_boost = min(0.3, change_magnitude * 2)  # Max 80% probability
            probability_up = base_probability + confidence_boost if direction == 'up' else base_probability - confidence_boost
            
            return {
                'predicted_price': final_prediction,
                'direction': direction,
                'probability_up': max(0.2, min(0.8, probability_up)),
                'prediction_components': [
                    {'type': pred_type, 'price': pred_price, 'weight': weight}
                    for pred_type, pred_price, weight in predictions
                ]
            }
            
        except Exception as e:
            self.logger.error(f"Error making prediction: {e}")
            # Fallback prediction
            current_price = await self._get_current_price(symbol)
            return {
                'predicted_price': current_price * 1.01,  # Small upward bias
                'direction': 'up',
                'probability_up': 0.55,
                'prediction_components': []
            }
            
    def _document_based_prediction(self, current_price: float, features: Dict, days: int) -> float:
        """Make prediction based on document analysis"""
        sentiment_score = features.get('document_sentiment', 0.5)
        business_outlook = features.get('business_outlook', 0.5)
        risk_score = features.get('risk_score', 0.5)
        
        # Convert sentiment and business factors to price change
        # Positive sentiment and low risk suggest price increase
        sentiment_impact = (sentiment_score - 0.5) * 0.1  # Max 5% impact
        business_impact = (business_outlook - 0.5) * 0.08  # Max 4% impact
        risk_impact = (0.5 - risk_score) * 0.06  # Lower risk = positive impact
        
        total_impact = (sentiment_impact + business_impact + risk_impact) * days / 5  # Scale by timeframe
        
        return current_price * (1 + total_impact)
        
    def _technical_prediction(self, current_price: float, features: Dict, days: int) -> float:
        """Make prediction based on technical analysis"""
        technical_score = features.get('technical_score', 0.5)
        momentum_5d = features.get('momentum_5d', 0)
        rsi_normalized = features.get('rsi_normalized', 0)
        
        # Technical momentum continuation
        momentum_impact = momentum_5d * 0.5  # Partial momentum continuation
        
        # RSI mean reversion (extreme values tend to revert)
        rsi_impact = -rsi_normalized * 0.03 if abs(rsi_normalized) > 0.6 else 0
        
        # Overall technical direction
        technical_impact = (technical_score - 0.5) * 0.08
        
        total_impact = (momentum_impact + rsi_impact + technical_impact) * days / 5
        
        return current_price * (1 + total_impact)
        
    def _market_based_prediction(self, current_price: float, features: Dict, days: int) -> float:
        """Make prediction based on market context"""
        market_momentum = features.get('market_momentum', 0)
        market_score = features.get('market_score', 0.5)
        
        # Market correlation (stocks tend to follow market direction)
        market_impact = market_momentum * 0.6  # High correlation with market
        
        # Market condition bonus/penalty
        condition_impact = (market_score - 0.5) * 0.04
        
        total_impact = (market_impact + condition_impact) * days / 5
        
        return current_price * (1 + total_impact)
        
    def _prepare_feature_vector(self, features: Dict) -> np.ndarray:
        """Prepare feature vector for ML models"""
        # Define feature order
        feature_names = [
            'technical_score', 'document_sentiment', 'market_score',
            'rsi_normalized', 'momentum_5d', 'volatility', 'volume_ratio',
            'document_count', 'risk_score', 'business_outlook',
            'market_momentum', 'ytd_return', 'max_drawdown'
        ]
        
        vector = []
        for feature_name in feature_names:
            value = features.get(feature_name, 0.5 if 'score' in feature_name else 0)
            vector.append(float(value))
            
        return np.array(vector).reshape(1, -1)
        
    async def _calculate_confidence_metrics(self, symbol: str, features: Dict, prediction: Dict) -> Dict:
        """Calculate confidence and uncertainty metrics"""
        try:
            # Base confidence from data availability
            base_confidence = 0.5
            
            # Document confidence boost
            doc_count = features.get('document_count', 0)
            if doc_count >= self.min_documents:
                doc_boost = min(0.3, doc_count / 20)  # Up to 30% boost
                base_confidence += doc_boost
                
            # Technical analysis confidence
            technical_clarity = abs(features.get('technical_score', 0.5) - 0.5) * 2
            base_confidence += technical_clarity * 0.2
            
            # Market alignment confidence
            market_clarity = abs(features.get('market_score', 0.5) - 0.5) * 2
            base_confidence += market_clarity * 0.1
            
            # Prediction consistency (how much components agree)
            if 'prediction_components' in prediction:
                prices = [comp['price'] for comp in prediction['prediction_components']]
                if len(prices) > 1:
                    price_std = np.std(prices)
                    current_price = await self._get_current_price(symbol)
                    consistency = 1 - (price_std / current_price)  # Lower std = higher consistency
                    base_confidence += consistency * 0.15
                    
            final_confidence = max(0.3, min(0.95, base_confidence))
            
            # Calculate uncertainty (for confidence intervals)
            volatility = features.get('volatility', 0.2)
            uncertainty_factor = volatility * (1 - final_confidence) * 0.5
            current_price = await self._get_current_price(symbol)
            uncertainty = current_price * uncertainty_factor
            
            return {
                'confidence': final_confidence,
                'uncertainty': uncertainty,
                'confidence_components': {
                    'document_availability': doc_count >= self.min_documents,
                    'technical_clarity': technical_clarity,
                    'market_alignment': market_clarity,
                    'prediction_consistency': len(prediction.get('prediction_components', [])) > 1
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating confidence: {e}")
            return {
                'confidence': 0.6,
                'uncertainty': (await self._get_current_price(symbol)) * 0.1,
                'confidence_components': {}
            }
            
    async def _get_document_evidence(self, symbol: str) -> Dict:
        """Get supporting evidence from document analysis"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get recent key insights
            cursor.execute('''
            SELECT d.title, da.analysis_type, da.analysis_result, da.confidence_score
            FROM document_analysis da
            JOIN documents d ON da.document_id = d.id
            WHERE d.symbol = ? 
            AND da.created_date > date('now', '-30 days')
            AND da.analysis_type IN ('sentiment', 'business_insights', 'summary')
            ORDER BY da.confidence_score DESC
            LIMIT 10
            ''', (symbol,))
            
            evidence = []
            for title, analysis_type, result_json, confidence in cursor.fetchall():
                try:
                    result = json.loads(result_json)
                    
                    if analysis_type == 'sentiment' and 'overall_sentiment' in result:
                        evidence.append({
                            'source': title,
                            'type': 'sentiment',
                            'finding': f"Sentiment: {result['overall_sentiment']}",
                            'confidence': confidence
                        })
                        
                    elif analysis_type == 'summary' and 'summary' in result:
                        evidence.append({
                            'source': title,
                            'type': 'summary',
                            'finding': result['summary'][:200] + "...",
                            'confidence': confidence
                        })
                        
                    elif analysis_type == 'business_insights' and 'forward_looking_statements' in result:
                        statements = result['forward_looking_statements'][:3]
                        if statements:
                            evidence.append({
                                'source': title,
                                'type': 'business_outlook',
                                'finding': '; '.join(statements),
                                'confidence': confidence
                            })
                            
                except Exception as e:
                    continue
                    
            conn.close()
            
            return {
                'evidence_count': len(evidence),
                'key_insights': evidence[:5],  # Top 5 insights
                'documents_analyzed': len(set(e['source'] for e in evidence))
            }
            
        except Exception as e:
            self.logger.error(f"Error getting document evidence: {e}")
            return {'evidence_count': 0, 'key_insights': [], 'documents_analyzed': 0}
            
    async def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current stock price"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1d", interval="1m")
            
            if not hist.empty:
                return float(hist['Close'].iloc[-1])
            else:
                # Fallback to daily data
                daily = ticker.history(period="5d")
                return float(daily['Close'].iloc[-1]) if not daily.empty else None
                
        except Exception as e:
            self.logger.error(f"Error fetching current price for {symbol}: {e}")
            return None
            
    async def _save_prediction(self, symbol: str, result: Dict):
        """Save prediction result to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            factors_used = {
                'technical_weight': self.technical_weight,
                'document_weight': self.document_weight,
                'market_weight': self.market_weight,
                'analysis_breakdown': result.get('analysis_breakdown', {}),
                'document_count': result['analysis_breakdown'].get('document_count', 0)
            }
            
            cursor.execute('''
            INSERT INTO predictions (
                symbol, timeframe, predicted_price, current_price, confidence_score,
                direction, factors_used, model_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                symbol,
                result['timeframe'],
                result['prediction']['predicted_price'],
                result['current_price'],
                result['prediction']['confidence_score'],
                result['prediction']['direction'],
                json.dumps(factors_used),
                self.model_version
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error saving prediction: {e}")
            
    def _load_models(self):
        """Load pre-trained models if available"""
        try:
            price_model_path = self.models_dir / "price_model.joblib"
            direction_model_path = self.models_dir / "direction_model.joblib"
            scaler_path = self.models_dir / "scaler.joblib"
            
            if price_model_path.exists():
                self.price_model = joblib.load(price_model_path)
                self.logger.info("✅ Loaded price prediction model")
                
            if direction_model_path.exists():
                self.direction_model = joblib.load(direction_model_path)
                self.logger.info("✅ Loaded direction prediction model")
                
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
                self.logger.info("✅ Loaded feature scaler")
                
        except Exception as e:
            self.logger.warning(f"Could not load existing models: {e}")
            
    async def train_models(self, symbols: List[str] = None) -> Dict:
        """Train prediction models on historical data"""
        self.logger.info("🎓 Starting model training...")
        
        if symbols is None:
            symbols = ['CBA.AX', 'WBC.AX', 'ANZ.AX', 'NAB.AX']  # Default training symbols
            
        training_results = {
            'symbols_processed': 0,
            'total_samples': 0,
            'model_performance': {}
        }
        
        try:
            # Collect training data
            all_features = []
            all_targets = []
            
            for symbol in symbols:
                self.logger.info(f"📊 Collecting training data for {symbol}")
                
                # Get historical predictions and actual outcomes
                features, targets = await self._collect_training_data(symbol)
                
                if len(features) > 0:
                    all_features.extend(features)
                    all_targets.extend(targets)
                    training_results['symbols_processed'] += 1
                    
            if len(all_features) < 10:
                return {'success': False, 'error': 'Insufficient training data'}
                
            # Prepare training data
            X = np.array(all_features)
            y = np.array(all_targets)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train price prediction model
            self.price_model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
            self.price_model.fit(X_scaled, y)
            
            # Calculate performance metrics
            y_pred = self.price_model.predict(X_scaled)
            mse = mean_squared_error(y, y_pred)
            r2 = r2_score(y, y_pred)
            
            training_results['total_samples'] = len(all_features)
            training_results['model_performance'] = {
                'mse': float(mse),
                'r2_score': float(r2),
                'rmse': float(np.sqrt(mse))
            }
            
            # Save models
            self._save_models()
            
            self.logger.info(f"✅ Model training completed: R² = {r2:.3f}, RMSE = {np.sqrt(mse):.3f}")
            
            return {
                'success': True,
                'training_results': training_results
            }
            
        except Exception as e:
            self.logger.error(f"Error training models: {e}")
            return {'success': False, 'error': str(e)}
            
    async def _collect_training_data(self, symbol: str) -> Tuple[List, List]:
        """Collect historical training data for a symbol"""
        # This would collect historical feature-target pairs
        # For now, return empty lists as this requires extensive historical data collection
        return [], []
        
    def _save_models(self):
        """Save trained models"""
        try:
            if self.price_model:
                joblib.dump(self.price_model, self.models_dir / "price_model.joblib")
                
            if self.direction_model:
                joblib.dump(self.direction_model, self.models_dir / "direction_model.joblib")
                
            joblib.dump(self.scaler, self.models_dir / "scaler.joblib")
            
            self.logger.info("✅ Models saved successfully")
            
        except Exception as e:
            self.logger.error(f"Error saving models: {e}")
            
    async def backtest_predictions(self, symbol: str, days_back: int = 30) -> Dict:
        """Backtest prediction accuracy"""
        self.logger.info(f"🔍 Backtesting predictions for {symbol} ({days_back} days)")
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get historical predictions
            cursor.execute('''
            SELECT prediction_date, predicted_price, current_price, direction, confidence_score
            FROM predictions 
            WHERE symbol = ? 
            AND prediction_date > date('now', '-{} days')
            ORDER BY prediction_date DESC
            '''.format(days_back), (symbol,))
            
            predictions = cursor.fetchall()
            conn.close()
            
            if not predictions:
                return {'success': False, 'error': 'No historical predictions found'}
                
            # Get actual prices for comparison
            ticker = yf.Ticker(symbol)
            hist_data = ticker.history(period=f"{days_back + 5}d")
            
            accuracies = []
            direction_accuracies = []
            
            for pred_date, pred_price, current_price, direction, confidence in predictions:
                # Find actual price on prediction date + timeframe
                try:
                    pred_datetime = datetime.strptime(pred_date, '%Y-%m-%d %H:%M:%S')
                    target_date = pred_datetime + timedelta(days=5)  # Assuming 5d prediction
                    
                    # Find closest actual price
                    closest_date = min(hist_data.index, key=lambda x: abs((x.date() - target_date.date()).days))
                    actual_price = hist_data.loc[closest_date, 'Close']
                    
                    # Calculate accuracy
                    price_accuracy = 1 - abs(pred_price - actual_price) / actual_price
                    accuracies.append(price_accuracy)
                    
                    # Direction accuracy
                    actual_direction = 'up' if actual_price > current_price else 'down'
                    direction_correct = direction == actual_direction
                    direction_accuracies.append(1 if direction_correct else 0)
                    
                except Exception as e:
                    continue
                    
            if not accuracies:
                return {'success': False, 'error': 'Could not calculate accuracies'}
                
            backtest_results = {
                'success': True,
                'symbol': symbol,
                'period_days': days_back,
                'predictions_evaluated': len(accuracies),
                'average_price_accuracy': np.mean(accuracies),
                'direction_accuracy': np.mean(direction_accuracies),
                'accuracy_std': np.std(accuracies),
                'best_accuracy': max(accuracies),
                'worst_accuracy': min(accuracies)
            }
            
            return backtest_results
            
        except Exception as e:
            self.logger.error(f"Error backtesting: {e}")
            return {'success': False, 'error': str(e)}