#!/usr/bin/env python3
"""
Quick Performance Analysis - Fast evaluation of model issues
"""

import asyncio
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import logging
import yfinance as yf

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import our prediction system
from market_prediction_llm import (
    MarketPredictionService,
    PredictionRequest
)

class QuickPerformanceAnalyzer:
    """Fast performance analysis focused on identifying key issues"""
    
    def __init__(self):
        self.prediction_service = MarketPredictionService()
    
    async def analyze_recent_performance(self, days_back: int = 7) -> Dict:
        """Quick analysis of recent predictions vs actual market performance"""
        
        logger.info(f"🔍 Analyzing model performance over last {days_back} days")
        
        # Get actual market data
        actual_data = await self._get_actual_market_data(days_back)
        
        # Test a few predictions
        prediction_tests = await self._run_sample_predictions()
        
        # Analyze factor consistency
        factor_analysis = await self._analyze_factor_consistency()
        
        # Performance insights
        insights = self._generate_performance_insights(actual_data, prediction_tests, factor_analysis)
        
        return {
            'actual_market_data': actual_data,
            'prediction_tests': prediction_tests,
            'factor_analysis': factor_analysis,
            'performance_insights': insights
        }
    
    async def _get_actual_market_data(self, days_back: int) -> Dict:
        """Get actual ASX All Ordinaries performance"""
        
        try:
            ticker = yf.Ticker("^AORD")
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back + 5)
            
            data = ticker.history(start=start_date, end=end_date)
            
            if data.empty:
                logger.warning("No real market data available")
                return self._simulate_market_data(days_back)
            
            # Calculate recent performance metrics
            recent_returns = data['Close'].pct_change().dropna()
            
            return {
                'total_days': len(data),
                'recent_returns': recent_returns.tolist()[-days_back:],
                'volatility': recent_returns.std() * 100,  # As percentage
                'avg_daily_return': recent_returns.mean() * 100,
                'max_gain': recent_returns.max() * 100,
                'max_loss': recent_returns.min() * 100,
                'positive_days': (recent_returns > 0).sum(),
                'negative_days': (recent_returns < 0).sum(),
                'current_price': float(data['Close'].iloc[-1]),
                'price_range': {
                    'high': float(data['High'].max()),
                    'low': float(data['Low'].min())
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to fetch market data: {e}")
            return self._simulate_market_data(days_back)
    
    def _simulate_market_data(self, days_back: int) -> Dict:
        """Simulate realistic market data for testing"""
        
        # Simulate recent ASX performance
        np.random.seed(42)
        returns = np.random.normal(0.0005, 0.015, days_back)  # Slight upward bias, 1.5% daily vol
        
        return {
            'total_days': days_back,
            'recent_returns': returns.tolist(),
            'volatility': np.std(returns) * 100,
            'avg_daily_return': np.mean(returns) * 100,
            'max_gain': np.max(returns) * 100,
            'max_loss': np.min(returns) * 100,
            'positive_days': (returns > 0).sum(),
            'negative_days': (returns < 0).sum(),
            'current_price': 7500.0,  # Simulated current price
            'price_range': {'high': 7650.0, 'low': 7350.0}
        }
    
    async def _run_sample_predictions(self) -> List[Dict]:
        """Run a few sample predictions to test current model behavior"""
        
        logger.info("🧠 Testing sample predictions...")
        
        test_cases = [
            {"timeframe": "1d", "description": "Next day prediction"},
            {"timeframe": "5d", "description": "5-day prediction"},
        ]
        
        predictions = []
        
        for test in test_cases:
            try:
                request = PredictionRequest(
                    symbol="^AORD",
                    timeframe=test["timeframe"],
                    include_factors=True,
                    include_news_intelligence=True
                )
                
                start_time = datetime.now()
                response = await self.prediction_service.get_market_prediction(request)
                end_time = datetime.now()
                
                processing_time = (end_time - start_time).total_seconds()
                
                if response.success:
                    pred_data = {
                        'timeframe': test["timeframe"],
                        'description': test["description"],
                        'direction': response.prediction['direction'],
                        'expected_change': response.prediction['expected_change_percent'],
                        'confidence': response.prediction['confidence_score'],
                        'processing_time': processing_time,
                        'tier1_factors_count': len(response.tier1_factors or {}),
                        'tier1_factors': response.tier1_factors,
                        'factor_attribution': response.factor_attribution,
                        'success': True
                    }
                else:
                    pred_data = {
                        'timeframe': test["timeframe"],
                        'description': test["description"],
                        'success': False,
                        'processing_time': processing_time
                    }
                
                predictions.append(pred_data)
                logger.info(f"✅ {test['description']}: {pred_data.get('direction', 'FAILED')} ({processing_time:.2f}s)")
                
            except Exception as e:
                logger.error(f"❌ Failed {test['description']}: {e}")
                predictions.append({
                    'timeframe': test["timeframe"],
                    'description': test["description"],
                    'success': False,
                    'error': str(e)
                })
        
        return predictions
    
    async def _analyze_factor_consistency(self) -> Dict:
        """Analyze consistency and quality of Tier 1 factors"""
        
        logger.info("📊 Analyzing factor consistency...")
        
        # Run multiple predictions to check factor stability
        factor_samples = []
        
        for i in range(3):  # Quick samples
            try:
                request = PredictionRequest(
                    symbol="^AORD",
                    timeframe="1d",
                    include_factors=True
                )
                
                response = await self.prediction_service.get_market_prediction(request)
                
                if response.success and response.tier1_factors:
                    factor_samples.append(response.tier1_factors)
                    
            except Exception as e:
                logger.warning(f"Factor sampling failed: {e}")
        
        if not factor_samples:
            return {'error': 'No factor samples collected'}
        
        # Analyze factor consistency
        all_factors = set()
        for sample in factor_samples:
            all_factors.update(sample.keys())
        
        factor_stats = {}
        for factor in all_factors:
            values = [sample.get(factor, 0) for sample in factor_samples if factor in sample]
            
            if values:
                factor_stats[factor] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'sample_count': len(values)
                }
        
        return {
            'total_factors': len(all_factors),
            'sample_count': len(factor_samples),
            'factor_statistics': factor_stats,
            'consistency_issues': self._identify_consistency_issues(factor_stats)
        }
    
    def _identify_consistency_issues(self, factor_stats: Dict) -> List[str]:
        """Identify potential issues with factor consistency"""
        
        issues = []
        
        for factor, stats in factor_stats.items():
            # Check for high volatility in factors that should be stable
            if stats['std'] > 0.3:  # High standard deviation
                issues.append(f"High volatility in {factor} (std: {stats['std']:.3f})")
            
            # Check for extreme values
            if abs(stats['max']) > 1.2 or abs(stats['min']) > 1.2:
                issues.append(f"Extreme values in {factor} (range: {stats['min']:.3f} to {stats['max']:.3f})")
            
            # Check for factors stuck at zero
            if stats['mean'] == 0 and stats['std'] == 0:
                issues.append(f"Factor {factor} appears inactive (constant zero)")
        
        return issues
    
    def _generate_performance_insights(self, market_data: Dict, predictions: List[Dict], factor_analysis: Dict) -> List[str]:
        """Generate actionable performance insights"""
        
        insights = []
        
        # Market context analysis
        market_vol = market_data.get('volatility', 0)
        if market_vol > 2.5:
            insights.append(f"🌊 HIGH MARKET VOLATILITY: {market_vol:.2f}% daily volatility detected. Model may struggle in volatile conditions.")
        elif market_vol < 0.8:
            insights.append(f"😴 LOW MARKET VOLATILITY: {market_vol:.2f}% daily volatility. Model may overfit to calm conditions.")
        
        # Prediction performance analysis
        successful_predictions = [p for p in predictions if p.get('success', False)]
        
        if len(successful_predictions) < len(predictions):
            failed_count = len(predictions) - len(successful_predictions)
            insights.append(f"❌ PREDICTION FAILURES: {failed_count}/{len(predictions)} predictions failed to generate.")
        
        # Processing time analysis
        avg_processing_time = np.mean([p.get('processing_time', 0) for p in successful_predictions])
        if avg_processing_time > 30:
            insights.append(f"⏱️ SLOW PROCESSING: Average {avg_processing_time:.1f}s per prediction. Consider optimization.")
        
        # Confidence analysis
        confidences = [p.get('confidence', 0) for p in successful_predictions]
        if confidences:
            avg_confidence = np.mean(confidences)
            if avg_confidence < 0.6:
                insights.append(f"🤔 LOW CONFIDENCE: Average confidence {avg_confidence:.1%}. Model uncertainty high.")
            elif avg_confidence > 0.9:
                insights.append(f"🎯 OVERCONFIDENT: Average confidence {avg_confidence:.1%}. May indicate overconfidence bias.")
        
        # Factor analysis insights
        if 'consistency_issues' in factor_analysis:
            issue_count = len(factor_analysis['consistency_issues'])
            if issue_count > 0:
                insights.append(f"⚠️ FACTOR ISSUES: {issue_count} factor consistency problems detected.")
                insights.extend(factor_analysis['consistency_issues'][:3])  # Top 3 issues
        
        # Direction bias analysis
        directions = [p.get('direction') for p in successful_predictions if p.get('direction')]
        if directions:
            bullish_count = directions.count('up')
            bearish_count = directions.count('down')
            total = len(directions)
            
            if bullish_count / total > 0.8:
                insights.append(f"📈 BULLISH BIAS: {bullish_count}/{total} predictions bullish. Check for optimism bias.")
            elif bearish_count / total > 0.8:
                insights.append(f"📉 BEARISH BIAS: {bearish_count}/{total} predictions bearish. Check for pessimism bias.")
        
        # Factor count analysis
        factor_counts = [p.get('tier1_factors_count', 0) for p in successful_predictions]
        if factor_counts:
            avg_factors = np.mean(factor_counts)
            if avg_factors < 20:
                insights.append(f"📊 LOW FACTOR COUNT: Only {avg_factors:.1f} average factors. Some systems may be failing.")
        
        return insights

async def main():
    """Run quick performance analysis"""
    
    analyzer = QuickPerformanceAnalyzer()
    
    print("🚀 Starting Quick Performance Analysis")
    print("=" * 50)
    
    results = await analyzer.analyze_recent_performance(days_back=7)
    
    # Display results
    print("\n📊 MARKET CONTEXT:")
    market = results['actual_market_data']
    print(f"  • Recent Volatility: {market['volatility']:.2f}% daily")
    print(f"  • Average Daily Return: {market['avg_daily_return']:+.3f}%")
    print(f"  • Positive/Negative Days: {market['positive_days']}/{market['negative_days']}")
    print(f"  • Price Range: ${market['price_range']['low']:.0f} - ${market['price_range']['high']:.0f}")
    
    print("\n🧠 PREDICTION TESTS:")
    for pred in results['prediction_tests']:
        if pred.get('success'):
            print(f"  • {pred['description']}: {pred['direction'].upper()} ({pred['confidence']:.1%} confidence, {pred['processing_time']:.1f}s)")
        else:
            print(f"  • {pred['description']}: FAILED ({pred.get('error', 'Unknown error')})")
    
    print("\n📈 FACTOR ANALYSIS:")
    factor_analysis = results['factor_analysis']
    if 'error' not in factor_analysis:
        print(f"  • Total Factors: {factor_analysis['total_factors']}")
        print(f"  • Samples Analyzed: {factor_analysis['sample_count']}")
        if factor_analysis['consistency_issues']:
            print("  • Issues Found:")
            for issue in factor_analysis['consistency_issues'][:3]:
                print(f"    - {issue}")
        else:
            print("  • No consistency issues detected")
    else:
        print(f"  • Error: {factor_analysis['error']}")
    
    print("\n💡 KEY INSIGHTS:")
    for i, insight in enumerate(results['performance_insights'], 1):
        print(f"  {i}. {insight}")
    
    print("\n✅ Quick analysis completed!")
    
    return results

if __name__ == "__main__":
    asyncio.run(main())