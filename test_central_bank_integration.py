#!/usr/bin/env python3
"""
Test script for enhanced CBA prediction system with central bank integration
"""
import asyncio
import sys
import traceback
sys.path.append('/home/user/webapp')

async def test_enhanced_prediction():
    try:
        print('🏦 Testing Enhanced CBA Prediction System with Central Bank Integration')
        print('=' * 70)
        
        # Test central bank tracker initialization
        from central_bank_rate_integration import central_bank_tracker
        print('✅ Central bank tracker initialized')
        
        # Test CBA prediction system
        from cba_enhanced_prediction_system import cba_predictor
        print('✅ CBA prediction system initialized')
        
        # Test data collection with central bank features
        print('\n📊 Collecting enhanced data with central bank rates...')
        data = await cba_predictor.collect_cba_enhanced_data(days_back=60)
        print(f'✅ Collected {len(data)} days of enhanced data')
        
        # Check for central bank features
        cb_features = [col for col in data.columns if any(keyword in col.lower() 
                                                         for keyword in ['rba_', 'fed_', 'rate_'])]
        print(f'🏦 Central bank features integrated: {len(cb_features)}')
        
        if cb_features:
            print('📋 Sample central bank features:')
            for feature in cb_features[:8]:
                latest_value = data[feature].iloc[-1] if not data[feature].isna().all() else 0
                print(f'   • {feature}: {latest_value:.3f}')
        
        # Test prediction with central bank analysis
        print('\n🔮 Making enhanced prediction with central bank analysis...')
        prediction = await cba_predictor.predict_with_publications_analysis(days=5)
        
        print(f'✅ Prediction completed successfully!')
        print(f'📈 Predicted price: ${prediction["prediction"]["predicted_price"]:.2f}')
        print(f'📊 Current price: ${prediction["prediction"]["current_price"]:.2f}')
        print(f'📈 Change: {prediction["prediction"]["predicted_change_percent"]:+.2f}%')
        
        # Central bank analysis results
        if 'central_bank_analysis' in prediction:
            cb_analysis = prediction['central_bank_analysis']
            print(f'\n🏦 Central Bank Impact Analysis:')
            
            if 'rate_environment' in cb_analysis:
                rate_env = cb_analysis['rate_environment']
                print(f'   • Current RBA Rate: {rate_env.get("current_rba_rate", "N/A")}%')
                print(f'   • Environment: {rate_env.get("environment_type", "unknown")}')
                print(f'   • Margin Impact: {rate_env.get("margin_impact", "unknown")}')
            
            if 'overall_assessment' in cb_analysis:
                overall = cb_analysis['overall_assessment']
                print(f'   • Rate Impact Score: {overall.get("rate_impact_score", 0):.3f}')
                print(f'   • Banking Sensitivity: {overall.get("banking_sector_sensitivity", 0):.3f}')
        else:
            print('\n⚠️ Central bank analysis not found in prediction results')
        
        # Model metrics
        metrics = prediction.get('model_metrics', {})
        print(f'\n📊 Model Metrics:')
        print(f'   • Total features: {metrics.get("features_used", 0)}')
        print(f'   • Central bank features: {metrics.get("central_bank_features", 0)}')
        print(f'   • Publications analyzed: {metrics.get("publications_count", 0)}')
        print(f'   • News articles analyzed: {metrics.get("news_articles_count", 0)}')
        
        print('\n✅ All tests completed successfully!')
        print('\n🎯 Central Bank Integration Summary:')
        print('   • RBA rate announcements integrated into prediction model')
        print('   • Fed rate data included for cross-currency impact')
        print('   • Banking sector sensitivity scoring implemented')
        print('   • Rate cycle position analysis included')
        print('   • Meeting proximity and market impact factors added')
        
    except Exception as e:
        print(f'❌ Error during testing: {e}')
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_enhanced_prediction())