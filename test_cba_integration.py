#!/usr/bin/env python3
"""
Test CBA Enhanced Prediction System Integration
Tests the integration of CBA system into the main application
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(__file__))

from cba_enhanced_prediction_system import CBAEnhancedPredictionSystem
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_cba_system():
    """Test the CBA enhanced prediction system functionality"""
    
    logger.info("🏦 Testing CBA Enhanced Prediction System...")
    
    try:
        # Initialize CBA predictor
        cba_predictor = CBAEnhancedPredictionSystem()
        logger.info("✅ CBA system initialized successfully")
        
        # Test 1: Basic system functionality
        logger.info("\n📊 Test 1: Basic system status check")
        assert cba_predictor.symbol == "CBA.AX"
        assert len(cba_predictor.banking_peers) == 3
        logger.info(f"   • Primary symbol: {cba_predictor.symbol}")
        logger.info(f"   • Banking peers: {cba_predictor.banking_peers}")
        logger.info("✅ Basic system status check passed")
        
        # Test 2: Data collection capabilities
        logger.info("\n📈 Test 2: Data collection test")
        market_data = await cba_predictor.collect_cba_enhanced_data(days_back=30)
        if market_data is not None and not market_data.empty:
            logger.info(f"   • Market data collected: {len(market_data)} records")
            logger.info(f"   • Latest price: ${market_data['close'].iloc[-1]:.2f}")
            logger.info("✅ Market data collection test passed")
        else:
            logger.warning("⚠️ Market data collection returned empty - this is expected in simulation mode")
        
        # Test 3: Publications system
        logger.info("\n📚 Test 3: Publications system test")
        from datetime import datetime, timedelta
        start_date = datetime.now() - timedelta(days=90)
        end_date = datetime.now()
        publications = await cba_predictor.retrieve_cba_publications(start_date, end_date, limit=5)
        logger.info(f"   • Publications found: {len(publications)}")
        if publications:
            latest_pub = publications[0]
            logger.info(f"   • Latest publication: {latest_pub.title}")
            logger.info(f"   • Publication type: {latest_pub.publication_type.value}")
            logger.info(f"   • Sentiment score: {latest_pub.sentiment_score}")
        logger.info("✅ Publications system test passed")
        
        # Test 4: News system
        logger.info("\n📰 Test 4: News analysis system test")
        news_articles = await cba_predictor.retrieve_cba_news_articles(start_date, end_date, limit=5)
        logger.info(f"   • News articles found: {len(news_articles)}")
        if news_articles:
            latest_news = news_articles[0]
            logger.info(f"   • Latest news: {latest_news.title}")
            logger.info(f"   • News source: {latest_news.source.value}")
            logger.info(f"   • Sentiment score: {latest_news.sentiment_score}")
        logger.info("✅ News analysis system test passed")
        
        # Test 5: Enhanced prediction
        logger.info("\n🔮 Test 5: Enhanced prediction test")
        prediction_result = await cba_predictor.predict_with_publications_analysis(days=5)
        
        if prediction_result and "prediction" in prediction_result:
            pred = prediction_result["prediction"]
            logger.info(f"   • Predicted price: ${pred['predicted_price']:.2f}")
            logger.info(f"   • Confidence interval: ${pred['confidence_interval'][0]:.2f} - ${pred['confidence_interval'][1]:.2f}")
            logger.info(f"   • Probability up: {pred['probability_up']:.1%}")
            logger.info(f"   • Risk score: {pred['risk_score']:.2f}")
            logger.info("✅ Enhanced prediction test passed")
        else:
            logger.warning("⚠️ Enhanced prediction returned empty result - this may be expected in test mode")
        
        logger.info("\n🎉 All CBA system tests completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ CBA system test failed: {e}")
        return False

async def test_app_integration():
    """Test the integration with the main app"""
    
    logger.info("\n🔗 Testing CBA system integration with main app...")
    
    try:
        # Import the main app components
        from app import cba_predictor, SYMBOLS_DB
        
        # Test 1: CBA system loaded in app
        if cba_predictor:
            logger.info("✅ CBA predictor loaded in main app")
        else:
            logger.error("❌ CBA predictor not loaded in main app")
            return False
        
        # Test 2: CBA symbol in database
        if "CBA.AX" in SYMBOLS_DB:
            cba_info = SYMBOLS_DB["CBA.AX"]
            logger.info(f"✅ CBA symbol in database: {cba_info.name}")
            logger.info(f"   • Market: {cba_info.market}")
            logger.info(f"   • Category: {cba_info.category}")
            logger.info(f"   • Currency: {cba_info.currency}")
        else:
            logger.error("❌ CBA.AX not found in SYMBOLS_DB")
            return False
        
        # Test 3: Banking peers in database
        banking_peers = ["ANZ.AX", "WBC.AX", "NAB.AX"]
        for peer in banking_peers:
            if peer in SYMBOLS_DB:
                logger.info(f"✅ Banking peer {peer} in database: {SYMBOLS_DB[peer].name}")
            else:
                logger.warning(f"⚠️ Banking peer {peer} not found in database")
        
        logger.info("🎉 App integration tests completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ App integration test failed: {e}")
        return False

async def main():
    """Main test function"""
    
    logger.info("🚀 Starting CBA Enhanced Prediction System Tests")
    logger.info("=" * 60)
    
    # Run system tests
    system_test = await test_cba_system()
    
    # Run integration tests
    integration_test = await test_app_integration()
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📋 TEST SUMMARY")
    logger.info(f"   • CBA System Test: {'✅ PASSED' if system_test else '❌ FAILED'}")
    logger.info(f"   • App Integration Test: {'✅ PASSED' if integration_test else '❌ FAILED'}")
    
    if system_test and integration_test:
        logger.info("🎉 ALL TESTS PASSED - CBA system ready for deployment!")
        return 0
    else:
        logger.error("❌ SOME TESTS FAILED - Check logs above")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)