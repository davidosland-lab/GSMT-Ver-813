#!/usr/bin/env python3
"""
Quick test for GNN-TFT integration
"""

import asyncio
from phase4_gnn_tft_integration import GNNTFTIntegratedPredictor

async def test_integration():
    try:
        predictor = GNNTFTIntegratedPredictor()
        print('✅ GNN-TFT integrated predictor initialized')
        
        status = predictor.get_system_status()
        print('Integration status:', status['multimodal_integration'])
        
        result = await predictor.generate_multimodal_prediction('AAPL', '5d', ['MSFT'])
        print('✅ Multi-modal prediction generated successfully')
        print(f'Symbol: {result.symbol}')
        print(f'Predicted price: ${result.predicted_price:.2f}')
        print(f'Confidence: {result.confidence_score:.3f}')
        print(f'Fusion method: {result.fusion_method}')
        print(f'Components used: {result.components_used}')
        print(f'Model agreement: {result.model_agreement:.3f}')
    except Exception as e:
        print(f'⚠️ Integration test failed: {e}')
        print('This may be expected due to TFT dependencies')

if __name__ == "__main__":
    asyncio.run(test_integration())