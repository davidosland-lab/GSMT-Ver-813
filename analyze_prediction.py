#!/usr/bin/env python3
import json

with open('cba_prediction.json', 'r') as f:
    data = json.load(f)

print('🚀 EXTENDED PHASE 3 PREDICTION RESULTS FOR CBA.AX')
print('=' * 55)
print(f'Symbol: {data.get("symbol", "N/A")}')
print(f'Timeframe: {data.get("timeframe", "N/A")}')
print(f'Processing Time: {data.get("processing_time", "N/A")}')
print(f'Prediction Type: {data.get("prediction_type", "N/A")}')

print('\n📊 PREDICTION RESULTS:')
pred = data.get('prediction', {})
print(f'Direction: {pred.get("direction", "N/A")}')
print(f'Current Price: AUD ${pred.get("current_price", "N/A")}')
print(f'Predicted Price: AUD ${pred.get("predicted_price", "N/A"):.2f}' if pred.get("predicted_price") else 'N/A')
print(f'Expected Return: {(pred.get("expected_return", 0) * 100):.2f}%')
print(f'Confidence Score: {(pred.get("confidence_score", 0) * 100):.1f}%')
print(f'Probability UP: {(pred.get("probability_up", 0) * 100):.1f}%')

print('\n🧠 PHASE 3 COMPONENTS STATUS:')
components = data.get('components_active', {})
print(f'P3-005 Advanced Features: {"✅ Active" if components.get("p3_005_advanced_features") else "❌ Inactive"}')
print(f'P3-006 Reinforcement Learning: {"✅ Active" if components.get("p3_006_reinforcement_learning") else "❌ Inactive"}')
print(f'P3-007 Risk Management: {"✅ Active" if components.get("p3_007_risk_management") else "❌ Inactive"}')

print('\n🔧 ADVANCED FEATURE ENGINEERING:')
afe = data.get('advanced_feature_engineering', {})
print(f'Total Features Engineered: {afe.get("total_features_engineered", "N/A")}')
print(f'Multimodal Fusion: {"✅ Active" if afe.get("multimodal_fusion_active") else "❌ Inactive"}')

print('\n🧠 REINFORCEMENT LEARNING:')
rl = data.get('reinforcement_learning', {})
if 'rl_recommendations' in rl:
    rl_rec = rl['rl_recommendations']
    print(f'Exploration Rate: {(rl_rec.get("exploration_rate", 0) * 100):.1f}%')
print(f'Adaptive Learning: {"✅ Active" if rl.get("adaptive_learning_active") else "❌ Inactive"}')

print('\n🛡️ RISK MANAGEMENT:')
risk = data.get('risk_management', {})
print(f'Risk Management: {"✅ Active" if risk.get("risk_management_active") else "❌ Inactive"}')
if 'var_calculation' in risk:
    var_calc = risk['var_calculation']
    print(f'Value at Risk (95%): AUD ${var_calc.get("var_95", "N/A")}')

print('\n📈 CONFIDENCE INTERVAL:')
ci = pred.get('confidence_interval', {})
if ci:
    print(f'Lower Bound: AUD ${ci.get("lower", "N/A"):.2f}' if ci.get("lower") else 'N/A')
    print(f'Upper Bound: AUD ${ci.get("upper", "N/A"):.2f}' if ci.get("upper") else 'N/A')