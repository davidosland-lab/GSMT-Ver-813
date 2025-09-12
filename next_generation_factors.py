"""
Next-Generation Market Prediction Factors
Implementation framework for advanced factors to enhance prediction robustness
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging
import json
import os
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FactorCategory(Enum):
    """Categories of enhancement factors"""
    TECHNICAL_ADVANCED = "technical_advanced"
    INSTITUTIONAL_FLOW = "institutional_flow"
    BEHAVIORAL_SENTIMENT = "behavioral_sentiment"
    ECONOMIC_LEADING = "economic_leading"
    GLOBAL_MACRO = "global_macro"
    ESG_SUSTAINABILITY = "esg_sustainability"
    SECTOR_SPECIFIC = "sector_specific"
    ALTERNATIVE_DATA = "alternative_data"

class FactorImpact(Enum):
    """Expected impact level of factors"""
    CRITICAL = "critical"       # 15-25% prediction accuracy improvement
    HIGH = "high"              # 8-15% improvement
    MEDIUM = "medium"          # 4-8% improvement
    LOW = "low"               # 1-4% improvement

class ImplementationComplexity(Enum):
    """Implementation difficulty assessment"""
    LOW = "low"               # 1-2 weeks implementation
    MEDIUM = "medium"         # 1-2 months implementation
    HIGH = "high"            # 3-6 months implementation
    EXTREME = "extreme"      # 6+ months implementation

@dataclass
class EnhancementFactor:
    """Individual factor for prediction enhancement"""
    name: str
    category: FactorCategory
    impact: FactorImpact
    complexity: ImplementationComplexity
    data_sources: List[str]
    update_frequency: str  # real-time, hourly, daily, weekly
    australian_relevance: float  # 0-1 relevance to ASX
    description: str
    implementation_priority: int  # 1-10 priority ranking

# Abstract base class for factor analyzers
class FactorAnalyzer(ABC):
    """Base class for all factor analysis modules"""
    
    @abstractmethod
    async def collect_data(self) -> Dict[str, Any]:
        """Collect raw data for this factor"""
        pass
    
    @abstractmethod
    async def analyze_factor(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Analyze factor and return standardized metrics"""
        pass
    
    @abstractmethod
    def get_factor_weight(self, timeframe: str) -> float:
        """Get factor importance weight for given timeframe"""
        pass

class OptionsFlowAnalyzer(FactorAnalyzer):
    """ASX Options Flow and Derivatives Intelligence"""
    
    def __init__(self):
        self.factor_info = EnhancementFactor(
            name="ASX Options Flow Analysis",
            category=FactorCategory.TECHNICAL_ADVANCED,
            impact=FactorImpact.HIGH,
            complexity=ImplementationComplexity.MEDIUM,
            data_sources=["ASX Market Data", "Options Analytics Providers"],
            update_frequency="real-time",
            australian_relevance=0.95,
            description="Put/call ratios, unusual options activity, and derivatives positioning analysis",
            implementation_priority=2
        )
    
    async def collect_data(self) -> Dict[str, Any]:
        """Collect ASX options and derivatives data"""
        
        # Simulated options data (replace with real ASX options API)
        return {
            "put_call_ratio": 0.75,  # Bullish when < 1.0
            "options_volume_spike": 1.8,  # Relative to 20-day average
            "unusual_options_activity": {
                "large_call_purchases": 145,
                "large_put_purchases": 89,
                "net_gamma_exposure": 2.3e9
            },
            "volatility_surface": {
                "atm_iv": 18.5,  # At-the-money implied volatility
                "skew": 3.2,     # Volatility skew indicator
                "term_structure": "contango"  # Forward vol structure
            },
            "dealer_positioning": {
                "net_delta": -1.2e7,  # Market maker hedging needs
                "gamma_exposure": 2.1e8,
                "vanna_exposure": 3.4e6
            }
        }
    
    async def analyze_factor(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Analyze options flow for market direction signals"""
        
        metrics = {}
        
        # Put/Call Ratio Analysis
        pc_ratio = data["put_call_ratio"]
        if pc_ratio < 0.5:
            metrics["pc_sentiment"] = 0.8  # Very bullish
        elif pc_ratio < 0.8:
            metrics["pc_sentiment"] = 0.3  # Moderately bullish
        elif pc_ratio < 1.2:
            metrics["pc_sentiment"] = 0.0  # Neutral
        else:
            metrics["pc_sentiment"] = -0.6  # Bearish
        
        # Volume Spike Analysis
        vol_spike = data["options_volume_spike"]
        metrics["volume_signal"] = min((vol_spike - 1.0) * 0.5, 1.0)
        
        # Unusual Activity Score
        unusual = data["unusual_options_activity"]
        call_put_flow = unusual["large_call_purchases"] / max(unusual["large_put_purchases"], 1)
        metrics["flow_sentiment"] = min((call_put_flow - 1.0) * 0.3, 0.8)
        
        # Volatility Surface Analysis
        vol_surface = data["volatility_surface"]
        skew_signal = -vol_surface["skew"] * 0.1  # High skew often bearish
        metrics["volatility_signal"] = max(min(skew_signal, 0.5), -0.5)
        
        # Dealer Positioning Impact
        dealer = data["dealer_positioning"]
        gamma_squeeze_potential = dealer["gamma_exposure"] / 1e8 * 0.2
        metrics["gamma_impact"] = min(gamma_squeeze_potential, 0.8)
        
        return metrics
    
    def get_factor_weight(self, timeframe: str) -> float:
        """Options flow more relevant for shorter timeframes"""
        weights = {
            "1d": 0.25,
            "5d": 0.20,
            "30d": 0.10,
            "90d": 0.05
        }
        return weights.get(timeframe, 0.15)

class InstitutionalFlowAnalyzer(FactorAnalyzer):
    """Australian Superannuation and Institutional Fund Flow Analysis"""
    
    def __init__(self):
        self.factor_info = EnhancementFactor(
            name="Superannuation Fund Flow Analysis", 
            category=FactorCategory.INSTITUTIONAL_FLOW,
            impact=FactorImpact.CRITICAL,
            complexity=ImplementationComplexity.MEDIUM,
            data_sources=["APRA Statistics", "Fund Manager Reports", "Industry Analysis"],
            update_frequency="monthly",
            australian_relevance=0.98,
            description="Australia's $3.3T superannuation system flow analysis and allocation changes",
            implementation_priority=1
        )
    
    async def collect_data(self) -> Dict[str, Any]:
        """Collect superannuation and institutional flow data"""
        
        # Simulated super fund data (replace with APRA API)
        return {
            "total_super_assets": 3.3e12,  # $3.3 trillion AUD
            "monthly_inflows": 12.5e9,     # $12.5 billion monthly contributions
            "asset_allocation_changes": {
                "australian_equities": {
                    "current_allocation": 0.24,
                    "monthly_change": 0.008,  # 0.8% increase
                    "flow_amount": 2.1e9
                },
                "international_equities": {
                    "current_allocation": 0.43,
                    "monthly_change": -0.003,
                    "flow_amount": -1.2e9
                },
                "fixed_income": {
                    "current_allocation": 0.19,
                    "monthly_change": -0.004,
                    "flow_amount": -1.8e9
                },
                "alternatives": {
                    "current_allocation": 0.14,
                    "monthly_change": 0.001,
                    "flow_amount": 0.9e9
                }
            },
            "performance_chasing": {
                "momentum_factor": 0.35,  # Funds moving to recent outperformers
                "rebalancing_pressure": -0.15  # Selling winners, buying losers
            },
            "demographic_flows": {
                "baby_boomer_retirement": -2.3e9,  # Net withdrawals
                "gen_x_peak_earning": 8.7e9,       # Peak contribution phase
                "millennial_growth": 6.1e9         # Growing workforce contributions
            }
        }
    
    async def analyze_factor(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Analyze institutional flows for market impact"""
        
        metrics = {}
        
        # Australian Equity Allocation Change
        aussie_equity = data["asset_allocation_changes"]["australian_equities"]
        allocation_change = aussie_equity["monthly_change"]
        flow_amount = aussie_equity["flow_amount"]
        
        # Normalize flow impact (billions to market impact score)
        metrics["allocation_momentum"] = min(allocation_change * 10, 1.0)
        metrics["flow_pressure"] = min(flow_amount / 5e9, 0.8)  # $5B = 0.8 impact
        
        # Performance Chasing vs Rebalancing
        performance = data["performance_chasing"]
        momentum = performance["momentum_factor"]
        rebalancing = performance["rebalancing_pressure"]
        
        metrics["momentum_chasing"] = momentum * 0.6
        metrics["rebalancing_drag"] = rebalancing * 0.4
        
        # Demographic Flow Analysis
        demographics = data["demographic_flows"]
        net_demographic_flow = sum(demographics.values())
        metrics["demographic_support"] = min(net_demographic_flow / 10e9, 0.7)
        
        # Total Assets Growth Impact
        total_assets = data["total_super_assets"]
        monthly_inflows = data["monthly_inflows"]
        growth_rate = (monthly_inflows * 12) / total_assets
        metrics["system_growth"] = min(growth_rate * 20, 0.5)  # 20x multiplier for impact
        
        return metrics
    
    def get_factor_weight(self, timeframe: str) -> float:
        """Super flows more relevant for longer timeframes"""
        weights = {
            "1d": 0.05,
            "5d": 0.10,
            "30d": 0.30,
            "90d": 0.40
        }
        return weights.get(timeframe, 0.25)

class SocialSentimentAnalyzer(FactorAnalyzer):
    """Social Media and Retail Investor Sentiment Analysis"""
    
    def __init__(self):
        self.factor_info = EnhancementFactor(
            name="Social Media & Retail Sentiment",
            category=FactorCategory.BEHAVIORAL_SENTIMENT,
            impact=FactorImpact.MEDIUM,
            complexity=ImplementationComplexity.LOW,
            data_sources=["Twitter API", "Reddit API", "Google Trends", "CommSec Data"],
            update_frequency="real-time",
            australian_relevance=0.80,
            description="Social media sentiment, retail investor behavior, and search trend analysis",
            implementation_priority=3
        )
    
    async def collect_data(self) -> Dict[str, Any]:
        """Collect social sentiment and retail behavior data"""
        
        # Simulated social sentiment data (replace with real APIs)
        return {
            "twitter_sentiment": {
                "asx_mentions": 1247,
                "sentiment_score": 0.23,  # -1 to +1
                "viral_threads": 3,
                "influencer_activity": 0.67
            },
            "reddit_analysis": {
                "asx_bets_activity": 2.1,  # Relative to average
                "wsb_australia_mentions": 89,
                "comment_sentiment": 0.15,
                "new_member_growth": 1.34
            },
            "google_trends": {
                "asx_search_volume": 1.8,  # Relative to baseline
                "stock_related_searches": 2.3,
                "economic_news_interest": 1.1,
                "trending_tickers": ["CBA", "BHP", "CSL", "WBC", "ANZ"]
            },
            "retail_brokerage": {
                "new_account_openings": 1.56,  # Relative to average
                "trading_volume_retail": 2.34,
                "margin_lending_growth": 0.87,
                "options_trading_retail": 3.21
            },
            "crypto_correlation": {
                "btc_aud_correlation": 0.34,
                "crypto_investor_crossover": 0.78,
                "defi_activity_aus": 1.23
            }
        }
    
    async def analyze_factor(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Analyze social sentiment for market direction"""
        
        metrics = {}
        
        # Twitter Sentiment Analysis
        twitter = data["twitter_sentiment"]
        mention_momentum = min(twitter["asx_mentions"] / 1000, 2.0) - 1.0  # Normalize around 0
        sentiment_signal = twitter["sentiment_score"] * 0.8
        viral_boost = twitter["viral_threads"] * 0.1
        
        metrics["twitter_sentiment"] = sentiment_signal + viral_boost * 0.3
        metrics["twitter_momentum"] = mention_momentum * 0.2
        
        # Reddit Analysis
        reddit = data["reddit_analysis"]
        reddit_activity = (reddit["asx_bets_activity"] - 1.0) * 0.4
        reddit_sentiment = reddit["comment_sentiment"] * 0.6
        member_growth_signal = (reddit["new_member_growth"] - 1.0) * 0.3
        
        metrics["reddit_sentiment"] = reddit_sentiment
        metrics["reddit_momentum"] = reddit_activity + member_growth_signal
        
        # Google Trends Analysis
        trends = data["google_trends"]
        search_momentum = (trends["asx_search_volume"] - 1.0) * 0.5
        interest_level = (trends["stock_related_searches"] - 1.0) * 0.3
        
        metrics["search_interest"] = search_momentum + interest_level
        
        # Retail Brokerage Activity
        retail = data["retail_brokerage"]
        new_money = (retail["new_account_openings"] - 1.0) * 0.4
        trading_intensity = (retail["trading_volume_retail"] - 1.0) * 0.3
        risk_appetite = (retail["options_trading_retail"] - 1.0) * 0.2
        
        metrics["retail_participation"] = new_money + trading_intensity
        metrics["retail_risk_appetite"] = risk_appetite
        
        # Crypto Correlation Factor
        crypto = data["crypto_correlation"]
        correlation_strength = crypto["btc_aud_correlation"]
        crossover_effect = (crypto["crypto_investor_crossover"] - 0.5) * 0.4
        
        metrics["crypto_spillover"] = correlation_strength * crossover_effect
        
        return metrics
    
    def get_factor_weight(self, timeframe: str) -> float:
        """Social sentiment more relevant for shorter timeframes"""
        weights = {
            "1d": 0.20,
            "5d": 0.25,
            "30d": 0.15,
            "90d": 0.08
        }
        return weights.get(timeframe, 0.17)

class EconomicLeadingAnalyzer(FactorAnalyzer):
    """Leading Economic Indicators for Australian Market Prediction"""
    
    def __init__(self):
        self.factor_info = EnhancementFactor(
            name="Leading Economic Indicators",
            category=FactorCategory.ECONOMIC_LEADING,
            impact=FactorImpact.HIGH,
            complexity=ImplementationComplexity.MEDIUM,
            data_sources=["ABS", "NAB", "RBA", "Westpac", "AI Group"],
            update_frequency="monthly",
            australian_relevance=0.95,
            description="Business confidence, employment quality, consumer spending, and forward-looking economic data",
            implementation_priority=4
        )
    
    async def collect_data(self) -> Dict[str, Any]:
        """Collect leading economic indicator data"""
        
        # Simulated economic data (replace with real APIs)
        return {
            "business_confidence": {
                "nab_business_confidence": 12,  # Index points
                "business_conditions": 8,
                "employment_index": 15,
                "capex_intentions": 6,
                "forward_orders": 4
            },
            "consumer_indicators": {
                "westpac_consumer_sentiment": 98.5,  # Index
                "spending_intentions": 105.2,
                "house_price_expectations": 112.8,
                "economic_conditions_next_12m": 91.3
            },
            "employment_quality": {
                "underemployment_rate": 6.1,  # Percentage
                "job_ads_growth": 8.3,
                "skills_shortage_index": 67.2,
                "wage_growth_expectation": 3.4
            },
            "manufacturing_pmi": {
                "ai_group_pmi": 52.1,  # Above 50 = expansion
                "new_orders": 54.3,
                "production": 51.8,
                "employment": 49.2,
                "supplier_deliveries": 47.9
            },
            "housing_indicators": {
                "building_approvals": -2.3,  # Monthly change %
                "housing_finance": 1.8,
                "mortgage_stress_index": 23.1,
                "rental_vacancy_rates": 1.2
            },
            "credit_growth": {
                "business_credit_growth": 4.2,  # Annual %
                "housing_credit_growth": 7.8,
                "personal_credit_growth": -1.1,
                "credit_standards_index": -15  # Negative = tightening
            }
        }
    
    async def analyze_factor(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Analyze leading economic indicators"""
        
        metrics = {}
        
        # Business Confidence Analysis
        business = data["business_confidence"]
        confidence_signal = min(max(business["nab_business_confidence"] / 20, -1), 1)
        conditions_signal = min(max(business["business_conditions"] / 15, -1), 1)
        forward_momentum = min(max(business["forward_orders"] / 10, -1), 1)
        
        metrics["business_outlook"] = (confidence_signal + conditions_signal + forward_momentum) / 3
        
        # Consumer Sentiment
        consumer = data["consumer_indicators"]
        sentiment_signal = (consumer["westpac_consumer_sentiment"] - 100) / 20  # Normalize around 100
        spending_signal = (consumer["spending_intentions"] - 100) / 15
        
        metrics["consumer_confidence"] = (sentiment_signal + spending_signal) / 2
        
        # Employment Quality
        employment = data["employment_quality"]
        underemployment_drag = -(employment["underemployment_rate"] - 6.5) / 5  # Lower is better
        job_ads_momentum = employment["job_ads_growth"] / 20
        wage_growth_signal = (employment["wage_growth_expectation"] - 3.0) / 4
        
        metrics["labor_market_strength"] = (underemployment_drag + job_ads_momentum + wage_growth_signal) / 3
        
        # Manufacturing PMI
        pmi = data["manufacturing_pmi"]
        pmi_signal = (pmi["ai_group_pmi"] - 50) / 10  # Above 50 = expansion
        orders_signal = (pmi["new_orders"] - 50) / 10
        
        metrics["manufacturing_momentum"] = (pmi_signal + orders_signal) / 2
        
        # Credit Growth Analysis
        credit = data["credit_growth"]
        business_credit_signal = credit["business_credit_growth"] / 10
        housing_credit_signal = credit["housing_credit_growth"] / 15
        credit_standards_signal = -credit["credit_standards_index"] / 30  # Negative tightening is bearish
        
        metrics["credit_conditions"] = (business_credit_signal + housing_credit_signal + credit_standards_signal) / 3
        
        return metrics
    
    def get_factor_weight(self, timeframe: str) -> float:
        """Economic indicators more relevant for longer timeframes"""
        weights = {
            "1d": 0.05,
            "5d": 0.15,
            "30d": 0.35,
            "90d": 0.45
        }
        return weights.get(timeframe, 0.30)

class GlobalMacroAnalyzer(FactorAnalyzer):
    """Global Macro Environment and Cross-Border Flow Analysis"""
    
    def __init__(self):
        self.factor_info = EnhancementFactor(
            name="Global Macro Environment",
            category=FactorCategory.GLOBAL_MACRO,
            impact=FactorImpact.HIGH,
            complexity=ImplementationComplexity.HIGH,
            data_sources=["Federal Reserve", "ECB", "BOJ", "PBOC", "BIS", "IMF"],
            update_frequency="daily",
            australian_relevance=0.85,
            description="Global central bank policy, international capital flows, and cross-border macro trends",
            implementation_priority=5
        )
    
    async def collect_data(self) -> Dict[str, Any]:
        """Collect global macro and capital flow data"""
        
        # Simulated global macro data
        return {
            "global_central_banks": {
                "fed_funds_rate": 5.25,
                "fed_dot_plot_terminal": 5.50,
                "ecb_deposit_rate": 4.00,
                "boj_policy_rate": -0.10,
                "pboc_policy_divergence": 0.85  # Relative to developed markets
            },
            "capital_flows": {
                "emerging_market_flows": -12.3e9,  # Weekly flows, negative = outflows
                "safe_haven_demand": 1.67,  # Relative demand index
                "risk_parity_rebalancing": -0.23,  # Portfolio rebalancing pressure
                "currency_carry_trades": {
                    "aud_carry_attractiveness": 0.34,
                    "jpy_funding_cost": -0.45,
                    "usd_funding_premium": 0.78
                }
            },
            "global_risk_indicators": {
                "vix": 18.4,
                "credit_spreads_ig": 145,  # Investment grade credit spreads (bps)
                "credit_spreads_hy": 487,  # High yield credit spreads
                "sovereign_cds_average": 234,  # Emerging market CDS spreads
                "term_premium_us": 0.34   # US treasury term premium
            },
            "commodity_cycles": {
                "dxy_strength": 103.2,  # US Dollar Index
                "commodity_momentum": -0.12,  # 3-month momentum
                "supply_chain_stress": 1.23,  # Supply chain pressure index
                "energy_transition_capex": 2.1e12  # Global renewable energy investment
            },
            "trade_relationships": {
                "us_china_trade_intensity": 0.78,  # Trade relationship strength
                "regional_trade_agreements": 1.15,  # RCEP, CPTPP activity level
                "supply_chain_reshoring": 0.67,   # Supply chain localization trend
                "critical_minerals_security": 0.45  # Strategic mineral supply security
            }
        }
    
    async def analyze_factor(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Analyze global macro environment"""
        
        metrics = {}
        
        # Central Bank Policy Divergence
        cb = data["global_central_banks"]
        rate_differential_us = cb["fed_funds_rate"] - 4.35  # vs RBA rate
        policy_divergence_signal = min(max(rate_differential_us / 2, -1), 1)
        
        pboc_divergence = cb["pboc_policy_divergence"]
        china_policy_signal = (pboc_divergence - 0.5) * 2  # Normalize around 0.5
        
        metrics["monetary_policy_divergence"] = (policy_divergence_signal + china_policy_signal) / 2
        
        # Global Capital Flow Analysis
        flows = data["capital_flows"]
        em_flow_signal = flows["emerging_market_flows"] / 50e9  # Normalize by $50B
        safe_haven_pressure = -(flows["safe_haven_demand"] - 1.0)  # Negative for AUD
        
        carry_trade = flows["currency_carry_trades"]
        aud_carry_signal = carry_trade["aud_carry_attractiveness"] * 2
        
        metrics["capital_flow_pressure"] = (em_flow_signal + safe_haven_pressure + aud_carry_signal) / 3
        
        # Global Risk Environment
        risk = data["global_risk_indicators"]
        vix_signal = -(risk["vix"] - 20) / 15  # Higher VIX negative for risk assets
        credit_spread_signal = -(risk["credit_spreads_ig"] - 100) / 100  # Wider spreads negative
        
        metrics["global_risk_appetite"] = (vix_signal + credit_spread_signal) / 2
        
        # Commodity Cycle Analysis
        commodities = data["commodity_cycles"]
        dxy_pressure = -(commodities["dxy_strength"] - 100) / 10  # Stronger USD negative for commodities
        commodity_momentum_signal = commodities["commodity_momentum"] * 3
        
        metrics["commodity_cycle_support"] = (dxy_pressure + commodity_momentum_signal) / 2
        
        # Trade Relationship Strength
        trade = data["trade_relationships"]
        us_china_stability = (trade["us_china_trade_intensity"] - 0.5) * 2
        regional_integration = (trade["regional_trade_agreements"] - 1.0)
        
        metrics["trade_environment"] = (us_china_stability + regional_integration) / 2
        
        return metrics
    
    def get_factor_weight(self, timeframe: str) -> float:
        """Global macro relevant across all timeframes"""
        weights = {
            "1d": 0.15,
            "5d": 0.25,
            "30d": 0.30,
            "90d": 0.35
        }
        return weights.get(timeframe, 0.26)

# Enhanced Factor Integration Manager
class EnhancedFactorManager:
    """Manages all enhanced prediction factors and their integration"""
    
    def __init__(self):
        self.analyzers = {
            "options_flow": OptionsFlowAnalyzer(),
            "institutional_flow": InstitutionalFlowAnalyzer(),
            "social_sentiment": SocialSentimentAnalyzer(),
            "economic_leading": EconomicLeadingAnalyzer(),
            "global_macro": GlobalMacroAnalyzer()
        }
        
        self.factor_history = {}  # Store historical factor values
        
    async def collect_all_factors(self) -> Dict[str, Dict[str, Any]]:
        """Collect data from all factor analyzers"""
        
        logger.info("🔍 Collecting enhanced prediction factors...")
        
        factor_data = {}
        
        for name, analyzer in self.analyzers.items():
            try:
                logger.info(f"📊 Collecting {name} data...")
                factor_data[name] = await analyzer.collect_data()
            except Exception as e:
                logger.error(f"❌ Error collecting {name}: {e}")
                factor_data[name] = {}
        
        return factor_data
    
    async def analyze_all_factors(self, factor_data: Dict[str, Dict[str, Any]], 
                                timeframe: str = "5d") -> Dict[str, Any]:
        """Analyze all factors and return weighted composite scores"""
        
        logger.info(f"🧮 Analyzing factors for {timeframe} timeframe...")
        
        factor_scores = {}
        weighted_composite = 0.0
        total_weight = 0.0
        
        for name, analyzer in self.analyzers.items():
            if name in factor_data and factor_data[name]:
                try:
                    # Get factor analysis
                    scores = await analyzer.analyze_factor(factor_data[name])
                    factor_scores[name] = scores
                    
                    # Calculate weighted contribution
                    weight = analyzer.get_factor_weight(timeframe)
                    factor_average = np.mean(list(scores.values()))
                    
                    weighted_composite += factor_average * weight
                    total_weight += weight
                    
                    logger.info(f"✅ {name}: {factor_average:.3f} (weight: {weight:.2f})")
                    
                except Exception as e:
                    logger.error(f"❌ Error analyzing {name}: {e}")
        
        # Normalize composite score
        if total_weight > 0:
            weighted_composite /= total_weight
        
        # Generate factor summary
        factor_summary = {
            "composite_score": weighted_composite,
            "individual_factors": factor_scores,
            "factor_count": len(factor_scores),
            "total_weight": total_weight,
            "timeframe": timeframe,
            "analysis_timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # Store in history
        self.factor_history[timeframe] = factor_summary
        
        logger.info(f"📈 Enhanced factors composite score: {weighted_composite:.3f}")
        
        return factor_summary
    
    def get_factor_impact_summary(self) -> Dict[str, Any]:
        """Generate summary of factor impacts and implementation priorities"""
        
        factor_info = []
        
        for name, analyzer in self.analyzers.items():
            info = analyzer.factor_info
            factor_info.append({
                "name": info.name,
                "category": info.category.value,
                "impact": info.impact.value,
                "complexity": info.complexity.value,
                "priority": info.implementation_priority,
                "australian_relevance": info.australian_relevance,
                "update_frequency": info.update_frequency,
                "description": info.description
            })
        
        # Sort by priority
        factor_info.sort(key=lambda x: x["priority"])
        
        return {
            "implemented_factors": len(factor_info),
            "high_impact_factors": len([f for f in factor_info if f["impact"] in ["critical", "high"]]),
            "real_time_factors": len([f for f in factor_info if f["update_frequency"] == "real-time"]),
            "factor_details": factor_info,
            "next_priorities": [f["name"] for f in factor_info[:3]]
        }

# Global enhanced factor manager instance
enhanced_factor_manager = EnhancedFactorManager()

# Export classes for integration
__all__ = [
    'EnhancedFactorManager',
    'FactorAnalyzer', 
    'OptionsFlowAnalyzer',
    'InstitutionalFlowAnalyzer',
    'SocialSentimentAnalyzer',
    'EconomicLeadingAnalyzer',
    'GlobalMacroAnalyzer',
    'enhanced_factor_manager'
]