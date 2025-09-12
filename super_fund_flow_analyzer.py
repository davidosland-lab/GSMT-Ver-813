"""
Australian Superannuation Fund Flow Analysis System
Critical Factor #1: Track Australia's $3.3T superannuation system flows and allocation changes
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import json
import os
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AssetClass(Enum):
    """Superannuation asset class categories"""
    AUSTRALIAN_EQUITIES = "australian_equities"
    INTERNATIONAL_EQUITIES = "international_equities"
    FIXED_INCOME = "fixed_income"
    PROPERTY = "property"
    INFRASTRUCTURE = "infrastructure"
    ALTERNATIVES = "alternatives"
    CASH = "cash"
    COMMODITIES = "commodities"

class FundType(Enum):
    """Types of superannuation funds"""
    INDUSTRY = "industry"           # Industry super funds (largest)
    RETAIL = "retail"              # Bank-owned retail funds
    PUBLIC_SECTOR = "public_sector" # Government employee funds
    CORPORATE = "corporate"         # Employer-specific funds
    SMSF = "smsf"                  # Self-managed super funds

@dataclass
class SuperFundAllocation:
    """Superannuation fund asset allocation data"""
    fund_name: str
    fund_type: FundType
    total_assets: float  # Total fund assets in AUD
    allocation_date: datetime
    asset_allocations: Dict[AssetClass, float]  # Percentage allocations
    monthly_change: Dict[AssetClass, float]     # Monthly allocation changes
    flows: Dict[AssetClass, float]              # Dollar flows by asset class
    member_demographics: Dict[str, Any]          # Age groups, contribution patterns
    performance_data: Dict[str, float]           # Recent performance metrics

@dataclass
class SuperSystemFlows:
    """Aggregate superannuation system flows"""
    reporting_date: datetime
    total_system_assets: float
    monthly_contributions: float
    monthly_withdrawals: float
    net_flows: float
    asset_class_flows: Dict[AssetClass, float]
    fund_type_flows: Dict[FundType, float]
    demographic_analysis: Dict[str, Any]
    seasonal_patterns: Dict[str, float]
    market_impact_score: float

class SuperFundDataCollector:
    """Collects superannuation fund data from multiple sources"""
    
    def __init__(self):
        self.session = None
        self.cache = {}
        self.cache_ttl = 86400  # 24 hours cache
        
        # APRA API endpoints (simulated - replace with actual endpoints)
        self.apra_endpoints = {
            'quarterly_statistics': 'https://www.apra.gov.au/sites/default/files/statistics_annual_superannuation_bulletin_data.xlsx',
            'monthly_funds': 'https://www.apra.gov.au/sites/default/files/monthly_superannuation_statistics.xlsx',
            'asset_allocations': 'https://www.apra.gov.au/sites/default/files/superannuation_asset_allocation_data.xlsx'
        }
        
        # Major super funds to track (top 20 by assets)
        self.major_funds = {
            'AustralianSuper': {'type': FundType.INDUSTRY, 'assets': 300e9},
            'Aware Super': {'type': FundType.INDUSTRY, 'assets': 140e9},
            'Sunsuper': {'type': FundType.INDUSTRY, 'assets': 90e9},
            'REST Industry Super': {'type': FundType.INDUSTRY, 'assets': 70e9},
            'UniSuper': {'type': FundType.INDUSTRY, 'assets': 65e9},
            'HESTA': {'type': FundType.INDUSTRY, 'assets': 60e9},
            'Cbus': {'type': FundType.INDUSTRY, 'assets': 55e9},
            'Colonial First State': {'type': FundType.RETAIL, 'assets': 50e9},
            'MLC Super': {'type': FundType.RETAIL, 'assets': 45e9},
            'BT Super': {'type': FundType.RETAIL, 'assets': 40e9},
            'Commonwealth Super': {'type': FundType.PUBLIC_SECTOR, 'assets': 250e9},
            'QSuper': {'type': FundType.PUBLIC_SECTOR, 'assets': 120e9},
            'VicSuper': {'type': FundType.PUBLIC_SECTOR, 'assets': 30e9},
            'NGS Super': {'type': FundType.INDUSTRY, 'assets': 25e9},
            'Media Super': {'type': FundType.INDUSTRY, 'assets': 20e9}
        }
        
        # Default asset allocations by fund type (industry averages)
        self.default_allocations = {
            FundType.INDUSTRY: {
                AssetClass.AUSTRALIAN_EQUITIES: 0.24,
                AssetClass.INTERNATIONAL_EQUITIES: 0.43,
                AssetClass.FIXED_INCOME: 0.18,
                AssetClass.PROPERTY: 0.08,
                AssetClass.INFRASTRUCTURE: 0.04,
                AssetClass.ALTERNATIVES: 0.02,
                AssetClass.CASH: 0.01
            },
            FundType.RETAIL: {
                AssetClass.AUSTRALIAN_EQUITIES: 0.22,
                AssetClass.INTERNATIONAL_EQUITIES: 0.38,
                AssetClass.FIXED_INCOME: 0.25,
                AssetClass.PROPERTY: 0.06,
                AssetClass.INFRASTRUCTURE: 0.03,
                AssetClass.ALTERNATIVES: 0.04,
                AssetClass.CASH: 0.02
            },
            FundType.PUBLIC_SECTOR: {
                AssetClass.AUSTRALIAN_EQUITIES: 0.26,
                AssetClass.INTERNATIONAL_EQUITIES: 0.40,
                AssetClass.FIXED_INCOME: 0.20,
                AssetClass.PROPERTY: 0.07,
                AssetClass.INFRASTRUCTURE: 0.05,
                AssetClass.ALTERNATIVES: 0.01,
                AssetClass.CASH: 0.01
            }
        }

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=60),
            headers={'User-Agent': 'SuperFund Analysis Bot 1.0'}
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def fetch_apra_statistics(self) -> Dict[str, Any]:
        """Fetch quarterly statistics from APRA"""
        
        logger.info("📊 Fetching APRA superannuation statistics...")
        
        # Simulate APRA data collection (replace with actual API calls)
        current_date = datetime.now(timezone.utc)
        
        apra_data = {
            'system_totals': {
                'total_assets': 3.5e12,  # $3.5 trillion
                'total_members': 16.8e6,  # 16.8 million members
                'total_funds': 180,       # Number of APRA-regulated funds
                'growth_rate_annual': 0.087,  # 8.7% annual growth
                'reporting_date': current_date
            },
            'fund_type_breakdown': {
                FundType.INDUSTRY.value: {'assets': 1.8e12, 'members': 8.2e6},
                FundType.RETAIL.value: {'assets': 800e9, 'members': 4.1e6}, 
                FundType.PUBLIC_SECTOR.value: {'assets': 750e9, 'members': 3.2e6},
                FundType.CORPORATE.value: {'assets': 150e9, 'members': 1.3e6}
            },
            'quarterly_flows': {
                'contributions': 42e9,    # $42B quarterly contributions
                'withdrawals': 28e9,      # $28B quarterly withdrawals  
                'net_flows': 14e9,        # $14B net inflows
                'investment_returns': 85e9 # $85B investment returns
            },
            'asset_allocation_aggregate': {
                AssetClass.AUSTRALIAN_EQUITIES.value: {'allocation': 0.24, 'assets': 840e9},
                AssetClass.INTERNATIONAL_EQUITIES.value: {'allocation': 0.43, 'assets': 1505e9},
                AssetClass.FIXED_INCOME.value: {'allocation': 0.19, 'assets': 665e9},
                AssetClass.PROPERTY.value: {'allocation': 0.08, 'assets': 280e9},
                AssetClass.INFRASTRUCTURE.value: {'allocation': 0.04, 'assets': 140e9},
                AssetClass.ALTERNATIVES.value: {'allocation': 0.01, 'assets': 35e9},
                AssetClass.CASH.value: {'allocation': 0.01, 'assets': 35e9}
            }
        }
        
        return apra_data

    async def fetch_fund_specific_data(self, fund_name: str) -> Optional[SuperFundAllocation]:
        """Fetch specific fund data including allocations and flows"""
        
        if fund_name not in self.major_funds:
            logger.warning(f"Fund {fund_name} not in major funds list")
            return None
        
        fund_info = self.major_funds[fund_name]
        fund_type = fund_info['type']
        total_assets = fund_info['assets']
        
        logger.info(f"📈 Fetching data for {fund_name} (${total_assets/1e9:.1f}B assets)")
        
        # Simulate fund-specific data collection
        base_allocations = self.default_allocations[fund_type].copy()
        
        # Add some realistic variation and recent changes
        current_date = datetime.now(timezone.utc)
        
        # Simulate recent allocation changes (monthly)
        monthly_changes = {}
        asset_flows = {}
        
        for asset_class in AssetClass:
            # Random walk allocation changes within realistic bounds
            base_change = np.random.normal(0, 0.002)  # 0.2% monthly volatility
            monthly_changes[asset_class] = base_change
            
            # Calculate dollar flows
            current_allocation = base_allocations.get(asset_class, 0)
            flow_amount = (base_change * total_assets) + (current_allocation * total_assets * 0.01)  # Include growth
            asset_flows[asset_class] = flow_amount
        
        # Ensure allocations sum to 1.0 after changes
        total_allocation = sum(base_allocations.values())
        if total_allocation != 1.0:
            # Normalize allocations
            for asset_class in base_allocations:
                base_allocations[asset_class] /= total_allocation
        
        # Generate member demographics
        demographics = self._generate_member_demographics(fund_type, total_assets)
        
        # Performance data (simulate recent returns)
        performance = self._generate_performance_data(fund_name)
        
        return SuperFundAllocation(
            fund_name=fund_name,
            fund_type=fund_type,
            total_assets=total_assets,
            allocation_date=current_date,
            asset_allocations=base_allocations,
            monthly_change=monthly_changes,
            flows=asset_flows,
            member_demographics=demographics,
            performance_data=performance
        )

    def _generate_member_demographics(self, fund_type: FundType, total_assets: float) -> Dict[str, Any]:
        """Generate realistic member demographic data"""
        
        # Demographics vary by fund type
        if fund_type == FundType.INDUSTRY:
            return {
                'average_age': 42,
                'age_distribution': {
                    'under_25': 0.18,
                    '25_34': 0.24,
                    '35_44': 0.22,
                    '45_54': 0.20,
                    '55_64': 0.12,
                    'over_65': 0.04
                },
                'average_balance': total_assets / 2.5e6,  # Estimated members
                'contribution_patterns': {
                    'salary_sacrifice': 0.35,
                    'employer_only': 0.45,
                    'additional_voluntary': 0.20
                },
                'member_growth_rate': 0.024  # 2.4% annual member growth
            }
        elif fund_type == FundType.PUBLIC_SECTOR:
            return {
                'average_age': 45,
                'age_distribution': {
                    'under_25': 0.08,
                    '25_34': 0.18,
                    '35_44': 0.25,
                    '45_54': 0.28,
                    '55_64': 0.18,
                    'over_65': 0.03
                },
                'average_balance': total_assets / 1.8e6,
                'contribution_patterns': {
                    'salary_sacrifice': 0.55,
                    'employer_only': 0.25,
                    'additional_voluntary': 0.20
                },
                'member_growth_rate': 0.012
            }
        else:  # RETAIL
            return {
                'average_age': 40,
                'age_distribution': {
                    'under_25': 0.22,
                    '25_34': 0.26,
                    '35_44': 0.20,
                    '45_54': 0.18,
                    '55_64': 0.11,
                    'over_65': 0.03
                },
                'average_balance': total_assets / 1.2e6,
                'contribution_patterns': {
                    'salary_sacrifice': 0.25,
                    'employer_only': 0.55,
                    'additional_voluntary': 0.20
                },
                'member_growth_rate': 0.008
            }

    def _generate_performance_data(self, fund_name: str) -> Dict[str, float]:
        """Generate realistic fund performance data"""
        
        # Simulate returns with some variation between funds
        base_return = 0.078  # 7.8% annual return baseline
        
        # Add fund-specific variation
        fund_variation = hash(fund_name) % 200 / 10000  # -1% to +1% variation
        
        return {
            '1_year_return': base_return + fund_variation + np.random.normal(0, 0.02),
            '3_year_return': 0.085 + fund_variation + np.random.normal(0, 0.015),
            '5_year_return': 0.092 + fund_variation + np.random.normal(0, 0.01),
            '10_year_return': 0.088 + fund_variation + np.random.normal(0, 0.008),
            'monthly_return': (base_return + fund_variation) / 12 + np.random.normal(0, 0.008),
            'volatility': 0.12 + abs(np.random.normal(0, 0.02)),  # ~12% volatility
            'sharpe_ratio': 0.65 + np.random.normal(0, 0.1),
            'tracking_error': abs(np.random.normal(0, 0.015))
        }

class SuperFundFlowAnalyzer:
    """Analyzes superannuation flows for market impact prediction"""
    
    def __init__(self):
        self.data_collector = SuperFundDataCollector()
        self.historical_data = {}
        
    async def collect_system_flows(self) -> SuperSystemFlows:
        """Collect and analyze aggregate superannuation system flows"""
        
        logger.info("🏦 Analyzing superannuation system flows...")
        
        async with self.data_collector as collector:
            # Get APRA system-wide data
            apra_data = await collector.fetch_apra_statistics()
            
            # Collect major fund data
            fund_allocations = []
            for fund_name in list(collector.major_funds.keys())[:10]:  # Top 10 funds
                fund_data = await collector.fetch_fund_specific_data(fund_name)
                if fund_data:
                    fund_allocations.append(fund_data)
        
        # Analyze aggregate flows
        return self._analyze_aggregate_flows(apra_data, fund_allocations)
    
    def _analyze_aggregate_flows(self, apra_data: Dict[str, Any], 
                                fund_allocations: List[SuperFundAllocation]) -> SuperSystemFlows:
        """Analyze aggregate superannuation flows and their market impact"""
        
        system_totals = apra_data['system_totals']
        quarterly_flows = apra_data['quarterly_flows']
        
        # Calculate asset class flows
        asset_class_flows = {}
        total_flow_impact = 0
        
        for asset_class in AssetClass:
            monthly_flows = []
            
            # Aggregate flows from major funds
            for fund in fund_allocations:
                if asset_class in fund.flows:
                    monthly_flows.append(fund.flows[asset_class])
            
            # Calculate aggregate flow for this asset class
            if monthly_flows:
                aggregate_flow = sum(monthly_flows)
                asset_class_flows[asset_class] = aggregate_flow
                
                # Weight Australian equity flows more heavily for market impact
                if asset_class == AssetClass.AUSTRALIAN_EQUITIES:
                    total_flow_impact += aggregate_flow * 2.0  # 2x weight
                else:
                    total_flow_impact += aggregate_flow * 0.3  # Indirect impact
        
        # Calculate fund type flows
        fund_type_flows = {}
        for fund_type in FundType:
            type_funds = [f for f in fund_allocations if f.fund_type == fund_type]
            if type_funds:
                aus_equity_flows = [f.flows.get(AssetClass.AUSTRALIAN_EQUITIES, 0) for f in type_funds]
                fund_type_flows[fund_type] = sum(aus_equity_flows)
        
        # Demographic analysis
        demographic_analysis = self._analyze_demographic_trends(fund_allocations)
        
        # Seasonal patterns
        seasonal_patterns = self._calculate_seasonal_patterns()
        
        # Market impact score (0-100)
        market_impact_score = self._calculate_market_impact_score(
            asset_class_flows, total_flow_impact, system_totals['total_assets']
        )
        
        return SuperSystemFlows(
            reporting_date=datetime.now(timezone.utc),
            total_system_assets=system_totals['total_assets'],
            monthly_contributions=quarterly_flows['contributions'] / 3,
            monthly_withdrawals=quarterly_flows['withdrawals'] / 3,
            net_flows=quarterly_flows['net_flows'] / 3,
            asset_class_flows=asset_class_flows,
            fund_type_flows=fund_type_flows,
            demographic_analysis=demographic_analysis,
            seasonal_patterns=seasonal_patterns,
            market_impact_score=market_impact_score
        )
    
    def _analyze_demographic_trends(self, fund_allocations: List[SuperFundAllocation]) -> Dict[str, Any]:
        """Analyze demographic trends affecting super flows"""
        
        total_assets = sum(f.total_assets for f in fund_allocations)
        
        # Weight demographics by fund size
        weighted_age_groups = {}
        weighted_contribution_patterns = {}
        
        for fund in fund_allocations:
            weight = fund.total_assets / total_assets
            demographics = fund.member_demographics
            
            # Weight age distributions
            for age_group, percentage in demographics['age_distribution'].items():
                if age_group not in weighted_age_groups:
                    weighted_age_groups[age_group] = 0
                weighted_age_groups[age_group] += percentage * weight
            
            # Weight contribution patterns
            for pattern, percentage in demographics['contribution_patterns'].items():
                if pattern not in weighted_contribution_patterns:
                    weighted_contribution_patterns[pattern] = 0
                weighted_contribution_patterns[pattern] += percentage * weight
        
        # Calculate demographic-driven flow predictions
        baby_boomer_retirement_impact = weighted_age_groups.get('55_64', 0) * 0.15  # 15% retirement rate
        millennial_growth_impact = weighted_age_groups.get('25_34', 0) * 1.8  # High contribution phase
        
        return {
            'age_distribution': weighted_age_groups,
            'contribution_patterns': weighted_contribution_patterns,
            'baby_boomer_retirement_pressure': baby_boomer_retirement_impact,
            'millennial_contribution_growth': millennial_growth_impact,
            'net_demographic_flow_bias': millennial_growth_impact - baby_boomer_retirement_impact
        }
    
    def _calculate_seasonal_patterns(self) -> Dict[str, float]:
        """Calculate seasonal patterns in superannuation flows"""
        
        # Historical seasonal patterns (normalized -1 to 1)
        return {
            'january': -0.2,   # Post-holiday, lower contributions
            'february': 0.1,   # Recovery
            'march': 0.3,      # End of fiscal year preparations
            'april': -0.1,     # Post-tax season
            'may': 0.0,        # Neutral
            'june': 0.8,       # End of financial year (strongest)
            'july': 0.2,       # New financial year
            'august': 0.0,     # Neutral
            'september': 0.1,  # Slight uptick
            'october': 0.0,    # Neutral
            'november': 0.2,   # Pre-holiday boost
            'december': -0.3   # Holiday season, reduced activity
        }
    
    def _calculate_market_impact_score(self, asset_class_flows: Dict[AssetClass, float], 
                                     total_flow_impact: float, system_assets: float) -> float:
        """Calculate market impact score from superannuation flows"""
        
        # Focus on Australian equity flows
        aus_equity_flow = asset_class_flows.get(AssetClass.AUSTRALIAN_EQUITIES, 0)
        
        # Calculate as percentage of total ASX market cap (~$2.5T)
        asx_market_cap = 2.5e12
        flow_percentage = abs(aus_equity_flow) / asx_market_cap * 100
        
        # Market impact score (0-100)
        # >1% of market cap = very high impact (80-100)
        # 0.1-1% = high impact (40-80) 
        # <0.1% = low impact (0-40)
        
        if flow_percentage > 1.0:
            impact_score = 80 + min(flow_percentage - 1.0, 1.0) * 20
        elif flow_percentage > 0.1:
            impact_score = 40 + (flow_percentage - 0.1) / 0.9 * 40
        else:
            impact_score = flow_percentage / 0.1 * 40
        
        # Directional bias (positive = buying pressure, negative = selling)
        direction_multiplier = 1.0 if aus_equity_flow > 0 else -1.0
        
        return min(impact_score, 100) * direction_multiplier

    async def get_market_prediction_factors(self) -> Dict[str, float]:
        """Get standardized factors for market prediction integration"""
        
        logger.info("🎯 Generating super fund prediction factors...")
        
        # Collect current system flows
        system_flows = await self.collect_system_flows()
        
        # Generate standardized factors (-1 to +1 scale)
        factors = {}
        
        # Flow momentum factor
        aus_equity_flow = system_flows.asset_class_flows.get(AssetClass.AUSTRALIAN_EQUITIES, 0)
        flow_momentum = np.tanh(aus_equity_flow / 5e9)  # Normalize around $5B monthly flow
        factors['super_flow_momentum'] = flow_momentum
        
        # Allocation trend factor
        # (Would compare current vs historical allocations)
        allocation_trend = 0.15  # Simulated positive trend
        factors['super_allocation_trend'] = allocation_trend
        
        # Demographic pressure factor
        demo = system_flows.demographic_analysis
        demographic_pressure = demo.get('net_demographic_flow_bias', 0) * 2
        factors['super_demographic_pressure'] = np.tanh(demographic_pressure)
        
        # Seasonal factor
        current_month = datetime.now().strftime('%B').lower()
        seasonal_bias = system_flows.seasonal_patterns.get(current_month, 0)
        factors['super_seasonal_factor'] = seasonal_bias
        
        # Market impact factor (absolute impact regardless of direction)
        market_impact = abs(system_flows.market_impact_score) / 100
        factors['super_market_impact'] = market_impact
        
        # Performance chasing factor (funds moving to recent outperformers)
        performance_chasing = 0.25  # Simulated moderate performance chasing
        factors['super_performance_chasing'] = performance_chasing
        
        logger.info(f"📊 Super fund factors: Flow momentum: {flow_momentum:.3f}, "
                   f"Market impact: {market_impact:.3f}, Seasonal: {seasonal_bias:.3f}")
        
        return factors

# Global instance
super_fund_analyzer = SuperFundFlowAnalyzer()

# Export classes
__all__ = [
    'SuperFundFlowAnalyzer',
    'SuperFundDataCollector', 
    'SuperFundAllocation',
    'SuperSystemFlows',
    'AssetClass',
    'FundType',
    'super_fund_analyzer'
]