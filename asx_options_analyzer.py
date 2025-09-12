"""
ASX Options Flow Analysis System
Critical Factor #2: Track ASX options flow, gamma exposure, and derivatives intelligence
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
import math
from scipy.stats import norm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptionType(Enum):
    """Option contract types"""
    CALL = "call"
    PUT = "put"

class ExpiryType(Enum):
    """Option expiry classifications"""
    WEEKLY = "weekly"
    MONTHLY = "monthly"  
    QUARTERLY = "quarterly"

class FlowType(Enum):
    """Types of options flow"""
    OPENING = "opening"      # Opening new positions
    CLOSING = "closing"      # Closing existing positions  
    ROLLING = "rolling"      # Rolling positions

@dataclass
class OptionContract:
    """Individual ASX option contract data"""
    symbol: str              # Underlying symbol (e.g., CBA, BHP, XJO)
    strike: float           # Strike price
    expiry: datetime        # Expiration date
    option_type: OptionType # Call or Put
    bid: float              # Bid price
    ask: float              # Ask price
    last_price: float       # Last traded price
    volume: int             # Daily volume
    open_interest: int      # Total open interest
    implied_volatility: float  # Implied volatility
    delta: float            # Option delta
    gamma: float            # Option gamma
    theta: float            # Option theta
    vega: float             # Option vega
    underlying_price: float # Current underlying price
    time_to_expiry: float   # Time to expiry in years

@dataclass
class OptionsFlowData:
    """Aggregated options flow analysis"""
    symbol: str
    analysis_date: datetime
    put_call_ratio: float           # Put volume / Call volume
    put_call_ratio_oi: float        # Put OI / Call OI
    total_volume: int               # Total options volume
    volume_spike_ratio: float       # Volume vs 20-day average
    unusual_activity_score: float   # 0-100 unusual activity score
    net_gamma_exposure: float       # Market maker net gamma exposure
    dealer_positioning: Dict[str, float]  # MM positioning metrics
    volatility_surface: Dict[str, float]  # Vol surface metrics
    flow_sentiment: str             # bullish, bearish, neutral
    market_impact_score: float      # Expected market impact 0-100

@dataclass
class GammaExposure:
    """Market maker gamma exposure analysis"""
    symbol: str
    total_gamma: float              # Total gamma exposure
    call_gamma: float               # Gamma from call options
    put_gamma: float                # Gamma from put options
    gamma_flip_level: float         # Price level where gamma flips
    dealer_position: str            # long_gamma, short_gamma, neutral
    hedging_pressure: float         # Expected hedging flow
    volatility_impact: float        # Impact on volatility
    price_magnetism: Dict[str, float]  # Strike levels with high gamma

class ASXOptionsDataCollector:
    """Collects ASX options market data and analytics"""
    
    def __init__(self):
        self.session = None
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes cache for options data
        
        # ASX symbols to track for options (most liquid)
        self.tracked_symbols = [
            'XJO',   # ASX 200 Index (most liquid)
            'CBA',   # Commonwealth Bank
            'BHP',   # BHP Billiton  
            'CSL',   # CSL Limited
            'WBC',   # Westpac Banking
            'ANZ',   # ANZ Banking
            'NAB',   # National Australia Bank
            'WES',   # Wesfarmers
            'WOW',   # Woolworths
            'TLS',   # Telstra
            'RIO',   # Rio Tinto
            'MQG',   # Macquarie Group
            'TCL',   # Transurban
            'STO',   # Santos
            'FMG'    # Fortescue Metals
        ]
        
        # Simulated ASX options data endpoints
        self.asx_endpoints = {
            'options_chain': 'https://www.asx.com.au/data/options/chain/{symbol}.json',
            'market_data': 'https://www.asx.com.au/data/options/market/{symbol}.json',
            'historical_vol': 'https://www.asx.com.au/data/options/volatility/{symbol}.json'
        }
        
        # Risk-free rate (RBA cash rate)
        self.risk_free_rate = 0.0435  # 4.35%

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={'User-Agent': 'ASX Options Analysis Bot 1.0'}
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def fetch_options_chain(self, symbol: str) -> List[OptionContract]:
        """Fetch complete options chain for a symbol"""
        
        logger.info(f"📊 Fetching options chain for {symbol}...")
        
        # Check cache first
        cache_key = f"options_chain_{symbol}"
        if cache_key in self.cache:
            cache_entry = self.cache[cache_key]
            if (datetime.now().timestamp() - cache_entry['timestamp']) < self.cache_ttl:
                return cache_entry['data']
        
        # Simulate options chain data (replace with actual ASX API)
        options_chain = await self._generate_simulated_options_chain(symbol)
        
        # Cache the results
        self.cache[cache_key] = {
            'data': options_chain,
            'timestamp': datetime.now().timestamp()
        }
        
        return options_chain

    async def _generate_simulated_options_chain(self, symbol: str) -> List[OptionContract]:
        """Generate realistic simulated options chain data"""
        
        # Get current underlying price (simulated)
        base_prices = {
            'XJO': 7600, 'CBA': 105, 'BHP': 45, 'CSL': 280, 'WBC': 22,
            'ANZ': 27, 'NAB': 32, 'WES': 55, 'WOW': 38, 'TLS': 4.2,
            'RIO': 115, 'MQG': 180, 'TCL': 14, 'STO': 8.5, 'FMG': 22
        }
        
        underlying_price = base_prices.get(symbol, 50.0)
        current_time = datetime.now(timezone.utc)
        
        options_contracts = []
        
        # Generate options for next 3 expiry cycles
        for expiry_weeks in [1, 2, 4, 8]:  # Weekly and monthly expirations
            expiry_date = current_time + timedelta(weeks=expiry_weeks)
            time_to_expiry = expiry_weeks / 52.0  # Convert to years
            
            # Generate strike prices around current price
            strike_range = np.arange(
                underlying_price * 0.8,   # 20% OTM puts
                underlying_price * 1.2,   # 20% OTM calls
                underlying_price * 0.02   # 2% strike intervals
            )
            
            for strike in strike_range:
                strike = round(strike, 2)
                
                # Calculate Black-Scholes Greeks
                call_greeks = self._calculate_black_scholes(
                    underlying_price, strike, time_to_expiry, self.risk_free_rate, 0.20, 'call'
                )
                put_greeks = self._calculate_black_scholes(
                    underlying_price, strike, time_to_expiry, self.risk_free_rate, 0.20, 'put'
                )
                
                # Generate realistic volume and open interest
                moneyness = abs(strike - underlying_price) / underlying_price
                liquidity_factor = max(0.1, 1.0 - moneyness * 3)  # More liquid ATM
                
                base_volume = int(np.random.exponential(50) * liquidity_factor)
                base_oi = int(np.random.exponential(200) * liquidity_factor)
                
                # Call option
                call_contract = OptionContract(
                    symbol=symbol,
                    strike=strike,
                    expiry=expiry_date,
                    option_type=OptionType.CALL,
                    bid=max(0.01, call_greeks['price'] - 0.02),
                    ask=call_greeks['price'] + 0.02,
                    last_price=call_greeks['price'],
                    volume=base_volume,
                    open_interest=base_oi,
                    implied_volatility=0.20 + np.random.normal(0, 0.02),
                    delta=call_greeks['delta'],
                    gamma=call_greeks['gamma'],
                    theta=call_greeks['theta'],
                    vega=call_greeks['vega'],
                    underlying_price=underlying_price,
                    time_to_expiry=time_to_expiry
                )
                options_contracts.append(call_contract)
                
                # Put option
                put_contract = OptionContract(
                    symbol=symbol,
                    strike=strike,
                    expiry=expiry_date,
                    option_type=OptionType.PUT,
                    bid=max(0.01, put_greeks['price'] - 0.02),
                    ask=put_greeks['price'] + 0.02,
                    last_price=put_greeks['price'],
                    volume=int(base_volume * 0.8),  # Puts typically less volume
                    open_interest=int(base_oi * 0.9),
                    implied_volatility=0.20 + np.random.normal(0, 0.02),
                    delta=put_greeks['delta'],
                    gamma=put_greeks['gamma'],
                    theta=put_greeks['theta'],
                    vega=put_greeks['vega'],
                    underlying_price=underlying_price,
                    time_to_expiry=time_to_expiry
                )
                options_contracts.append(put_contract)
        
        return options_contracts

    def _calculate_black_scholes(self, S: float, K: float, T: float, r: float, 
                                sigma: float, option_type: str) -> Dict[str, float]:
        """Calculate Black-Scholes option price and Greeks"""
        
        if T <= 0:
            # Handle expired options
            if option_type.lower() == 'call':
                intrinsic = max(0, S - K)
            else:
                intrinsic = max(0, K - S)
            return {
                'price': intrinsic,
                'delta': 1.0 if intrinsic > 0 else 0.0,
                'gamma': 0.0,
                'theta': 0.0,
                'vega': 0.0
            }
        
        # Black-Scholes calculation
        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        
        if option_type.lower() == 'call':
            price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
            delta = norm.cdf(d1)
        else:  # put
            price = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            delta = -norm.cdf(-d1)
        
        # Greeks
        gamma = norm.pdf(d1) / (S * sigma * math.sqrt(T))
        theta_divisor = 365.25  # Convert to per day
        theta = -(S * norm.pdf(d1) * sigma / (2 * math.sqrt(T)) + 
                 r * K * math.exp(-r * T) * norm.cdf(d2 if option_type.lower() == 'call' else -d2)) / theta_divisor
        vega = S * norm.pdf(d1) * math.sqrt(T) / 100  # Per 1% vol change
        
        return {
            'price': max(0.01, price),
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega
        }

class OptionsFlowAnalyzer:
    """Analyzes ASX options flow for market prediction signals"""
    
    def __init__(self):
        self.data_collector = ASXOptionsDataCollector()
        self.flow_history = {}
        
    async def analyze_options_flow(self, symbol: str) -> OptionsFlowData:
        """Analyze options flow for a specific symbol"""
        
        logger.info(f"🔍 Analyzing options flow for {symbol}...")
        
        async with self.data_collector as collector:
            options_chain = await collector.fetch_options_chain(symbol)
        
        if not options_chain:
            logger.warning(f"No options data available for {symbol}")
            return None
        
        # Calculate flow metrics
        flow_data = self._calculate_flow_metrics(symbol, options_chain)
        
        # Calculate gamma exposure
        gamma_exposure = self._calculate_gamma_exposure(symbol, options_chain)
        
        # Combine into comprehensive flow analysis
        return self._create_flow_analysis(symbol, options_chain, flow_data, gamma_exposure)
    
    def _calculate_flow_metrics(self, symbol: str, options: List[OptionContract]) -> Dict[str, Any]:
        """Calculate basic options flow metrics"""
        
        calls = [opt for opt in options if opt.option_type == OptionType.CALL]
        puts = [opt for opt in options if opt.option_type == OptionType.PUT]
        
        # Volume analysis
        call_volume = sum(opt.volume for opt in calls)
        put_volume = sum(opt.volume for opt in puts)
        total_volume = call_volume + put_volume
        
        # Open interest analysis
        call_oi = sum(opt.open_interest for opt in calls)
        put_oi = sum(opt.open_interest for opt in puts)
        
        # Put/Call ratios
        pc_ratio = put_volume / max(call_volume, 1)
        pc_ratio_oi = put_oi / max(call_oi, 1)
        
        # Volume spike analysis (simulate 20-day average)
        avg_volume = total_volume * 0.7  # Simulate average being 30% lower
        volume_spike_ratio = total_volume / avg_volume
        
        return {
            'call_volume': call_volume,
            'put_volume': put_volume,
            'total_volume': total_volume,
            'call_oi': call_oi,
            'put_oi': put_oi,
            'pc_ratio': pc_ratio,
            'pc_ratio_oi': pc_ratio_oi,
            'volume_spike_ratio': volume_spike_ratio
        }
    
    def _calculate_gamma_exposure(self, symbol: str, options: List[OptionContract]) -> GammaExposure:
        """Calculate market maker gamma exposure"""
        
        if not options:
            return GammaExposure(
                symbol=symbol, total_gamma=0, call_gamma=0, put_gamma=0,
                gamma_flip_level=0, dealer_position="neutral", hedging_pressure=0,
                volatility_impact=0, price_magnetism={}
            )
        
        underlying_price = options[0].underlying_price
        
        # Calculate total gamma exposure
        # Assume market makers are short options (typical)
        call_gamma = -sum(opt.gamma * opt.open_interest for opt in options 
                         if opt.option_type == OptionType.CALL)
        put_gamma = -sum(opt.gamma * opt.open_interest for opt in options 
                        if opt.option_type == OptionType.PUT)
        total_gamma = call_gamma + put_gamma
        
        # Find gamma flip level (where total gamma = 0)
        gamma_flip_level = self._find_gamma_flip_level(options, underlying_price)
        
        # Determine dealer positioning
        if total_gamma > 1000:
            dealer_position = "long_gamma"
            hedging_pressure = -0.3  # Dealers sell into rallies
        elif total_gamma < -1000:
            dealer_position = "short_gamma" 
            hedging_pressure = 0.5   # Dealers buy into rallies (gamma squeeze potential)
        else:
            dealer_position = "neutral"
            hedging_pressure = 0.0
        
        # Calculate volatility impact
        volatility_impact = abs(total_gamma) / 10000 * 0.2  # Gamma per 10k contracts adds 20% vol impact
        
        # Find high gamma strikes (price magnetism)
        price_magnetism = self._find_high_gamma_strikes(options, underlying_price)
        
        return GammaExposure(
            symbol=symbol,
            total_gamma=total_gamma,
            call_gamma=call_gamma,
            put_gamma=put_gamma,
            gamma_flip_level=gamma_flip_level,
            dealer_position=dealer_position,
            hedging_pressure=hedging_pressure,
            volatility_impact=min(volatility_impact, 1.0),
            price_magnetism=price_magnetism
        )
    
    def _find_gamma_flip_level(self, options: List[OptionContract], current_price: float) -> float:
        """Find price level where dealer gamma exposure flips sign"""
        
        # Simplified calculation - find strike with highest total gamma
        strike_gamma = {}
        
        for opt in options:
            strike = opt.strike
            if strike not in strike_gamma:
                strike_gamma[strike] = 0
            
            # Market makers typically short options
            gamma_contribution = -opt.gamma * opt.open_interest
            if opt.option_type == OptionType.PUT:
                gamma_contribution *= -1  # Put gamma has opposite sign effect
            
            strike_gamma[strike] += gamma_contribution
        
        if not strike_gamma:
            return current_price
        
        # Find strike where gamma is closest to zero
        flip_level = min(strike_gamma.keys(), key=lambda k: abs(strike_gamma[k]))
        return flip_level
    
    def _find_high_gamma_strikes(self, options: List[OptionContract], 
                                underlying_price: float) -> Dict[str, float]:
        """Find strikes with high gamma concentration (price magnetism)"""
        
        strike_gamma = {}
        
        for opt in options:
            strike = opt.strike
            if abs(strike - underlying_price) / underlying_price > 0.15:  # Skip far OTM
                continue
                
            if strike not in strike_gamma:
                strike_gamma[strike] = 0
            
            strike_gamma[strike] += opt.gamma * opt.open_interest
        
        # Sort by absolute gamma and return top 5
        sorted_strikes = sorted(strike_gamma.items(), 
                               key=lambda x: abs(x[1]), reverse=True)
        
        return {f"strike_{strike}": gamma for strike, gamma in sorted_strikes[:5]}
    
    def _create_flow_analysis(self, symbol: str, options: List[OptionContract],
                             flow_data: Dict[str, Any], gamma_exposure: GammaExposure) -> OptionsFlowData:
        """Create comprehensive options flow analysis"""
        
        # Calculate unusual activity score
        unusual_score = self._calculate_unusual_activity_score(flow_data, options)
        
        # Determine flow sentiment
        sentiment = self._determine_flow_sentiment(flow_data, gamma_exposure)
        
        # Calculate market impact score
        market_impact = self._calculate_market_impact_score(flow_data, gamma_exposure)
        
        # Create volatility surface summary
        vol_surface = self._analyze_volatility_surface(options)
        
        # Dealer positioning metrics
        dealer_positioning = {
            'net_delta_exposure': sum(opt.delta * opt.open_interest * 
                                    (-1 if opt.option_type == OptionType.CALL else 1) for opt in options),
            'net_gamma_exposure': gamma_exposure.total_gamma,
            'hedging_flow_pressure': gamma_exposure.hedging_pressure,
            'volatility_impact': gamma_exposure.volatility_impact
        }
        
        return OptionsFlowData(
            symbol=symbol,
            analysis_date=datetime.now(timezone.utc),
            put_call_ratio=flow_data['pc_ratio'],
            put_call_ratio_oi=flow_data['pc_ratio_oi'],
            total_volume=flow_data['total_volume'],
            volume_spike_ratio=flow_data['volume_spike_ratio'],
            unusual_activity_score=unusual_score,
            net_gamma_exposure=gamma_exposure.total_gamma,
            dealer_positioning=dealer_positioning,
            volatility_surface=vol_surface,
            flow_sentiment=sentiment,
            market_impact_score=market_impact
        )
    
    def _calculate_unusual_activity_score(self, flow_data: Dict[str, Any], 
                                        options: List[OptionContract]) -> float:
        """Calculate unusual options activity score (0-100)"""
        
        score = 0.0
        
        # Volume spike component (40% weight)
        volume_spike = flow_data['volume_spike_ratio']
        if volume_spike > 3.0:
            score += 40
        elif volume_spike > 2.0:
            score += 20 + (volume_spike - 2.0) * 20
        elif volume_spike > 1.5:
            score += (volume_spike - 1.5) * 40
        
        # Put/Call ratio extremes (30% weight)
        pc_ratio = flow_data['pc_ratio']
        if pc_ratio > 2.0 or pc_ratio < 0.3:  # Extreme ratios
            score += 30
        elif pc_ratio > 1.5 or pc_ratio < 0.5:  # Moderate extremes
            score += 15
        
        # Large single trades (30% weight) - simulate detection
        large_trades_detected = np.random.random() > 0.7  # 30% chance of detection
        if large_trades_detected:
            score += np.random.uniform(10, 30)
        
        return min(score, 100)
    
    def _determine_flow_sentiment(self, flow_data: Dict[str, Any], 
                                 gamma_exposure: GammaExposure) -> str:
        """Determine overall sentiment from options flow"""
        
        pc_ratio = flow_data['pc_ratio']
        pc_ratio_oi = flow_data['pc_ratio_oi']
        hedging_pressure = gamma_exposure.hedging_pressure
        
        # Composite sentiment score
        sentiment_score = 0.0
        
        # Put/Call ratio analysis (bearish if high put activity)
        if pc_ratio > 1.5:
            sentiment_score -= 0.4
        elif pc_ratio < 0.5:
            sentiment_score += 0.4
        elif pc_ratio < 0.8:
            sentiment_score += 0.2
        
        # Open interest analysis
        if pc_ratio_oi > 1.2:
            sentiment_score -= 0.2
        elif pc_ratio_oi < 0.8:
            sentiment_score += 0.2
        
        # Gamma hedging pressure
        sentiment_score += hedging_pressure * 0.4
        
        # Classify sentiment
        if sentiment_score > 0.3:
            return "bullish"
        elif sentiment_score < -0.3:
            return "bearish"
        else:
            return "neutral"
    
    def _calculate_market_impact_score(self, flow_data: Dict[str, Any], 
                                      gamma_exposure: GammaExposure) -> float:
        """Calculate expected market impact score (0-100)"""
        
        impact_score = 0.0
        
        # Volume impact (30% weight)
        total_volume = flow_data['total_volume']
        if total_volume > 10000:  # High volume threshold
            impact_score += 30
        elif total_volume > 5000:
            impact_score += (total_volume - 5000) / 5000 * 30
        
        # Gamma exposure impact (40% weight)
        gamma_impact = min(abs(gamma_exposure.total_gamma) / 50000, 1.0) * 40
        impact_score += gamma_impact
        
        # Unusual activity impact (20% weight)
        unusual_score = flow_data.get('unusual_activity_score', 0)
        impact_score += (unusual_score / 100) * 20
        
        # Volume spike impact (10% weight)
        volume_spike = flow_data['volume_spike_ratio']
        spike_impact = min((volume_spike - 1.0) * 10, 10)
        impact_score += max(spike_impact, 0)
        
        return min(impact_score, 100)
    
    def _analyze_volatility_surface(self, options: List[OptionContract]) -> Dict[str, float]:
        """Analyze the implied volatility surface"""
        
        if not options:
            return {}
        
        # Calculate ATM implied volatility
        underlying_price = options[0].underlying_price
        atm_options = [opt for opt in options 
                      if abs(opt.strike - underlying_price) / underlying_price < 0.02]
        
        if atm_options:
            atm_iv = np.mean([opt.implied_volatility for opt in atm_options])
        else:
            atm_iv = 0.20  # Default 20%
        
        # Calculate volatility skew (25-delta put vs call)
        # Simplified: compare OTM puts vs OTM calls
        otm_puts = [opt for opt in options 
                   if opt.option_type == OptionType.PUT and opt.delta < -0.2 and opt.delta > -0.35]
        otm_calls = [opt for opt in options 
                    if opt.option_type == OptionType.CALL and opt.delta > 0.2 and opt.delta < 0.35]
        
        if otm_puts and otm_calls:
            put_iv = np.mean([opt.implied_volatility for opt in otm_puts])
            call_iv = np.mean([opt.implied_volatility for opt in otm_calls])
            skew = put_iv - call_iv
        else:
            skew = 0.03  # Typical 3% skew
        
        # Calculate term structure (near vs far expiry)
        near_term = [opt for opt in options if opt.time_to_expiry < 0.1]  # < 5 weeks
        far_term = [opt for opt in options if opt.time_to_expiry > 0.2]   # > 10 weeks
        
        if near_term and far_term:
            near_iv = np.mean([opt.implied_volatility for opt in near_term])
            far_iv = np.mean([opt.implied_volatility for opt in far_term])
            term_structure = near_iv - far_iv
        else:
            term_structure = 0.02  # Typical contango
        
        return {
            'atm_iv': atm_iv,
            'volatility_skew': skew,
            'term_structure': term_structure,
            'iv_percentile': min(max((atm_iv - 0.15) / 0.20, 0), 1)  # IV percentile estimate
        }

    async def get_market_prediction_factors(self, symbols: List[str] = None) -> Dict[str, float]:
        """Get standardized factors for market prediction integration"""
        
        if symbols is None:
            # Focus on most liquid options for market-wide signals
            symbols = ['XJO', 'CBA', 'BHP', 'CSL', 'WBC']
        
        logger.info(f"🎯 Analyzing options flow for {len(symbols)} symbols...")
        
        all_flows = []
        
        # Collect flow data for all symbols
        for symbol in symbols:
            try:
                flow_data = await self.analyze_options_flow(symbol)
                if flow_data:
                    all_flows.append(flow_data)
            except Exception as e:
                logger.warning(f"Failed to analyze options for {symbol}: {e}")
        
        if not all_flows:
            logger.warning("No options flow data available")
            return {}
        
        # Calculate aggregate factors
        factors = {}
        
        # Put/Call ratio factor (bearish when high)
        avg_pc_ratio = np.mean([flow.put_call_ratio for flow in all_flows])
        pc_factor = -np.tanh((avg_pc_ratio - 0.8) * 2)  # Normalize around 0.8
        factors['options_pc_sentiment'] = pc_factor
        
        # Gamma exposure factor
        total_gamma = sum([flow.net_gamma_exposure for flow in all_flows])
        gamma_factor = np.tanh(total_gamma / 100000)  # Normalize
        factors['options_gamma_pressure'] = gamma_factor
        
        # Volume spike factor
        avg_volume_spike = np.mean([flow.volume_spike_ratio for flow in all_flows])
        volume_factor = min((avg_volume_spike - 1.0) * 0.5, 1.0)
        factors['options_volume_momentum'] = volume_factor
        
        # Unusual activity factor
        avg_unusual = np.mean([flow.unusual_activity_score for flow in all_flows])
        unusual_factor = avg_unusual / 100
        factors['options_unusual_activity'] = unusual_factor
        
        # Market impact factor
        avg_market_impact = np.mean([flow.market_impact_score for flow in all_flows])
        impact_factor = avg_market_impact / 100
        factors['options_market_impact'] = impact_factor
        
        # Volatility surface factor
        vol_surfaces = [flow.volatility_surface for flow in all_flows if flow.volatility_surface]
        if vol_surfaces:
            avg_skew = np.mean([vs.get('volatility_skew', 0) for vs in vol_surfaces])
            skew_factor = np.tanh(avg_skew * 10)  # High skew often bearish
            factors['options_volatility_skew'] = skew_factor
        
        logger.info(f"📊 Options factors: PC sentiment: {pc_factor:.3f}, "
                   f"Gamma pressure: {gamma_factor:.3f}, Volume: {volume_factor:.3f}")
        
        return factors

# Global instance
asx_options_analyzer = OptionsFlowAnalyzer()

# Export classes
__all__ = [
    'OptionsFlowAnalyzer',
    'ASXOptionsDataCollector',
    'OptionContract',
    'OptionsFlowData', 
    'GammaExposure',
    'OptionType',
    'ExpiryType',
    'FlowType',
    'asx_options_analyzer'
]