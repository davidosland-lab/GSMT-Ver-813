#!/usr/bin/env python3
"""
Global Conflict Monitor - Comprehensive Worldwide Threat Assessment
Uses multiple data sources including ACLED, GDELT, and real-time news feeds

Monitors:
- Armed conflicts worldwide (not just Ukraine/Gaza) 
- Political instability and protests
- Terrorism and security threats
- Economic sanctions and trade wars
- Natural disasters with economic impact
- Cyber warfare and infrastructure threats
- Energy supply disruptions
- Food security crises
"""

import asyncio
import aiohttp
import json
import re
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Any
import logging
import os
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import time
from collections import defaultdict
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ThreatLevel(Enum):
    """Global threat levels for market impact"""
    MINIMAL = 1      # <5% market volatility impact
    LOW = 2          # 5-15% volatility impact  
    MODERATE = 3     # 15-30% volatility impact
    HIGH = 4         # 30-60% volatility impact
    CRITICAL = 5     # >60% volatility impact

class ConflictType(Enum):
    """Types of global conflicts and threats"""
    ARMED_CONFLICT = "armed_conflict"
    POLITICAL_INSTABILITY = "political_instability"
    TERRORISM = "terrorism"
    ECONOMIC_SANCTIONS = "economic_sanctions"
    TRADE_WAR = "trade_war"
    CYBER_WARFARE = "cyber_warfare"
    NATURAL_DISASTER = "natural_disaster"
    ENERGY_CRISIS = "energy_crisis"
    FOOD_SECURITY = "food_security"
    PANDEMIC = "pandemic"
    FINANCIAL_CRISIS = "financial_crisis"

class RegionImpact(Enum):
    """Regional economic importance weighting"""
    GLOBAL_MAJOR = 1.0      # US, China, EU
    REGIONAL_MAJOR = 0.8    # Japan, India, Brazil, Russia
    EMERGING_KEY = 0.6      # Indonesia, Mexico, South Korea
    REGIONAL_MODERATE = 0.4 # Other significant economies
    LOCAL = 0.2             # Smaller economies

@dataclass
class GlobalThreat:
    """Represents a global threat/conflict affecting markets"""
    threat_id: str
    title: str
    description: str
    threat_type: ConflictType
    threat_level: ThreatLevel
    countries_affected: List[str]
    region_impact_weight: float
    market_impact_score: float  # -1.0 to +1.0
    volatility_multiplier: float  # 1.0 to 10.0
    affected_sectors: List[str]  # Energy, Defense, Tech, etc.
    start_date: datetime
    last_update: datetime
    is_escalating: bool
    escalation_probability: float  # 0.0 to 1.0
    keywords: List[str]
    data_sources: List[str]
    economic_impact_estimate: float  # USD billions

class GlobalConflictMonitor:
    """Comprehensive global threat and conflict monitoring system"""
    
    def __init__(self):
        self.session = None
        self.active_threats = {}
        self.cache = {}
        self.cache_ttl = 1800  # 30 minutes
        
        # Multiple data sources for comprehensive coverage
        self.data_sources = {
            # News Sources - Global Coverage
            'reuters_world': {
                'url': 'https://feeds.reuters.com/reuters/worldNews',
                'type': 'rss',
                'reliability': 0.95
            },
            'bbc_world': {
                'url': 'http://feeds.bbci.co.uk/news/world/rss.xml',
                'type': 'rss',
                'reliability': 0.90
            },
            'guardian_world': {
                'url': 'https://www.theguardian.com/world/rss',
                'type': 'rss',
                'reliability': 0.85
            },
            'ap_world': {
                'url': 'https://feeds.apnews.com/apnews/rss/world',
                'type': 'rss',
                'reliability': 0.90
            },
            'cnn_world': {
                'url': 'http://rss.cnn.com/rss/edition.rss',
                'type': 'rss',
                'reliability': 0.80
            },
            
            # Regional Specialized Sources
            'al_jazeera': {
                'url': 'https://www.aljazeera.com/xml/rss/all.xml',
                'type': 'rss',
                'reliability': 0.85
            },
            'scmp_asia': {
                'url': 'https://www.scmp.com/rss/91/feed',
                'type': 'rss',
                'reliability': 0.80
            }
        }
        
        # Comprehensive threat patterns - much more than just Ukraine/Gaza
        self.threat_patterns = {
            # Active Armed Conflicts
            'ukraine_russia': {
                'keywords': ['ukraine', 'russia', 'putin', 'zelensky', 'crimea', 'donetsk', 'mariupol', 'kherson'],
                'type': ConflictType.ARMED_CONFLICT,
                'threat_level': ThreatLevel.HIGH,
                'region_impact': RegionImpact.GLOBAL_MAJOR,
                'sectors': ['energy', 'agriculture', 'defense', 'commodities'],
                'market_impact': -0.8
            },
            'israel_palestine': {
                'keywords': ['gaza', 'israel', 'palestine', 'hamas', 'west bank', 'netanyahu', 'hezbollah'],
                'type': ConflictType.ARMED_CONFLICT,
                'threat_level': ThreatLevel.HIGH,
                'region_impact': RegionImpact.REGIONAL_MAJOR,
                'sectors': ['energy', 'defense', 'technology'],
                'market_impact': -0.6
            },
            
            # China-Taiwan Tensions
            'china_taiwan': {
                'keywords': ['taiwan', 'china taiwan', 'strait', 'tsmc', 'semiconductor', 'xi jinping'],
                'type': ConflictType.POLITICAL_INSTABILITY,
                'threat_level': ThreatLevel.HIGH,
                'region_impact': RegionImpact.GLOBAL_MAJOR,
                'sectors': ['technology', 'semiconductors', 'manufacturing'],
                'market_impact': -0.9
            },
            
            # Korea Peninsula
            'north_korea': {
                'keywords': ['north korea', 'kim jong', 'missile', 'nuclear', 'pyongyang', 'dmz'],
                'type': ConflictType.POLITICAL_INSTABILITY,
                'threat_level': ThreatLevel.MODERATE,
                'region_impact': RegionImpact.REGIONAL_MAJOR,
                'sectors': ['defense', 'technology'],
                'market_impact': -0.4
            },
            
            # Middle East Broader Conflicts
            'iran_tensions': {
                'keywords': ['iran', 'tehran', 'nuclear deal', 'sanctions iran', 'ayatollah', 'strait hormuz'],
                'type': ConflictType.ECONOMIC_SANCTIONS,
                'threat_level': ThreatLevel.MODERATE,
                'region_impact': RegionImpact.GLOBAL_MAJOR,
                'sectors': ['energy', 'oil', 'shipping'],
                'market_impact': -0.5
            },
            
            # India-Pakistan
            'india_pakistan': {
                'keywords': ['india pakistan', 'kashmir', 'border clash', 'modi', 'pakistan india'],
                'type': ConflictType.ARMED_CONFLICT,
                'threat_level': ThreatLevel.MODERATE,
                'region_impact': RegionImpact.REGIONAL_MAJOR,
                'sectors': ['defense', 'technology'],
                'market_impact': -0.4
            },
            
            # China-India Border
            'china_india_border': {
                'keywords': ['china india', 'ladakh', 'galwan', 'border dispute', 'himalaya'],
                'type': ConflictType.POLITICAL_INSTABILITY,
                'threat_level': ThreatLevel.LOW,
                'region_impact': RegionImpact.REGIONAL_MAJOR,
                'sectors': ['defense', 'technology'],
                'market_impact': -0.3
            },
            
            # African Conflicts
            'ethiopia_conflict': {
                'keywords': ['ethiopia', 'tigray', 'addis ababa', 'civil war ethiopia'],
                'type': ConflictType.ARMED_CONFLICT,
                'threat_level': ThreatLevel.LOW,
                'region_impact': RegionImpact.LOCAL,
                'sectors': ['commodities', 'mining'],
                'market_impact': -0.2
            },
            
            # Venezuela Crisis
            'venezuela_crisis': {
                'keywords': ['venezuela', 'maduro', 'caracas', 'oil venezuela', 'economic crisis venezuela'],
                'type': ConflictType.POLITICAL_INSTABILITY,
                'threat_level': ThreatLevel.LOW,
                'region_impact': RegionImpact.REGIONAL_MODERATE,
                'sectors': ['energy', 'oil'],
                'market_impact': -0.3
            },
            
            # Trade Wars
            'us_china_trade': {
                'keywords': ['us china trade', 'tariffs', 'trade war', 'biden china', 'trump china'],
                'type': ConflictType.TRADE_WAR,
                'threat_level': ThreatLevel.MODERATE,
                'region_impact': RegionImpact.GLOBAL_MAJOR,
                'sectors': ['technology', 'manufacturing', 'agriculture'],
                'market_impact': -0.6
            },
            
            # Cyber Warfare
            'cyber_attacks': {
                'keywords': ['cyber attack', 'ransomware', 'hacking', 'infrastructure hack', 'state sponsored'],
                'type': ConflictType.CYBER_WARFARE,
                'threat_level': ThreatLevel.MODERATE,
                'region_impact': RegionImpact.GLOBAL_MAJOR,
                'sectors': ['technology', 'financial services', 'energy'],
                'market_impact': -0.4
            },
            
            # Energy Crises
            'energy_supply': {
                'keywords': ['energy crisis', 'oil shortage', 'gas pipeline', 'opec', 'energy security'],
                'type': ConflictType.ENERGY_CRISIS,
                'threat_level': ThreatLevel.HIGH,
                'region_impact': RegionImpact.GLOBAL_MAJOR,
                'sectors': ['energy', 'utilities', 'transportation'],
                'market_impact': -0.7
            },
            
            # Food Security
            'food_crisis': {
                'keywords': ['food crisis', 'grain shortage', 'famine', 'food security', 'agricultural'],
                'type': ConflictType.FOOD_SECURITY,
                'threat_level': ThreatLevel.MODERATE,
                'region_impact': RegionImpact.GLOBAL_MAJOR,
                'sectors': ['agriculture', 'food', 'commodities'],
                'market_impact': -0.5
            },
            
            # Economic Sanctions
            'economic_sanctions': {
                'keywords': ['sanctions', 'economic embargo', 'financial sanctions', 'swift ban'],
                'type': ConflictType.ECONOMIC_SANCTIONS,
                'threat_level': ThreatLevel.MODERATE,
                'region_impact': RegionImpact.GLOBAL_MAJOR,
                'sectors': ['banking', 'energy', 'technology'],
                'market_impact': -0.5
            },
            
            # Natural Disasters with Economic Impact
            'natural_disasters': {
                'keywords': ['earthquake', 'tsunami', 'hurricane', 'flood', 'natural disaster', 'climate'],
                'type': ConflictType.NATURAL_DISASTER,
                'threat_level': ThreatLevel.MODERATE,
                'region_impact': RegionImpact.REGIONAL_MODERATE,
                'sectors': ['insurance', 'construction', 'utilities'],
                'market_impact': -0.3
            }
        }
        
        # Country economic importance weights
        self.country_weights = {
            # Major Global Economies
            'united states': 1.0, 'china': 1.0, 'germany': 0.9, 'japan': 0.9,
            'united kingdom': 0.8, 'france': 0.8, 'india': 0.8, 'italy': 0.7,
            'brazil': 0.7, 'canada': 0.7, 'russia': 0.7, 'south korea': 0.6,
            'spain': 0.6, 'mexico': 0.6, 'indonesia': 0.6, 'netherlands': 0.5,
            'saudi arabia': 0.6, 'turkey': 0.5, 'taiwan': 0.8, 'switzerland': 0.5,
            'belgium': 0.4, 'poland': 0.4, 'ireland': 0.4, 'israel': 0.4,
            'austria': 0.4, 'nigeria': 0.4, 'egypt': 0.4, 'south africa': 0.4,
            'philippines': 0.4, 'bangladesh': 0.3, 'vietnam': 0.4, 'chile': 0.3,
            'finland': 0.3, 'romania': 0.3, 'czech republic': 0.3, 'portugal': 0.3,
            'new zealand': 0.3, 'peru': 0.3, 'greece': 0.3, 'ukraine': 0.4,
            'venezuela': 0.3, 'qatar': 0.3, 'kuwait': 0.3, 'uae': 0.4
        }
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={'User-Agent': 'GlobalConflictMonitor/2.0'}
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid"""
        if cache_key not in self.cache:
            return False
        return time.time() - self.cache[cache_key]['timestamp'] < self.cache_ttl
    
    async def collect_global_threats(self, hours_back: int = 48) -> List[GlobalThreat]:
        """Collect global threats from all monitoring sources"""
        
        cache_key = f"global_threats_{hours_back}"
        if self._is_cache_valid(cache_key):
            logger.info(f"🗂️  Using cached global threats data")
            return self.cache[cache_key]['data']
        
        logger.info(f"🌍 Collecting global threats from {len(self.data_sources)} sources")
        
        all_threats = []
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours_back)
        
        # Collect from all news sources
        for source_name, source_config in self.data_sources.items():
            try:
                threats = await self._fetch_threats_from_source(
                    source_config, source_name, cutoff_time
                )
                all_threats.extend(threats)
                logger.info(f"✓ {source_name}: Found {len(threats)} potential threats")
                
            except Exception as e:
                logger.warning(f"⚠️  Failed to fetch from {source_name}: {e}")
        
        # Process and consolidate threats
        consolidated_threats = self._consolidate_threats(all_threats)
        
        # Add escalation probability analysis
        for threat in consolidated_threats:
            threat.escalation_probability = self._calculate_escalation_probability(threat)
        
        # Cache results
        self.cache[cache_key] = {
            'data': consolidated_threats,
            'timestamp': time.time()
        }
        
        logger.info(f"🌍 Processed {len(consolidated_threats)} unique global threats")
        return consolidated_threats
    
    async def _fetch_threats_from_source(self, 
                                       source_config: Dict, 
                                       source_name: str,
                                       cutoff_time: datetime) -> List[GlobalThreat]:
        """Fetch potential threats from a news source"""
        
        threats = []
        
        try:
            async with self.session.get(source_config['url']) as response:
                if response.status != 200:
                    logger.warning(f"Failed to fetch from {source_name}: {response.status}")
                    return []
                
                content = await response.text()
                
                # Parse RSS/XML content
                root = ET.fromstring(content)
                items = root.findall('.//item')
                
                for item in items:
                    threat = await self._parse_item_to_threat(
                        item, source_name, source_config['reliability'], cutoff_time
                    )
                    if threat:
                        threats.append(threat)
                        
        except Exception as e:
            logger.error(f"Error parsing threats from {source_name}: {e}")
        
        return threats
    
    async def _parse_item_to_threat(self, 
                                  item: ET.Element, 
                                  source_name: str,
                                  reliability: float,
                                  cutoff_time: datetime) -> Optional[GlobalThreat]:
        """Parse RSS item into GlobalThreat"""
        
        try:
            title = item.findtext('title', '').strip()
            description = item.findtext('description', '').strip()
            pub_date_str = item.findtext('pubDate', '')
            
            if not title:
                return None
            
            # Parse publication date
            try:
                pub_date = self._parse_date(pub_date_str)
            except:
                pub_date = datetime.now(timezone.utc)
            
            if pub_date < cutoff_time:
                return None
            
            # Check against all threat patterns
            full_text = f"{title} {description}".lower()
            
            for threat_id, pattern in self.threat_patterns.items():
                if any(keyword.lower() in full_text for keyword in pattern['keywords']):
                    
                    # Calculate threat intensity based on keyword matches and source reliability
                    keyword_matches = sum(1 for kw in pattern['keywords'] if kw.lower() in full_text)
                    intensity = (keyword_matches / len(pattern['keywords'])) * reliability
                    
                    # Extract affected countries
                    affected_countries = self._extract_countries(full_text)
                    if not affected_countries:
                        affected_countries = self._infer_countries_from_pattern(threat_id)
                    
                    # Calculate region impact weight
                    region_weight = max(self.country_weights.get(country.lower(), 0.2) 
                                      for country in affected_countries)
                    
                    # Adjust market impact based on intensity and region
                    adjusted_impact = pattern['market_impact'] * intensity * region_weight
                    
                    # Calculate volatility multiplier
                    base_volatility = {
                        ThreatLevel.MINIMAL: 1.1,
                        ThreatLevel.LOW: 1.3,
                        ThreatLevel.MODERATE: 1.8,
                        ThreatLevel.HIGH: 2.5,
                        ThreatLevel.CRITICAL: 4.0
                    }
                    
                    volatility_mult = base_volatility[pattern['threat_level']] * (0.5 + intensity)
                    
                    # Create unique threat ID
                    unique_id = hashlib.md5(f"{threat_id}_{source_name}_{pub_date.isoformat()}".encode()).hexdigest()[:12]
                    
                    threat = GlobalThreat(
                        threat_id=unique_id,
                        title=title,
                        description=description,
                        threat_type=pattern['type'],
                        threat_level=pattern['threat_level'],
                        countries_affected=affected_countries,
                        region_impact_weight=region_weight,
                        market_impact_score=adjusted_impact,
                        volatility_multiplier=volatility_mult,
                        affected_sectors=pattern['sectors'],
                        start_date=pub_date,
                        last_update=datetime.now(timezone.utc),
                        is_escalating=self._detect_escalation_keywords(full_text),
                        escalation_probability=0.0,  # Will be calculated later
                        keywords=pattern['keywords'],
                        data_sources=[source_name],
                        economic_impact_estimate=self._estimate_economic_impact(pattern, region_weight)
                    )
                    
                    logger.debug(f"🔍 Detected threat: {threat_id} - {title[:50]}...")
                    return threat
        
        except Exception as e:
            logger.error(f"Error parsing threat item: {e}")
        
        return None
    
    def _parse_date(self, date_str: str) -> datetime:
        """Parse various RSS date formats"""
        
        from dateutil import parser
        try:
            return parser.parse(date_str).replace(tzinfo=timezone.utc)
        except:
            return datetime.now(timezone.utc)
    
    def _extract_countries(self, text: str) -> List[str]:
        """Extract country names from text"""
        
        # Simple country extraction (could be enhanced with NLP)
        countries_mentioned = []
        for country in self.country_weights.keys():
            if country in text:
                countries_mentioned.append(country.title())
        
        return list(set(countries_mentioned))[:5]  # Limit to top 5
    
    def _infer_countries_from_pattern(self, threat_id: str) -> List[str]:
        """Infer countries based on threat pattern"""
        
        country_mapping = {
            'ukraine_russia': ['Ukraine', 'Russia'],
            'israel_palestine': ['Israel', 'Palestine'],
            'china_taiwan': ['China', 'Taiwan'],
            'north_korea': ['North Korea', 'South Korea'],
            'iran_tensions': ['Iran', 'United States'],
            'india_pakistan': ['India', 'Pakistan'],
            'china_india_border': ['China', 'India'],
            'us_china_trade': ['United States', 'China'],
            'ethiopia_conflict': ['Ethiopia'],
            'venezuela_crisis': ['Venezuela']
        }
        
        return country_mapping.get(threat_id, ['Global'])
    
    def _detect_escalation_keywords(self, text: str) -> bool:
        """Detect if threat is escalating based on keywords"""
        
        escalation_keywords = [
            'escalat', 'worsen', 'intensif', 'spread', 'expand', 'increas',
            'attack', 'strike', 'bomb', 'invasion', 'mobiliz', 'deploy'
        ]
        
        return any(keyword in text for keyword in escalation_keywords)
    
    def _estimate_economic_impact(self, pattern: Dict, region_weight: float) -> float:
        """Estimate economic impact in USD billions"""
        
        # Base impact estimates by threat type
        base_impacts = {
            ConflictType.ARMED_CONFLICT: 100.0,
            ConflictType.TRADE_WAR: 200.0,
            ConflictType.ECONOMIC_SANCTIONS: 50.0,
            ConflictType.ENERGY_CRISIS: 150.0,
            ConflictType.CYBER_WARFARE: 30.0,
            ConflictType.NATURAL_DISASTER: 75.0,
            ConflictType.POLITICAL_INSTABILITY: 25.0
        }
        
        base_impact = base_impacts.get(pattern['type'], 10.0)
        return base_impact * region_weight
    
    def _consolidate_threats(self, threats: List[GlobalThreat]) -> List[GlobalThreat]:
        """Consolidate similar threats and calculate aggregate impacts"""
        
        if not threats:
            return []
        
        # Group by threat type and affected countries
        threat_groups = defaultdict(list)
        
        for threat in threats:
            # Create grouping key
            countries_key = '_'.join(sorted(threat.countries_affected))
            group_key = f"{threat.threat_type.value}_{countries_key}"
            threat_groups[group_key].append(threat)
        
        consolidated = []
        
        for group_key, group_threats in threat_groups.items():
            if not group_threats:
                continue
                
            # Sort by recency and impact
            group_threats.sort(key=lambda x: (x.last_update, abs(x.market_impact_score)), reverse=True)
            
            # Take the most significant threat as base
            primary_threat = group_threats[0]
            
            # Aggregate multiple recent threats
            if len(group_threats) > 1:
                recent_threats = [t for t in group_threats if 
                               (datetime.now(timezone.utc) - t.last_update).total_seconds() < 86400]
                
                if len(recent_threats) > 1:
                    # Aggregate impact scores
                    total_impact = sum(abs(t.market_impact_score) for t in recent_threats[:3])
                    primary_threat.market_impact_score = -total_impact if primary_threat.market_impact_score < 0 else total_impact
                    
                    # Increase volatility for multiple events
                    primary_threat.volatility_multiplier = min(primary_threat.volatility_multiplier * 1.3, 8.0)
                    
                    # Combine data sources
                    all_sources = set()
                    for t in recent_threats:
                        all_sources.update(t.data_sources)
                    primary_threat.data_sources = list(all_sources)
                    
                    logger.info(f"📊 Consolidated {len(recent_threats)} threats for {group_key}: impact={primary_threat.market_impact_score:.2f}")
            
            consolidated.append(primary_threat)
        
        # Sort by threat level and impact
        consolidated.sort(key=lambda x: (x.threat_level.value, abs(x.market_impact_score)), reverse=True)
        
        return consolidated
    
    def _calculate_escalation_probability(self, threat: GlobalThreat) -> float:
        """Calculate probability of threat escalation"""
        
        base_probability = {
            ThreatLevel.MINIMAL: 0.1,
            ThreatLevel.LOW: 0.2,
            ThreatLevel.MODERATE: 0.4,
            ThreatLevel.HIGH: 0.6,
            ThreatLevel.CRITICAL: 0.8
        }
        
        prob = base_probability[threat.threat_level]
        
        # Increase if already escalating
        if threat.is_escalating:
            prob += 0.2
        
        # Increase for certain threat types
        if threat.threat_type in [ConflictType.ARMED_CONFLICT, ConflictType.TRADE_WAR]:
            prob += 0.1
        
        return min(prob, 0.95)
    
    async def calculate_global_threat_index(self, threats: Optional[List[GlobalThreat]] = None) -> Dict[str, Any]:
        """Calculate comprehensive global threat index"""
        
        if threats is None:
            threats = await self.collect_global_threats(hours_back=72)
        
        if not threats:
            return {
                'global_threat_index': 15.0,  # Base threat level (15%)
                'threat_level': 'low',
                'active_conflicts': 0,
                'major_threats': [],
                'sector_risks': {},
                'regional_breakdown': {},
                'total_economic_impact': 0.0,
                'volatility_estimate': 20.0
            }
        
        # Calculate composite threat index
        base_threat = 15.0  # 15% baseline global threat
        conflict_threat = 0.0
        total_economic_impact = 0.0
        
        # Sector risk accumulation
        sector_risks = defaultdict(float)
        
        # Regional breakdown
        regional_threats = defaultdict(list)
        
        major_threats = []
        
        for threat in threats:
            # Weight by region importance and recency
            time_weight = max(0.1, 1.0 - (datetime.now(timezone.utc) - threat.last_update).total_seconds() / (7 * 86400))
            
            # Add to threat index
            threat_contribution = (
                abs(threat.market_impact_score) * 
                threat.region_impact_weight * 
                time_weight * 
                threat.threat_level.value * 5  # Scale by threat level
            )
            conflict_threat += threat_contribution
            
            # Economic impact
            total_economic_impact += threat.economic_impact_estimate * time_weight
            
            # Sector risks
            for sector in threat.affected_sectors:
                sector_risks[sector] += abs(threat.market_impact_score) * time_weight
            
            # Regional classification
            for country in threat.countries_affected:
                regional_threats[country].append(threat)
            
            # Major threats (high impact)
            if threat.threat_level.value >= 3:  # MODERATE or higher
                major_threats.append({
                    'title': threat.title,
                    'type': threat.threat_type.name,
                    'level': threat.threat_level.name,
                    'countries': threat.countries_affected,
                    'impact': threat.market_impact_score,
                    'escalation_prob': threat.escalation_probability,
                    'economic_impact': threat.economic_impact_estimate,
                    'hours_ago': (datetime.now(timezone.utc) - threat.last_update).total_seconds() / 3600
                })
        
        # Calculate final global threat index
        global_threat_index = min(base_threat + conflict_threat * 100, 100.0)  # Cap at 100%
        
        # Determine overall threat level
        if global_threat_index < 25:
            threat_level = 'low'
        elif global_threat_index < 45:
            threat_level = 'moderate'  
        elif global_threat_index < 70:
            threat_level = 'high'
        else:
            threat_level = 'critical'
        
        # Count active conflicts
        active_conflicts = sum(1 for t in threats if t.threat_type == ConflictType.ARMED_CONFLICT)
        
        # Volatility estimate (enhanced from baseline)
        volatility_estimate = 20.0 + (global_threat_index - 15.0) * 1.5  # Scale volatility with threats
        
        logger.info(f"🌍 Global threat assessment: {global_threat_index:.1f}% ({threat_level}) - {active_conflicts} conflicts, ${total_economic_impact:.1f}B impact")
        
        return {
            'global_threat_index': global_threat_index,
            'threat_level': threat_level,
            'active_conflicts': active_conflicts,
            'total_threats': len(threats),
            'major_threats': sorted(major_threats, key=lambda x: x['impact'], reverse=True)[:10],
            'sector_risks': dict(sorted(sector_risks.items(), key=lambda x: x[1], reverse=True)[:10]),
            'regional_breakdown': {region: len(threats) for region, threats in regional_threats.items()},
            'total_economic_impact': total_economic_impact,
            'volatility_estimate': volatility_estimate,
            'assessment_timestamp': datetime.now(timezone.utc).isoformat(),
            'data_sources_used': len(self.data_sources)
        }

# Global instance
global_conflict_monitor = GlobalConflictMonitor()

async def test_global_monitor():
    """Test the global conflict monitoring system"""
    
    print("🌍 Testing Global Conflict Monitor")
    print("=" * 60)
    
    async with global_conflict_monitor as monitor:
        # Test threat collection
        threats = await monitor.collect_global_threats(hours_back=48)
        print(f"📊 Found {len(threats)} global threats")
        
        # Show top threats
        for i, threat in enumerate(threats[:5], 1):
            print(f"{i}. {threat.threat_type.name}: {threat.title[:60]}...")
            print(f"   Countries: {', '.join(threat.countries_affected[:3])}")
            print(f"   Impact: {threat.market_impact_score:+.2f}, Level: {threat.threat_level.name}")
            print()
        
        # Test global threat index
        threat_index = await monitor.calculate_global_threat_index(threats)
        print(f"🚨 Global Threat Index: {threat_index['global_threat_index']:.1f}% ({threat_index['threat_level'].upper()})")
        print(f"⚔️  Active Conflicts: {threat_index['active_conflicts']}")
        print(f"💰 Economic Impact: ${threat_index['total_economic_impact']:.1f}B")
        print(f"📈 Volatility Estimate: {threat_index['volatility_estimate']:.1f}%")
        
        if threat_index['major_threats']:
            print("\n🎯 Major Threats:")
            for i, threat in enumerate(threat_index['major_threats'][:3], 1):
                print(f"{i}. {threat['type']}: {threat['title'][:50]}...")
                print(f"   Level: {threat['level']}, Escalation Prob: {threat['escalation_prob']:.1%}")
        
        if threat_index['sector_risks']:
            print(f"\n📊 Top Sector Risks: {list(threat_index['sector_risks'].keys())[:5]}")

if __name__ == "__main__":
    asyncio.run(test_global_monitor())