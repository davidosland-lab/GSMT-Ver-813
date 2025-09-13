#!/usr/bin/env python3
"""
Geopolitical Events Monitoring System
Real-time monitoring of global conflicts and geopolitical events for market volatility assessment
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
from dataclasses import dataclass
from enum import Enum
import hashlib
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConflictSeverity(Enum):
    """Severity levels for geopolitical conflicts"""
    LOW = 1        # Minor tensions, diplomatic disputes
    MODERATE = 2   # Economic sanctions, trade disputes
    HIGH = 3       # Active conflicts, regional instability
    CRITICAL = 4   # Major wars, global security threats

class EventType(Enum):
    """Types of geopolitical events"""
    ARMED_CONFLICT = "armed_conflict"
    ECONOMIC_SANCTIONS = "economic_sanctions"
    DIPLOMATIC_CRISIS = "diplomatic_crisis"
    TERRORISM = "terrorism"
    NATURAL_DISASTER = "natural_disaster"
    POLITICAL_INSTABILITY = "political_instability"

@dataclass
class GeopoliticalEvent:
    """Represents a geopolitical event affecting markets"""
    event_id: str
    title: str
    description: str
    event_type: EventType
    severity: ConflictSeverity
    countries_involved: List[str]
    market_impact_score: float  # -1.0 to +1.0 (negative = bearish)
    volatility_multiplier: float  # 1.0 to 5.0
    start_date: datetime
    last_update: datetime
    is_active: bool
    keywords: List[str]

class GeopoliticalEventsMonitor:
    """Monitors real-time geopolitical events for market impact assessment"""
    
    def __init__(self):
        self.session = None
        self.active_events = {}
        self.cache = {}
        self.cache_ttl = 3600  # 1 hour cache for geopolitical data
        
        # RSS news sources for geopolitical monitoring
        self.geopolitical_sources = {
            'bbc_world': {
                'url': 'http://feeds.bbci.co.uk/news/world/rss.xml',
                'relevance': 0.9
            },
            'reuters_world': {
                'url': 'https://feeds.reuters.com/reuters/worldNews',
                'relevance': 0.95
            },
            'guardian_world': {
                'url': 'https://www.theguardian.com/world/rss',
                'relevance': 0.8
            },
            'reuters_politics': {
                'url': 'https://feeds.reuters.com/reuters/politicsNews',
                'relevance': 0.85
            }
        }
        
        # Critical geopolitical keywords for market impact
        self.geopolitical_keywords = {
            'ukraine_russia_conflict': {
                'keywords': ['ukraine', 'russia', 'putin', 'zelensky', 'kiev', 'moscow', 'invasion', 'war ukraine'],
                'severity': ConflictSeverity.HIGH,
                'event_type': EventType.ARMED_CONFLICT,
                'market_impact': -0.7,  # Highly bearish for global markets
                'volatility_multiplier': 3.5,
                'countries': ['Ukraine', 'Russia']
            },
            'gaza_israel_conflict': {
                'keywords': ['gaza', 'israel', 'hamas', 'palestine', 'netanyahu', 'gaza strip', 'israel palestine'],
                'severity': ConflictSeverity.HIGH,
                'event_type': EventType.ARMED_CONFLICT,
                'market_impact': -0.6,  # Bearish for global markets
                'volatility_multiplier': 2.8,
                'countries': ['Israel', 'Palestine']
            },
            'china_taiwan_tensions': {
                'keywords': ['taiwan', 'china taiwan', 'strait', 'xi jinping', 'tsai ing-wen'],
                'severity': ConflictSeverity.MODERATE,
                'event_type': EventType.DIPLOMATIC_CRISIS,
                'market_impact': -0.4,
                'volatility_multiplier': 2.2,
                'countries': ['China', 'Taiwan']
            },
            'north_korea_threats': {
                'keywords': ['north korea', 'kim jong', 'missile test', 'nuclear test', 'pyongyang'],
                'severity': ConflictSeverity.MODERATE,
                'event_type': EventType.POLITICAL_INSTABILITY,
                'market_impact': -0.3,
                'volatility_multiplier': 1.8,
                'countries': ['North Korea']
            },
            'iran_tensions': {
                'keywords': ['iran', 'tehran', 'nuclear deal', 'sanctions iran', 'ayatollah'],
                'severity': ConflictSeverity.MODERATE,
                'event_type': EventType.ECONOMIC_SANCTIONS,
                'market_impact': -0.2,
                'volatility_multiplier': 1.5,
                'countries': ['Iran']
            },
            'energy_crisis': {
                'keywords': ['energy crisis', 'oil embargo', 'gas shortage', 'pipeline sabotage'],
                'severity': ConflictSeverity.HIGH,
                'event_type': EventType.ECONOMIC_SANCTIONS,
                'market_impact': -0.8,
                'volatility_multiplier': 4.0,
                'countries': ['Global']
            }
        }
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={'User-Agent': 'GeopoliticalEventsMonitor/1.0'}
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
    
    async def collect_geopolitical_events(self, hours_back: int = 24) -> List[GeopoliticalEvent]:
        """Collect real-time geopolitical events from news sources"""
        
        cache_key = f"geopolitical_events_{hours_back}"
        if self._is_cache_valid(cache_key):
            logger.info(f"🗂️  Using cached geopolitical events data")
            return self.cache[cache_key]['data']
        
        logger.info(f"🌍 Collecting real geopolitical events from {len(self.geopolitical_sources)} sources")
        
        all_events = []
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours_back)
        
        for source_name, source_config in self.geopolitical_sources.items():
            try:
                events = await self._fetch_events_from_rss(
                    source_config['url'], 
                    source_name,
                    cutoff_time,
                    source_config['relevance']
                )
                all_events.extend(events)
                logger.info(f"✓ {source_name}: Found {len(events)} geopolitical events")
                
            except Exception as e:
                logger.warning(f"⚠️  Failed to fetch from {source_name}: {e}")
        
        # Process and deduplicate events
        processed_events = self._process_and_deduplicate_events(all_events)
        
        # Cache the results
        self.cache[cache_key] = {
            'data': processed_events,
            'timestamp': time.time()
        }
        
        logger.info(f"🌍 Collected {len(processed_events)} unique geopolitical events")
        return processed_events
    
    async def _fetch_events_from_rss(self, rss_url: str, source_name: str, cutoff_time: datetime, relevance: float) -> List[GeopoliticalEvent]:
        """Fetch geopolitical events from RSS feed"""
        
        events = []
        
        try:
            async with self.session.get(rss_url) as response:
                if response.status != 200:
                    logger.warning(f"Failed to fetch RSS from {source_name}: {response.status}")
                    return []
                
                content = await response.text()
                
                # Parse RSS XML
                root = ET.fromstring(content)
                
                # Find all item elements (RSS 2.0 format)
                items = root.findall('.//item')
                
                for item in items:
                    event = await self._parse_rss_item_to_event(item, source_name, cutoff_time, relevance)
                    if event:
                        events.append(event)
                        
        except Exception as e:
            logger.error(f"Error parsing RSS from {source_name}: {e}")
        
        return events
    
    async def _parse_rss_item_to_event(self, item: ET.Element, source_name: str, cutoff_time: datetime, relevance: float) -> Optional[GeopoliticalEvent]:
        """Parse RSS item into GeopoliticalEvent"""
        
        try:
            title = item.findtext('title', '').strip()
            description = item.findtext('description', '').strip()
            pub_date_str = item.findtext('pubDate', '')
            
            if not title:
                return None
            
            # Parse publication date
            try:
                # Handle various RSS date formats
                pub_date = datetime.strptime(pub_date_str, '%a, %d %b %Y %H:%M:%S %Z')
                pub_date = pub_date.replace(tzinfo=timezone.utc)
            except:
                try:
                    pub_date = datetime.strptime(pub_date_str, '%a, %d %b %Y %H:%M:%S %z')
                except:
                    pub_date = datetime.now(timezone.utc)  # Fallback to current time
            
            # Filter by time
            if pub_date < cutoff_time:
                return None
            
            # Check for geopolitical relevance
            full_text = f"{title} {description}".lower()
            
            for conflict_id, conflict_data in self.geopolitical_keywords.items():
                # Check if any keywords match
                if any(keyword.lower() in full_text for keyword in conflict_data['keywords']):
                    
                    # Calculate event impact
                    keyword_matches = sum(1 for keyword in conflict_data['keywords'] if keyword.lower() in full_text)
                    impact_intensity = min(keyword_matches / len(conflict_data['keywords']), 1.0)
                    
                    # Adjust market impact based on relevance and intensity
                    adjusted_impact = conflict_data['market_impact'] * relevance * impact_intensity
                    adjusted_volatility = conflict_data['volatility_multiplier'] * (0.5 + 0.5 * impact_intensity)
                    
                    # Create event ID
                    event_id = hashlib.md5(f"{conflict_id}_{title}_{pub_date.isoformat()}".encode()).hexdigest()[:12]
                    
                    event = GeopoliticalEvent(
                        event_id=event_id,
                        title=title,
                        description=description,
                        event_type=conflict_data['event_type'],
                        severity=conflict_data['severity'],
                        countries_involved=conflict_data['countries'],
                        market_impact_score=adjusted_impact,
                        volatility_multiplier=adjusted_volatility,
                        start_date=pub_date,
                        last_update=datetime.now(timezone.utc),
                        is_active=True,
                        keywords=conflict_data['keywords']
                    )
                    
                    logger.debug(f"🔍 Detected geopolitical event: {conflict_id} - {title[:50]}...")
                    return event
        
        except Exception as e:
            logger.error(f"Error parsing RSS item: {e}")
            
        return None
    
    def _process_and_deduplicate_events(self, events: List[GeopoliticalEvent]) -> List[GeopoliticalEvent]:
        """Process and deduplicate similar events"""
        
        if not events:
            return []
        
        # Group by event type and countries
        event_groups = {}
        
        for event in events:
            # Create grouping key
            countries_key = '_'.join(sorted(event.countries_involved))
            group_key = f"{event.event_type.value}_{countries_key}"
            
            if group_key not in event_groups:
                event_groups[group_key] = []
            event_groups[group_key].append(event)
        
        # For each group, keep the most recent and impactful event
        processed_events = []
        
        for group_key, group_events in event_groups.items():
            # Sort by recency and impact
            group_events.sort(key=lambda x: (x.last_update, abs(x.market_impact_score)), reverse=True)
            
            # Keep the top event from each group
            best_event = group_events[0]
            
            # If there are multiple recent events, aggregate their impact
            if len(group_events) > 1:
                recent_events = [e for e in group_events if 
                               (datetime.now(timezone.utc) - e.last_update).total_seconds() < 86400]  # Last 24 hours
                
                if len(recent_events) > 1:
                    # Aggregate impact scores
                    total_impact = sum(abs(e.market_impact_score) for e in recent_events[:3])  # Top 3 events
                    best_event.market_impact_score = -total_impact if best_event.market_impact_score < 0 else total_impact
                    
                    # Increase volatility for multiple events
                    best_event.volatility_multiplier = min(best_event.volatility_multiplier * 1.2, 5.0)
                    
                    logger.info(f"📊 Aggregated {len(recent_events)} events for {group_key}: impact={best_event.market_impact_score:.2f}")
            
            processed_events.append(best_event)
        
        # Sort by impact severity
        processed_events.sort(key=lambda x: (x.severity.value, abs(x.market_impact_score)), reverse=True)
        
        return processed_events
    
    async def calculate_global_volatility_score(self, events: Optional[List[GeopoliticalEvent]] = None) -> Dict[str, Any]:
        """Calculate global market volatility score based on active geopolitical events"""
        
        if events is None:
            events = await self.collect_geopolitical_events(hours_back=48)
        
        if not events:
            return {
                'global_volatility_score': 20.0,  # Base volatility (20%)
                'risk_level': 'low',
                'active_conflicts': 0,
                'major_events': [],
                'total_market_impact': 0.0
            }
        
        # Calculate composite volatility score
        base_volatility = 20.0  # 20% base market volatility
        conflict_volatility = 0.0
        total_market_impact = 0.0
        major_events = []
        
        for event in events:
            # Weight recent events more heavily
            time_weight = max(0.1, 1.0 - (datetime.now(timezone.utc) - event.last_update).total_seconds() / 86400)
            
            # Add to volatility based on severity and recency
            event_volatility = event.volatility_multiplier * time_weight * 10  # Convert to percentage points
            conflict_volatility += event_volatility
            
            # Accumulate market impact
            total_market_impact += event.market_impact_score * time_weight
            
            # Track major events
            if event.severity.value >= 3:  # HIGH or CRITICAL
                major_events.append({
                    'title': event.title,
                    'countries': event.countries_involved,
                    'severity': event.severity.name,
                    'impact': event.market_impact_score,
                    'hours_ago': (datetime.now(timezone.utc) - event.last_update).total_seconds() / 3600
                })
        
        # Calculate final global volatility score
        global_volatility = min(base_volatility + conflict_volatility, 100.0)  # Cap at 100%
        
        # Determine risk level
        if global_volatility < 30:
            risk_level = 'low'
        elif global_volatility < 50:
            risk_level = 'moderate'
        elif global_volatility < 75:
            risk_level = 'high'
        else:
            risk_level = 'critical'
        
        active_conflicts = sum(1 for e in events if e.event_type == EventType.ARMED_CONFLICT)
        
        logger.info(f"🌍 Global volatility assessment: {global_volatility:.1f}% ({risk_level}) - {active_conflicts} active conflicts")
        
        return {
            'global_volatility_score': global_volatility,
            'risk_level': risk_level,
            'active_conflicts': active_conflicts,
            'major_events': major_events[:5],  # Top 5 events
            'total_market_impact': total_market_impact,
            'base_volatility': base_volatility,
            'conflict_added_volatility': conflict_volatility,
            'assessment_timestamp': datetime.now(timezone.utc).isoformat()
        }

# Test the geopolitical monitor
async def main():
    """Test the geopolitical events monitoring system"""
    
    async with GeopoliticalEventsMonitor() as monitor:
        logger.info("🧪 Testing Geopolitical Events Monitor...")
        
        # Test event collection
        events = await monitor.collect_geopolitical_events(hours_back=24)
        logger.info(f"📰 Found {len(events)} geopolitical events")
        
        for event in events[:3]:
            logger.info(f"  • {event.title[:80]}... (Impact: {event.market_impact_score:+.2f})")
        
        # Test volatility scoring
        volatility_assessment = await monitor.calculate_global_volatility_score(events)
        logger.info(f"🌍 Global volatility: {volatility_assessment['global_volatility_score']:.1f}% ({volatility_assessment['risk_level']})")
        logger.info(f"📊 Active conflicts: {volatility_assessment['active_conflicts']}")
        
        if volatility_assessment['major_events']:
            logger.info("🚨 Major events:")
            for event in volatility_assessment['major_events']:
                logger.info(f"  • {event['title'][:60]}... ({event['severity']})")

if __name__ == "__main__":
    asyncio.run(main())