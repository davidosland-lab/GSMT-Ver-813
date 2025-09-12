"""
News Intelligence & Global Affairs Monitoring System
Integrates real-world volatility through news sentiment and global events analysis
"""

import asyncio
import aiohttp
import json
import re
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import hashlib

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NewsImpactLevel(Enum):
    """News impact levels on market volatility"""
    MINIMAL = "minimal"          # 0-2% expected volatility
    LOW = "low"                 # 2-5% expected volatility  
    MODERATE = "moderate"       # 5-10% expected volatility
    HIGH = "high"              # 10-20% expected volatility
    EXTREME = "extreme"        # 20%+ expected volatility

class NewsSentiment(Enum):
    """News sentiment classifications"""
    VERY_NEGATIVE = "very_negative"  # -1.0 to -0.6
    NEGATIVE = "negative"            # -0.6 to -0.2
    NEUTRAL = "neutral"              # -0.2 to 0.2
    POSITIVE = "positive"            # 0.2 to 0.6
    VERY_POSITIVE = "very_positive"  # 0.6 to 1.0

class NewsCategory(Enum):
    """Categories of news affecting market volatility"""
    GEOPOLITICAL = "geopolitical"
    ECONOMIC = "economic"
    MONETARY_POLICY = "monetary_policy"
    TRADE_WAR = "trade_war"
    NATURAL_DISASTER = "natural_disaster"
    PANDEMIC = "pandemic"
    TECHNOLOGY = "technology"
    ENERGY = "energy"
    COMMODITIES = "commodities"
    CORPORATE = "corporate"
    REGULATORY = "regulatory"
    SOCIAL_UNREST = "social_unrest"

class GeopoliticalRegion(Enum):
    """Key geopolitical regions affecting Australian markets"""
    AUSTRALIA = "australia"
    CHINA = "china"
    USA = "usa"
    EUROPE = "europe"
    ASIA_PACIFIC = "asia_pacific"
    MIDDLE_EAST = "middle_east"
    GLOBAL = "global"

@dataclass
class NewsArticle:
    """Individual news article with analysis"""
    title: str
    content: str
    source: str
    published_at: datetime
    url: str
    sentiment_score: float = 0.0
    sentiment: NewsSentiment = NewsSentiment.NEUTRAL
    impact_level: NewsImpactLevel = NewsImpactLevel.MINIMAL
    categories: List[NewsCategory] = None
    regions: List[GeopoliticalRegion] = None
    market_relevance: float = 0.0  # 0-1 relevance to Australian markets
    volatility_score: float = 0.0  # Expected volatility contribution
    key_entities: List[str] = None
    analysis_summary: str = ""

    def __post_init__(self):
        if self.categories is None:
            self.categories = []
        if self.regions is None:
            self.regions = []
        if self.key_entities is None:
            self.key_entities = []

@dataclass
class GlobalEvent:
    """Major global events with market impact analysis"""
    event_id: str
    title: str
    description: str
    event_type: NewsCategory
    start_date: datetime
    end_date: Optional[datetime]
    affected_regions: List[GeopoliticalRegion]
    severity: NewsImpactLevel
    market_impact_score: float  # 0-100
    australian_relevance: float  # 0-1
    related_articles: List[str] = None  # Article URLs
    key_indicators: Dict[str, float] = None  # Market indicators affected
    
    def __post_init__(self):
        if self.related_articles is None:
            self.related_articles = []
        if self.key_indicators is None:
            self.key_indicators = {}

@dataclass
class VolatilityAssessment:
    """Comprehensive volatility assessment from news analysis"""
    overall_sentiment: float  # -1 to 1
    volatility_score: float   # 0-100
    impact_level: NewsImpactLevel
    confidence: float         # 0-1
    time_horizon: str         # short_term, medium_term, long_term
    key_drivers: List[str]
    risk_factors: List[str]
    opportunity_factors: List[str]
    geographic_focus: Dict[GeopoliticalRegion, float]
    category_breakdown: Dict[NewsCategory, float]
    recent_events_count: int
    trend_direction: str      # increasing, stable, decreasing

class NewsIntelligenceEngine:
    """Core engine for news intelligence and sentiment analysis"""
    
    def __init__(self):
        self.session = None
        self.cache = {}
        self.cache_ttl = 3600  # 1 hour cache
        
        # Australian market specific keywords
        self.australian_keywords = [
            'australia', 'australian', 'asx', 'rba', 'reserve bank australia',
            'aud', 'sydney', 'melbourne', 'iron ore', 'mining', 'resources',
            'commonwealth bank', 'westpac', 'anz', 'nab', 'bhp', 'rio tinto',
            'csl', 'commonwealth', 'telstra', 'woolworths', 'wesfarmers'
        ]
        
        # High-impact keywords for volatility detection
        self.volatility_keywords = {
            NewsImpactLevel.EXTREME: [
                'war', 'invasion', 'nuclear', 'terrorist', 'collapse', 'crash',
                'pandemic', 'lockdown', 'emergency', 'crisis', 'catastrophe'
            ],
            NewsImpactLevel.HIGH: [
                'inflation', 'recession', 'interest rate', 'trade war', 'sanctions',
                'default', 'bankruptcy', 'outbreak', 'strike', 'protest'
            ],
            NewsImpactLevel.MODERATE: [
                'gdp', 'unemployment', 'policy', 'regulation', 'merger',
                'acquisition', 'earnings', 'forecast', 'outlook'
            ],
            NewsImpactLevel.LOW: [
                'growth', 'investment', 'expansion', 'partnership', 'agreement'
            ]
        }
        
        # Sentiment indicators
        self.positive_indicators = [
            'growth', 'increase', 'rise', 'boost', 'surge', 'gain', 'profit',
            'success', 'strong', 'bullish', 'optimistic', 'recovery', 'expansion'
        ]
        
        self.negative_indicators = [
            'decline', 'fall', 'drop', 'crash', 'loss', 'weak', 'bearish',
            'pessimistic', 'recession', 'crisis', 'concern', 'risk', 'threat'
        ]

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={'User-Agent': 'News Intelligence Bot 1.0'}
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    def _generate_cache_key(self, *args) -> str:
        """Generate cache key from arguments"""
        key_string = "_".join(str(arg) for arg in args)
        return hashlib.md5(key_string.encode()).hexdigest()

    def _is_cache_valid(self, cache_entry: dict) -> bool:
        """Check if cache entry is still valid"""
        timestamp = cache_entry.get('timestamp', 0)
        return (datetime.now().timestamp() - timestamp) < self.cache_ttl

    async def fetch_news_feeds(self, 
                             sources: List[str] = None,
                             keywords: List[str] = None,
                             hours_back: int = 24) -> List[NewsArticle]:
        """
        Fetch news from multiple sources with Australian market focus
        """
        if sources is None:
            sources = ['reuters', 'bloomberg', 'afr', 'abc_news', 'smh', 'guardian']
        
        if keywords is None:
            keywords = self.australian_keywords + ['market', 'economy', 'finance']

        logger.info(f"🗞️ Fetching news from {len(sources)} sources for last {hours_back} hours")
        
        # Cache key
        cache_key = self._generate_cache_key('news_feeds', str(sources), str(keywords), hours_back)
        
        # Check cache
        if cache_key in self.cache and self._is_cache_valid(self.cache[cache_key]):
            logger.info("📦 Returning cached news data")
            return self.cache[cache_key]['data']

        articles = []
        
        # Simulate news fetching (in production, replace with real APIs)
        sample_articles = await self._generate_sample_news_articles(hours_back)
        
        for article_data in sample_articles:
            article = NewsArticle(**article_data)
            
            # Analyze each article
            await self._analyze_article(article)
            
            # Filter by relevance
            if article.market_relevance > 0.3:  # Only include relevant articles
                articles.append(article)
        
        # Sort by relevance and impact
        articles.sort(key=lambda x: (x.market_relevance * x.volatility_score), reverse=True)
        
        # Cache results
        self.cache[cache_key] = {
            'data': articles,
            'timestamp': datetime.now().timestamp()
        }
        
        logger.info(f"✅ Processed {len(articles)} relevant news articles")
        return articles

    async def _generate_sample_news_articles(self, hours_back: int) -> List[dict]:
        """Generate realistic sample news articles for testing"""
        
        base_time = datetime.now(timezone.utc) - timedelta(hours=hours_back)
        
        sample_articles = [
            {
                'title': 'RBA Holds Interest Rates Steady Amid Global Economic Uncertainty',
                'content': 'The Reserve Bank of Australia maintained the official cash rate at 4.35% following its monthly board meeting, citing ongoing inflationary pressures and global economic headwinds from China and Europe.',
                'source': 'Reuters',
                'published_at': base_time + timedelta(hours=2),
                'url': 'https://reuters.com/sample/rba-rates'
            },
            {
                'title': 'Iron Ore Prices Surge 5% on China Infrastructure Stimulus Announcement',
                'content': 'Iron ore futures jumped to $120/tonne after China announced a $500 billion infrastructure spending package, boosting demand outlook for Australian mining exports.',
                'source': 'Bloomberg',
                'published_at': base_time + timedelta(hours=6),
                'url': 'https://bloomberg.com/sample/iron-ore'
            },
            {
                'title': 'US-China Trade Tensions Escalate with New Tariff Threats',
                'content': 'Rising trade tensions between the US and China are creating uncertainty for global supply chains, with potential implications for Australian commodity exports and currency stability.',
                'source': 'AFR',
                'published_at': base_time + timedelta(hours=8),
                'url': 'https://afr.com/sample/trade-tensions'
            },
            {
                'title': 'Commonwealth Bank Reports Strong Q4 Earnings Beat Expectations',
                'content': 'CBA posted quarterly cash earnings of $2.7 billion, beating analyst forecasts driven by strong home lending growth and improved net interest margins.',
                'source': 'SMH',
                'published_at': base_time + timedelta(hours=12),
                'url': 'https://smh.com.au/sample/cba-earnings'
            },
            {
                'title': 'European Energy Crisis Deepens Amid Russian Gas Supply Cuts',
                'content': 'European natural gas prices soared 15% as Russia further reduced pipeline supplies, raising concerns about global energy security and inflation.',
                'source': 'Guardian',
                'published_at': base_time + timedelta(hours=14),
                'url': 'https://guardian.com/sample/energy-crisis'
            },
            {
                'title': 'Australian Dollar Weakens Against USD on Global Risk-Off Sentiment',
                'content': 'The AUD/USD pair fell 1.2% to 0.6650 as investors sought safe-haven assets amid concerns about slowing global growth and geopolitical tensions.',
                'source': 'ABC News',
                'published_at': base_time + timedelta(hours=18),
                'url': 'https://abc.net.au/sample/aud-weakens'
            },
            {
                'title': 'BHP Announces Major Copper Discovery in South Australia',
                'content': 'Mining giant BHP revealed a significant copper deposit discovery that could boost Australian mining output and support the green energy transition.',
                'source': 'Reuters',
                'published_at': base_time + timedelta(hours=20),
                'url': 'https://reuters.com/sample/bhp-copper'
            }
        ]
        
        return sample_articles

    async def _analyze_article(self, article: NewsArticle):
        """Comprehensive analysis of news article for market impact"""
        
        text = f"{article.title} {article.content}".lower()
        
        # 1. Sentiment Analysis
        article.sentiment_score = self._calculate_sentiment_score(text)
        article.sentiment = self._classify_sentiment(article.sentiment_score)
        
        # 2. Impact Level Assessment
        article.impact_level = self._assess_impact_level(text)
        
        # 3. Category Classification
        article.categories = self._classify_categories(text)
        
        # 4. Regional Relevance
        article.regions = self._identify_regions(text)
        
        # 5. Australian Market Relevance
        article.market_relevance = self._calculate_australian_relevance(text)
        
        # 6. Volatility Score
        article.volatility_score = self._calculate_volatility_score(article)
        
        # 7. Key Entity Extraction
        article.key_entities = self._extract_key_entities(text)
        
        # 8. Generate Analysis Summary
        article.analysis_summary = self._generate_analysis_summary(article)

    def _calculate_sentiment_score(self, text: str) -> float:
        """Calculate sentiment score from -1 to 1"""
        
        positive_count = sum(1 for word in self.positive_indicators if word in text)
        negative_count = sum(1 for word in self.negative_indicators if word in text)
        
        total_words = len(text.split())
        
        # Normalize by text length
        positive_ratio = positive_count / max(total_words, 1) * 100
        negative_ratio = negative_count / max(total_words, 1) * 100
        
        # Calculate sentiment score
        sentiment_score = (positive_ratio - negative_ratio) / 10
        
        # Clamp to [-1, 1]
        return max(-1.0, min(1.0, sentiment_score))

    def _classify_sentiment(self, score: float) -> NewsSentiment:
        """Classify sentiment based on score"""
        if score >= 0.6:
            return NewsSentiment.VERY_POSITIVE
        elif score >= 0.2:
            return NewsSentiment.POSITIVE
        elif score >= -0.2:
            return NewsSentiment.NEUTRAL
        elif score >= -0.6:
            return NewsSentiment.NEGATIVE
        else:
            return NewsSentiment.VERY_NEGATIVE

    def _assess_impact_level(self, text: str) -> NewsImpactLevel:
        """Assess potential market impact level"""
        
        for level, keywords in self.volatility_keywords.items():
            if any(keyword in text for keyword in keywords):
                return level
        
        return NewsImpactLevel.MINIMAL

    def _classify_categories(self, text: str) -> List[NewsCategory]:
        """Classify news into relevant categories"""
        
        category_keywords = {
            NewsCategory.GEOPOLITICAL: ['war', 'conflict', 'diplomacy', 'sanctions', 'tension'],
            NewsCategory.ECONOMIC: ['gdp', 'growth', 'recession', 'inflation', 'economy'],
            NewsCategory.MONETARY_POLICY: ['interest rate', 'rba', 'fed', 'central bank', 'monetary'],
            NewsCategory.TRADE_WAR: ['trade war', 'tariff', 'export', 'import', 'trade deal'],
            NewsCategory.NATURAL_DISASTER: ['earthquake', 'flood', 'hurricane', 'disaster', 'climate'],
            NewsCategory.PANDEMIC: ['pandemic', 'virus', 'covid', 'health crisis', 'lockdown'],
            NewsCategory.TECHNOLOGY: ['tech', 'ai', 'digital', 'innovation', 'cyber'],
            NewsCategory.ENERGY: ['oil', 'gas', 'energy', 'renewable', 'electricity'],
            NewsCategory.COMMODITIES: ['iron ore', 'gold', 'copper', 'coal', 'mining'],
            NewsCategory.CORPORATE: ['earnings', 'merger', 'acquisition', 'bankruptcy', 'ipo'],
            NewsCategory.REGULATORY: ['regulation', 'policy', 'law', 'compliance', 'government'],
            NewsCategory.SOCIAL_UNREST: ['protest', 'strike', 'unrest', 'riot', 'demonstration']
        }
        
        categories = []
        for category, keywords in category_keywords.items():
            if any(keyword in text for keyword in keywords):
                categories.append(category)
        
        return categories

    def _identify_regions(self, text: str) -> List[GeopoliticalRegion]:
        """Identify relevant geopolitical regions"""
        
        region_keywords = {
            GeopoliticalRegion.AUSTRALIA: ['australia', 'australian', 'sydney', 'melbourne', 'asx'],
            GeopoliticalRegion.CHINA: ['china', 'chinese', 'beijing', 'shanghai', 'yuan'],
            GeopoliticalRegion.USA: ['usa', 'america', 'us', 'dollar', 'fed', 'wall street'],
            GeopoliticalRegion.EUROPE: ['europe', 'eu', 'euro', 'ecb', 'germany', 'france'],
            GeopoliticalRegion.ASIA_PACIFIC: ['japan', 'south korea', 'india', 'asia', 'pacific'],
            GeopoliticalRegion.MIDDLE_EAST: ['saudi', 'iran', 'israel', 'middle east', 'oil'],
            GeopoliticalRegion.GLOBAL: ['global', 'worldwide', 'international', 'world']
        }
        
        regions = []
        for region, keywords in region_keywords.items():
            if any(keyword in text for keyword in keywords):
                regions.append(region)
        
        return regions

    def _calculate_australian_relevance(self, text: str) -> float:
        """Calculate relevance to Australian markets (0-1)"""
        
        relevance_score = 0.0
        
        # Direct Australian mentions
        australian_mentions = sum(1 for keyword in self.australian_keywords if keyword in text)
        relevance_score += min(australian_mentions * 0.2, 0.6)
        
        # China mentions (high relevance due to trade relationship)
        china_mentions = sum(1 for word in ['china', 'chinese', 'yuan'] if word in text)
        relevance_score += min(china_mentions * 0.1, 0.3)
        
        # Commodities mentions (iron ore, gold, mining)
        commodity_mentions = sum(1 for word in ['iron ore', 'mining', 'gold', 'copper', 'coal'] if word in text)
        relevance_score += min(commodity_mentions * 0.15, 0.4)
        
        # Global economic mentions
        global_mentions = sum(1 for word in ['global', 'world', 'international'] if word in text)
        relevance_score += min(global_mentions * 0.05, 0.2)
        
        return min(relevance_score, 1.0)

    def _calculate_volatility_score(self, article: NewsArticle) -> float:
        """Calculate expected volatility contribution (0-100)"""
        
        base_score = 0.0
        
        # Impact level contribution
        impact_multipliers = {
            NewsImpactLevel.MINIMAL: 1.0,
            NewsImpactLevel.LOW: 2.0,
            NewsImpactLevel.MODERATE: 5.0,
            NewsImpactLevel.HIGH: 10.0,
            NewsImpactLevel.EXTREME: 20.0
        }
        
        base_score += impact_multipliers.get(article.impact_level, 1.0)
        
        # Sentiment extremity (high positive or negative increases volatility)
        sentiment_volatility = abs(article.sentiment_score) * 10
        base_score += sentiment_volatility
        
        # Market relevance multiplier
        base_score *= article.market_relevance
        
        # Category-specific multipliers
        category_multipliers = {
            NewsCategory.GEOPOLITICAL: 1.5,
            NewsCategory.ECONOMIC: 1.3,
            NewsCategory.MONETARY_POLICY: 1.8,
            NewsCategory.TRADE_WAR: 1.6,
            NewsCategory.PANDEMIC: 2.0,
            NewsCategory.ENERGY: 1.4,
            NewsCategory.COMMODITIES: 1.7
        }
        
        for category in article.categories:
            multiplier = category_multipliers.get(category, 1.0)
            base_score *= multiplier
            break  # Apply highest multiplier once
        
        return min(base_score, 100.0)

    def _extract_key_entities(self, text: str) -> List[str]:
        """Extract key entities from text"""
        
        entities = []
        
        # Extract company names (simple pattern matching)
        companies = ['commonwealth bank', 'cba', 'westpac', 'anz', 'nab', 'bhp', 'rio tinto', 
                    'csl', 'telstra', 'woolworths', 'wesfarmers', 'qantas', 'santos']
        
        for company in companies:
            if company in text:
                entities.append(company.title())
        
        # Extract countries/regions
        regions = ['australia', 'china', 'usa', 'america', 'europe', 'japan', 'india']
        for region in regions:
            if region in text:
                entities.append(region.title())
        
        # Extract commodities
        commodities = ['iron ore', 'gold', 'copper', 'coal', 'oil', 'gas']
        for commodity in commodities:
            if commodity in text:
                entities.append(commodity.title())
        
        return list(set(entities))  # Remove duplicates

    def _generate_analysis_summary(self, article: NewsArticle) -> str:
        """Generate human-readable analysis summary"""
        
        sentiment_desc = {
            NewsSentiment.VERY_POSITIVE: "very positive",
            NewsSentiment.POSITIVE: "positive", 
            NewsSentiment.NEUTRAL: "neutral",
            NewsSentiment.NEGATIVE: "negative",
            NewsSentiment.VERY_NEGATIVE: "very negative"
        }
        
        impact_desc = {
            NewsImpactLevel.MINIMAL: "minimal market impact",
            NewsImpactLevel.LOW: "low market impact",
            NewsImpactLevel.MODERATE: "moderate market impact", 
            NewsImpactLevel.HIGH: "high market impact",
            NewsImpactLevel.EXTREME: "extreme market impact"
        }
        
        summary = f"News sentiment: {sentiment_desc[article.sentiment]} ({article.sentiment_score:.2f}). "
        summary += f"Expected {impact_desc[article.impact_level]} "
        summary += f"with {article.market_relevance:.0%} relevance to Australian markets. "
        
        if article.categories:
            categories_str = ", ".join([cat.value.replace('_', ' ') for cat in article.categories[:2]])
            summary += f"Categories: {categories_str}. "
        
        if article.key_entities:
            entities_str = ", ".join(article.key_entities[:3])
            summary += f"Key entities: {entities_str}."
        
        return summary

    async def analyze_global_events(self, time_window_hours: int = 168) -> List[GlobalEvent]:
        """Analyze major global events affecting markets"""
        
        logger.info(f"🌍 Analyzing global events for last {time_window_hours} hours")
        
        # Fetch recent news
        articles = await self.fetch_news_feeds(hours_back=time_window_hours)
        
        # Group articles by event themes
        events = self._identify_major_events(articles)
        
        logger.info(f"✅ Identified {len(events)} major global events")
        return events

    def _identify_major_events(self, articles: List[NewsArticle]) -> List[GlobalEvent]:
        """Identify major events from news articles"""
        
        # Group articles by similar themes/entities
        event_groups = defaultdict(list)
        
        for article in articles:
            # Create event key based on entities and categories
            key_elements = []
            key_elements.extend([entity.lower() for entity in article.key_entities[:2]])
            key_elements.extend([cat.value for cat in article.categories[:1]])
            
            event_key = "_".join(sorted(key_elements))
            if event_key:
                event_groups[event_key].append(article)
        
        # Convert groups to events
        events = []
        for event_key, group_articles in event_groups.items():
            if len(group_articles) >= 2:  # Require multiple articles for major event
                event = self._create_global_event(event_key, group_articles)
                events.append(event)
        
        # Sort by impact score
        events.sort(key=lambda x: x.market_impact_score, reverse=True)
        
        return events[:10]  # Return top 10 events

    def _create_global_event(self, event_key: str, articles: List[NewsArticle]) -> GlobalEvent:
        """Create GlobalEvent from grouped articles"""
        
        # Calculate aggregate metrics
        avg_impact = np.mean([article.volatility_score for article in articles])
        avg_relevance = np.mean([article.market_relevance for article in articles])
        
        # Determine event details
        all_categories = [cat for article in articles for cat in article.categories]
        most_common_category = Counter(all_categories).most_common(1)[0][0] if all_categories else NewsCategory.ECONOMIC
        
        all_regions = [region for article in articles for region in article.regions]
        affected_regions = list(set(all_regions))
        
        # Determine severity
        max_impact_level = max([article.impact_level for article in articles], 
                              key=lambda x: list(NewsImpactLevel).index(x))
        
        # Create event
        event = GlobalEvent(
            event_id=hashlib.md5(event_key.encode()).hexdigest()[:8],
            title=self._generate_event_title(articles),
            description=self._generate_event_description(articles),
            event_type=most_common_category,
            start_date=min([article.published_at for article in articles]),
            end_date=None,  # Ongoing events
            affected_regions=affected_regions,
            severity=max_impact_level,
            market_impact_score=min(avg_impact * 2, 100.0),  # Scale to 0-100
            australian_relevance=avg_relevance,
            related_articles=[article.url for article in articles]
        )
        
        return event

    def _generate_event_title(self, articles: List[NewsArticle]) -> str:
        """Generate event title from articles"""
        
        # Extract common themes from titles
        all_words = []
        for article in articles:
            words = article.title.lower().split()
            all_words.extend([word for word in words if len(word) > 3])
        
        # Find most common significant words
        word_counts = Counter(all_words)
        common_words = [word for word, count in word_counts.most_common(3) if count > 1]
        
        if common_words:
            return " ".join(word.title() for word in common_words[:3]) + " Event"
        else:
            return f"Market Event ({articles[0].published_at.strftime('%Y-%m-%d')})"

    def _generate_event_description(self, articles: List[NewsArticle]) -> str:
        """Generate event description from articles"""
        
        # Combine key information
        entities = set()
        categories = set()
        
        for article in articles:
            entities.update(article.key_entities)
            categories.update([cat.value.replace('_', ' ') for cat in article.categories])
        
        description = f"Multi-faceted event involving {', '.join(list(entities)[:3])} "
        description += f"across {', '.join(list(categories)[:2])} sectors. "
        description += f"Based on {len(articles)} news reports over "
        
        time_span = max([article.published_at for article in articles]) - min([article.published_at for article in articles])
        description += f"{time_span.days} days."
        
        return description

    async def generate_volatility_assessment(self, 
                                           articles: List[NewsArticle] = None,
                                           events: List[GlobalEvent] = None,
                                           time_horizon: str = "medium_term") -> VolatilityAssessment:
        """Generate comprehensive volatility assessment"""
        
        if articles is None:
            articles = await self.fetch_news_feeds(hours_back=48)
        
        if events is None:
            events = await self.analyze_global_events(time_window_hours=168)
        
        logger.info(f"📊 Generating volatility assessment from {len(articles)} articles and {len(events)} events")
        
        # Calculate overall metrics
        overall_sentiment = np.mean([article.sentiment_score for article in articles]) if articles else 0.0
        
        volatility_scores = [article.volatility_score for article in articles] + \
                          [event.market_impact_score * 0.5 for event in events]  # Scale event scores
        
        volatility_score = np.mean(volatility_scores) if volatility_scores else 0.0
        
        # Determine impact level
        if volatility_score >= 60:
            impact_level = NewsImpactLevel.EXTREME
        elif volatility_score >= 40:
            impact_level = NewsImpactLevel.HIGH
        elif volatility_score >= 20:
            impact_level = NewsImpactLevel.MODERATE
        elif volatility_score >= 10:
            impact_level = NewsImpactLevel.LOW
        else:
            impact_level = NewsImpactLevel.MINIMAL
        
        # Calculate confidence based on data quality
        confidence = min(len(articles) / 20.0, 1.0) * 0.7 + min(len(events) / 5.0, 1.0) * 0.3
        
        # Identify key drivers
        key_drivers = self._identify_key_volatility_drivers(articles, events)
        
        # Risk and opportunity factors
        risk_factors, opportunity_factors = self._analyze_risk_opportunities(articles, events)
        
        # Geographic and categorical breakdown
        geographic_focus = self._calculate_geographic_focus(articles, events)
        category_breakdown = self._calculate_category_breakdown(articles, events)
        
        # Trend analysis
        trend_direction = self._analyze_trend_direction(articles)
        
        assessment = VolatilityAssessment(
            overall_sentiment=overall_sentiment,
            volatility_score=volatility_score,
            impact_level=impact_level,
            confidence=confidence,
            time_horizon=time_horizon,
            key_drivers=key_drivers,
            risk_factors=risk_factors,
            opportunity_factors=opportunity_factors,
            geographic_focus=geographic_focus,
            category_breakdown=category_breakdown,
            recent_events_count=len(articles) + len(events),
            trend_direction=trend_direction
        )
        
        return assessment

    def _identify_key_volatility_drivers(self, 
                                       articles: List[NewsArticle], 
                                       events: List[GlobalEvent]) -> List[str]:
        """Identify main drivers of market volatility"""
        
        drivers = []
        
        # From articles
        high_impact_articles = [a for a in articles if a.volatility_score > 30]
        for article in high_impact_articles[:3]:
            if article.key_entities:
                drivers.append(f"{article.key_entities[0]} - {article.impact_level.value}")
        
        # From events  
        high_impact_events = [e for e in events if e.market_impact_score > 40]
        for event in high_impact_events[:3]:
            drivers.append(f"{event.title} - {event.severity.value} impact")
        
        return drivers[:5]  # Top 5 drivers

    def _analyze_risk_opportunities(self, 
                                  articles: List[NewsArticle], 
                                  events: List[GlobalEvent]) -> Tuple[List[str], List[str]]:
        """Analyze risk factors and opportunities"""
        
        risks = []
        opportunities = []
        
        # Analyze sentiment and categories
        negative_articles = [a for a in articles if a.sentiment_score < -0.2]
        positive_articles = [a for a in articles if a.sentiment_score > 0.2]
        
        # Risk factors from negative sentiment
        for article in negative_articles[:3]:
            if article.categories:
                category_name = article.categories[0].value.replace('_', ' ')
                risks.append(f"{category_name.title()} concerns")
        
        # Opportunities from positive sentiment  
        for article in positive_articles[:3]:
            if article.categories:
                category_name = article.categories[0].value.replace('_', ' ')
                opportunities.append(f"{category_name.title()} growth potential")
        
        # Add event-based risks
        high_risk_events = [e for e in events if e.severity in [NewsImpactLevel.HIGH, NewsImpactLevel.EXTREME]]
        for event in high_risk_events[:2]:
            risks.append(f"{event.event_type.value.replace('_', ' ').title()} volatility")
        
        return risks[:5], opportunities[:3]

    def _calculate_geographic_focus(self, 
                                  articles: List[NewsArticle], 
                                  events: List[GlobalEvent]) -> Dict[GeopoliticalRegion, float]:
        """Calculate geographic distribution of market influence"""
        
        region_scores = defaultdict(float)
        total_weight = 0
        
        # Weight by article relevance and volatility
        for article in articles:
            weight = article.market_relevance * article.volatility_score
            total_weight += weight
            
            for region in article.regions:
                region_scores[region] += weight
        
        # Weight by event impact
        for event in events:
            weight = event.australian_relevance * event.market_impact_score
            total_weight += weight
            
            for region in event.affected_regions:
                region_scores[region] += weight
        
        # Normalize
        if total_weight > 0:
            return {region: score / total_weight for region, score in region_scores.items()}
        else:
            return {}

    def _calculate_category_breakdown(self, 
                                    articles: List[NewsArticle], 
                                    events: List[GlobalEvent]) -> Dict[NewsCategory, float]:
        """Calculate category distribution of market influence"""
        
        category_scores = defaultdict(float)
        total_weight = 0
        
        # Weight by article impact
        for article in articles:
            weight = article.volatility_score
            total_weight += weight
            
            for category in article.categories:
                category_scores[category] += weight
        
        # Weight by event impact
        for event in events:
            weight = event.market_impact_score
            total_weight += weight
            category_scores[event.event_type] += weight
        
        # Normalize
        if total_weight > 0:
            return {category: score / total_weight for category, score in category_scores.items()}
        else:
            return {}

    def _analyze_trend_direction(self, articles: List[NewsArticle]) -> str:
        """Analyze if volatility is increasing, stable, or decreasing"""
        
        if len(articles) < 4:
            return "stable"
        
        # Sort articles by time
        sorted_articles = sorted(articles, key=lambda x: x.published_at)
        
        # Split into early and late periods
        mid_point = len(sorted_articles) // 2
        early_period = sorted_articles[:mid_point]
        late_period = sorted_articles[mid_point:]
        
        # Calculate average volatility for each period
        early_volatility = np.mean([a.volatility_score for a in early_period])
        late_volatility = np.mean([a.volatility_score for a in late_period])
        
        # Determine trend
        change = late_volatility - early_volatility
        
        if change > 5:
            return "increasing"
        elif change < -5:
            return "decreasing" 
        else:
            return "stable"

# Initialize global news intelligence service
news_intelligence_service = NewsIntelligenceEngine()

# Export classes for integration
__all__ = [
    'NewsIntelligenceEngine',
    'NewsArticle',
    'GlobalEvent', 
    'VolatilityAssessment',
    'NewsImpactLevel',
    'NewsSentiment',
    'NewsCategory',
    'GeopoliticalRegion',
    'news_intelligence_service'
]