#!/usr/bin/env python3
"""
Live News Integration System - Real-World News Feeds
Replaces simulated data with actual news sources for research purposes
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
from urllib.parse import urljoin, urlparse
import hashlib
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LiveNewsCollector:
    """Collects real news from multiple free sources for market analysis"""
    
    def __init__(self):
        self.session = None
        self.cache = {}
        self.cache_ttl = 1800  # 30 minutes cache for live data
        
        # Free news sources (no API key required)
        self.news_sources = {
            'reuters_rss': {
                'url': 'https://feeds.reuters.com/reuters/businessNews',
                'type': 'rss',
                'relevance': 0.9
            },
            'bbc_business_rss': {
                'url': 'http://feeds.bbci.co.uk/news/business/rss.xml',
                'type': 'rss',
                'relevance': 0.8
            },
            'reuters_markets_rss': {
                'url': 'https://feeds.reuters.com/news/wealth',
                'type': 'rss',
                'relevance': 0.9
            },
            'guardian_business_rss': {
                'url': 'https://www.theguardian.com/business/rss',
                'type': 'rss',
                'relevance': 0.7
            },
            'abc_news_business_rss': {
                'url': 'https://www.abc.net.au/news/feed/business/rss.xml',
                'type': 'rss',
                'relevance': 0.95  # High relevance for Australian news
            },
            'smh_business_rss': {
                'url': 'https://www.smh.com.au/rss/business.xml',
                'type': 'rss',
                'relevance': 0.95  # High relevance for Australian news
            }
        }
        
        # Australian market keywords for filtering
        self.australian_keywords = [
            'australia', 'australian', 'asx', 'rba', 'reserve bank australia',
            'aud', 'sydney', 'melbourne', 'iron ore', 'mining', 'resources',
            'commonwealth bank', 'westpac', 'anz', 'nab', 'bhp', 'rio tinto',
            'csl', 'telstra', 'woolworths', 'wesfarmers', 'fortescue',
            'all ordinaries', 'aord', 'china trade', 'commodity'
        ]
        
        # Market impact keywords
        self.market_impact_keywords = {
            'high': [
                'crash', 'collapse', 'crisis', 'recession', 'inflation', 'interest rate',
                'trade war', 'sanctions', 'pandemic', 'lockdown', 'war', 'invasion',
                'central bank', 'policy', 'gdp', 'unemployment'
            ],
            'medium': [
                'earnings', 'profit', 'revenue', 'merger', 'acquisition', 'ipo',
                'dividend', 'forecast', 'outlook', 'guidance', 'regulation'
            ],
            'low': [
                'partnership', 'agreement', 'expansion', 'investment', 'launch',
                'appointment', 'resignation', 'contract', 'deal'
            ]
        }
        
        # Sentiment indicators
        self.sentiment_indicators = {
            'positive': [
                'growth', 'increase', 'rise', 'boost', 'surge', 'gain', 'profit',
                'success', 'strong', 'bullish', 'optimistic', 'recovery', 'expansion',
                'beat', 'exceed', 'outperform', 'breakthrough', 'milestone'
            ],
            'negative': [
                'decline', 'fall', 'drop', 'crash', 'loss', 'weak', 'bearish',
                'pessimistic', 'recession', 'crisis', 'concern', 'risk', 'threat',
                'miss', 'disappoint', 'underperform', 'warning', 'alert'
            ]
        }
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
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
        return (time.time() - timestamp) < self.cache_ttl
    
    async def collect_live_news(self, hours_back: int = 24, max_articles: int = 50) -> List[Dict]:
        """Collect real news articles from live sources"""
        
        logger.info(f"🌐 Collecting LIVE news from {len(self.news_sources)} sources")
        
        # Check cache
        cache_key = self._generate_cache_key('live_news', hours_back, max_articles)
        if cache_key in self.cache and self._is_cache_valid(self.cache[cache_key]):
            logger.info("📦 Returning cached live news data")
            return self.cache[cache_key]['data']
        
        all_articles = []
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours_back)
        
        for source_name, source_config in self.news_sources.items():
            try:
                logger.info(f"📡 Fetching from {source_name}")
                articles = await self._fetch_rss_feed(source_config, cutoff_time)
                
                # Add source relevance
                for article in articles:
                    article['source_relevance'] = source_config['relevance']
                    article['source_name'] = source_name
                
                all_articles.extend(articles)
                
                # Small delay to be respectful to servers
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.warning(f"Failed to fetch from {source_name}: {e}")
        
        # Process and filter articles
        processed_articles = []
        
        for article in all_articles:
            try:
                processed = await self._process_article(article)
                if processed and self._is_relevant_to_australian_markets(processed):
                    processed_articles.append(processed)
            except Exception as e:
                logger.warning(f"Failed to process article: {e}")
        
        # Sort by relevance and recency
        processed_articles.sort(
            key=lambda x: (x.get('market_relevance', 0) * x.get('source_relevance', 0), x.get('published_at', datetime.min)),
            reverse=True
        )
        
        # Limit results
        final_articles = processed_articles[:max_articles]
        
        # Cache results
        self.cache[cache_key] = {
            'data': final_articles,
            'timestamp': time.time()
        }
        
        logger.info(f"✅ Collected {len(final_articles)} relevant live articles")
        return final_articles
    
    async def _fetch_rss_feed(self, source_config: Dict, cutoff_time: datetime) -> List[Dict]:
        """Fetch articles from RSS feed"""
        
        articles = []
        
        try:
            async with self.session.get(source_config['url']) as response:
                if response.status == 200:
                    content = await response.text()
                    
                    # Parse RSS/XML
                    try:
                        root = ET.fromstring(content)
                        
                        # Handle different RSS formats
                        items = root.findall('.//item') or root.findall('.//{http://www.w3.org/2005/Atom}entry')
                        
                        for item in items:
                            try:
                                article = self._parse_rss_item(item)
                                
                                # Filter by time
                                if article and article.get('published_at', datetime.min) > cutoff_time:
                                    articles.append(article)
                                    
                            except Exception as e:
                                logger.debug(f"Failed to parse RSS item: {e}")
                                
                    except ET.ParseError as e:
                        logger.warning(f"Failed to parse RSS XML: {e}")
                        
        except Exception as e:
            logger.warning(f"Failed to fetch RSS feed {source_config['url']}: {e}")
        
        return articles
    
    def _parse_rss_item(self, item) -> Optional[Dict]:
        """Parse individual RSS item"""
        
        try:
            # Extract basic fields
            title = self._get_element_text(item, ['title'])
            description = self._get_element_text(item, ['description', 'summary', '{http://www.w3.org/2005/Atom}summary'])
            link = self._get_element_text(item, ['link', 'guid', '{http://www.w3.org/2005/Atom}id'])
            pub_date = self._get_element_text(item, ['pubDate', 'published', '{http://www.w3.org/2005/Atom}published'])
            
            if not title or not description:
                return None
            
            # Parse publication date
            published_at = self._parse_date(pub_date) if pub_date else datetime.now(timezone.utc)
            
            return {
                'title': title.strip(),
                'content': self._clean_html(description),
                'url': link or '',
                'published_at': published_at,
                'raw_pub_date': pub_date
            }
            
        except Exception as e:
            logger.debug(f"Error parsing RSS item: {e}")
            return None
    
    def _get_element_text(self, item, tag_names: List[str]) -> Optional[str]:
        """Get text from element with multiple possible tag names"""
        
        for tag_name in tag_names:
            element = item.find(tag_name)
            if element is not None:
                if element.text:
                    return element.text
                # For links, might be in href attribute
                if 'link' in tag_name.lower() and element.get('href'):
                    return element.get('href')
        return None
    
    def _parse_date(self, date_str: str) -> datetime:
        """Parse various date formats"""
        
        if not date_str:
            return datetime.now(timezone.utc)
        
        # Common RSS date formats
        date_formats = [
            '%a, %d %b %Y %H:%M:%S %z',  # RFC 2822
            '%a, %d %b %Y %H:%M:%S %Z',
            '%Y-%m-%dT%H:%M:%S%z',       # ISO 8601
            '%Y-%m-%dT%H:%M:%SZ',
            '%Y-%m-%d %H:%M:%S',
            '%d %b %Y %H:%M:%S',
        ]
        
        for fmt in date_formats:
            try:
                return datetime.strptime(date_str.strip(), fmt)
            except ValueError:
                continue
        
        # If all else fails, return current time
        logger.debug(f"Could not parse date: {date_str}")
        return datetime.now(timezone.utc)
    
    def _clean_html(self, text: str) -> str:
        """Remove HTML tags and clean up text"""
        
        if not text:
            return ""
        
        # Use BeautifulSoup if available, otherwise regex
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(text, 'html.parser')
            cleaned = soup.get_text()
        except:
            # Fallback to regex
            cleaned = re.sub(r'<[^>]+>', '', text)
        
        # Clean up whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned
    
    async def _process_article(self, article: Dict) -> Optional[Dict]:
        """Process and analyze article for market relevance"""
        
        try:
            text = f"{article.get('title', '')} {article.get('content', '')}".lower()
            
            # Calculate market relevance
            market_relevance = self._calculate_market_relevance(text)
            
            # Calculate sentiment
            sentiment_score = self._calculate_sentiment(text)
            
            # Assess market impact
            impact_level = self._assess_market_impact(text)
            
            # Extract key topics
            topics = self._extract_topics(text)
            
            # Enhanced article data
            processed = {
                **article,
                'market_relevance': market_relevance,
                'sentiment_score': sentiment_score,
                'impact_level': impact_level,
                'topics': topics,
                'australian_relevance': self._calculate_australian_relevance(text),
                'analysis_timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            return processed
            
        except Exception as e:
            logger.debug(f"Error processing article: {e}")
            return None
    
    def _calculate_market_relevance(self, text: str) -> float:
        """Calculate how relevant the article is to financial markets"""
        
        financial_keywords = [
            'market', 'stock', 'share', 'trading', 'investment', 'finance',
            'economy', 'economic', 'bank', 'central bank', 'interest rate',
            'inflation', 'gdp', 'earnings', 'revenue', 'profit', 'dividend',
            'commodity', 'currency', 'exchange', 'bond', 'yield'
        ]
        
        matches = sum(1 for keyword in financial_keywords if keyword in text)
        total_words = len(text.split())
        
        # Normalize by text length and boost for multiple matches
        relevance = min((matches / max(total_words, 1)) * 100, 1.0)
        
        # Boost if multiple financial keywords found
        if matches >= 3:
            relevance = min(relevance * 1.5, 1.0)
        
        return relevance
    
    def _calculate_australian_relevance(self, text: str) -> float:
        """Calculate relevance to Australian markets"""
        
        matches = sum(1 for keyword in self.australian_keywords if keyword in text)
        
        # Direct Australian mentions get high scores
        if any(keyword in text for keyword in ['australia', 'australian', 'asx', 'rba']):
            base_score = 0.8
        else:
            base_score = 0.3
        
        # Boost for additional Australian-specific terms
        boost = min(matches * 0.1, 0.5)
        
        return min(base_score + boost, 1.0)
    
    def _calculate_sentiment(self, text: str) -> float:
        """Calculate sentiment score from -1 to 1"""
        
        positive_count = sum(1 for word in self.sentiment_indicators['positive'] if word in text)
        negative_count = sum(1 for word in self.sentiment_indicators['negative'] if word in text)
        
        if positive_count == 0 and negative_count == 0:
            return 0.0
        
        total_sentiment = positive_count + negative_count
        sentiment_score = (positive_count - negative_count) / total_sentiment
        
        return max(-1.0, min(1.0, sentiment_score))
    
    def _assess_market_impact(self, text: str) -> str:
        """Assess potential market impact level"""
        
        high_impact = sum(1 for word in self.market_impact_keywords['high'] if word in text)
        medium_impact = sum(1 for word in self.market_impact_keywords['medium'] if word in text)
        low_impact = sum(1 for word in self.market_impact_keywords['low'] if word in text)
        
        if high_impact >= 2:
            return 'high'
        elif high_impact >= 1 or medium_impact >= 2:
            return 'medium'
        elif medium_impact >= 1 or low_impact >= 1:
            return 'low'
        else:
            return 'minimal'
    
    def _extract_topics(self, text: str) -> List[str]:
        """Extract key topics from the text"""
        
        topic_keywords = {
            'monetary_policy': ['interest rate', 'central bank', 'rba', 'fed', 'monetary policy'],
            'trade': ['trade', 'export', 'import', 'tariff', 'china trade'],
            'commodities': ['iron ore', 'gold', 'oil', 'gas', 'copper', 'coal'],
            'banking': ['bank', 'lending', 'mortgage', 'credit'],
            'mining': ['mining', 'bhp', 'rio tinto', 'fortescue'],
            'currency': ['aud', 'dollar', 'exchange rate', 'currency'],
            'geopolitical': ['war', 'sanctions', 'tension', 'conflict']
        }
        
        topics = []
        for topic, keywords in topic_keywords.items():
            if any(keyword in text for keyword in keywords):
                topics.append(topic)
        
        return topics
    
    def _is_relevant_to_australian_markets(self, article: Dict) -> bool:
        """Filter articles for Australian market relevance"""
        
        # Must have some market relevance
        if article.get('market_relevance', 0) < 0.1:
            return False
        
        # Must be recent (within timeframe)
        pub_date = article.get('published_at')
        if pub_date and isinstance(pub_date, datetime):
            hours_old = (datetime.now(timezone.utc) - pub_date.replace(tzinfo=timezone.utc)).total_seconds() / 3600
            if hours_old > 48:  # More than 48 hours old
                return False
        
        return True

# Global live news collector instance
live_news_collector = LiveNewsCollector()

async def test_live_news():
    """Test the live news collection system"""
    
    print("🌐 Testing Live News Collection System")
    print("=" * 50)
    
    async with live_news_collector as collector:
        articles = await collector.collect_live_news(hours_back=24, max_articles=10)
        
        print(f"📰 Collected {len(articles)} live articles")
        
        for i, article in enumerate(articles[:5], 1):
            print(f"\n📄 Article {i}:")
            print(f"  Title: {article['title'][:80]}...")
            print(f"  Source: {article.get('source_name', 'Unknown')}")
            print(f"  Published: {article.get('published_at', 'Unknown')}")
            print(f"  Market Relevance: {article.get('market_relevance', 0):.2f}")
            print(f"  Australian Relevance: {article.get('australian_relevance', 0):.2f}")
            print(f"  Sentiment: {article.get('sentiment_score', 0):+.2f}")
            print(f"  Impact: {article.get('impact_level', 'unknown')}")
            print(f"  Topics: {', '.join(article.get('topics', []))}")

if __name__ == "__main__":
    asyncio.run(test_live_news())