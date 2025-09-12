"""
Social Media Sentiment Tracking Infrastructure
Critical Factor #3: Track Twitter, Reddit, and social sentiment for Australian markets
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
import re
import hashlib
from collections import defaultdict, Counter
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentScore(Enum):
    """Sentiment classification levels"""
    VERY_NEGATIVE = -2
    NEGATIVE = -1
    NEUTRAL = 0
    POSITIVE = 1
    VERY_POSITIVE = 2

class PlatformType(Enum):
    """Social media platform types"""
    TWITTER = "twitter"
    REDDIT = "reddit"
    YOUTUBE = "youtube"
    LINKEDIN = "linkedin"
    FACEBOOK = "facebook"
    TELEGRAM = "telegram"

class ContentType(Enum):
    """Types of social media content"""
    POST = "post"
    COMMENT = "comment"
    RETWEET = "retweet"
    REPLY = "reply"
    SHARE = "share"
    MENTION = "mention"

@dataclass
class SocialPost:
    """Individual social media post with sentiment analysis"""
    platform: PlatformType
    content_type: ContentType
    author: str
    content: str
    timestamp: datetime
    engagement: Dict[str, int]  # likes, shares, comments, etc.
    sentiment_score: float      # -1.0 to 1.0
    sentiment_class: SentimentScore
    confidence: float           # 0-1 confidence in sentiment
    asx_mentions: List[str]     # ASX symbols mentioned
    market_relevance: float     # 0-1 relevance to market
    influence_score: float      # Author influence score
    viral_potential: float      # 0-1 viral potential score

@dataclass
class SentimentTrend:
    """Trending sentiment analysis"""
    platform: PlatformType
    time_window: str           # 1h, 4h, 24h, 7d
    total_posts: int
    sentiment_distribution: Dict[SentimentScore, int]
    average_sentiment: float   # -1.0 to 1.0
    momentum: float           # Sentiment change rate
    trending_topics: List[str]
    trending_symbols: List[str]
    engagement_velocity: float # Posts per hour
    viral_events: List[Dict[str, Any]]

@dataclass
class RetailInvestorBehavior:
    """Retail investor behavior analysis from social signals"""
    platform_activity: Dict[PlatformType, float]  # Activity levels
    risk_appetite: float                          # 0-1 risk appetite score
    fomo_indicator: float                         # Fear of missing out score
    contrarian_signal: float                      # Contrarian indicator
    new_investor_influx: float                    # New investor activity
    options_interest: float                       # Interest in options trading
    meme_stock_activity: float                    # Meme stock discussion level
    education_seeking: float                      # Questions/learning posts

class TwitterSentimentCollector:
    """Collects sentiment data from Twitter/X"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('TWITTER_API_KEY')
        self.session = None
        self.rate_limits = {'search': 300, 'user_timeline': 900}  # Per 15 minutes
        
        # Australian finance Twitter accounts to monitor
        self.finance_accounts = [
            '@ASX', '@RBA_Gov', '@ComSec', '@NABmarkets', '@WesBankNews',
            '@AFR', '@FinancialReview', '@ausbusinessnews', '@InvestmentNews',
            '@MarketsHerald', '@CommBank', '@ANZ_AU', '@WestpacNews'
        ]
        
        # ASX-related hashtags and keywords
        self.asx_keywords = [
            'ASX', 'All Ordinaries', 'Australian shares', 'Aussie stocks',
            'CommSec', 'NAB trade', 'Westpac Invest', 'ANZ invest',
            'ASIC', 'RBA', 'interest rates', 'inflation', 'iron ore',
            'mining stocks', 'bank stocks', 'dividend yield'
        ]
        
        # Stock symbols to track
        self.tracked_symbols = [
            'CBA', 'BHP', 'CSL', 'WBC', 'ANZ', 'NAB', 'WES', 'WOW', 'TLS',
            'RIO', 'MQG', 'TCL', 'STO', 'FMG', 'XRO', 'APT', 'A2M', 'WTC'
        ]

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={
                'User-Agent': 'Social Sentiment Tracker 1.0',
                'Authorization': f'Bearer {self.api_key}' if self.api_key else None
            }
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def fetch_twitter_sentiment(self, time_window: int = 24) -> List[SocialPost]:
        """Fetch Twitter posts related to ASX and Australian markets"""
        
        logger.info(f"🐦 Fetching Twitter sentiment for last {time_window} hours...")
        
        # Simulate Twitter API calls (replace with actual Twitter API v2)
        posts = await self._simulate_twitter_data(time_window)
        
        return posts

    async def _simulate_twitter_data(self, hours: int) -> List[SocialPost]:
        """Generate realistic simulated Twitter data"""
        
        posts = []
        current_time = datetime.now(timezone.utc)
        
        # Generate posts over time window
        num_posts = int(np.random.poisson(hours * 15))  # ~15 posts per hour
        
        for _ in range(num_posts):
            # Random timestamp within window
            post_time = current_time - timedelta(
                seconds=np.random.randint(0, hours * 3600)
            )
            
            # Generate realistic post content
            post_data = self._generate_twitter_post(post_time)
            if post_data:
                posts.append(post_data)
        
        return posts

    def _generate_twitter_post(self, timestamp: datetime) -> Optional[SocialPost]:
        """Generate a realistic Twitter post"""
        
        # Sample post templates
        post_templates = [
            "Just bought more {symbol} shares. This dip is a gift! 📈 #ASX #investing",
            "{symbol} earnings looking strong. Could see a breakout soon 🚀",
            "Market volatility has me worried about my {symbol} position 😰 #ASX",
            "RBA decision tomorrow - expecting impact on bank stocks like {symbol}",
            "Iron ore prices affecting {symbol}. China demand looking weak 📉",
            "Dividend season! {symbol} yielding {yield}% looks attractive 💰",
            "Technical analysis on {symbol} shows bullish pattern forming 📊",
            "{symbol} just broke resistance! Target price ${target} #breakout",
            "Sold my {symbol} position. Taking profits before earnings 💵",
            "Long-term hold on {symbol}. Australian market has strong fundamentals 🇦🇺"
        ]
        
        # Select random elements
        symbol = np.random.choice(self.tracked_symbols)
        template = np.random.choice(post_templates)
        
        # Generate post content with realistic data
        content = template.format(
            symbol=symbol,
            **{"yield": f"{np.random.uniform(2.5, 8.0):.1f}"},  # Use dict unpacking for reserved keyword
            target=f"{np.random.uniform(20, 200):.2f}"
        )
        
        # Calculate sentiment from content
        sentiment_score, sentiment_class = self._analyze_post_sentiment(content)
        
        # Generate engagement metrics
        engagement = {
            'likes': max(0, int(np.random.exponential(5))),
            'retweets': max(0, int(np.random.exponential(2))),
            'replies': max(0, int(np.random.exponential(1))),
            'views': max(10, int(np.random.exponential(50)))
        }
        
        # Calculate influence and relevance scores
        influence_score = min(np.random.exponential(0.3), 1.0)
        market_relevance = 0.8 + np.random.uniform(-0.2, 0.2)
        
        return SocialPost(
            platform=PlatformType.TWITTER,
            content_type=ContentType.POST,
            author=f"user_{hash(content) % 10000}",
            content=content,
            timestamp=timestamp,
            engagement=engagement,
            sentiment_score=sentiment_score,
            sentiment_class=sentiment_class,
            confidence=0.75 + np.random.uniform(-0.15, 0.15),
            asx_mentions=[symbol] if symbol in content else [],
            market_relevance=market_relevance,
            influence_score=influence_score,
            viral_potential=min(sum(engagement.values()) / 100, 1.0)
        )

    def _analyze_post_sentiment(self, content: str) -> Tuple[float, SentimentScore]:
        """Analyze sentiment of social media post"""
        
        # Positive sentiment indicators
        positive_words = [
            'bullish', 'buy', 'long', 'breakout', 'strong', 'growth', 'profit',
            'gains', 'rally', 'uptrend', 'target', 'moon', 'rocket', '🚀', '📈',
            'diamond hands', 'HODL', 'dip buying', 'opportunity'
        ]
        
        # Negative sentiment indicators  
        negative_words = [
            'bearish', 'sell', 'short', 'crash', 'weak', 'loss', 'decline',
            'dump', 'breakdown', 'resistance', 'bear market', '📉', '💀',
            'paper hands', 'panic selling', 'worried', 'scared'
        ]
        
        # Neutral/uncertainty indicators
        neutral_words = [
            'holding', 'watching', 'waiting', 'uncertain', 'maybe', 'could',
            'might', 'sideways', 'consolidation', 'range bound'
        ]
        
        content_lower = content.lower()
        
        # Count sentiment indicators
        positive_count = sum(1 for word in positive_words if word in content_lower)
        negative_count = sum(1 for word in negative_words if word in content_lower)
        neutral_count = sum(1 for word in neutral_words if word in content_lower)
        
        # Calculate sentiment score
        total_sentiment_words = positive_count + negative_count + neutral_count
        
        if total_sentiment_words == 0:
            return 0.0, SentimentScore.NEUTRAL
        
        # Weight sentiment
        raw_score = (positive_count - negative_count) / max(total_sentiment_words, 1)
        
        # Add emoji sentiment
        if '🚀' in content or '📈' in content or '💰' in content:
            raw_score += 0.3
        if '📉' in content or '💀' in content or '😰' in content:
            raw_score -= 0.3
        
        # Normalize to -1 to 1
        sentiment_score = np.tanh(raw_score)
        
        # Classify sentiment
        if sentiment_score >= 0.4:
            sentiment_class = SentimentScore.VERY_POSITIVE
        elif sentiment_score >= 0.1:
            sentiment_class = SentimentScore.POSITIVE
        elif sentiment_score <= -0.4:
            sentiment_class = SentimentScore.VERY_NEGATIVE
        elif sentiment_score <= -0.1:
            sentiment_class = SentimentScore.NEGATIVE
        else:
            sentiment_class = SentimentScore.NEUTRAL
        
        return sentiment_score, sentiment_class

class RedditSentimentCollector:
    """Collects sentiment data from Reddit"""
    
    def __init__(self, client_id: Optional[str] = None, client_secret: Optional[str] = None):
        self.client_id = client_id or os.getenv('REDDIT_CLIENT_ID')
        self.client_secret = client_secret or os.getenv('REDDIT_CLIENT_SECRET')
        self.session = None
        
        # Australian finance subreddits
        self.finance_subreddits = [
            'ASX_Bets', 'AusFinance', 'fiaustralia', 'AusInvesting',
            'SecurityAnalysis', 'ValueInvesting', 'stocks', 'investing',
            'StockMarket', 'SecurityHolder', 'AustralianStocks'
        ]
        
        # Track these for ASX-specific content
        self.asx_symbols = [
            'CBA', 'BHP', 'CSL', 'WBC', 'ANZ', 'NAB', 'WES', 'WOW',
            'TLS', 'RIO', 'MQG', 'TCL', 'STO', 'FMG', 'XRO', 'APT'
        ]

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={'User-Agent': 'Social Sentiment Tracker 1.0'}
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def fetch_reddit_sentiment(self, time_window: int = 24) -> List[SocialPost]:
        """Fetch Reddit posts and comments from finance subreddits"""
        
        logger.info(f"🔴 Fetching Reddit sentiment for last {time_window} hours...")
        
        # Simulate Reddit API calls (replace with actual Reddit API)
        posts = await self._simulate_reddit_data(time_window)
        
        return posts

    async def _simulate_reddit_data(self, hours: int) -> List[SocialPost]:
        """Generate realistic simulated Reddit data"""
        
        posts = []
        current_time = datetime.now(timezone.utc)
        
        # Generate posts and comments
        num_posts = int(np.random.poisson(hours * 8))  # ~8 posts per hour
        
        for _ in range(num_posts):
            post_time = current_time - timedelta(
                seconds=np.random.randint(0, hours * 3600)
            )
            
            post_data = self._generate_reddit_post(post_time)
            if post_data:
                posts.append(post_data)
        
        return posts

    def _generate_reddit_post(self, timestamp: datetime) -> Optional[SocialPost]:
        """Generate a realistic Reddit post"""
        
        # Reddit post templates (more detailed than Twitter)
        post_templates = [
            "DD: {symbol} - Why I think this is undervalued\n\nLooking at the fundamentals, P/E ratio of {pe} seems reasonable given the growth prospects...",
            "YOLO: All in on {symbol} 🚀\n\nJust put my entire portfolio into {symbol}. Diamond hands! This is going to the moon!",
            "Loss porn: Down {loss}% on {symbol} 😭\n\nBought at the top like a true retard. Should have listened to my wife's boyfriend...",
            "Gain porn: {symbol} printed money today 💰\n\nUp {gain}% on my {symbol} position. Sometimes being an autist pays off!",
            "Discussion: Thoughts on {symbol} after latest earnings?\n\nEPS beat expectations but revenue guidance was weak. What's everyone's take?",
            "News: {symbol} announces {news_type}\n\nThis could be a game changer. Bullish or bearish? What are your positions?",
            "TA: {symbol} technical analysis - breakout incoming?\n\nLooking at the charts, we're at a key resistance level. Volume is picking up...",
            "Question: Is {symbol} a good long-term hold?\n\nNew to investing, thinking of buying {symbol} for my super. Good idea or nah?",
        ]
        
        # Select random elements
        symbol = np.random.choice(self.asx_symbols)
        template = np.random.choice(post_templates)
        
        # Generate realistic content
        content = template.format(
            symbol=symbol,
            pe=f"{np.random.uniform(8, 25):.1f}",
            loss=f"{np.random.uniform(10, 80):.0f}",
            gain=f"{np.random.uniform(15, 200):.0f}",
            news_type=np.random.choice(['dividend increase', 'merger', 'expansion', 'cost cutting'])
        )
        
        # Analyze sentiment
        sentiment_score, sentiment_class = self._analyze_reddit_sentiment(content)
        
        # Reddit engagement metrics
        upvote_ratio = 0.6 + np.random.uniform(-0.2, 0.3)
        total_votes = max(1, int(np.random.exponential(20)))
        upvotes = int(total_votes * upvote_ratio)
        
        engagement = {
            'upvotes': upvotes,
            'downvotes': total_votes - upvotes,
            'comments': max(0, int(np.random.exponential(5))),
            'awards': max(0, int(np.random.exponential(0.5)))
        }
        
        # Determine content type and subreddit influence
        if 'DD:' in content:
            content_type = ContentType.POST
            influence_multiplier = 1.5  # DD posts more influential
        elif 'YOLO:' in content or 'Loss porn:' in content or 'Gain porn:' in content:
            content_type = ContentType.POST
            influence_multiplier = 1.2  # YOLO posts get attention
        else:
            content_type = ContentType.POST
            influence_multiplier = 1.0
        
        influence_score = min(np.random.exponential(0.4) * influence_multiplier, 1.0)
        
        return SocialPost(
            platform=PlatformType.REDDIT,
            content_type=content_type,
            author=f"u/autist_{hash(content) % 10000}",
            content=content,
            timestamp=timestamp,
            engagement=engagement,
            sentiment_score=sentiment_score,
            sentiment_class=sentiment_class,
            confidence=0.70 + np.random.uniform(-0.15, 0.20),
            asx_mentions=[symbol] if symbol in content else [],
            market_relevance=0.85 + np.random.uniform(-0.15, 0.15),
            influence_score=influence_score,
            viral_potential=min((upvotes + engagement['comments'] * 2) / 100, 1.0)
        )

    def _analyze_reddit_sentiment(self, content: str) -> Tuple[float, SentimentScore]:
        """Analyze Reddit post sentiment (similar to Twitter but adjusted for Reddit culture)"""
        
        # Reddit-specific positive indicators
        reddit_positive = [
            'DD', 'to the moon', 'diamond hands', 'HODL', 'stonks', 'tendies',
            'bullish', 'calls', 'yolo', 'apes together strong', '🚀', '💎🙌',
            'this is the way', 'buy the dip', 'undervalued', 'fundamentals strong'
        ]
        
        # Reddit-specific negative indicators
        reddit_negative = [
            'loss porn', 'bag holder', 'paper hands', 'puts', 'bearish',
            'overvalued', 'bubble', 'crash incoming', 'sell everything',
            'wife\'s boyfriend', 'retard', 'autist', 'FUD', 'manipulation'
        ]
        
        content_lower = content.lower()
        
        positive_count = sum(1 for phrase in reddit_positive if phrase in content_lower)
        negative_count = sum(1 for phrase in reddit_negative if phrase in content_lower)
        
        # Reddit-specific adjustments
        if 'DD:' in content:
            positive_count += 1  # Due diligence posts tend to be bullish
        if 'Loss porn:' in content:
            negative_count += 2  # Loss posts are definitely negative
        if 'Gain porn:' in content:
            positive_count += 2  # Gain posts are very positive
        
        total_indicators = positive_count + negative_count
        
        if total_indicators == 0:
            return 0.0, SentimentScore.NEUTRAL
        
        raw_score = (positive_count - negative_count) / max(total_indicators, 1)
        sentiment_score = np.tanh(raw_score)
        
        # Classify sentiment
        if sentiment_score >= 0.5:
            sentiment_class = SentimentScore.VERY_POSITIVE
        elif sentiment_score >= 0.15:
            sentiment_class = SentimentScore.POSITIVE
        elif sentiment_score <= -0.5:
            sentiment_class = SentimentScore.VERY_NEGATIVE
        elif sentiment_score <= -0.15:
            sentiment_class = SentimentScore.NEGATIVE
        else:
            sentiment_class = SentimentScore.NEUTRAL
        
        return sentiment_score, sentiment_class

class SocialSentimentAnalyzer:
    """Analyzes social media sentiment for market prediction"""
    
    def __init__(self):
        self.twitter_collector = TwitterSentimentCollector()
        self.reddit_collector = RedditSentimentCollector()
        self.sentiment_history = {}
        
    async def collect_social_sentiment(self, time_window: int = 24) -> Dict[PlatformType, List[SocialPost]]:
        """Collect sentiment data from all social platforms"""
        
        logger.info(f"📱 Collecting social sentiment for last {time_window} hours...")
        
        social_data = {}
        
        # Collect Twitter data
        try:
            async with self.twitter_collector as twitter:
                twitter_posts = await twitter.fetch_twitter_sentiment(time_window)
                social_data[PlatformType.TWITTER] = twitter_posts
                logger.info(f"✅ Collected {len(twitter_posts)} Twitter posts")
        except Exception as e:
            logger.error(f"❌ Twitter collection failed: {e}")
            social_data[PlatformType.TWITTER] = []
        
        # Collect Reddit data
        try:
            async with self.reddit_collector as reddit:
                reddit_posts = await reddit.fetch_reddit_sentiment(time_window)
                social_data[PlatformType.REDDIT] = reddit_posts
                logger.info(f"✅ Collected {len(reddit_posts)} Reddit posts")
        except Exception as e:
            logger.error(f"❌ Reddit collection failed: {e}")
            social_data[PlatformType.REDDIT] = []
        
        return social_data

    def analyze_sentiment_trends(self, social_data: Dict[PlatformType, List[SocialPost]]) -> Dict[PlatformType, SentimentTrend]:
        """Analyze sentiment trends across platforms"""
        
        trends = {}
        
        for platform, posts in social_data.items():
            if not posts:
                continue
                
            # Calculate sentiment distribution
            sentiment_dist = Counter([post.sentiment_class for post in posts])
            
            # Calculate average sentiment
            avg_sentiment = np.mean([post.sentiment_score for post in posts])
            
            # Find trending topics and symbols
            all_mentions = []
            for post in posts:
                all_mentions.extend(post.asx_mentions)
            
            symbol_counts = Counter(all_mentions)
            trending_symbols = [symbol for symbol, count in symbol_counts.most_common(10)]
            
            # Calculate engagement velocity
            time_span = 24  # hours
            engagement_velocity = len(posts) / time_span
            
            # Identify viral events (high engagement posts)
            viral_threshold = np.percentile([p.viral_potential for p in posts], 90) if posts else 0
            viral_events = []
            
            for post in posts:
                if post.viral_potential > viral_threshold and post.viral_potential > 0.5:
                    viral_events.append({
                        'content': post.content[:100] + '...',
                        'engagement': sum(post.engagement.values()),
                        'sentiment': post.sentiment_score,
                        'timestamp': post.timestamp.isoformat()
                    })
            
            trends[platform] = SentimentTrend(
                platform=platform,
                time_window='24h',
                total_posts=len(posts),
                sentiment_distribution=dict(sentiment_dist),
                average_sentiment=avg_sentiment,
                momentum=self._calculate_sentiment_momentum(posts),
                trending_topics=[],  # Would extract from content analysis
                trending_symbols=trending_symbols,
                engagement_velocity=engagement_velocity,
                viral_events=viral_events[:5]  # Top 5 viral events
            )
        
        return trends

    def _calculate_sentiment_momentum(self, posts: List[SocialPost]) -> float:
        """Calculate sentiment momentum (change rate)"""
        
        if len(posts) < 10:
            return 0.0
        
        # Sort posts by timestamp
        sorted_posts = sorted(posts, key=lambda p: p.timestamp)
        
        # Split into early and recent periods
        mid_point = len(sorted_posts) // 2
        early_posts = sorted_posts[:mid_point]
        recent_posts = sorted_posts[mid_point:]
        
        # Calculate average sentiment for each period
        early_sentiment = np.mean([p.sentiment_score for p in early_posts])
        recent_sentiment = np.mean([p.sentiment_score for p in recent_posts])
        
        # Momentum is the change in sentiment
        momentum = recent_sentiment - early_sentiment
        
        return momentum

    def analyze_retail_behavior(self, social_data: Dict[PlatformType, List[SocialPost]]) -> RetailInvestorBehavior:
        """Analyze retail investor behavior patterns from social data"""
        
        all_posts = []
        for posts in social_data.values():
            all_posts.extend(posts)
        
        if not all_posts:
            return RetailInvestorBehavior(
                platform_activity={}, risk_appetite=0.5, fomo_indicator=0.5,
                contrarian_signal=0.5, new_investor_influx=0.5, options_interest=0.5,
                meme_stock_activity=0.5, education_seeking=0.5
            )
        
        # Platform activity levels
        platform_activity = {}
        for platform, posts in social_data.items():
            activity_score = len(posts) / 100  # Normalize
            platform_activity[platform] = min(activity_score, 1.0)
        
        # Risk appetite (from YOLO posts, options mentions)
        risk_indicators = ['yolo', 'all in', 'moon', 'rocket', 'diamond hands', 'options', 'calls', 'puts']
        risk_posts = [p for p in all_posts if any(indicator in p.content.lower() for indicator in risk_indicators)]
        risk_appetite = min(len(risk_posts) / len(all_posts) * 2, 1.0)
        
        # FOMO indicator (urgency, fear of missing out)
        fomo_indicators = ['moon', 'rocket', 'fomo', 'missing out', 'last chance', 'too late']
        fomo_posts = [p for p in all_posts if any(indicator in p.content.lower() for indicator in fomo_indicators)]
        fomo_score = min(len(fomo_posts) / len(all_posts) * 3, 1.0)
        
        # Contrarian signal (when retail is very bullish, often bearish signal)
        very_positive_posts = [p for p in all_posts if p.sentiment_score > 0.5]
        contrarian_signal = min(len(very_positive_posts) / len(all_posts) * 1.5, 1.0)
        
        # New investor activity (questions, basic terms)
        newbie_indicators = ['new to', 'beginner', 'help', 'advice', 'how to', 'should i', 'first time']
        newbie_posts = [p for p in all_posts if any(indicator in p.content.lower() for indicator in newbie_indicators)]
        new_investor_influx = min(len(newbie_posts) / len(all_posts) * 2, 1.0)
        
        # Options interest
        options_indicators = ['options', 'calls', 'puts', 'strike', 'expiry', 'theta', 'delta']
        options_posts = [p for p in all_posts if any(indicator in p.content.lower() for indicator in options_indicators)]
        options_interest = min(len(options_posts) / len(all_posts) * 4, 1.0)
        
        # Meme stock activity
        meme_indicators = ['meme', 'ape', 'stonks', 'tendies', 'wife\'s boyfriend', 'retard']
        meme_posts = [p for p in all_posts if any(indicator in p.content.lower() for indicator in meme_indicators)]
        meme_activity = min(len(meme_posts) / len(all_posts) * 2, 1.0)
        
        # Education seeking
        education_indicators = ['dd', 'analysis', 'fundamentals', 'valuation', 'research', 'study']
        education_posts = [p for p in all_posts if any(indicator in p.content.lower() for indicator in education_indicators)]
        education_seeking = min(len(education_posts) / len(all_posts) * 1.5, 1.0)
        
        return RetailInvestorBehavior(
            platform_activity=platform_activity,
            risk_appetite=risk_appetite,
            fomo_indicator=fomo_score,
            contrarian_signal=contrarian_signal,
            new_investor_influx=new_investor_influx,
            options_interest=options_interest,
            meme_stock_activity=meme_activity,
            education_seeking=education_seeking
        )

    async def get_market_prediction_factors(self, time_window: int = 24) -> Dict[str, float]:
        """Get standardized factors for market prediction integration"""
        
        logger.info("🎯 Generating social sentiment prediction factors...")
        
        # Collect social data
        social_data = await self.collect_social_sentiment(time_window)
        
        # Analyze trends and behavior
        sentiment_trends = self.analyze_sentiment_trends(social_data)
        retail_behavior = self.analyze_retail_behavior(social_data)
        
        factors = {}
        
        # Overall sentiment factor
        all_posts = []
        for posts in social_data.values():
            all_posts.extend(posts)
        
        if all_posts:
            avg_sentiment = np.mean([p.sentiment_score for p in all_posts])
            factors['social_sentiment'] = avg_sentiment
            
            # Sentiment momentum
            momentum = np.mean([trend.momentum for trend in sentiment_trends.values()])
            factors['social_momentum'] = momentum
            
            # Engagement velocity (activity level)
            avg_velocity = np.mean([trend.engagement_velocity for trend in sentiment_trends.values()])
            velocity_factor = min(avg_velocity / 50, 1.0)  # Normalize around 50 posts/hour
            factors['social_activity_level'] = velocity_factor
        else:
            factors['social_sentiment'] = 0.0
            factors['social_momentum'] = 0.0
            factors['social_activity_level'] = 0.0
        
        # Retail behavior factors
        factors['social_risk_appetite'] = retail_behavior.risk_appetite
        factors['social_fomo_indicator'] = retail_behavior.fomo_indicator
        factors['social_contrarian_signal'] = retail_behavior.contrarian_signal
        factors['social_new_investor_influx'] = retail_behavior.new_investor_influx
        factors['social_options_interest'] = retail_behavior.options_interest
        
        # Platform-specific factors
        for platform, trend in sentiment_trends.items():
            platform_name = platform.value
            factors[f'social_{platform_name}_sentiment'] = trend.average_sentiment
            
            # Viral events indicator
            viral_factor = min(len(trend.viral_events) / 5, 1.0)  # 5+ viral events = max score
            factors[f'social_{platform_name}_viral'] = viral_factor
        
        logger.info(f"📊 Social factors: Sentiment: {factors.get('social_sentiment', 0):.3f}, "
                   f"FOMO: {factors.get('social_fomo_indicator', 0):.3f}, "
                   f"Activity: {factors.get('social_activity_level', 0):.3f}")
        
        return factors

# Global instance
social_sentiment_analyzer = SocialSentimentAnalyzer()

# Export classes
__all__ = [
    'SocialSentimentAnalyzer',
    'TwitterSentimentCollector',
    'RedditSentimentCollector',
    'SocialPost',
    'SentimentTrend',
    'RetailInvestorBehavior',
    'SentimentScore',
    'PlatformType',
    'ContentType',
    'social_sentiment_analyzer'
]