"""
GSMT Ver 7.0 - Market Sessions Manager
Manages global market sessions with Sydney timezone focus
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

# Import timezone handler - handle as module in same directory
try:
    from timezone_handler import sydney_tz_handler
except ImportError:
    # Fallback for different import patterns
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    from timezone_handler import sydney_tz_handler

logger = logging.getLogger(__name__)

@dataclass
class MarketSession:
    """Represents a trading session for a specific market"""
    market: str
    display_name: str
    open_sydney: datetime
    close_sydney: datetime
    is_active: bool
    color: str
    indices: List[str]
    timezone_name: str

class MarketSessionsManager:
    """
    Manages global market sessions with Sydney-centric view
    """
    
    def __init__(self):
        # Market-to-symbol mappings for default indices
        self.market_symbols = {
            "Australia": {
                "primary": "^AXJO",
                "symbols": ["^AXJO", "CBA.AX", "BHP.AX", "CSL.AX", "WBC.AX", "ANZ.AX"],
                "display": "ASX 200",
                "color": "#10b981"  # emerald-500
            },
            "Japan": {
                "primary": "^N225", 
                "symbols": ["^N225"],
                "display": "Nikkei 225",
                "color": "#3b82f6"  # blue-500
            },
            "Hong Kong": {
                "primary": "^HSI",
                "symbols": ["^HSI"], 
                "display": "Hang Seng",
                "color": "#8b5cf6"  # purple-500
            },
            "China": {
                "primary": "000001.SS",
                "symbols": ["000001.SS"],
                "display": "Shanghai Composite", 
                "color": "#ef4444"  # red-500
            },
            "UK": {
                "primary": "^FTSE",
                "symbols": ["^FTSE"],
                "display": "FTSE 100",
                "color": "#f59e0b"  # amber-500
            },
            "Germany": {
                "primary": "^GDAXI", 
                "symbols": ["^GDAXI"],
                "display": "DAX",
                "color": "#f97316"  # orange-500
            },
            "France": {
                "primary": "^FCHI",
                "symbols": ["^FCHI"], 
                "display": "CAC 40",
                "color": "#6366f1"  # indigo-500
            },
            "US": {
                "primary": "^GSPC",
                "symbols": ["^GSPC", "^IXIC", "^DJI", "AAPL", "GOOGL", "MSFT", "AMZN"],
                "display": "S&P 500",
                "color": "#06b6d4"  # cyan-500
            }
        }
    
    def get_current_market_sessions(self, reference_date: Optional[datetime] = None) -> List[MarketSession]:
        """
        Get current market sessions with Sydney timezone context
        
        Args:
            reference_date: Optional reference date for calculations
            
        Returns:
            List[MarketSession]: Active and upcoming market sessions
        """
        if reference_date is None:
            reference_date = sydney_tz_handler.get_sydney_now()
        
        sessions = []
        timeline = sydney_tz_handler.get_market_session_timeline()
        
        for market, session_data in timeline.items():
            if market in self.market_symbols:
                market_info = self.market_symbols[market]
                
                session = MarketSession(
                    market=market,
                    display_name=session_data["display_name"],
                    open_sydney=session_data["open_datetime"], 
                    close_sydney=session_data["close_datetime"],
                    is_active=session_data["is_active"],
                    color=market_info["color"],
                    indices=[market_info["primary"]],
                    timezone_name=session_data["local_timezone"]
                )
                
                sessions.append(session)
        
        # Sort by open time in Sydney timezone
        sessions.sort(key=lambda x: x.open_sydney.hour + (x.open_sydney.minute / 60))
        
        return sessions
    
    def get_24h_market_flow_data(self) -> Dict:
        """
        Get comprehensive 24-hour market flow data starting from 10am Sydney
        
        Returns:
            Dict: Market flow data optimized for visualization
        """
        start_time, end_time = sydney_tz_handler.get_24h_period_from_sydney_10am()
        sessions = self.get_current_market_sessions()
        
        # Create hourly timeline
        hourly_timeline = []
        current_hour = start_time
        
        while current_hour < end_time:
            hour_data = {
                "hour": current_hour.hour,
                "datetime": current_hour,
                "sydney_time": sydney_tz_handler.format_sydney_time(current_hour),
                "active_markets": [],
                "transitioning_markets": [],
                "major_indices": []
            }
            
            # Check which markets are active at this hour
            for session in sessions:
                # Handle sessions that span midnight
                session_start_hour = session.open_sydney.hour + (session.open_sydney.minute / 60)
                session_end_hour = session.close_sydney.hour + (session.close_sydney.minute / 60)
                current_decimal_hour = current_hour.hour + (current_hour.minute / 60)
                
                # Check if current hour falls within session
                is_active = False
                if session_end_hour > session_start_hour:
                    # Same day session
                    is_active = session_start_hour <= current_decimal_hour <= session_end_hour
                else:
                    # Session spans midnight
                    is_active = current_decimal_hour >= session_start_hour or current_decimal_hour <= session_end_hour
                
                if is_active:
                    hour_data["active_markets"].append({
                        "market": session.market,
                        "display": session.display_name,
                        "color": session.color,
                        "primary_symbol": self.market_symbols[session.market]["primary"]
                    })
                    
                    # Add major indices for this market
                    hour_data["major_indices"].extend(self.market_symbols[session.market]["symbols"][:2])
                
                # Check for opening/closing transitions (within 1 hour)
                if (abs(current_decimal_hour - session_start_hour) <= 1 or 
                    abs(current_decimal_hour - session_end_hour) <= 1):
                    hour_data["transitioning_markets"].append({
                        "market": session.market,
                        "type": "opening" if abs(current_decimal_hour - session_start_hour) <= 1 else "closing",
                        "display": session.display_name
                    })
            
            hourly_timeline.append(hour_data)
            current_hour += timedelta(hours=1)
        
        return {
            "start_time": start_time,
            "end_time": end_time,
            "timeline": hourly_timeline,
            "market_sessions": [
                {
                    "market": session.market,
                    "display_name": session.display_name,
                    "open_sydney": session.open_sydney,
                    "close_sydney": session.close_sydney,
                    "is_active": session.is_active,
                    "color": session.color,
                    "primary_symbol": self.market_symbols[session.market]["primary"],
                    "timezone": session.timezone_name
                } for session in sessions
            ],
            "default_symbols": [
                "^AXJO",  # Australia - ASX 200 (starts at 10am Sydney - perfect timing)
                "^N225",  # Japan - Nikkei 225 (usually active in early Sydney hours)
                "^FTSE",  # UK - FTSE 100 (active in Sydney evening)
                "^GSPC"   # US - S&P 500 (active late Sydney evening/night)
            ]
        }
    
    def get_optimal_refresh_schedule(self) -> Dict[str, int]:
        """
        Get optimal refresh schedule based on active markets
        
        Returns:
            Dict: Refresh intervals for different time periods
        """
        active_markets = sydney_tz_handler.get_active_markets()
        active_count = sum(active_markets.values())
        
        # More frequent updates during high-activity periods
        if active_count >= 3:
            # High activity - multiple markets open
            return {
                "primary_refresh": 60,     # 1 minute for primary data
                "secondary_refresh": 300,  # 5 minutes for secondary data
                "chart_refresh": 180       # 3 minutes for chart updates
            }
        elif active_count >= 1:
            # Moderate activity - some markets open
            return {
                "primary_refresh": 180,    # 3 minutes for primary data
                "secondary_refresh": 600,  # 10 minutes for secondary data  
                "chart_refresh": 300       # 5 minutes for chart updates
            }
        else:
            # Low activity - most markets closed
            return {
                "primary_refresh": 600,    # 10 minutes for primary data
                "secondary_refresh": 1800, # 30 minutes for secondary data
                "chart_refresh": 900       # 15 minutes for chart updates
            }
    
    def get_market_priority_order(self) -> List[str]:
        """
        Get market priority order starting with Australia (Sydney-centric)
        
        Returns:
            List[str]: Ordered list of markets by priority for Sydney traders
        """
        # Australia first as it's the primary focus, then by typical trading flow
        return [
            "Australia",    # 10:00 - 16:00 Sydney (primary market)
            "Japan",        # Usually 08:00 - 14:00 Sydney
            "Hong Kong",    # Usually 07:00 - 14:00 Sydney  
            "China",        # Usually 07:30 - 13:00 Sydney
            "UK",           # Usually 19:00 - 03:00 Sydney (next day)
            "Germany",      # Usually 18:00 - 02:30 Sydney (next day)
            "France",       # Usually 18:00 - 02:30 Sydney (next day)
            "US"            # Usually 00:30 - 07:00 Sydney (next day)
        ]
    
    def get_sydney_market_context(self) -> Dict:
        """
        Get specific context for Sydney market focus
        
        Returns:
            Dict: Sydney-specific market context and information
        """
        sydney_now = sydney_tz_handler.get_sydney_now()
        
        return {
            "current_sydney_time": sydney_tz_handler.format_sydney_time(sydney_now),
            "sydney_market_status": sydney_tz_handler.is_market_open_now("Australia"),
            "time_to_10am": self._calculate_time_to_next_10am(),
            "current_period_start": sydney_tz_handler.get_sydney_10am_start(),
            "market_day_phase": self._get_market_day_phase(),
            "recommended_focus_markets": self._get_recommended_focus_markets(),
            "sydney_trading_tips": {
                "best_analysis_time": "10:00 AM AEST/AEDT - Start of new 24h period",
                "peak_activity": "Multiple overlaps: Japan+Australia (morning), UK+US (evening)",
                "quiet_periods": "Early morning (4-8 AM Sydney time)",
                "optimal_refresh": "Every 3-5 minutes during active trading"
            }
        }
    
    def _calculate_time_to_next_10am(self) -> Dict:
        """Calculate time remaining until next 10am Sydney"""
        sydney_now = sydney_tz_handler.get_sydney_now()
        next_10am = sydney_tz_handler.get_sydney_10am_start()
        
        if sydney_now.hour >= 10:
            # Move to tomorrow's 10am
            next_10am = next_10am + timedelta(days=1)
        
        time_diff = next_10am - sydney_now
        
        return {
            "next_10am": sydney_tz_handler.format_sydney_time(next_10am),
            "hours_remaining": time_diff.seconds // 3600,
            "minutes_remaining": (time_diff.seconds % 3600) // 60,
            "total_seconds": time_diff.total_seconds()
        }
    
    def _get_market_day_phase(self) -> str:
        """Determine current phase of Sydney market day"""
        sydney_now = sydney_tz_handler.get_sydney_now()
        hour = sydney_now.hour
        
        if 6 <= hour < 10:
            return "Pre-market (Asian markets active)"
        elif 10 <= hour < 16:
            return "Sydney market hours" 
        elif 16 <= hour < 20:
            return "Post-Sydney, Pre-European"
        elif 20 <= hour <= 23 or 0 <= hour < 4:
            return "European/US markets active"
        else:
            return "Quiet period (most markets closed)"
    
    def _get_recommended_focus_markets(self) -> List[str]:
        """Get recommended markets to focus on based on current Sydney time"""
        sydney_now = sydney_tz_handler.get_sydney_now()
        hour = sydney_now.hour
        
        if 6 <= hour < 10:
            return ["Japan", "Hong Kong", "China"]
        elif 10 <= hour < 16: 
            return ["Australia", "Japan", "Hong Kong"]
        elif 16 <= hour < 20:
            return ["Australia", "UK", "Germany", "France"]
        elif 20 <= hour <= 23 or 0 <= hour < 4:
            return ["UK", "Germany", "France", "US"]
        else:
            return ["US"] if 0 <= hour < 6 else ["Australia"]

# Global instance
market_sessions_manager = MarketSessionsManager()