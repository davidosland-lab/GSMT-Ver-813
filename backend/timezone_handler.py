"""
GSMT Ver 7.0 - Sydney Timezone Handler
Handles all timezone calculations for Sydney-centric market analysis
"""

from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional
import pytz
from zoneinfo import ZoneInfo
import logging

logger = logging.getLogger(__name__)

class SydneyTimezoneHandler:
    """
    Handles all timezone operations for Sydney-centric market analysis
    """
    
    def __init__(self):
        # Primary timezone for Sydney
        self.sydney_tz = ZoneInfo("Australia/Sydney")
        self.utc_tz = ZoneInfo("UTC")
        
        # Market timezone mappings
        self.market_timezones = {
            "Australia": ZoneInfo("Australia/Sydney"),
            "Japan": ZoneInfo("Asia/Tokyo"), 
            "Hong Kong": ZoneInfo("Asia/Hong_Kong"),
            "China": ZoneInfo("Asia/Shanghai"),
            "UK": ZoneInfo("Europe/London"),
            "Germany": ZoneInfo("Europe/Berlin"),
            "France": ZoneInfo("Europe/Paris"),
            "US": ZoneInfo("America/New_York")
        }
        
        # Standard market hours in local time (24-hour format)
        self.local_market_hours = {
            "Australia": {"open": 10, "close": 16},  # 10:00 - 16:00 AEST/AEDT
            "Japan": {"open": 9, "close": 15},       # 09:00 - 15:00 JST  
            "Hong Kong": {"open": 9, "close": 16},   # 09:00 - 16:00 HKT
            "China": {"open": 9, "close": 15},       # 09:30 - 15:00 CST (simplified to 9-15)
            "UK": {"open": 8, "close": 16},          # 08:00 - 16:30 GMT/BST (simplified to 8-16)
            "Germany": {"open": 9, "close": 17},     # 09:00 - 17:30 CET/CEST (simplified to 9-17)
            "France": {"open": 9, "close": 17},      # 09:00 - 17:30 CET/CEST (simplified to 9-17)  
            "US": {"open": 9, "close": 16}           # 09:30 - 16:00 EST/EDT (simplified to 9-16)
        }
    
    def get_sydney_now(self) -> datetime:
        """Get current time in Sydney timezone"""
        return datetime.now(self.sydney_tz)
    
    def get_sydney_10am_start(self, reference_date: Optional[datetime] = None) -> datetime:
        """
        Get the 10am Sydney time start for 24-hour analysis period
        
        Args:
            reference_date: Optional reference date, defaults to today
            
        Returns:
            datetime: 10am Sydney time for the reference date
        """
        if reference_date is None:
            reference_date = self.get_sydney_now()
        elif reference_date.tzinfo is None:
            # Assume UTC if no timezone info
            reference_date = reference_date.replace(tzinfo=self.utc_tz).astimezone(self.sydney_tz)
        elif reference_date.tzinfo != self.sydney_tz:
            # Convert to Sydney timezone
            reference_date = reference_date.astimezone(self.sydney_tz)
        
        # Set to 10:00 AM Sydney time
        start_time = reference_date.replace(hour=10, minute=0, second=0, microsecond=0)
        
        return start_time
    
    def get_24h_period_from_sydney_10am(self, reference_date: Optional[datetime] = None) -> Tuple[datetime, datetime]:
        """
        Get 24-hour period starting from 10am Sydney time
        
        Args:
            reference_date: Optional reference date, defaults to today
            
        Returns:
            Tuple[datetime, datetime]: (start_time, end_time) both in Sydney timezone
        """
        start_time = self.get_sydney_10am_start(reference_date)
        end_time = start_time + timedelta(hours=24)
        
        return start_time, end_time
    
    def convert_to_sydney_time(self, dt: datetime, source_timezone: str = "UTC") -> datetime:
        """
        Convert datetime from source timezone to Sydney time
        
        Args:
            dt: datetime object to convert
            source_timezone: source timezone string (e.g., "UTC", "US/Eastern")
            
        Returns:
            datetime: converted to Sydney timezone
        """
        if dt.tzinfo is None:
            # Assume source timezone if no timezone info
            if source_timezone == "UTC":
                dt = dt.replace(tzinfo=self.utc_tz)
            else:
                source_tz = ZoneInfo(source_timezone)
                dt = dt.replace(tzinfo=source_tz)
        
        return dt.astimezone(self.sydney_tz)
    
    def get_market_hours_in_sydney_time(self) -> Dict[str, Dict[str, int]]:
        """
        Get all market hours converted to Sydney time
        
        Returns:
            Dict: Market hours in Sydney timezone (24-hour format)
        """
        sydney_market_hours = {}
        sydney_now = self.get_sydney_now()
        today_sydney = sydney_now.date()
        
        for market, hours in self.local_market_hours.items():
            market_tz = self.market_timezones[market]
            
            # Create market open/close times in their local timezone for today
            market_today = sydney_now.astimezone(market_tz).date()
            
            open_local = datetime.combine(market_today, datetime.min.time().replace(hour=hours["open"]))
            close_local = datetime.combine(market_today, datetime.min.time().replace(hour=hours["close"]))
            
            open_local = open_local.replace(tzinfo=market_tz)
            close_local = close_local.replace(tzinfo=market_tz)
            
            # Convert to Sydney time
            open_sydney = open_local.astimezone(self.sydney_tz)
            close_sydney = close_local.astimezone(self.sydney_tz)
            
            sydney_market_hours[market] = {
                "open": open_sydney.hour + (open_sydney.minute / 60),  # Decimal hours
                "close": close_sydney.hour + (close_sydney.minute / 60),
                "open_datetime": open_sydney,
                "close_datetime": close_sydney,
                "is_next_day": open_sydney.date() != today_sydney or close_sydney.date() != today_sydney
            }
        
        return sydney_market_hours
    
    def is_market_open_now(self, market: str) -> bool:
        """
        Check if a specific market is currently open
        
        Args:
            market: Market name (e.g., "US", "Japan", "Australia")
            
        Returns:
            bool: True if market is currently open
        """
        if market not in self.market_timezones:
            return False
        
        market_tz = self.market_timezones[market]
        market_now = datetime.now(market_tz)
        market_hours = self.local_market_hours[market]
        
        current_hour = market_now.hour + (market_now.minute / 60)
        
        # Check if within market hours
        open_time = market_hours["open"]
        close_time = market_hours["close"]
        
        # Handle cases where market closes next day (shouldn't happen with current hours but good to have)
        if close_time > open_time:
            return open_time <= current_hour <= close_time
        else:
            return current_hour >= open_time or current_hour <= close_time
    
    def get_active_markets(self) -> Dict[str, bool]:
        """
        Get status of all markets (open/closed)
        
        Returns:
            Dict[str, bool]: Market name -> is_open status
        """
        active_status = {}
        
        for market in self.market_timezones.keys():
            active_status[market] = self.is_market_open_now(market)
        
        return active_status
    
    def format_sydney_time(self, dt: datetime, include_timezone: bool = True) -> str:
        """
        Format datetime in Sydney timezone for display
        
        Args:
            dt: datetime object
            include_timezone: whether to include timezone abbreviation
            
        Returns:
            str: formatted time string
        """
        if dt.tzinfo != self.sydney_tz:
            dt = dt.astimezone(self.sydney_tz)
        
        if include_timezone:
            # Show AEST or AEDT appropriately
            tz_name = dt.strftime('%Z')  # Will show AEST or AEDT
            return dt.strftime(f'%Y-%m-%d %H:%M:%S {tz_name}')
        else:
            return dt.strftime('%Y-%m-%d %H:%M:%S')
    
    def get_market_session_timeline(self) -> Dict[str, Dict]:
        """
        Get detailed timeline of market sessions for visualization
        
        Returns:
            Dict: Comprehensive market session data for charts
        """
        sydney_hours = self.get_market_hours_in_sydney_time()
        active_markets = self.get_active_markets()
        
        timeline_data = {}
        
        for market, hours in sydney_hours.items():
            timeline_data[market] = {
                "open_hour": hours["open"],
                "close_hour": hours["close"], 
                "open_datetime": hours["open_datetime"],
                "close_datetime": hours["close_datetime"],
                "is_active": active_markets[market],
                "is_next_day": hours["is_next_day"],
                "local_timezone": str(self.market_timezones[market]),
                "color_class": self._get_market_color_class(market),
                "display_name": self._get_market_display_name(market)
            }
        
        return timeline_data
    
    def _get_market_color_class(self, market: str) -> str:
        """Get CSS color class for market visualization"""
        color_map = {
            "Australia": "emerald",
            "Japan": "blue", 
            "Hong Kong": "purple",
            "China": "red",
            "UK": "yellow",
            "Germany": "orange",
            "France": "indigo",
            "US": "cyan"
        }
        return color_map.get(market, "gray")
    
    def _get_market_display_name(self, market: str) -> str:
        """Get display name for market"""
        display_map = {
            "Australia": "🇦🇺 Sydney",
            "Japan": "🇯🇵 Tokyo", 
            "Hong Kong": "🇭🇰 Hong Kong",
            "China": "🇨🇳 Shanghai",
            "UK": "🇬🇧 London",
            "Germany": "🇩🇪 Frankfurt",
            "France": "🇫🇷 Paris",
            "US": "🇺🇸 New York"
        }
        return display_map.get(market, market)

# Global instance for easy access
sydney_tz_handler = SydneyTimezoneHandler()

def get_sydney_24h_period(reference_date: Optional[datetime] = None) -> Tuple[datetime, datetime]:
    """
    Convenience function to get 24-hour period from 10am Sydney time
    
    Args:
        reference_date: Optional reference date
        
    Returns:
        Tuple[datetime, datetime]: (start, end) times in Sydney timezone
    """
    return sydney_tz_handler.get_24h_period_from_sydney_10am(reference_date)

def format_for_sydney_display(dt: datetime) -> str:
    """
    Convenience function to format datetime for Sydney display
    
    Args:
        dt: datetime object
        
    Returns:
        str: formatted Sydney time string
    """
    return sydney_tz_handler.format_sydney_time(dt)