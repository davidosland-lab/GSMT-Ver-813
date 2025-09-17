"""
Market Holiday Detection System
Comprehensive holiday calendar for major global stock markets
"""

from datetime import date, datetime, timedelta
from typing import Dict, List, Optional
import pytz
from enum import Enum

class MarketHolidayType(Enum):
    FULL_CLOSURE = "full_closure"
    EARLY_CLOSE = "early_close"
    LATE_OPEN = "late_open"

class MarketHoliday:
    def __init__(self, name: str, date_obj: date, holiday_type: MarketHolidayType = MarketHolidayType.FULL_CLOSURE, early_close_time: str = None):
        self.name = name
        self.date = date_obj
        self.holiday_type = holiday_type
        self.early_close_time = early_close_time

def get_easter_date(year: int) -> date:
    """Calculate Easter date using the algorithm"""
    # Easter calculation algorithm
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)

def get_nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """Get the nth occurrence of a weekday in a month (e.g., 3rd Monday)"""
    first_day = date(year, month, 1)
    first_weekday = first_day.weekday()
    
    # Calculate the first occurrence of the target weekday
    days_ahead = weekday - first_weekday
    if days_ahead < 0:
        days_ahead += 7
    
    # Calculate the nth occurrence
    target_date = first_day + timedelta(days=days_ahead + (n-1) * 7)
    
    # Make sure it's still in the same month
    if target_date.month == month:
        return target_date
    else:
        return None

def get_last_weekday(year: int, month: int, weekday: int) -> date:
    """Get the last occurrence of a weekday in a month"""
    # Start from the last day of the month and work backwards
    if month == 12:
        next_month = date(year + 1, 1, 1)
    else:
        next_month = date(year, month + 1, 1)
    
    last_day = next_month - timedelta(days=1)
    
    # Find the last occurrence of the weekday
    days_back = (last_day.weekday() - weekday) % 7
    return last_day - timedelta(days=days_back)

class MarketHolidayCalendar:
    """Comprehensive market holiday calendar for major global exchanges"""
    
    @staticmethod
    def get_japanese_holidays(year: int) -> List[MarketHoliday]:
        """Tokyo Stock Exchange (TSE) holidays"""
        holidays = []
        
        # Fixed holidays
        holidays.extend([
            MarketHoliday("New Year's Day", date(year, 1, 1)),
            MarketHoliday("Bank Holiday (Jan 2)", date(year, 1, 2)),
            MarketHoliday("Bank Holiday (Jan 3)", date(year, 1, 3)),
            MarketHoliday("National Foundation Day", date(year, 2, 11)),
            MarketHoliday("Emperor's Birthday", date(year, 2, 23)),
            MarketHoliday("Showa Day", date(year, 4, 29)),
            MarketHoliday("Constitution Memorial Day", date(year, 5, 3)),
            MarketHoliday("Greenery Day", date(year, 5, 4)),
            MarketHoliday("Children's Day", date(year, 5, 5)),
            MarketHoliday("Culture Day", date(year, 11, 3)),
            MarketHoliday("Labor Thanksgiving Day", date(year, 11, 23)),
        ])
        
        # Variable holidays
        # Coming of Age Day (2nd Monday in January)
        coming_of_age = get_nth_weekday(year, 1, 0, 2)  # 0 = Monday
        if coming_of_age:
            holidays.append(MarketHoliday("Coming of Age Day", coming_of_age))
        
        # Marine Day (3rd Monday in July)
        marine_day = get_nth_weekday(year, 7, 0, 3)
        if marine_day:
            holidays.append(MarketHoliday("Marine Day", marine_day))
        
        # Mountain Day (August 11 or observed)
        mountain_day = date(year, 8, 11)
        if mountain_day.weekday() == 6:  # Sunday
            mountain_day += timedelta(days=1)
        holidays.append(MarketHoliday("Mountain Day", mountain_day))
        
        # Respect for the Aged Day (3rd Monday in September)
        aged_day = get_nth_weekday(year, 9, 0, 3)
        if aged_day:
            holidays.append(MarketHoliday("Respect for the Aged Day", aged_day))
        
        # Health and Sports Day (2nd Monday in October)
        sports_day = get_nth_weekday(year, 10, 0, 2)
        if sports_day:
            holidays.append(MarketHoliday("Health and Sports Day", sports_day))
        
        # Equinoxes (approximate dates)
        vernal_equinox = date(year, 3, 20)  # Approximate
        autumn_equinox = date(year, 9, 23)  # Approximate
        holidays.extend([
            MarketHoliday("Vernal Equinox Day", vernal_equinox),
            MarketHoliday("Autumn Equinox Day", autumn_equinox),
        ])
        
        return holidays
    
    @staticmethod
    def get_australian_holidays(year: int) -> List[MarketHoliday]:
        """Australian Securities Exchange (ASX) holidays"""
        holidays = []
        easter = get_easter_date(year)
        
        # Fixed holidays
        holidays.extend([
            MarketHoliday("New Year's Day", date(year, 1, 1)),
            MarketHoliday("ANZAC Day", date(year, 4, 25)),
            MarketHoliday("Christmas Day", date(year, 12, 25)),
            MarketHoliday("Boxing Day", date(year, 12, 26)),
        ])
        
        # Australia Day (January 26 or observed)
        australia_day = date(year, 1, 26)
        if australia_day.weekday() >= 5:  # Weekend
            # Move to Monday
            australia_day += timedelta(days=(7 - australia_day.weekday()))
        holidays.append(MarketHoliday("Australia Day", australia_day))
        
        # Easter holidays
        good_friday = easter - timedelta(days=2)
        easter_monday = easter + timedelta(days=1)
        holidays.extend([
            MarketHoliday("Good Friday", good_friday),
            MarketHoliday("Easter Monday", easter_monday),
        ])
        
        # King's Birthday (2nd Monday in June)
        kings_birthday = get_nth_weekday(year, 6, 0, 2)
        if kings_birthday:
            holidays.append(MarketHoliday("King's Birthday", kings_birthday))
        
        return holidays
    
    @staticmethod
    def get_uk_holidays(year: int) -> List[MarketHoliday]:
        """London Stock Exchange (LSE) holidays"""
        holidays = []
        easter = get_easter_date(year)
        
        # Fixed holidays
        holidays.extend([
            MarketHoliday("New Year's Day", date(year, 1, 1)),
            MarketHoliday("Christmas Day", date(year, 12, 25)),
            MarketHoliday("Boxing Day", date(year, 12, 26)),
        ])
        
        # Easter holidays
        good_friday = easter - timedelta(days=2)
        easter_monday = easter + timedelta(days=1)
        holidays.extend([
            MarketHoliday("Good Friday", good_friday),
            MarketHoliday("Easter Monday", easter_monday),
        ])
        
        # May Day Bank Holiday (1st Monday in May)
        may_day = get_nth_weekday(year, 5, 0, 1)
        if may_day:
            holidays.append(MarketHoliday("Early May Bank Holiday", may_day))
        
        # Spring Bank Holiday (last Monday in May)
        spring_bank = get_last_weekday(year, 5, 0)
        holidays.append(MarketHoliday("Late May Bank Holiday", spring_bank))
        
        # Summer Bank Holiday (last Monday in August)
        summer_bank = get_last_weekday(year, 8, 0)
        holidays.append(MarketHoliday("Summer Bank Holiday", summer_bank))
        
        return holidays
    
    @staticmethod
    def get_us_holidays(year: int) -> List[MarketHoliday]:
        """NYSE/NASDAQ holidays"""
        holidays = []
        
        # Fixed holidays
        holidays.extend([
            MarketHoliday("New Year's Day", date(year, 1, 1)),
            MarketHoliday("Independence Day", date(year, 7, 4)),
            MarketHoliday("Christmas Day", date(year, 12, 25)),
        ])
        
        # Martin Luther King Jr. Day (3rd Monday in January)
        mlk_day = get_nth_weekday(year, 1, 0, 3)
        if mlk_day:
            holidays.append(MarketHoliday("Martin Luther King Jr. Day", mlk_day))
        
        # Presidents' Day (3rd Monday in February)
        presidents_day = get_nth_weekday(year, 2, 0, 3)
        if presidents_day:
            holidays.append(MarketHoliday("Presidents' Day", presidents_day))
        
        # Good Friday
        easter = get_easter_date(year)
        good_friday = easter - timedelta(days=2)
        holidays.append(MarketHoliday("Good Friday", good_friday))
        
        # Memorial Day (last Monday in May)
        memorial_day = get_last_weekday(year, 5, 0)
        holidays.append(MarketHoliday("Memorial Day", memorial_day))
        
        # Juneteenth (June 19)
        holidays.append(MarketHoliday("Juneteenth National Independence Day", date(year, 6, 19)))
        
        # Labor Day (1st Monday in September)
        labor_day = get_nth_weekday(year, 9, 0, 1)
        if labor_day:
            holidays.append(MarketHoliday("Labor Day", labor_day))
        
        # Thanksgiving (4th Thursday in November)
        thanksgiving = get_nth_weekday(year, 11, 3, 4)  # 3 = Thursday
        if thanksgiving:
            holidays.append(MarketHoliday("Thanksgiving Day", thanksgiving))
            # Day after Thanksgiving (early close)
            black_friday = thanksgiving + timedelta(days=1)
            holidays.append(MarketHoliday("Day after Thanksgiving", black_friday, MarketHolidayType.EARLY_CLOSE, "13:00"))
        
        # Early close days
        # Day before Independence Day (if July 4 is not Monday)
        july_4 = date(year, 7, 4)
        if july_4.weekday() != 0:  # Not Monday
            day_before_july4 = july_4 - timedelta(days=1)
            if day_before_july4.weekday() < 5:  # Weekday
                holidays.append(MarketHoliday("Day before Independence Day", day_before_july4, MarketHolidayType.EARLY_CLOSE, "13:00"))
        
        # Christmas Eve (if not weekend)
        christmas_eve = date(year, 12, 24)
        if christmas_eve.weekday() < 5:  # Weekday
            holidays.append(MarketHoliday("Christmas Eve", christmas_eve, MarketHolidayType.EARLY_CLOSE, "13:00"))
        
        return holidays
    
    @staticmethod
    def get_market_holidays(market: str, year: int) -> List[MarketHoliday]:
        """Get holidays for a specific market"""
        market_holiday_map = {
            "Japan": MarketHolidayCalendar.get_japanese_holidays,
            "Australia": MarketHolidayCalendar.get_australian_holidays,
            "UK": MarketHolidayCalendar.get_uk_holidays,
            "US": MarketHolidayCalendar.get_us_holidays,
        }
        
        if market in market_holiday_map:
            return market_holiday_map[market](year)
        else:
            return []
    
    @staticmethod
    def is_market_holiday(market: str, check_date: date) -> Optional[MarketHoliday]:
        """Check if a specific date is a market holiday"""
        holidays = MarketHolidayCalendar.get_market_holidays(market, check_date.year)
        for holiday in holidays:
            if holiday.date == check_date:
                return holiday
        return None
    
    @staticmethod
    def get_next_market_holiday(market: str, from_date: date = None) -> Optional[MarketHoliday]:
        """Get the next market holiday for a given market"""
        if from_date is None:
            from_date = date.today()
        
        # Check current year and next year
        for year in [from_date.year, from_date.year + 1]:
            holidays = MarketHolidayCalendar.get_market_holidays(market, year)
            upcoming_holidays = [h for h in holidays if h.date > from_date]
            if upcoming_holidays:
                return min(upcoming_holidays, key=lambda h: h.date)
        
        return None
    
    @staticmethod
    def get_market_status_with_holidays(market: str, current_datetime: datetime) -> Dict:
        """Get comprehensive market status including holiday information"""
        current_date = current_datetime.date()
        current_hour = current_datetime.hour
        
        # Check if today is a holiday
        today_holiday = MarketHolidayCalendar.is_market_holiday(market, current_date)
        
        # Get next holiday
        next_holiday = MarketHolidayCalendar.get_next_market_holiday(market, current_date)
        
        # Base market hours (this should come from your existing MARKET_HOURS)
        market_hours_map = {
            "Japan": {"open": 0, "close": 6},
            "Australia": {"open": 0, "close": 6}, 
            "UK": {"open": 8, "close": 16},
            "US": {"open": 14, "close": 21},
        }
        
        base_hours = market_hours_map.get(market, {"open": 0, "close": 23})
        
        # Determine market status
        if today_holiday:
            if today_holiday.holiday_type == MarketHolidayType.FULL_CLOSURE:
                status = "HOLIDAY"
                is_open = False
                next_event = "Reopens"
                # Calculate next trading day
                next_date = current_date + timedelta(days=1)
                while (next_date.weekday() >= 5 or  # Weekend
                       MarketHolidayCalendar.is_market_holiday(market, next_date)):
                    next_date += timedelta(days=1)
                next_time = f"{base_hours['open']:02d}:00 UTC"
            elif today_holiday.holiday_type == MarketHolidayType.EARLY_CLOSE:
                # Check if we're past the early close time
                early_close_hour = int(today_holiday.early_close_time.split(':')[0])
                if current_hour >= early_close_hour:
                    status = "CLOSED (Early)"
                    is_open = False
                    next_event = "Opens"
                    next_time = f"next trading day at {base_hours['open']:02d}:00 UTC"
                else:
                    status = f"OPEN (Early close at {today_holiday.early_close_time})"
                    is_open = True
                    next_event = "Early close"
                    next_time = today_holiday.early_close_time + " UTC"
            else:
                # Normal market hours logic for other special cases
                is_open = base_hours["open"] <= current_hour <= base_hours["close"]
                status = "OPEN" if is_open else "CLOSED"
                next_event = "Closes" if is_open else "Opens"
                if is_open:
                    next_time = f"{base_hours['close']:02d}:00 UTC"
                else:
                    next_time = f"{base_hours['open']:02d}:00 UTC"
        else:
            # Weekend check
            if current_datetime.weekday() >= 5:  # Weekend
                status = "WEEKEND"
                is_open = False
                next_event = "Opens"
                # Find next Monday (or next trading day if Monday is a holiday)
                next_date = current_date
                while (next_date.weekday() >= 5 or
                       MarketHolidayCalendar.is_market_holiday(market, next_date)):
                    next_date += timedelta(days=1)
                next_time = f"Monday at {base_hours['open']:02d}:00 UTC"
            else:
                # Normal trading day logic
                is_open = base_hours["open"] <= current_hour <= base_hours["close"]
                status = "OPEN" if is_open else "CLOSED"
                next_event = "Closes" if is_open else "Opens"
                if is_open:
                    next_time = f"{base_hours['close']:02d}:00 UTC"
                else:
                    next_time = f"{base_hours['open']:02d}:00 UTC"
        
        result = {
            "market": market,
            "is_open": is_open,
            "status": status,
            "hours_utc": f"{base_hours['open']:02d}:00-{base_hours['close']:02d}:00",
            "next_event": next_event,
            "next_time": next_time,
            "current_date": current_date.strftime("%Y-%m-%d"),
            "current_time_utc": current_datetime.strftime("%H:%M:%S")
        }
        
        # Add holiday information
        if today_holiday:
            result["today_holiday"] = {
                "name": today_holiday.name,
                "type": today_holiday.holiday_type.value,
                "early_close_time": today_holiday.early_close_time
            }
        
        if next_holiday:
            result["next_holiday"] = {
                "name": next_holiday.name,
                "date": next_holiday.date.strftime("%Y-%m-%d"),
                "days_until": (next_holiday.date - current_date).days,
                "type": next_holiday.holiday_type.value
            }
        
        return result

def get_all_market_holidays_summary(year: int) -> Dict:
    """Get a summary of all market holidays for the year"""
    markets = ["Japan", "Australia", "UK", "US"]
    summary = {}
    
    for market in markets:
        holidays = MarketHolidayCalendar.get_market_holidays(market, year)
        summary[market] = [
            {
                "name": h.name,
                "date": h.date.strftime("%Y-%m-%d"),
                "type": h.holiday_type.value,
                "early_close_time": h.early_close_time
            }
            for h in sorted(holidays, key=lambda x: x.date)
        ]
    
    return summary