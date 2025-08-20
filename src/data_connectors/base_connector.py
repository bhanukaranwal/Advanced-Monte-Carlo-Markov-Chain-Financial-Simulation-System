"""
Base class for all market data connectors
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, AsyncIterator
import pandas as pd
from datetime import datetime, timedelta
import asyncio
import logging

logger = logging.getLogger(__name__)

class BaseDataConnector(ABC):
    """Abstract base class for market data connectors"""
    
    def __init__(self, api_key: Optional[str] = None, rate_limit: float = 1.0):
        self.api_key = api_key
        self.rate_limit = rate_limit  # requests per second
        self.last_request_time = 0.0
        
    @abstractmethod
    async def get_historical_data(
        self, 
        symbol: str, 
        start_date: datetime, 
        end_date: datetime,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """Get historical market data"""
        pass
        
    @abstractmethod
    async def get_real_time_quote(self, symbol: str) -> Dict[str, Any]:
        """Get real-time quote for a symbol"""
        pass
        
    @abstractmethod
    async def subscribe_real_time(self, symbols: List[str]) -> AsyncIterator[Dict[str, Any]]:
        """Subscribe to real-time data stream"""
        pass
        
    async def _rate_limit_check(self):
        """Enforce rate limiting"""
        current_time = asyncio.get_event_loop().time()
        time_since_last = current_time - self.last_request_time
        min_interval = 1.0 / self.rate_limit
        
        if time_since_last < min_interval:
            await asyncio.sleep(min_interval - time_since_last)
            
        self.last_request_time = asyncio.get_event_loop().time()
        
    def validate_symbol(self, symbol: str) -> bool:
        """Validate symbol format"""
        return isinstance(symbol, str) and len(symbol) > 0
        
    def standardize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize DataFrame format"""
        # Ensure consistent column names
        column_mapping = {
            'Open': 'open',
            'High': 'high', 
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume',
            'Adj Close': 'adj_close'
        }
        
        df = df.rename(columns=column_mapping)
        
        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
            
        return df
