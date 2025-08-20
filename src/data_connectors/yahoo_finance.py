"""
Yahoo Finance data connector
"""

import yfinance as yf
import pandas as pd
from typing import Dict, List, Optional, Any, AsyncIterator
from datetime import datetime
import asyncio
import aiohttp
from .base_connector import BaseDataConnector

class YahooFinanceConnector(BaseDataConnector):
    """Yahoo Finance data connector"""
    
    def __init__(self):
        super().__init__(rate_limit=2.0)  # Yahoo allows ~2 requests/second
        
    async def get_historical_data(
        self, 
        symbol: str, 
        start_date: datetime, 
        end_date: datetime,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """Get historical data from Yahoo Finance"""
        await self._rate_limit_check()
        
        if not self.validate_symbol(symbol):
            raise ValueError(f"Invalid symbol: {symbol}")
            
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(
                start=start_date,
                end=end_date,
                interval=interval,
                auto_adjust=True,
                prepost=True
            )
            
            if data.empty:
                raise ValueError(f"No data found for symbol {symbol}")
                
            return self.standardize_dataframe(data)
            
        except Exception as e:
            logger.error(f"Error fetching Yahoo Finance data for {symbol}: {e}")
            raise
            
    async def get_real_time_quote(self, symbol: str) -> Dict[str, Any]:
        """Get real-time quote from Yahoo Finance"""
        await self._rate_limit_check()
        
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            return {
                'symbol': symbol,
                'price': info.get('currentPrice', info.get('regularMarketPrice')),
                'change': info.get('regularMarketChange'),
                'change_percent': info.get('regularMarketChangePercent'),
                'volume': info.get('regularMarketVolume'),
                'timestamp': datetime.now(),
                'source': 'yahoo_finance'
            }
            
        except Exception as e:
            logger.error(f"Error fetching real-time quote for {symbol}: {e}")
            raise
            
    async def subscribe_real_time(self, symbols: List[str]) -> AsyncIterator[Dict[str, Any]]:
        """Subscribe to real-time data (polling-based for Yahoo)"""
        while True:
            for symbol in symbols:
                try:
                    quote = await self.get_real_time_quote(symbol)
                    yield quote
                except Exception as e:
                    logger.error(f"Error in real-time stream for {symbol}: {e}")
                    
            await asyncio.sleep(1.0)  # Poll every second
            
    async def get_multiple_quotes(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """Get quotes for multiple symbols efficiently"""
        results = {}
        
        # Process in batches to respect rate limits
        batch_size = 5
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            tasks = [self.get_real_time_quote(symbol) for symbol in batch]
            
            try:
                quotes = await asyncio.gather(*tasks, return_exceptions=True)
                for symbol, quote in zip(batch, quotes):
                    if isinstance(quote, Exception):
                        logger.error(f"Error fetching quote for {symbol}: {quote}")
                        results[symbol] = None
                    else:
                        results[symbol] = quote
            except Exception as e:
                logger.error(f"Batch quote error: {e}")
                
        return results
