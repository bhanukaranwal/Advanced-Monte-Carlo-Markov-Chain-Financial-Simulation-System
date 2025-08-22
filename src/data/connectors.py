"""
Market data connectors
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import logging
import asyncio
import aiohttp
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class BaseDataConnector(ABC):
    """Base class for data connectors"""
    
    @abstractmethod
    async def get_price_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Get price data for symbol"""
        pass
        
    @abstractmethod
    async def get_current_price(self, symbol: str) -> float:
        """Get current price for symbol"""
        pass

class YahooFinanceConnector(BaseDataConnector):
    """Yahoo Finance data connector"""
    
    def __init__(self):
        self.session = None
        
    async def get_price_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Get historical price data from Yahoo Finance"""
        
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date)
            
            if data.empty:
                logger.warning(f"No data found for symbol {symbol}")
                return pd.DataFrame()
                
            # Standardize column names
            data = data.rename(columns={
                'Open': 'open',
                'High': 'high', 
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            
            data['symbol'] = symbol
            data.reset_index(inplace=True)
            data['date'] = data['Date']
            
            return data[['date', 'symbol', 'open', 'high', 'low', 'close', 'volume']]
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()
            
    async def get_current_price(self, symbol: str) -> float:
        """Get current price"""
        
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            return float(info.get('regularMarketPrice', 0))
            
        except Exception as e:
            logger.error(f"Error fetching current price for {symbol}: {e}")
            return 0.0
            
    async def get_multiple_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Get current prices for multiple symbols"""
        
        prices = {}
        
        for symbol in symbols:
            price = await self.get_current_price(symbol)
            prices[symbol] = price
            
        return prices
        
    async def get_market_data_bulk(self, symbols: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """Get bulk market data for multiple symbols"""
        
        data = {}
        
        tasks = [
            self.get_price_data(symbol, start_date, end_date) 
            for symbol in symbols
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for symbol, result in zip(symbols, results):
            if isinstance(result, Exception):
                logger.error(f"Error fetching data for {symbol}: {result}")
                data[symbol] = pd.DataFrame()
            else:
                data[symbol] = result
                
        return data

class AlphaVantageConnector(BaseDataConnector):
    """Alpha Vantage data connector"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"
        
    async def get_price_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Get price data from Alpha Vantage"""
        
        params = {
            'function': 'TIME_SERIES_DAILY_ADJUSTED',
            'symbol': symbol,
            'apikey': self.api_key,
            'datatype': 'json',
            'outputsize': 'full'
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(self.base_url, params=params) as response:
                    data = await response.json()
                    
                if 'Time Series (Daily)' not in data:
                    logger.error(f"Invalid response for {symbol}")
                    return pd.DataFrame()
                    
                time_series = data['Time Series (Daily)']
                
                # Convert to DataFrame
                df_data = []
                for date, values in time_series.items():
                    df_data.append({
                        'date': pd.to_datetime(date),
                        'symbol': symbol,
                        'open': float(values['1. open']),
                        'high': float(values['2. high']),
                        'low': float(values['3. low']),
                        'close': float(values['4. close']),
                        'volume': int(values['6. volume'])
                    })
                    
                df = pd.DataFrame(df_data)
                df = df.sort_values('date')
                
                # Filter by date range
                start_dt = pd.to_datetime(start_date)
                end_dt = pd.to_datetime(end_date)
                df = df[(df['date'] >= start_dt) & (df['date'] <= end_dt)]
                
                return df
                
            except Exception as e:
                logger.error(f"Error fetching Alpha Vantage data for {symbol}: {e}")
                return pd.DataFrame()
                
    async def get_current_price(self, symbol: str) -> float:
        """Get current price from Alpha Vantage"""
        
        params = {
            'function': 'GLOBAL_QUOTE',
            'symbol': symbol,
            'apikey': self.api_key
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(self.base_url, params=params) as response:
                    data = await response.json()
                    
                if 'Global Quote' in data:
                    return float(data['Global Quote']['05. price'])
                    
                return 0.0
                
            except Exception as e:
                logger.error(f"Error fetching current price for {symbol}: {e}")
                return 0.0

class CryptoDataConnector:
    """Cryptocurrency data connector"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.base_url = "https://api.coingecko.com/api/v3"
        
    async def get_crypto_price_data(self, symbol: str, days: int = 365) -> pd.DataFrame:
        """Get cryptocurrency price data"""
        
        # Convert symbol to CoinGecko format
        symbol_map = {
            'BTC': 'bitcoin',
            'ETH': 'ethereum', 
            'ADA': 'cardano',
            'SOL': 'solana'
        }
        
        coin_id = symbol_map.get(symbol.upper(), symbol.lower())
        
        params = {
            'vs_currency': 'usd',
            'days': days,
            'interval': 'daily'
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                url = f"{self.base_url}/coins/{coin_id}/market_chart"
                async with session.get(url, params=params) as response:
                    data = await response.json()
                    
                if 'prices' not in data:
                    logger.error(f"Invalid response for {symbol}")
                    return pd.DataFrame()
                    
                # Convert to DataFrame
                prices = data['prices']
                df_data = []
                
                for timestamp, price in prices:
                    df_data.append({
                        'date': pd.to_datetime(timestamp, unit='ms'),
                        'symbol': symbol.upper(),
                        'close': price,
                        'open': price,  # CoinGecko doesn't provide OHLC for free tier
                        'high': price,
                        'low': price,
                        'volume': 0
                    })
                    
                return pd.DataFrame(df_data)
                
            except Exception as e:
                logger.error(f"Error fetching crypto data for {symbol}: {e}")
                return pd.DataFrame()

class MarketDataConnector:
    """Main market data connector that aggregates multiple sources"""
    
    def __init__(self, yahoo_finance=True, alpha_vantage_key=None, crypto_data=True):
        self.connectors = {}
        
        if yahoo_finance:
            self.connectors['yahoo'] = YahooFinanceConnector()
            
        if alpha_vantage_key:
            self.connectors['alphavantage'] = AlphaVantageConnector(alpha_vantage_key)
            
        if crypto_data:
            self.connectors['crypto'] = CryptoDataConnector()
            
        self.primary_connector = 'yahoo'  # Default primary connector
        
    async def get_price_data(self, symbol: str, start_date: str, end_date: str, 
                           source: Optional[str] = None) -> pd.DataFrame:
        """Get price data with fallback sources"""
        
        if source and source in self.connectors:
            return await self.connectors[source].get_price_data(symbol, start_date, end_date)
            
        # Try primary connector first
        if self.primary_connector in self.connectors:
            data = await self.connectors[self.primary_connector].get_price_data(
                symbol, start_date, end_date
            )
            
            if not data.empty:
                return data
                
        # Fallback to other connectors
        for name, connector in self.connectors.items():
            if name != self.primary_connector:
                try:
                    data = await connector.get_price_data(symbol, start_date, end_date)
                    if not data.empty:
                        logger.info(f"Using fallback connector {name} for {symbol}")
                        return data
                except Exception as e:
                    logger.warning(f"Fallback connector {name} failed for {symbol}: {e}")
                    continue
                    
        logger.error(f"All connectors failed for symbol {symbol}")
        return pd.DataFrame()
        
    async def get_current_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Get current prices for multiple symbols"""
        
        if 'yahoo' in self.connectors:
            return await self.connectors['yahoo'].get_multiple_prices(symbols)
            
        # Fallback to individual calls
        prices = {}
        for symbol in symbols:
            for connector in self.connectors.values():
                try:
                    price = await connector.get_current_price(symbol)
                    if price > 0:
                        prices[symbol] = price
                        break
                except Exception:
                    continue
                    
        return prices
