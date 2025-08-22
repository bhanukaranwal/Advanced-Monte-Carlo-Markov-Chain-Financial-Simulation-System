"""
Data management package
"""

from .connectors import MarketDataConnector, YahooFinanceConnector
from .processors import DataProcessor, ReturnCalculator

__all__ = [
    'MarketDataConnector',
    'YahooFinanceConnector', 
    'DataProcessor',
    'ReturnCalculator'
]
