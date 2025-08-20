"""
Real-time Engine for Live Market Data Processing and Analytics

Stream processing, Kalman filters, and real-time analytics for financial markets
"""

from .stream_processor import StreamProcessor, MarketDataStream, RealTimeAnalyzer
from .kalman_filters import KalmanFilter, ExtendedKalmanFilter, UnscentedKalmanFilter
from .real_time_analytics import RealTimeRiskAnalytics, RealTimePortfolioManager

__all__ = [
    "StreamProcessor",
    "MarketDataStream",
    "RealTimeAnalyzer",
    "KalmanFilter",
    "ExtendedKalmanFilter", 
    "UnscentedKalmanFilter",
    "RealTimeRiskAnalytics",
    "RealTimePortfolioManager"
]
