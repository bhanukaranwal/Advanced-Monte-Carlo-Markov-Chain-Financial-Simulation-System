"""
High-performance stream processing for real-time market data
"""
import asyncio
import threading
import queue
import time
import json
import logging
from typing import Dict, List, Optional, Callable, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import websockets
import pandas as pd
import numpy as np
from collections import deque, defaultdict
import redis
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import zmq
import pickle

logger = logging.getLogger(__name__)

@dataclass
class MarketTick:
    """Real-time market tick data"""
    symbol: str
    timestamp: datetime
    price: float
    volume: float
    bid: Optional[float] = None
    ask: Optional[float] = None
    bid_size: Optional[float] = None
    ask_size: Optional[float] = None
    trade_type: Optional[str] = None  # 'buy', 'sell', 'unknown'
    exchange: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        d = asdict(self)
        d['timestamp'] = self.timestamp.isoformat()
        return d
        
    @classmethod
    def from_dict(cls, data: Dict) -> 'MarketTick':
        """Create from dictionary"""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)

@dataclass
class AggregatedData:
    """Aggregated market data over time window"""
    symbol: str
    start_time: datetime
    end_time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: float
    num_trades: int
    
class MarketDataStream:
    """Real-time market data stream processor"""
    
    def __init__(
        self,
        symbols: List[str],
        data_sources: List[str] = ['websocket'],
        buffer_size: int = 10000,
        redis_config: Optional[Dict] = None
    ):
        self.symbols = symbols
        self.data_sources = data_sources
        self.buffer_size = buffer_size
        
        # Data buffers
        self.tick_buffer = deque(maxlen=buffer_size)
        self.symbol_buffers = {symbol: deque(maxlen=1000) for symbol in symbols}
        
        # Redis connection for persistence
        self.redis_client = None
        if redis_config:
            self.redis_client = redis.Redis(**redis_config)
            
        # Subscribers and callbacks
        self.subscribers = []
        self.tick_callbacks = []
        self.aggregation_callbacks = []
        
        # Processing state
        self.running = False
        self.processing_thread = None
        self.websocket_tasks = []
        
        # Statistics
        self.stats = {
            'total_ticks': 0,
            'ticks_per_second': 0,
            'processing_latency_ms': 0,
            'last_update': datetime.now()
        }
        
    def add_tick_callback(self, callback: Callable[[MarketTick], None]):
        """Add callback for individual ticks"""
        self.tick_callbacks.append(callback)
        
    def add_aggregation_callback(self, callback: Callable[[AggregatedData], None]):
        """Add callback for aggregated data"""
        self.aggregation_callbacks.append(callback)
        
    async def start_streaming(self):
        """Start real-time data streaming"""
        logger.info(f"Starting market data stream for {len(self.symbols)} symbols")
        self.running = True
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.start()
        
        # Start WebSocket connections
        if 'websocket' in self.data_sources:
            await self._start_websocket_streams()
            
        # Start other data sources
        if 'rest_api' in self.data_sources:
            await self._start_rest_api_polling()
            
    async def _start_websocket_streams(self):
        """Start WebSocket connections for real-time data"""
        for symbol in self.symbols:
            task = asyncio.create_task(self._websocket_handler(symbol))
            self.websocket_tasks.append(task)
            
        # Wait for all WebSocket connections
        if self.websocket_tasks:
            await asyncio.gather(*self.websocket_tasks, return_exceptions=True)
            
    async def _websocket_handler(self, symbol: str):
        """Handle WebSocket connection for a specific symbol"""
        # Example WebSocket URL (would be specific to data provider)
        ws_url = f"wss://stream.example.com/ws/{symbol.lower()}"
        
        retry_count = 0
        max_retries = 5
        
        while self.running and retry_count < max_retries:
            try:
                async with websockets.connect(ws_url) as websocket:
                    logger.info(f"Connected to WebSocket for {symbol}")
                    retry_count = 0  # Reset on successful connection
                    
                    async for message in websocket:
                        if not self.running:
                            break
                            
                        try:
                            # Parse message (format depends on provider)
                            data = json.loads(message)
                            tick = self._parse_websocket_message(symbol, data)
                            
                            if tick:
                                await self._process_tick(tick)
                                
                        except json.JSONDecodeError:
                            logger.warning(f"Failed to parse WebSocket message for {symbol}")
                        except Exception as e:
                            logger.error(f"Error processing WebSocket message for {symbol}: {e}")
                            
            except websockets.exceptions.ConnectionClosed:
                retry_count += 1
                wait_time = min(2 ** retry_count, 30)  # Exponential backoff
                logger.warning(f"WebSocket connection lost for {symbol}, retrying in {wait_time}s")
                await asyncio.sleep(wait_time)
                
            except Exception as e:
                logger.error(f"WebSocket error for {symbol}: {e}")
                retry_count += 1
                await asyncio.sleep(5)
                
    def _parse_websocket_message(self, symbol: str, data: Dict) -> Optional[MarketTick]:
        """Parse WebSocket message into MarketTick"""
        try:
            # Example parsing (format depends on data provider)
            tick = MarketTick(
                symbol=symbol,
                timestamp=datetime.fromtimestamp(data.get('timestamp', time.time())),
                price=float(data.get('price', 0)),
                volume=float(data.get('volume', 0)),
                bid=data.get('bid'),
                ask=data.get('ask'),
                bid_size=data.get('bid_size'),
                ask_size=data.get('ask_size'),
                trade_type=data.get('trade_type'),
                exchange=data.get('exchange')
            )
            return tick
            
        except (KeyError, ValueError, TypeError) as e:
            logger.warning(f"Failed to parse WebSocket data for {symbol}: {e}")
            return None
            
    async def _start_rest_api_polling(self):
        """Start REST API polling for data sources that don't support WebSocket"""
        polling_tasks = []
        
        for symbol in self.symbols:
            task = asyncio.create_task(self._rest_api_poller(symbol))
            polling_tasks.append(task)
            
        if polling_tasks:
            await asyncio.gather(*polling_tasks, return_exceptions=True)
            
    async def _rest_api_poller(self, symbol: str, interval: int = 1):
        """Poll REST API for symbol data"""
        url = f"https://api.example.com/ticker/{symbol}"
        
        async with aiohttp.ClientSession() as session:
            while self.running:
                try:
                    async with session.get(url) as response:
                        if response.status == 200:
                            data = await response.json()
                            tick = self._parse_rest_api_data(symbol, data)
                            
                            if tick:
                                await self._process_tick(tick)
                                
                except Exception as e:
                    logger.error(f"REST API error for {symbol}: {e}")
                    
                await asyncio.sleep(interval)
                
    def _parse_rest_api_data(self, symbol: str, data: Dict) -> Optional[MarketTick]:
        """Parse REST API data into MarketTick"""
        try:
            tick = MarketTick(
                symbol=symbol,
                timestamp=datetime.now(),
                price=float(data.get('last_price', 0)),
                volume=float(data.get('volume_24h', 0)),
                bid=data.get('bid'),
                ask=data.get('ask')
            )
            return tick
            
        except (KeyError, ValueError) as e:
            logger.warning(f"Failed to parse REST API data for {symbol}: {e}")
            return None
            
    async def _process_tick(self, tick: MarketTick):
        """Process incoming tick data"""
        # Add to buffers
        self.tick_buffer.append(tick)
        self.symbol_buffers[tick.symbol].append(tick)
        
        # Update statistics
        self.stats['total_ticks'] += 1
        
        # Store in Redis if available
        if self.redis_client:
            try:
                key = f"tick:{tick.symbol}:{tick.timestamp.isoformat()}"
                self.redis_client.setex(key, 300, pickle.dumps(tick))  # 5 minute expiry
            except Exception as e:
                logger.warning(f"Failed to store tick in Redis: {e}")
                
        # Trigger callbacks asynchronously
        for callback in self.tick_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    asyncio.create_task(callback(tick))
                else:
                    # Run in thread pool for sync callbacks
                    loop = asyncio.get_event_loop()
                    loop.run_in_executor(None, callback, tick)
            except Exception as e:
                logger.error(f"Error in tick callback: {e}")
                
    def _processing_loop(self):
        """Background processing loop"""
        logger.info("Starting data processing loop")
        
        last_stats_update = time.time()
        tick_count_since_last_update = 0
        
        while self.running:
            try:
                current_time = time.time()
                
                # Update statistics
                if current_time - last_stats_update >= 1.0:  # Every second
                    self.stats['ticks_per_second'] = tick_count_since_last_update
                    self.stats['last_update'] = datetime.now()
                    
                    tick_count_since_last_update = 0
                    last_stats_update = current_time
                    
                # Process aggregations
                self._process_aggregations()
                
                # Cleanup old data
                self._cleanup_buffers()
                
                time.sleep(0.1)  # Small delay to prevent busy waiting
                
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                time.sleep(1)
                
        logger.info("Data processing loop stopped")
        
    def _process_aggregations(self):
        """Process data aggregations"""
        current_time = datetime.now()
        
        for symbol in self.symbols:
            symbol_buffer = self.symbol_buffers[symbol]
            
            if len(symbol_buffer) == 0:
                continue
                
            # 1-minute aggregations
            minute_start = current_time.replace(second=0, microsecond=0)
            minute_data = [
                tick for tick in symbol_buffer 
                if minute_start <= tick.timestamp < minute_start + timedelta(minutes=1)
            ]
            
            if len(minute_data) >= 2:  # Need at least 2 ticks for meaningful aggregation
                agg_data = self._create_aggregation(symbol, minute_data, minute_start)
                
                # Trigger aggregation callbacks
                for callback in self.aggregation_callbacks:
                    try:
                        callback(agg_data)
                    except Exception as e:
                        logger.error(f"Error in aggregation callback: {e}")
                        
    def _create_aggregation(
        self, 
        symbol: str, 
        ticks: List[MarketTick], 
        start_time: datetime
    ) -> AggregatedData:
        """Create aggregated data from ticks"""
        prices = [tick.price for tick in ticks]
        volumes = [tick.volume for tick in ticks]
        
        # OHLC calculation
        open_price = ticks[0].price
        close_price = ticks[-1].price
        high_price = max(prices)
        low_price = min(prices)
        
        # Volume and VWAP
        total_volume = sum(volumes)
        vwap = sum(p * v for p, v in zip(prices, volumes)) / total_volume if total_volume > 0 else close_price
        
        return AggregatedData(
            symbol=symbol,
            start_time=start_time,
            end_time=start_time + timedelta(minutes=1),
            open=open_price,
            high=high_price,
            low=low_price,
            close=close_price,
            volume=total_volume,
            vwap=vwap,
            num_trades=len(ticks)
        )
        
    def _cleanup_buffers(self):
        """Remove old data from buffers"""
        cutoff_time = datetime.now() - timedelta(hours=1)
        
        # Clean symbol buffers
        for symbol_buffer in self.symbol_buffers.values():
            while symbol_buffer and symbol_buffer[0].timestamp < cutoff_time:
                symbol_buffer.popleft()
                
    def get_latest_tick(self, symbol: str) -> Optional[MarketTick]:
        """Get latest tick for symbol"""
        symbol_buffer = self.symbol_buffers.get(symbol)
        return symbol_buffer[-1] if symbol_buffer else None
        
    def get_recent_ticks(
        self, 
        symbol: str, 
        minutes: int = 5
    ) -> List[MarketTick]:
        """Get recent ticks for symbol"""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        symbol_buffer = self.symbol_buffers.get(symbol, deque())
        
        return [
            tick for tick in symbol_buffer 
            if tick.timestamp >= cutoff_time
        ]
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get streaming statistics"""
        return self.stats.copy()
        
    async def stop_streaming(self):
        """Stop data streaming"""
        logger.info("Stopping market data stream")
        self.running = False
        
        # Cancel WebSocket tasks
        for task in self.websocket_tasks:
            task.cancel()
            
        # Wait for processing thread to finish
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=5)
            
        logger.info("Market data stream stopped")

class StreamProcessor:
    """High-level stream processing orchestrator"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.streams = {}
        self.analyzers = {}
        self.running = False
        
    def create_stream(
        self,
        stream_name: str,
        symbols: List[str],
        **kwargs
    ) -> MarketDataStream:
        """Create a new market data stream"""
        stream = MarketDataStream(symbols, **kwargs)
        self.streams[stream_name] = stream
        return stream
        
    def add_analyzer(
        self,
        analyzer_name: str,
        analyzer: 'RealTimeAnalyzer'
    ):
        """Add real-time analyzer"""
        self.analyzers[analyzer_name] = analyzer
        
    async def start_all_streams(self):
        """Start all configured streams"""
        self.running = True
        
        tasks = []
        for stream_name, stream in self.streams.items():
            task = asyncio.create_task(stream.start_streaming())
            tasks.append(task)
            
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
            
    async def stop_all_streams(self):
        """Stop all streams"""
        self.running = False
        
        for stream in self.streams.values():
            await stream.stop_streaming()

class RealTimeAnalyzer:
    """Real-time market data analyzer"""
    
    def __init__(self, analysis_functions: List[Callable] = None):
        self.analysis_functions = analysis_functions or []
        self.results_buffer = deque(maxlen=1000)
        
    def add_analysis_function(self, func: Callable):
        """Add analysis function"""
        self.analysis_functions.append(func)
        
    async def analyze_tick(self, tick: MarketTick):
        """Analyze individual tick"""
        results = {}
        
        for func in self.analysis_functions:
            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(tick)
                else:
                    result = func(tick)
                    
                results[func.__name__] = result
                
            except Exception as e:
                logger.error(f"Error in analysis function {func.__name__}: {e}")
                
        if results:
            self.results_buffer.append({
                'timestamp': tick.timestamp,
                'symbol': tick.symbol,
                'results': results
            })
            
        return results
        
    def get_recent_results(self, minutes: int = 5) -> List[Dict]:
        """Get recent analysis results"""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        
        return [
            result for result in self.results_buffer
            if result['timestamp'] >= cutoff_time
        ]

# Example usage and testing
if __name__ == "__main__":
    print("Testing Stream Processing...")
    
    async def test_stream_processor():
        # Test configuration
        symbols = ['AAPL', 'GOOGL', 'MSFT']
        
        # Create stream processor
        config = {'buffer_size': 5000}
        processor = StreamProcessor(config)
        
        # Create market data stream
        stream = processor.create_stream(
            'test_stream',
            symbols,
            data_sources=['websocket'],  # Would need real WebSocket URLs
            buffer_size=1000
        )
        
        # Add tick callback for testing
        def tick_callback(tick: MarketTick):
            print(f"Received tick: {tick.symbol} @ {tick.price} [{tick.timestamp}]")
            
        stream.add_tick_callback(tick_callback)
        
        # Add aggregation callback
        def aggregation_callback(agg_data: AggregatedData):
            print(f"Aggregation: {agg_data.symbol} OHLC=({agg_data.open:.2f}, "
                  f"{agg_data.high:.2f}, {agg_data.low:.2f}, {agg_data.close:.2f})")
                  
        stream.add_aggreg
              stream.add_aggregation_callback(aggregation_callback)
        
        # Create real-time analyzer
        def momentum_analysis(tick: MarketTick) -> float:
            """Simple momentum indicator"""
            return tick.price * tick.volume if tick.volume else 0
            
        analyzer = RealTimeAnalyzer([momentum_analysis])
        processor.add_analyzer('momentum_analyzer', analyzer)
        
        # Add analyzer to stream
        stream.add_tick_callback(analyzer.analyze_tick)
        
        print("Stream processor setup completed")
        
        # In a real implementation, this would connect to actual data sources
        # For testing, we'll simulate some ticks
        for i in range(10):
            test_tick = MarketTick(
                symbol='AAPL',
                timestamp=datetime.now(),
                price=150.0 + np.random.normal(0, 1),
                volume=1000 + np.random.randint(0, 500)
            )
            await stream._process_tick(test_tick)
            await asyncio.sleep(0.1)
            
        # Get statistics
        stats = stream.get_statistics()
        print(f"Stream statistics: {stats}")
        
        print("Stream processing test completed!")
    
    # Run test
    asyncio.run(test_stream_processor())

