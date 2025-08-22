"""
Real-time data stream processing
"""

import asyncio
import logging
from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import json
import websocket
from collections import deque
import threading
import time

logger = logging.getLogger(__name__)

@dataclass
class MarketTick:
    """Market data tick"""
    symbol: str
    timestamp: datetime
    price: float
    volume: int
    bid: Optional[float] = None
    ask: Optional[float] = None

class MarketDataStream:
    """Market data streaming client"""
    
    def __init__(self, symbols: List[str], api_key: Optional[str] = None):
        self.symbols = symbols
        self.api_key = api_key
        self.ws = None
        self.is_connected = False
        self.subscribers = []
        
    def subscribe(self, callback: Callable[[MarketTick], None]):
        """Subscribe to market data updates"""
        self.subscribers.append(callback)
        
    def start_stream(self):
        """Start the market data stream"""
        
        # Example WebSocket connection (Finnhub)
        if self.api_key:
            ws_url = f"wss://ws.finnhub.io?token={self.api_key}"
        else:
            # Mock websocket for demonstration
            threading.Thread(target=self._mock_data_stream, daemon=True).start()
            return
            
        self.ws = websocket.WebSocketApp(
            ws_url,
            on_open=self.on_open,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close
        )
        
        # Start websocket in separate thread
        ws_thread = threading.Thread(target=self.ws.run_forever, daemon=True)
        ws_thread.start()
        
    def on_open(self, ws):
        """WebSocket connection opened"""
        self.is_connected = True
        logger.info("Market data stream connected")
        
        # Subscribe to symbols
        for symbol in self.symbols:
            subscribe_msg = json.dumps({'type': 'subscribe', 'symbol': symbol})
            ws.send(subscribe_msg)
            
    def on_message(self, ws, message):
        """Handle incoming market data"""
        try:
            data = json.loads(message)
            
            if data['type'] == 'trade':
                tick = MarketTick(
                    symbol=data['s'],
                    timestamp=datetime.fromtimestamp(data['t'] / 1000),
                    price=data['p'],
                    volume=data['v']
                )
                
                # Notify subscribers
                for callback in self.subscribers:
                    try:
                        callback(tick)
                    except Exception as e:
                        logger.error(f"Error in subscriber callback: {e}")
                        
        except Exception as e:
            logger.error(f"Error processing market data: {e}")
            
    def on_error(self, ws, error):
        """Handle WebSocket errors"""
        logger.error(f"WebSocket error: {error}")
        self.is_connected = False
        
    def on_close(self, ws):
        """Handle WebSocket close"""
        logger.info("Market data stream disconnected")
        self.is_connected = False
        
    def _mock_data_stream(self):
        """Mock market data stream for testing"""
        logger.info("Starting mock market data stream")
        
        import random
        
        prices = {symbol: random.uniform(50, 200) for symbol in self.symbols}
        
        while True:
            for symbol in self.symbols:
                # Simulate price movement
                change = random.uniform(-0.02, 0.02)  # ±2% change
                prices[symbol] *= (1 + change)
                
                tick = MarketTick(
                    symbol=symbol,
                    timestamp=datetime.now(),
                    price=prices[symbol],
                    volume=random.randint(100, 10000)
                )
                
                # Notify subscribers
                for callback in self.subscribers:
                    try:
                        callback(tick)
                    except Exception as e:
                        logger.error(f"Error in subscriber callback: {e}")
                        
            time.sleep(1)  # 1 second interval

class StreamProcessor:
    """Real-time stream processor with buffering and analytics"""
    
    def __init__(self, buffer_size: int = 10000, batch_size: int = 100):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.data_buffer = deque(maxlen=buffer_size)
        self.processors = []
        self.is_running = False
        self.processing_thread = None
        self.lock = threading.Lock()
        self.stats = {
            'ticks_processed': 0,
            'batches_processed': 0,
            'errors': 0,
            'start_time': None
        }
        
    def add_processor(self, processor_func: Callable[[List[MarketTick]], Any]):
        """Add a data processing function"""
        self.processors.append(processor_func)
        
    def start(self):
        """Start stream processing"""
        logger.info("Starting stream processor")
        self.is_running = True
        self.stats['start_time'] = time.time()
        
        self.processing_thread = threading.Thread(
            target=self._processing_loop,
            daemon=True
        )
        self.processing_thread.start()
        
    def stop(self):
        """Stop stream processing"""
        logger.info("Stopping stream processor") 
        self.is_running = False
        
        if self.processing_thread:
            self.processing_thread.join()
            
    def add_tick(self, tick: MarketTick):
        """Add tick to processing buffer"""
        with self.lock:
            self.data_buffer.append(tick)
            self.stats['ticks_processed'] += 1
            
    def _processing_loop(self):
        """Main processing loop"""
        
        while self.is_running:
            try:
                batch = self._get_batch()
                
                if batch:
                    # Process with all registered processors
                    for processor in self.processors:
                        try:
                            processor(batch)
                        except Exception as e:
                            logger.error(f"Processor error: {e}")
                            self.stats['errors'] += 1
                            
                    self.stats['batches_processed'] += 1
                    
                time.sleep(0.1)  # Small delay to prevent busy waiting
                
            except Exception as e:
                logger.error(f"Processing loop error: {e}")
                self.stats['errors'] += 1
                
    def _get_batch(self) -> List[MarketTick]:
        """Get batch of data for processing"""
        
        with self.lock:
            if len(self.data_buffer) >= self.batch_size:
                batch = []
                for _ in range(min(self.batch_size, len(self.data_buffer))):
                    batch.append(self.data_buffer.popleft())
                return batch
                
        return []
        
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        
        stats = self.stats.copy()
        if stats['start_time']:
            stats['uptime_seconds'] = time.time() - stats['start_time']
            stats['ticks_per_second'] = stats['ticks_processed'] / stats['uptime_seconds']
            
        return stats

class RealTimeRiskCalculator:
    """Real-time risk calculations"""
    
    def __init__(self, lookback_window: int = 252):
        self.lookback_window = lookback_window
        self.price_history = {}
        self.returns_history = {}
        
    def process_ticks(self, ticks: List[MarketTick]):
        """Process incoming ticks and update risk metrics"""
        
        for tick in ticks:
            symbol = tick.symbol
            
            # Initialize if new symbol
            if symbol not in self.price_history:
                self.price_history[symbol] = deque(maxlen=self.lookback_window)
                self.returns_history[symbol] = deque(maxlen=self.lookback_window-1)
                
            # Add new price
            prev_price = self.price_history[symbol][-1] if self.price_history[symbol] else tick.price
            self.price_history[symbol].append(tick.price)
            
            # Calculate return
            if len(self.price_history[symbol]) > 1:
                return_val = (tick.price / prev_price) - 1
                self.returns_history[symbol].append(return_val)
                
    def calculate_real_time_var(self, symbol: str, confidence_level: float = 0.95) -> Optional[float]:
        """Calculate real-time VaR"""
        
        if symbol not in self.returns_history or len(self.returns_history[symbol]) < 30:
            return None
            
        returns = list(self.returns_history[symbol])
        return np.percentile(returns, (1 - confidence_level) * 100)
        
    def calculate_real_time_volatility(self, symbol: str) -> Optional[float]:
        """Calculate real-time volatility"""
        
        if symbol not in self.returns_history or len(self.returns_history[symbol]) < 30:
            return None
            
        returns = list(self.returns_history[symbol])
        return np.std(returns) * np.sqrt(252)  # Annualized

class PriceAlertSystem:
    """Real-time price alert system"""
    
    def __init__(self):
        self.alerts = {}
        self.triggered_alerts = []
        
    def add_price_alert(self, symbol: str, target_price: float, 
                       condition: str = 'above', alert_id: str = None):
        """Add price alert"""
        
        if alert_id is None:
            alert_id = f"{symbol}_{target_price}_{condition}_{int(time.time())}"
            
        self.alerts[alert_id] = {
            'symbol': symbol,
            'target_price': target_price,
            'condition': condition,
            'created_at': datetime.now(),
            'triggered': False
        }
        
        return alert_id
        
    def process_ticks(self, ticks: List[MarketTick]):
        """Check alerts against incoming ticks"""
        
        for tick in ticks:
            for alert_id, alert in self.alerts.items():
                if alert['symbol'] != tick.symbol or alert['triggered']:
                    continue
                    
                triggered = False
                
                if alert['condition'] == 'above' and tick.price >= alert['target_price']:
                    triggered = True
                elif alert['condition'] == 'below' and tick.price <= alert['target_price']:
                    triggered = True
                    
                if triggered:
                    alert['triggered'] = True
                    alert['triggered_at'] = datetime.now()
                    alert['triggered_price'] = tick.price
                    
                    self.triggered_alerts.append({
                        'alert_id': alert_id,
                        'symbol': tick.symbol,
                        'target_price': alert['target_price'],
                        'actual_price': tick.price,
                        'condition': alert['condition'],
                        'timestamp': datetime.now()
                    })
                    
                    logger.info(f"Price alert triggered: {alert_id}")
                    
    def get_triggered_alerts(self) -> List[Dict[str, Any]]:
        """Get and clear triggered alerts"""
        
        alerts = self.triggered_alerts.copy()
        self.triggered_alerts.clear()
        return alerts
