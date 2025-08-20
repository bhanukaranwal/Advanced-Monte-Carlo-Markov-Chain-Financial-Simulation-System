"""
General utility functions and helpers
"""

import os
import sys
import json
import pickle
import hashlib
import logging
from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

def ensure_directory_exists(path: str) -> None:
    """Create directory if it doesn't exist"""
    Path(path).mkdir(parents=True, exist_ok=True)

def get_timestamp_string(format_str: str = "%Y%m%d_%H%M%S") -> str:
    """Get formatted timestamp string"""
    return datetime.now().strftime(format_str)

def serialize_to_json(obj: Any, filepath: str) -> None:
    """Serialize object to JSON file"""
    ensure_directory_exists(os.path.dirname(filepath))
    with open(filepath, 'w') as f:
        json.dump(obj, f, indent=2, default=str)

def deserialize_from_json(filepath: str) -> Any:
    """Deserialize object from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)

def serialize_to_pickle(obj: Any, filepath: str) -> None:
    """Serialize object to pickle file"""
    ensure_directory_exists(os.path.dirname(filepath))
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)

def deserialize_from_pickle(filepath: str) -> Any:
    """Deserialize object from pickle file"""
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def calculate_hash(data: Union[str, bytes]) -> str:
    """Calculate SHA256 hash of data"""
    if isinstance(data, str):
        data = data.encode('utf-8')
    return hashlib.sha256(data).hexdigest()

def flatten_dict(d: Dict, parent_key: str = '', sep: str = '.') -> Dict:
    """Flatten nested dictionary"""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage statistics"""
    import psutil
    process = psutil.Process()
    memory_info = process.memory_info()
    
    return {
        'rss_mb': memory_info.rss / 1024 / 1024,
        'vms_mb': memory_info.vms / 1024 / 1024,
        'percent': process.memory_percent()
    }

def timer(func: Callable) -> Callable:
    """Decorator to time function execution"""
    def wrapper(*args, **kwargs):
        start_time = datetime.now()
        result = func(*args, **kwargs)
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        logger.info(f"{func.__name__} executed in {duration:.3f} seconds")
        return result
    return wrapper

def chunk_array(arr: np.ndarray, chunk_size: int) -> List[np.ndarray]:
    """Split array into chunks"""
    return [arr[i:i + chunk_size] for i in range(0, len(arr), chunk_size)]

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division with default value for zero denominator"""
    if denominator == 0:
        return default
    return numerator / denominator

def format_large_number(num: float, precision: int = 2) -> str:
    """Format large numbers with appropriate suffix (K, M, B, T)"""
    if abs(num) >= 1e12:
        return f"{num / 1e12:.{precision}f}T"
    elif abs(num) >= 1e9:
        return f"{num / 1e9:.{precision}f}B"
    elif abs(num) >= 1e6:
        return f"{num / 1e6:.{precision}f}M"
    elif abs(num) >= 1e3:
        return f"{num / 1e3:.{precision}f}K"
    else:
        return f"{num:.{precision}f}"

def retry_on_exception(max_retries: int = 3, delay: float = 1.0, 
                      backoff: float = 2.0, exceptions: tuple = (Exception,)):
    """Decorator to retry function on exception"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            retries = 0
            current_delay = delay
            
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    retries += 1
                    if retries >= max_retries:
                        raise e
                    
                    logger.warning(f"Retry {retries}/{max_retries} for {func.__name__}: {e}")
                    import time
                    time.sleep(current_delay)
                    current_delay *= backoff
                    
            return None
        return wrapper
    return decorator
