"""Configuration management for the Advanced MC-Markov Finance system."""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path

class ConfigManager:
    """Centralized configuration management"""
    
    def __init__(self, config_dir: Optional[str] = None):
        self.config_dir = Path(config_dir) if config_dir else Path(__file__).parent
        self._configs: Dict[str, Dict[str, Any]] = {}
        
    def load_config(self, config_name: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if config_name in self._configs:
            return self._configs[config_name]
            
        config_path = self.config_dir / f"{config_name}.yaml"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Environment variable substitution
        config = self._substitute_env_vars(config)
        
        self._configs[config_name] = config
        return config
        
    def _substitute_env_vars(self, obj: Any) -> Any:
        """Recursively substitute environment variables"""
        if isinstance(obj, str):
            if obj.startswith('${') and obj.endswith('}'):
                env_var = obj[2:-1]
                return os.getenv(env_var, obj)
            return obj
        elif isinstance(obj, dict):
            return {k: self._substitute_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._substitute_env_vars(item) for item in obj]
        return obj
        
    def get(self, config_name: str, key: str = None, default: Any = None) -> Any:
        """Get configuration value"""
        config = self.load_config(config_name)
        
        if key is None:
            return config
            
        keys = key.split('.')
        value = config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
                
        return value

# Global config manager instance
config_manager = ConfigManager()

# Convenience functions
def load_config(config_name: str) -> Dict[str, Any]:
    """Load configuration file"""
    return config_manager.load_config(config_name)
    
def get_config(config_name: str, key: str = None, default: Any = None) -> Any:
    """Get configuration value"""
    return config_manager.get(config_name, key, default)

__all__ = ["ConfigManager", "config_manager", "load_config", "get_config"]
