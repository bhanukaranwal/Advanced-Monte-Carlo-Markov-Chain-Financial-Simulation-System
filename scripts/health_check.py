"""
System health check script
"""

import sys
import requests
import psycopg2
import redis
import logging
from typing import Dict, List
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HealthChecker:
    def __init__(self, config: Dict):
        self.config = config
        self.checks = []
        
    def check_api_health(self) -> bool:
        """Check API health"""
        try:
            response = requests.get(f"{self.config['api_url']}/health", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"API health check failed: {e}")
            return False
            
    def check_database_health(self) -> bool:
        """Check database connectivity"""
        try:
            conn = psycopg2.connect(
                host=self.config['db_host'],
                port=self.config['db_port'],
                database=self.config['db_name'],
                user=self.config['db_user'],
                password=self.config['db_password']
            )
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            conn.close()
            return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False
            
    def check_redis_health(self) -> bool:
        """Check Redis connectivity"""
        try:
            r = redis.Redis(
                host=self.config['redis_host'],
                port=self.config['redis_port'],
                decode_responses=True
            )
            r.ping()
            return True
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return False
            
    def run_all_checks(self) -> Dict[str, bool]:
        """Run all health checks"""
        results = {
            'api': self.check_api_health(),
            'database': self.check_database_health(),
            'redis': self.check_redis_health()
        }
        
        all_healthy = all(results.values())
        logger.info(f"Health check results: {results}")
        logger.info(f"Overall health: {'HEALTHY' if all_healthy else 'UNHEALTHY'}")
        
        return results

def main():
    parser = argparse.ArgumentParser(description="MCMF System Health Check")
    parser.add_argument("--config", required=True, help="Configuration file")
    args = parser.parse_args()
    
    # Load configuration
    import json
    with open(args.config) as f:
        config = json.load(f)
    
    checker = HealthChecker(config)
    results = checker.run_all_checks()
    
    # Exit with error code if any check fails
    if not all(results.values()):
        sys.exit(1)

if __name__ == "__main__":
    main()
