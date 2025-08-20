"""
Comprehensive performance monitoring and profiling script
"""

import psutil
import time
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import asyncio
import aiohttp
import json

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """System performance monitoring and profiling"""
    
    def __init__(self, api_base_url: str = "http://localhost:5000"):
        self.api_base_url = api_base_url
        self.metrics_history = []
        self.monitoring = False
        
    async def monitor_system_resources(self, duration_minutes: int = 60):
        """Monitor system resources over time"""
        logger.info(f"Starting system monitoring for {duration_minutes} minutes")
        
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        
        self.monitoring = True
        
        while time.time() < end_time and self.monitoring:
            timestamp = datetime.now()
            
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            load_avg = psutil.getloadavg()
            
            # Memory metrics
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            
            # Network metrics
            network = psutil.net_io_counters()
            
            # Process-specific metrics
            current_process = psutil.Process()
            process_memory = current_process.memory_info()
            process_cpu = current_process.cpu_percent()
            
            metrics = {
                'timestamp': timestamp,
                'cpu': {
                    'percent': cpu_percent,
                    'count': cpu_count,
                    'load_avg_1m': load_avg[0],
                    'load_avg_5m': load_avg[11],
                    'load_avg_15m': load_avg[12]
                },
                'memory': {
                    'total_gb': memory.total / 1024**3,
                    'available_gb': memory.available / 1024**3,
                    'used_gb': memory.used / 1024**3,
                    'percent': memory.percent,
                    'swap_used_gb': swap.used / 1024**3,
                    'swap_percent': swap.percent
                },
                'disk': {
                    'total_gb': disk.total / 1024**3,
                    'used_gb': disk.used / 1024**3,
                    'free_gb': disk.free / 1024**3,
                    'percent': (disk.used / disk.total) * 100
                },
                'network': {
                    'bytes_sent': network.bytes_sent,
                    'bytes_recv': network.bytes_recv,
                    'packets_sent': network.packets_sent,
                    'packets_recv': network.packets_recv
                },
                'process': {
                    'memory_rss_mb': process_memory.rss / 1024**2,
                    'memory_vms_mb': process_memory.vms / 1024**2,
                    'cpu_percent': process_cpu
                }
            }
            
            self.metrics_history.append(metrics)
            logger.debug(f"Recorded metrics at {timestamp}")
            
            await asyncio.sleep(10)  # Record every 10 seconds
            
        logger.info("System monitoring completed")
        
    async def benchmark_api_performance(self, concurrent_users: int = 10, requests_per_user: int = 100):
        """Benchmark API performance under load"""
        logger.info(f"Starting API benchmark: {concurrent_users} users, {requests_per_user} requests each")
        
        async def user_session(user_id: int, session: aiohttp.ClientSession):
            """Simulate a user session"""
            user_results = []
            
            # Login
            login_start = time.time()
            async with session.post(f"{self.api_base_url}/api/v1/auth/login", json={
                "user_id": "demo_user",
                "password": "demo_password"
            }) as response:
                if response.status == 200:
                    token_data = await response.json()
                    token = token_data["access_token"]
                    headers = {"Authorization": f"Bearer {token}"}
                else:
                    logger.error(f"User {user_id} failed to login")
                    return []
            login_time = time.time() - login_start
            
            # Make requests
            for request_id in range(requests_per_user):
                request_start = time.time()
                
                try:
                    # Random request type
                    if request_id % 3 == 0:
                        # Monte Carlo simulation
                        payload = {
                            "n_simulations": 1000,
                            "n_steps": 100,
                            "initial_price": 100.0,
                            "drift": 0.05,
                            "volatility": 0.2
                        }
                        async with session.post(
                            f"{self.api_base_url}/api/v1/simulations/monte-carlo",
                            json=payload,
                            headers=headers
                        ) as response:
                            status = response.status
                            
                    elif request_id % 3 == 1:
                        # Risk analysis
                        payload = {
                            "returns_data": np.random.normal(0.001, 0.02, 100).tolist(),
                            "confidence_levels": [0.95, 0.99]
                        }
                        async with session.post(
                            f"{self.api_base_url}/api/v1/analytics/risk",
                            json=payload,
                            headers=headers
                        ) as response:
                            status = response.status
                            
                    else:
                        # Health check
                        async with session.get(f"{self.api_base_url}/health") as response:
                            status = response.status
                            
                    request_time = time.time() - request_start
                    
                    user_results.append({
                        'user_id': user_id,
                        'request_id': request_id,
                        'status_code': status,
                        'response_time': request_time,
                        'timestamp': datetime.now()
                    })
                    
                except Exception as e:
                    request_time = time.time() - request_start
                    logger.error(f"Request error for user {user_id}: {e}")
                    user_results.append({
                        'user_id': user_id,
                        'request_id': request_id,
                        'status_code': 0,
                        'response_time': request_time,
                        'error': str(e),
                        'timestamp': datetime.now()
                    })
                    
            return user_results
            
        # Run concurrent user sessions
        connector = aiohttp.TCPConnector(limit=100)
        timeout = aiohttp.ClientTimeout(total=30)
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            tasks = [user_session(user_id, session) for user_id in range(concurrent_users)]
            results = await asyncio.gather(*tasks)
            
        # Flatten results
        all_results = []
        for user_results in results:
            all_results.extend(user_results)
            
        # Analyze results
        df = pd.DataFrame(all_results)
        
        if not df.empty:
            analysis = {
                'total_requests': len(df),
                'successful_requests': len(df[df['status_code'] == 200]),
                'error_rate': len(df[df['status_code'] != 200]) / len(df),
                'mean_response_time': df['response_time'].mean(),
                'median_response_time': df['response_time'].median(),
                'p95_response_time': df['response_time'].quantile(0.95),
                'p99_response_time': df['response_time'].quantile(0.99),
                'requests_per_second': len(df) / df['response_time'].sum() * concurrent_users
            }
            
            logger.info("API Benchmark Results:")
            for key, value in analysis.items():
                logger.info(f"  {key}: {value}")
                
            return df, analysis
        else:
            logger.error("No benchmark results collected")
            return None, None
            
    def generate_performance_report(self, output_file: str = "performance_report.html"):
        """Generate comprehensive performance report"""
        if not self.metrics_history:
            logger.error("No metrics data available for report")
            return
            
        # Convert metrics to DataFrame
        metrics_data = []
        for entry in self.metrics_history:
            row = {
                'timestamp': entry['timestamp'],
                'cpu_percent': entry['cpu']['percent'],
                'memory_percent': entry['memory']['percent'],
                'memory_used_gb': entry['memory']['used_gb'],
                'disk_percent': entry['disk']['percent'],
                'process_memory_mb': entry['process']['memory_rss_mb'],
                'process_cpu_percent': entry['process']['cpu_percent']
            }
            metrics_data.append(row)
            
        df = pd.DataFrame(metrics_data)
        
        # Create visualizations
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('System Performance Metrics', fontsize=16)
        
        # CPU usage
        axes[0, 0].plot(df['timestamp'], df['cpu_percent'])
        axes[0, 0].set_title('CPU Usage (%)')
        axes[0, 0].set_ylabel('Percentage')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Memory usage
        axes[0, 1].plot(df['timestamp'], df['memory_percent'])
        axes[0, 1].set_title('Memory Usage (%)')
        axes[0, 1].set_ylabel('Percentage')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Disk usage
        axes[0, 2].plot(df['timestamp'], df['disk_percent'])
        axes[0, 2].set_title('Disk Usage (%)')
        axes[0, 2].set_ylabel('Percentage')
        axes[0, 2].tick_params(axis='x', rotation=45)
        
        # Process memory
        axes[1, 0].plot(df['timestamp'], df['process_memory_mb'])
        axes[1, 0].set_title('Process Memory (MB)')
        axes[1, 0].set_ylabel('MB')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Process CPU
        axes[1, 1].plot(df['timestamp'], df['process_cpu_percent'])
        axes[1, 1].set_title('Process CPU (%)')
        axes[1, 1].set_ylabel('Percentage')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # Memory usage over time
        axes[1, 2].plot(df['timestamp'], df['memory_used_gb'])
        axes[1, 2].set_title('Memory Used (GB)')
        axes[1, 2].set_ylabel('GB')
        axes[1, 2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('performance_metrics.png', dpi=300, bbox_inches='tight')
        
        # Generate HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>MCMF Performance Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .metric {{ margin: 20px 0; }}
                .chart {{ text-align: center; margin: 30px 0; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>Monte Carlo-Markov Finance System Performance Report</h1>
            <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>Summary Statistics</h2>
            <table>
                <tr><th>Metric</th><th>Mean</th><th>Max</th><th>Min</th><th>Std Dev</th></tr>
                <tr><td>CPU Usage (%)</td><td>{df['cpu_percent'].mean():.2f}</td><td>{df['cpu_percent'].max():.2f}</td><td>{df['cpu_percent'].min():.2f}</td><td>{df['cpu_percent'].std():.2f}</td></tr>
                <tr><td>Memory Usage (%)</td><td>{df['memory_percent'].mean():.2f}</td><td>{df['memory_percent'].max():.2f}</td><td>{df['memory_percent'].min():.2f}</td><td>{df['memory_percent'].std():.2f}</td></tr>
                <tr><td>Process Memory (MB)</td><td>{df['process_memory_mb'].mean():.2f}</td><td>{df['process_memory_mb'].max():.2f}</td><td>{df['process_memory_mb'].min():.2f}</td><td>{df['process_memory_mb'].std():.2f}</td></tr>
            </table>
            
            <div class="chart">
                <h2>Performance Metrics Over Time</h2>
                <img src="performance_metrics.png" alt="Performance Metrics Chart" style="max-width: 100%;">
            </div>
            
            <h2>Monitoring Duration</h2>
            <p>Start Time: {df['timestamp'].min()}</p>
            <p>End Time: {df['timestamp'].max()}</p>
            <p>Duration: {df['timestamp'].max() - df['timestamp'].min()}</p>
            
        </body>
        </html>
        """
        
        with open(output_file, 'w') as f:
            f.write(html_content)
            
        logger.info(f"Performance report generated: {output_file}")

async def main():
    monitor = PerformanceMonitor()
    
    # Start system monitoring in background
    monitor_task = asyncio.create_task(monitor.monitor_system_resources(duration_minutes=5))
    
    # Run API benchmark
    await asyncio.sleep(30)  # Let monitoring collect some baseline data
    df, analysis = await monitor.benchmark_api_performance(concurrent_users=5, requests_per_user=20)
    
    # Wait for monitoring to complete
    await monitor_task
    
    # Generate report
    monitor.generate_performance_report()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
