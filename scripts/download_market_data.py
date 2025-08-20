"""
Script to download historical market data
"""

import argparse
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logger = logging.getLogger(__name__)

def download_data(symbols, start_date, end_date, output_dir):
    """Download market data for given symbols"""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for symbol in symbols:
        logger.info(f"Downloading data for {symbol}...")
        
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date)
            
            if data.empty:
                logger.warning(f"No data found for {symbol}")
                continue
                
            # Clean and prepare data
            data.index.name = 'date'
            data.columns = data.columns.str.lower()
            
            # Save to CSV
            output_file = output_path / f"{symbol.lower()}_data.csv"
            data.to_csv(output_file)
            
            logger.info(f"Saved {len(data)} rows for {symbol} to {output_file}")
            
        except Exception as e:
            logger.error(f"Error downloading {symbol}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Download historical market data")
    parser.add_argument("--symbols", nargs="+", default=["AAPL", "GOOGL", "MSFT"], 
                       help="Stock symbols to download")
    parser.add_argument("--start-date", default="2020-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument("--output-dir", default="data/raw", help="Output directory")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    # Configure logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")
    
    # Set end date to today if not provided
    end_date = args.end_date or datetime.now().strftime("%Y-%m-%d")
    
    logger.info(f"Downloading data from {args.start_date} to {end_date}")
    logger.info(f"Symbols: {args.symbols}")
    
    download_data(args.symbols, args.start_date, end_date, args.output_dir)
    
    logger.info("Download complete!")

if __name__ == "__main__":
    main()
