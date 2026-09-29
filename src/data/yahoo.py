import yfinance as yf
import pandas as pd
import logging
from pathlib import Path
import datetime
import os

logger = logging.getLogger(__name__)

def fetch_yahoo_data(ticker, start_date, end_date=None, cache_dir=None, force_refresh=False):
    """
    Fetch historical data for a given ticker from Yahoo Finance.
    Saves to cache to avoid repeated requests.
    """
    if end_date is None:
        end_date = datetime.date.today().strftime('%Y-%m-%d')
        
    cache_file = None
    if cache_dir:
        cache_file = Path(cache_dir) / f"{ticker}_{start_date}_{end_date}.csv"
        if cache_file.exists() and not force_refresh:
            logger.info(f"Loading {ticker} from cache.")
            return pd.read_csv(cache_file, index_col='Date', parse_dates=True)
            
    logger.info(f"Fetching {ticker} from {start_date} to {end_date}...")
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if data.empty:
            logger.warning(f"No data fetched for {ticker}. Check if ticker is valid.")
            return pd.DataFrame()
        
        # Flatten multi-index columns if present (yfinance sometimes does this)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [c[0] for c in data.columns]
            
        if cache_file:
            data.to_csv(cache_file)
        return data
    except Exception as e:
        logger.error(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()
