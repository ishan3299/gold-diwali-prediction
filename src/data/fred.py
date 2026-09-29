import pandas_datareader.data as web
import pandas as pd
import logging
from pathlib import Path
import datetime

logger = logging.getLogger(__name__)

def fetch_fred_data(series_id, start_date, end_date=None, cache_dir=None, force_refresh=False):
    """
    Fetch historical data for a given series from FRED.
    """
    if end_date is None:
        end_date = datetime.date.today().strftime('%Y-%m-%d')
        
    cache_file = None
    if cache_dir:
        cache_file = Path(cache_dir) / f"fred_{series_id}_{start_date}_{end_date}.csv"
        if cache_file.exists() and not force_refresh:
            logger.info(f"Loading {series_id} from cache.")
            return pd.read_csv(cache_file, index_col='DATE', parse_dates=True)
            
    logger.info(f"Fetching {series_id} from FRED ({start_date} to {end_date})...")
    try:
        data = web.DataReader(series_id, 'fred', start_date, end_date)
        if data.empty:
            logger.warning(f"No data fetched for {series_id}.")
            return pd.DataFrame()
            
        if cache_file:
            data.to_csv(cache_file)
        return data
    except Exception as e:
        logger.error(f"Error fetching data for {series_id}: {e}")
        return pd.DataFrame()
