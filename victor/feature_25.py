"""
Helpers for engineering the features to be input to JMs.
"""

from utils_dir import *
include_home_dir()

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from jumpmodels.utils import *

############################################
## Feature Engineering
############################################

def compute_ewm_DD(ret_ser: pd.Series, hl: float) -> pd.Series:
    """
    Compute the exponentially weighted moving downside deviation (DD) for a return series.
    """
    ret_ser_neg: pd.Series = np.minimum(ret_ser, 0.)
    sq_mean = ret_ser_neg.pow(2).ewm(halflife=hl).mean()
    return np.sqrt(sq_mean)

def compute_rsi(ret_ser: pd.Series, window: int) -> pd.Series:
    delta = ret_ser.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_stochastic_k(ret_ser: pd.Series, window: int) -> pd.Series:
    rolling_min = ret_ser.rolling(window=window).min()
    rolling_max = ret_ser.rolling(window=window).max()
    return 100 * (ret_ser - rolling_min) / (rolling_max - rolling_min)

def compute_macd(ret_ser: pd.Series, fast: int, slow: int) -> pd.Series:
    ema_fast = ret_ser.ewm(span=fast, adjust=False).mean()
    ema_slow = ret_ser.ewm(span=slow, adjust=False).mean()
    return ema_fast - ema_slow

def compute_beta(asset: pd.Series, market: pd.Series, window: int = 21) -> pd.Series:
    cov = asset.rolling(window).cov(market)
    var = market.rolling(window).var()
    return cov / var

def feature_engineer(ret_ser: pd.Series, market: pd.Series = None) -> pd.DataFrame:
    feat_dict = {}
    
    # Active Return (EWMA) for windows 8, 21, 63
    for window in [8, 21, 63]:
        feat_dict[f"ActiveReturn_EWMA_{window}"] = ret_ser.ewm(halflife=window).mean()
    
    # Relative Strength Index for windows 8, 21, 63
    for window in [8, 21, 63]:
        feat_dict[f"RSI_{window}"] = compute_rsi(ret_ser, window)
    
    # Stochastic Oscillator %K for windows 8, 21, 63
    for window in [8, 21, 63]:
        feat_dict[f"Stochastic_%K_{window}"] = compute_stochastic_k(ret_ser, window)
    
    # MACD for pairs (8,21) and (21,63)
    for fast, slow in [(8, 21), (21, 63)]:
        feat_dict[f"MACD_{fast}_{slow}"] = compute_macd(ret_ser, fast, slow)
    
    # Downside deviation (log) for window 21
    feat_dict["DownsideDeviation_log_21"] = np.log(compute_ewm_DD(ret_ser, hl=21))
    
    # Active market beta for window 21 (requires a market return series)
    if market is not None:
        feat_dict["ActiveMarketBeta_21"] = compute_beta(ret_ser, market, window=21)
    else:
        feat_dict["ActiveMarketBeta_21"] = np.nan  # or handle appropriately
    
    return pd.DataFrame(feat_dict)

############################################
## DataLoader Class
############################################

class DataLoader(BaseEstimator):
    """
    Flexible class for loading the feature matrix from CSV or Pickle.
    """
    def __init__(self, file_path: str, ver: str = "v0"):
        self.file_path = file_path
        self.ver = ver
    
    def load(self, start_date: str = None, end_date: str = None):
        """
        Load raw data, compute features, and filter by date.
        """
        # Detect file type and load data
        if self.file_path.endswith(".pkl"):
            df_raw = pd.read_pickle(self.file_path).dropna()
        elif self.file_path.endswith(".csv"):
            df_raw = pd.read_csv(self.file_path, parse_dates=["Date"], index_col="Date").dropna()
        else:
            raise ValueError("Unsupported file format. Use .csv or .pkl")

        # Assume first column is returns
        ret_ser_raw = df_raw.iloc[:, 0]
        
        # Compute features
        df_features_all = feature_engineer(ret_ser_raw)
        
        # Filter by date and drop rows with NaNs
        X = filter_date_range(df_features_all, start_date, end_date).dropna()
        valid_no_nan(X)


        # Save attributes
        self.X = X
        self.ret_ser = filter_date_range(ret_ser_raw, start_date, end_date)
        
        return self