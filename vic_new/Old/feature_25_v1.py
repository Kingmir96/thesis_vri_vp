import os
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from jumpmodels.utils import *
# Assume filter_date_range and valid_no_nan are available from utils_dir

############################################
# Indicator Functions
############################################

def compute_ewm_DD(ret_ser: pd.Series, hl: float) -> pd.Series:
    """
    Compute the exponentially weighted moving downside deviation (DD) for a return series.
    """
    ret_ser_neg: pd.Series = np.minimum(ret_ser, 0.)
    sq_mean = ret_ser_neg.pow(2).ewm(halflife=hl).mean()
    return np.sqrt(sq_mean)

def compute_active_return(factor_return: pd.Series, market_return: pd.Series) -> pd.Series:
    """
    Compute active return as the difference between factor and market returns.
    """
    return factor_return - market_return

def compute_active_return_ewma(active_return: pd.Series, window: int) -> pd.Series:
    """
    Compute EWMA of the active return.
    """
    return active_return.ewm(halflife=window).mean()

def compute_rsi(price: pd.Series, window: int) -> pd.Series:
    """
    Compute the Relative Strength Index (RSI) for a price series.
    """
    delta = price.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window, min_periods=window).mean()
    avg_loss = loss.rolling(window, min_periods=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_stoch_osc(price: pd.Series, window: int) -> pd.Series:
    """
    Compute the Stochastic Oscillator %K for a price series.
    """
    rolling_min = price.rolling(window, min_periods=window).min()
    rolling_max = price.rolling(window, min_periods=window).max()
    stoch_k = (price - rolling_min) / (rolling_max - rolling_min) * 100
    return stoch_k

def compute_macd(price: pd.Series, fast: int, slow: int) -> pd.Series:
    """
    Compute the Moving Average Convergence/Divergence (MACD) on a price series.
    """
    ema_fast = price.ewm(halflife=fast).mean()
    ema_slow = price.ewm(halflife=slow).mean()
    return ema_fast - ema_slow

def compute_downside_dev_log(active_return: pd.Series, window: int) -> pd.Series:
    """
    Compute the log-transformed downside deviation over a rolling window.
    """
    def downside_dev_window(x):
        dd = np.mean(np.square(np.maximum(0 - x, 0)))
        return np.log(dd) if dd > 0 else np.nan
    return active_return.rolling(window=window, min_periods=window).apply(downside_dev_window, raw=True)

def compute_active_market_beta(active_return: pd.Series, market_return: pd.Series, window: int) -> pd.Series:
    """
    Compute the EWMA-based active market beta over a rolling window.
    """
    ewma_cov = active_return.ewm(halflife=window).cov(market_return)
    ewma_var = market_return.ewm(halflife=window).var()
    return ewma_cov / ewma_var

############################################
# Feature Engineering Function
############################################

def feature_engineer(ret_ser: pd.Series, df_raw: pd.DataFrame = None, ver: str = "v0") -> pd.DataFrame:
    """
    v1 computes the following features:
      1. Active Return (EWMA) over windows [8, 21, 63].
      2. Relative Strength Index (RSI) on a price series computed from factor returns.
      3. Stochastic Oscillator %K on the same price series.
      4. MACD for window pairs (8,21) and (21,63) on the price series.
      5. Downside Deviation (log-transformed) on active return (21-day window).
      6. Active Market Beta (EWMA) over a 21-day window.
    """
    if ver == "v1":
        if df_raw is None or "PBUS" not in df_raw.columns:
            raise ValueError("df_raw must contain a 'PBUS' column for market return.")
        
        # Extract factor and market returns
        factor_return = ret_ser
        market_return = df_raw["PBUS"]
        
        # Compute active return (factor - market)
        active_return = compute_active_return(factor_return, market_return)
        
        # Construct a synthetic factor price series (assume initial price = 100)
        price = 100 * (1 + factor_return).cumprod()
        
        features = {}
        
        # 1. Active Return (EWMA)
        for window in [8, 21, 63]:
            features[f"ActiveReturn_EWMA_{window}"] = compute_active_return_ewma(active_return, window)
        
        # 2. RSI on factor price
        for window in [8, 21, 63]:
            features[f"RSI_{window}"] = compute_rsi(price, window)
        
        # 3. Stochastic Oscillator %K on factor price
        for window in [8, 21, 63]:
            features[f"StochOsc_%K_{window}"] = compute_stoch_osc(price, window)
        
        # 4. MACD on factor price for two window pairs
        for fast, slow in [(8, 21), (21, 63)]:
            features[f"MACD_{fast}_{slow}"] = compute_macd(price, fast, slow)
        
        # 5. Downside Deviation (log-transformed) on active return (21-day)
        features["DownsideDev_log_21"] = compute_downside_dev_log(active_return, 21)
        
        # 6. Active Market Beta (EWMA) over 21-day
        features["ActiveMarketBeta_EWMA_21"] = compute_active_market_beta(active_return, market_return, 21)
        
        return pd.DataFrame(features)
    
    else:
        # Fallback to original v0 feature engineering
        feat_dict = {}
        hls = [5, 20, 60]
        for hl in hls:
            feat_dict[f"ret_{hl}"] = ret_ser.ewm(halflife=hl).mean()
            DD = compute_ewm_DD(ret_ser, hl)
            feat_dict[f"DD-log_{hl}"] = np.log(DD)
            feat_dict[f"sortino_{hl}"] = feat_dict[f"ret_{hl}"].div(DD)
        return pd.DataFrame(feat_dict)

############################################
# DataLoader Class
############################################

class DataLoader(BaseEstimator):
    """
    Modified DataLoader to support v1 feature engineering.
    """
    def __init__(self, file_path: str, ver: str = "v0", factor_col: str = None):
        self.file_path = file_path
        self.ver = ver
        self.factor_col = factor_col  # New argument to specify the factor column
    
    def load(self, start_date: str = None, end_date: str = None):
        # Load raw data
        if self.file_path.endswith(".pkl"):
            df_raw = pd.read_pickle(self.file_path).dropna()
        elif self.file_path.endswith(".csv"):
            df_raw = pd.read_csv(self.file_path, parse_dates=["Date"], index_col="Date").dropna()
        else:
            raise ValueError("Unsupported file format. Use .csv or .pkl")

        # Use specified column or default to first column
        if self.factor_col:
            if self.factor_col not in df_raw.columns:
                raise ValueError(f"Factor column '{self.factor_col}' not found in data.")
            ret_ser_raw = df_raw[self.factor_col]  # Select chosen column
        else:
            ret_ser_raw = df_raw.iloc[:, 0]  # Default to first column

        # Compute features using the appropriate version
        if self.ver == "v1":
            df_features_all = feature_engineer(ret_ser_raw, df_raw, self.ver)
        else:
            df_features_all = feature_engineer(ret_ser_raw, ver=self.ver)

        # Filter by date and validate no NaNs
        X = filter_date_range(df_features_all, start_date, end_date)
        X = X.dropna()
        valid_no_nan(X)

        self.X = X
        self.ret_ser = filter_date_range(ret_ser_raw, start_date, end_date)
        return self
