"""
feature_25.py Module
Fully aligned with standard EMA (span=8,21,63). 
All half-life calls removed for factor features.
"""

import os
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from jumpmodels.utils import *  # Assume filter_date_range and valid_no_nan

############################################
# Indicator Functions (Factor-Focused)
############################################

def compute_ewma_active_return(active_return: pd.Series, span: int) -> pd.Series:
    """
    Standard EMA of 'active_return' using the given span (not half-life).
    E.g., .ewm(span=8) for short, 21 for medium, 63 for long.
    """
    return active_return.ewm(span=span, adjust=False).mean()

def compute_rsi(price: pd.Series, window: int) -> pd.Series:
    """
    RSI typically uses a simple rolling average (not exponential).
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
    Stochastic Oscillator %K with rolling window.
    """
    rolling_min = price.rolling(window, min_periods=window).min()
    rolling_max = price.rolling(window, min_periods=window).max()
    return (price - rolling_min) / (rolling_max - rolling_min) * 100

def compute_macd(price: pd.Series, fast: int, slow: int) -> pd.Series:
    """
    MACD = EMA_fast - EMA_slow, each with span=fast, span=slow.
    For example, MACD(8,21) or MACD(21,63).
    """
    ema_fast = price.ewm(span=fast, adjust=False).mean()
    ema_slow = price.ewm(span=slow, adjust=False).mean()
    return ema_fast - ema_slow

def compute_downside_dev_ewma(active_return: pd.Series, span: int, do_log: bool = True) -> pd.Series:
    """
    Downside deviation with exponential weighting (span).
    Optionally log-transform (the paper often logs vol).
    """
    neg_returns = np.minimum(active_return, 0.0)
    neg_sq = neg_returns ** 2
    ewm_neg_sq = neg_sq.ewm(span=span, adjust=False).mean()
    downside_dev = np.sqrt(ewm_neg_sq)
    if do_log:
        downside_dev = np.log(downside_dev.clip(lower=1e-10))
    return downside_dev

def compute_active_market_beta(active_return: pd.Series, market_return: pd.Series, span: int) -> pd.Series:
    """
    Computes the EMA-based covariance and divides by EMA-based variance,
    both with .ewm(span=...).
    """
    ewma_cov = active_return.ewm(span=span, adjust=False).cov(market_return)
    ewma_var = market_return.ewm(span=span, adjust=False).var()
    return ewma_cov / ewma_var

############################################
# Indicator Functions (Macro-Focused)
############################################

def _log_diff_ewma(series: pd.Series, span: int) -> pd.Series:
    """
    log -> diff -> ewm(span=...).
    """
    logged = np.log(series)
    differenced = logged.diff()
    return differenced.ewm(span=span, adjust=False).mean()

def _diff_ewma(series: pd.Series, span: int) -> pd.Series:
    """
    diff -> ewm(span=...).
    """
    differenced = series.diff()
    return differenced.ewm(span=span, adjust=False).mean()

def compute_macro_features(df_raw: pd.DataFrame, span: int = 21) -> pd.DataFrame:
    """
    Build the macro features if columns are present in df_raw:
      1. Market return (EWMA of daily returns)
      2. VIX => log -> diff -> ewm(span)
      3. 2-year yield => diff -> ewm(span)
      4. 10-year minus 2-year => diff -> ewm(span)
    """
    features = {}
    
    # (1) Market return => ewm(span=...)
    if "mkt" in df_raw.columns:
        features["MarketReturn_EWMA_21"] = df_raw["mkt"].ewm(span=span, adjust=False).mean()
    
    # (2) VIX => log -> diff -> ewm
    if "VIX" in df_raw.columns:
        features["VIX_log_diff_EWMA_21"] = _log_diff_ewma(df_raw["VIX"], span=span)
    
    # (3) 2-year yield => diff -> ewm
    if "2Y_Yield" in df_raw.columns:
        features["2Y_diff_EWMA_21"] = _diff_ewma(df_raw["2Y_Yield"], span=span)
    
    # (4) 10Y-2Y => diff -> ewm
    if "10Y-2Y_Spread" in df_raw.columns:
        features["10Y-2Y_EWMA_21"] = _diff_ewma(df_raw["10Y-2Y_Spread"], span=span)
    
    return pd.DataFrame(features)

############################################
# Main Feature-Engineering Logic
############################################

def feature_engineer(
    factor_return: pd.Series,
    df_raw: pd.DataFrame,
    ver: str = "v0"
) -> pd.DataFrame:
    """
    A dispatcher that calls either:
      - feature_engineer_v2(...) if ver=='v2'
      - otherwise do something else or fallback
    """
    # We'll define a simple fallback for 'v0', 'v1', etc.
    # but the real logic is in 'v2'.
    if ver == "v2":
        # We assume 'mkt' is in df_raw for the market return,
        # plus a 'price' series for RSI, stoch, MACD, etc.
        if "mkt" not in df_raw.columns:
            raise ValueError("df_raw must contain 'mkt' for the market returns in v2.")
        
        # Build 'price' from factor_return for the technical indicators:
        # e.g. a pseudo-price:  price(t) = 100 * product(1+factor_return)
        price = 100 * (1 + factor_return).cumprod()
        return feature_engineer_v2(factor_return, df_raw["mkt"], price)
    
    # For older versions v0/v1, just do a trivial feature set or raise an error:
    raise NotImplementedError(f"feature_engineer(..., ver='{ver}') not implemented except for 'v2'.")

def feature_engineer_v2(
    factor_return: pd.Series,
    market_return: pd.Series,
    price: pd.Series,
    window_short: int = 8,
    window_med: int = 21,
    window_long: int = 63
) -> pd.DataFrame:
    """
    Example factor feature set using standard EMA (span).
    Fully aligned with the paper's 8/21/63 approach.
    """
    active_return = factor_return - market_return
    feat = {}
    
    # 1) EWMA of active returns (short, medium, long)
    for w in [window_short, window_med, window_long]:
        feat[f"ActiveReturn_EWMA_{w}"] = compute_ewma_active_return(active_return, span=w)
    
    # 2) RSI, Stoch Osc (both rolling)
    feat[f"RSI_{window_short}"] = compute_rsi(price, window_short)
    feat[f"RSI_{window_med}"]   = compute_rsi(price, window_med)
    feat[f"RSI_{window_long}"]  = compute_rsi(price, window_long)
    
    feat[f"StochOsc_%K_{window_short}"] = compute_stoch_osc(price, window_short)
    feat[f"StochOsc_%K_{window_med}"]   = compute_stoch_osc(price, window_med)
    feat[f"StochOsc_%K_{window_long}"]  = compute_stoch_osc(price, window_long)
    
    # 3) MACD
    feat["MACD_8_21"]  = compute_macd(price, 8, 21)
    feat["MACD_21_63"] = compute_macd(price, 21, 63)
    
    # 4) Downside Deviation (21-day span, log‐transformed)
    feat["DownsideDev_21"] = compute_downside_dev_ewma(active_return, span=21, do_log=True)
    
    # 5) Active Market Beta (21-day span)
    feat["ActiveMarketBeta_21"] = compute_active_market_beta(active_return, market_return, span=21)
    
    return pd.DataFrame(feat)


############################################
# MergedDataLoader
############################################

class MergedDataLoader(BaseEstimator):
    """
    Loads factor_data.csv and market_data.csv, merges on date index,
    and runs feature_engineer(...) with the chosen 'ver' (e.g., 'v2').
    """
    def __init__(self, factor_file: str, market_file: str, ver: str = "v2", factor_col: str = None):
        self.factor_file = factor_file
        self.market_file = market_file
        self.ver = ver
        self.factor_col = factor_col
        self.dropped_obs = {"factor_file": {}, "market_file": {}}
        self.dropped_pipeline = {}

    def load(self, start_date: str = None, end_date: str = None):
        # 1) Read factor/market data
        df_factors = pd.read_csv(self.factor_file, parse_dates=["date"], index_col="date")
        df_market  = pd.read_csv(self.market_file, parse_dates=["date"], index_col="date")
        
        # ...Check missing & dropna...
        df_factors_clean = df_factors.dropna()
        df_market_clean  = df_market.dropna()

        # 2) Merge on date index
        df_merged = df_factors_clean.join(df_market_clean, how="inner")

        # 3) Decide factor column
        if self.factor_col is None:
            ret_ser_raw = df_factors_clean.iloc[:, 0]
        else:
            if self.factor_col not in df_factors_clean.columns:
                raise ValueError(f"Factor column '{self.factor_col}' not found in factor_data.csv.")
            ret_ser_raw = df_factors_clean[self.factor_col]
        
        # 4) Compute features
        df_features_all = feature_engineer(ret_ser_raw, df_merged, ver=self.ver)

        # 5) Filter date & dropna
        X = filter_date_range(df_features_all, start_date, end_date).dropna()
        valid_no_nan(X)

        # Similarly filter ret_ser
        ret_ser_filtered = filter_date_range(ret_ser_raw, start_date, end_date).dropna()

        # Align indices
        common_idx = X.index.intersection(ret_ser_filtered.index)
        X = X.loc[common_idx]
        ret_ser_filtered = ret_ser_filtered.loc[common_idx]

        # 6) Also keep a market series, e.g. "mkt"
        if "mkt" not in df_merged.columns:
            raise ValueError("Market column 'mkt' not found in market_data.csv.")
        market_ser = df_merged["mkt"]
        market_ser = filter_date_range(market_ser, start_date, end_date).dropna()
        market_ser = market_ser.loc[common_idx]

        # 7) Save final
        self.X = X
        self.ret_ser = ret_ser_filtered
        self.market_ser = market_ser
        self.active_ret = ret_ser_filtered - market_ser
        return self
