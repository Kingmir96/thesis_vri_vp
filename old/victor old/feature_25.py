"""
feature_25.py Module

"""


import os
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from jumpmodels.utils import *
# Assume filter_date_range and valid_no_nan are available from utils_dir

############################################
# Indicator Functions (Factor-Focused)
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
# Indicator Functions (Macro-Focused)
############################################

def _log_diff_ewma(series: pd.Series, window: int) -> pd.Series:
    """
    Helper that does log -> diff -> EWMA(window).
    """
    logged = np.log(series)
    differenced = logged.diff()
    return differenced.ewm(halflife=window).mean()

def _diff_ewma(series: pd.Series, window: int) -> pd.Series:
    """
    Helper that does diff -> EWMA(window).
    """
    differenced = series.diff()
    return differenced.ewm(halflife=window).mean()

############################################
# Macro Feature Builder
############################################

def compute_macro_features(df_raw: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    """
    Build the macro features if columns are present in df_raw:
      1. Market return (EWMA of daily returns) => column "PBUS" or "SP500" or something
      2. VIX => log -> diff -> EWMA
      3. 2-year yield => diff -> EWMA
      4. 10-year minus 2-year => diff -> EWMA
    
    The user can adapt the column names as needed. Returns a DataFrame of macro features.
    """
    features = {}
    
    # 1) Market return (EWMA of daily returns)
    if "PBUS" in df_raw.columns:
        features["MarketReturn_EWMA_21"] = df_raw["PBUS"].ewm(halflife=window).mean()
    
    # 2) VIX => log -> diff -> EWMA(21)
    if "VIX" in df_raw.columns:
        # If VIX is zero-based (strictly positive) you can do log:
        features["VIX_log_diff_EWMA_21"] = _log_diff_ewma(df_raw["VIX"], window)
    
    # 3) 2-year yield => diff -> EWMA(21)
    if "2Y_Yield" in df_raw.columns:
        features["2Y_diff_EWMA_21"] = _diff_ewma(df_raw["2Y_Yield"], window)
    
    # 4) 10-year minus 2-year => diff -> EWMA(21)
    #    We'll assume columns "10Y" and "2Y" exist
    if "10Y-2Y_Spread" in df_raw.columns:
        features["10Y-2Y_EWMA_21"] = _diff_ewma(df_raw["10Y-2Y_Spread"], window)
    
    return pd.DataFrame(features)

############################################
# Feature Engineering Function
############################################

def feature_engineer(ret_ser: pd.Series, df_raw: pd.DataFrame = None, ver: str = "v0") -> pd.DataFrame:
    """
    v0, v1, or v2. 
    - v1: Factor-based features 
    - v2: Factor-based features + macro features (if columns exist)
    """
    
    if ver == "v1":
        if df_raw is None or "PBUS" not in df_raw.columns:
            raise ValueError("df_raw must contain a 'PBUS' column for market return.")
        
        # --------------------
        # Factor-based Features
        # --------------------
        factor_return = ret_ser
        market_return = df_raw["PBUS"]
        
        # Active return
        active_return = compute_active_return(factor_return, market_return)
        
        # Synthetic price for RSI/Stoch/MACD
        price = 100 * (1 + factor_return).cumprod()
        
        features = {}
        
        # 1) Active Return (EWMA)
        for window in [8, 21, 63]:
            features[f"ActiveReturn_EWMA_{window}"] = compute_active_return_ewma(active_return, window)
        
        # 2) RSI
        for window in [8, 21, 63]:
            features[f"RSI_{window}"] = compute_rsi(price, window)
        
        # 3) Stochastic Oscillator %K
        for window in [8, 21, 63]:
            features[f"StochOsc_%K_{window}"] = compute_stoch_osc(price, window)
        
        # 4) MACD
        for fast, slow in [(8, 21), (21, 63)]:
            features[f"MACD_{fast}_{slow}"] = compute_macd(price, fast, slow)
        
        # 5) Downside Deviation (log)
        features["DownsideDev_log_21"] = compute_downside_dev_log(active_return, 21)
        
        # 6) Active Market Beta
        features["ActiveMarketBeta_EWMA_21"] = compute_active_market_beta(active_return, market_return, 21)
        
        return pd.DataFrame(features)
    
    elif ver == "v2":
        # --------------------
        # Factor-based Features (from v1)
        # --------------------
        if df_raw is None or "PBUS" not in df_raw.columns:
            raise ValueError("df_raw must contain a 'PBUS' column for market return.")
        
        factor_return = ret_ser
        market_return = df_raw["PBUS"]
        
        active_return = compute_active_return(factor_return, market_return)
        price = 100 * (1 + factor_return).cumprod()
        
        factor_features = {}
        
        for window in [8, 21, 63]:
            factor_features[f"ActiveReturn_EWMA_{window}"] = compute_active_return_ewma(active_return, window)
        for window in [8, 21, 63]:
            factor_features[f"RSI_{window}"] = compute_rsi(price, window)
        for window in [8, 21, 63]:
            factor_features[f"StochOsc_%K_{window}"] = compute_stoch_osc(price, window)
        for fast, slow in [(8, 21), (21, 63)]:
            factor_features[f"MACD_{fast}_{slow}"] = compute_macd(price, fast, slow)
        factor_features["DownsideDev_log_21"] = compute_downside_dev_log(active_return, 21)
        factor_features["ActiveMarketBeta_EWMA_21"] = compute_active_market_beta(active_return, market_return, 21)
        
        df_factor_features = pd.DataFrame(factor_features)
        
        # --------------------
        # Macro Features
        # --------------------
        df_macro_features = compute_macro_features(df_raw, window=21)
        
        # Merge them together on the date index
        df_all_features = pd.concat([df_factor_features, df_macro_features], axis=1)
        return df_all_features
    
    else:
        # v0 fallback
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
    Modified DataLoader to support v1 (factor only) or v2 (factor + macro).
    """
    def __init__(self, file_path: str, ver: str = "v0", factor_col: str = None):
        self.file_path = file_path
        self.ver = ver
        self.factor_col = factor_col  # Which factor to treat as ret_ser
    
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
            ret_ser_raw = df_raw[self.factor_col]
        else:
            ret_ser_raw = df_raw.iloc[:, 0]

        # Compute features using the appropriate version
        df_features_all = feature_engineer(ret_ser_raw, df_raw, ver=self.ver)

        # Filter by date, drop rows with NaNs, then validate
        X = filter_date_range(df_features_all, start_date, end_date)
        X.dropna(inplace=True)
        valid_no_nan(X)

        self.X = X
        self.ret_ser = filter_date_range(ret_ser_raw, start_date, end_date).dropna()
        return self



############################################
# DataLoader Class When Market and Factor Data
############################################

class MergedDataLoader(BaseEstimator):
    """
    Loads factor_data.csv and market_data.csv, merges on date index,
    and runs feature_engineer with the chosen 'ver' (e.g., 'v1', 'v2').
    """
    def __init__(
        self,
        factor_file: str,
        market_file: str,
        ver: str = "v2",
        factor_col: str = None,
    ):
        """
        factor_file : path to factor_data.csv
        market_file : path to market_data.csv
        ver         : version of feature_engineer ('v0', 'v1', or 'v2')
        factor_col  : which column in factor_data.csv to treat as the factor return
        """
        self.factor_file = factor_file
        self.market_file = market_file
        self.ver = ver
        self.factor_col = factor_col

    def load(self, start_date: str = None, end_date: str = None):
        # 1) Read both CSVs
        df_factors = pd.read_csv(
            self.factor_file,
            parse_dates=["Date"],
            index_col="Date"
        ).dropna()

        df_market = pd.read_csv(
            self.market_file,
            parse_dates=["Date"],
            index_col="Date"
        ).dropna()

        # 2) Merge on date index (inner join so we only keep matching dates)
        df_merged = df_factors.join(df_market, how="inner")

        # 3) Decide which factor column to use for ret_ser
        if self.factor_col is None:
            # Default to first column in the factor file
            # (Since factor_data is on the left side of join, it’s still in the merged DataFrame.)
            ret_ser_raw = df_factors.iloc[:, 0]
        else:
            # Use the specified factor column from df_factors
            if self.factor_col not in df_factors.columns:
                raise ValueError(f"Factor column '{self.factor_col}' not found in factor_data.csv.")
            ret_ser_raw = df_factors[self.factor_col]

        # 4) Pass ret_ser and the merged DataFrame to feature_engineer
        df_features_all = feature_engineer(ret_ser_raw, df_merged, ver=self.ver)

        # 5) Filter by date range, drop NaNs and zeros, then validate
        X = filter_date_range(df_features_all, start_date, end_date)
        #X.replace(0, np.nan, inplace=True)  # Convert 0s to NaNs
        X.dropna(inplace=True)  # Drop rows with NaNs (including original NaNs and converted 0s)
        valid_no_nan(X)



        # Also filter ret_ser by date and drop NA
        ret_ser_filtered = filter_date_range(ret_ser_raw, start_date, end_date).dropna()

        # Align the indices of X and ret_ser_filtered by taking the intersection
        common_index = X.index.intersection(ret_ser_filtered.index)
        X = X.loc[common_index]
        ret_ser_filtered = ret_ser_filtered.loc[common_index]

        # -----  Create market_ser attribute  -----
        if "PBUS" not in df_merged.columns:
            raise ValueError("Market column 'PBUS' not found in market_data.csv.")
        market_ser_raw = df_merged["PBUS"]
        market_ser_filtered = filter_date_range(market_ser_raw, start_date, end_date).dropna()
        market_ser_filtered = market_ser_filtered.loc[common_index]  # align to common_index

        self.X = X
        self.ret_ser = ret_ser_filtered
        self.market_ser = market_ser_filtered  # <--- now you have a separate attribute
        self.active_ret = compute_active_return(ret_ser_filtered, market_ser_filtered)

        return self