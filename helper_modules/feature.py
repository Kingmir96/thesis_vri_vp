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

def feature_engineer(ret_ser: pd.Series, ver: str = "v0") -> pd.DataFrame:
    """
    Engineer a set of features based on a return series.
    """
    if ver == "v0":
        feat_dict = {}
        hls = [5, 20, 60]
        for hl in hls:
            feat_dict[f"ret_{hl}"] = ret_ser.ewm(halflife=hl).mean()
            DD = compute_ewm_DD(ret_ser, hl)
            feat_dict[f"DD-log_{hl}"] = np.log(DD)
            feat_dict[f"sortino_{hl}"] = feat_dict[f"ret_{hl}"].div(DD)
        return pd.DataFrame(feat_dict)
    else:
        raise NotImplementedError()

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
        df_features_all = feature_engineer(ret_ser_raw, self.ver)
        
        # Filter by date
        X = filter_date_range(df_features_all, start_date, end_date)
        valid_no_nan(X)

        # Save attributes
        self.X = X
        self.ret_ser = filter_date_range(ret_ser_raw, start_date, end_date)
        
        return self
