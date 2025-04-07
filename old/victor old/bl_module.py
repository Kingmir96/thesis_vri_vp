import numpy as np
import pandas as pd
from jumpmodels.utils import filter_date_range
from jumpmodels.preprocess import StandardScalerPD, DataClipperStd
from jumpmodels.sparse_jump import SparseJumpModel
import cvxpy as cp

def run_multifactor_BL_portfolio(
    factor_dict,
    market_series,
    cov_matrix,
    test_index,
    tracking_error=0.03,
    delta=2.5,  # risk aversion
    annual_threshold=0.05,
    cost_per_turnover=0.0005
):
    """
    Constructs a long-only Black-Litterman portfolio using regime signals from SJMs.
    
    Parameters:
    -----------
    factor_dict : dict
        Mapping from factor name to dict with keys:
        'ret': pd.Series of factor returns
        'states': pd.Series of online inferred regime state in test period
        'regime_returns': dict mapping state to average daily active return (train period)

    market_series : pd.Series
        Market return series for the test period

    cov_matrix : pd.DataFrame
        Covariance matrix of factor + market returns

    test_index : pd.DatetimeIndex
        Date index for the test period

    tracking_error : float
        Target tracking error vs. EW benchmark

    Returns:
    --------
    weights_df : pd.DataFrame
        Daily portfolio weights across factors (long-only, fully invested)
    """
    factors = list(factor_dict.keys())
    n_assets = len(factors) + 1  # 6 factors + 1 market
    assets = factors + ['Market']

    # Compute equally weighted (EW) benchmark
    ew_weights = np.ones(n_assets) / n_assets

    # Compute view matrix P and Omega scaling
    P = np.zeros((len(factors), n_assets))
    for i, fac in enumerate(factors):
        P[i, i] = 1    # long factor
        P[i, -1] = -1  # short market

    # Portfolio storage
    weights = pd.DataFrame(index=test_index, columns=assets)

    for t in test_index:
        q = []
        for fac in factors:
            state = factor_dict[fac]['states'].loc[t]
            ann_ret = factor_dict[fac]['regime_returns'].get(state, 0.0) * 252
            q.append(ann_ret)
        q = np.array(q)

        # Confidence scaling (Omega) based on target tracking error
        tau = 0.05
        omega_diag = np.diag(np.diag(P @ cov_matrix.values @ P.T)) * (1 / tau)
        omega = np.diag(np.diag(omega_diag))

        # BL posterior return
        pi = delta * cov_matrix @ pd.Series(ew_weights, index=assets)
        middle = np.linalg.inv(np.linalg.inv(tau * cov_matrix.values) + P.T @ np.linalg.inv(omega) @ P)
        mu_bl = middle @ (np.linalg.inv(tau * cov_matrix.values) @ pi.values + P.T @ np.linalg.inv(omega) @ q)

        # Solve long-only, fully invested mean-variance problem
        w = cp.Variable(n_assets)
        ret_expr = mu_bl.T @ w
        risk_expr = cp.quad_form(w, cov_matrix.values)
        prob = cp.Problem(
            cp.Maximize(ret_expr - (delta / 2) * risk_expr),
            [cp.sum(w) == 1, w >= 0]
        )
        prob.solve()

        weights.loc[t] = w.value

    return weights
