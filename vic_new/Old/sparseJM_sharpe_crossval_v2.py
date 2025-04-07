#!/usr/bin/env python
# coding: utf-8

# In[23]:


# General
import numpy as np
import pandas as pd
import os
import jumpmodels.utils

# For loading data and feature engineering
from feature_25 import DataLoader, MergedDataLoader

# For data prep and pre-processing
from jumpmodels.utils import filter_date_range 
from jumpmodels.preprocess import StandardScalerPD, DataClipperStd 

# For model fit and prediction
from joblib import Parallel, delayed # allows parallel grid search on all 4 cores
from jumpmodels.sparse_jump import SparseJumpModel

# For plotting
from jumpmodels.plot import plot_regimes_and_cumret, plot_cumret
import matplotlib.pyplot as plt


# In[24]:


import importlib
import feature_25 
importlib.reload(feature_25)


# In[25]:


# Import data
# -------------------------

# Define file path
directory = r"C:\Users\victo\0_thesis_repo\thesis_vri_vp\data"
factor_file = os.path.join(directory, "factor_data.csv")
market_file = os.path.join(directory, "market_data.csv")

# Use DataLoader to generate features when we use factor and market data
data = MergedDataLoader(
    factor_file=factor_file,
    market_file=market_file,
    ver="v2",
    factor_col="VLUE"  # specify which column in factor_data.csv is your factor return
).load(start_date="2002-05-31", end_date="2025-02-24")

# Ensure all attributes have the same index before filtering
common_index = data.X.index.intersection(data.ret_ser.index).intersection(data.market_ser.index)
data.X = data.X.loc[common_index]
data.ret_ser = data.ret_ser.loc[common_index]
data.market_ser = data.market_ser.loc[common_index]

# Identify and drop dates where returns are exactly 0.0
zero_return_dates = data.ret_ser[data.ret_ser == 0.0].index

# Drop from all attributes to maintain alignment
data.X = data.X.drop(zero_return_dates, errors='ignore')
data.ret_ser = data.ret_ser.drop(zero_return_dates, errors='ignore')
data.market_ser = data.market_ser.drop(zero_return_dates, errors='ignore')  # Ensure market returns match


# print("Daily returns:", data.ret_ser)
# print("Engineered features:", data.X)

# factor_data = pd.read_csv(factor_file, parse_dates=["Date"], index_col="Date")

# Plot cumulative returns
# plot_cumret(factor_data["IWF"])
# plt.show()


# In[26]:


# Train/test split
# -------------------------

train_start, test_start = "2002-05-31", "2022-01-01" #actual start date of training will be in August due to 63 trading days required for EWMA
# filter dates
X_train = filter_date_range(data.X, start_date=train_start, end_date=test_start)
X_test = filter_date_range(data.X, start_date=test_start)
# print time split
train_start, train_end = X_train.index[[0, -1]]
test_start, test_end = X_test.index[[0, -1]]
print("Training starts at:", train_start, "and ends at:", train_end)
print("Testing starts at:", test_start, "and ends at:", test_end)

# Preprocessing
# -------------------------

# Clip the data within 3 standard deviations to mitigate the impact of outliers and standardize the clipped data (zero mean and unit variance)
clipper = DataClipperStd(mul=3.)
scalar = StandardScalerPD()
# fit on training data
X_train_processed = scalar.fit_transform(clipper.fit_transform(X_train))
# transform the test data
X_test_processed = scalar.transform(clipper.transform(X_test))


# In[27]:


def rolling_window_cv_sjm_long_short(
    lam, kappa, 
    X, 
    factor_returns, 
    market_returns, 
    n_splits=5, 
    min_train_size=5*252,
    cost_per_100pct=0.0005,  # 5 bps cost per 100% turnover
    annual_threshold=0.05    # ±5% annual threshold
):
    """
    Perform rolling expanding-window cross-validation using a
    long–short strategy evaluation as described in Shu et al. (2025).

    Parameters
    ----------
    lam : float
        The jump penalty parameter (lambda).
    kappa : float
        The sqrt of max features (kappa). The model will use int(kappa^2) features.
    X : pd.DataFrame
        Feature matrix (indexed by date).
    factor_returns : pd.Series
        Daily returns for the factor (indexed by date).
    market_returns : pd.Series
        Daily returns for the corresponding market or benchmark (indexed by date).
    n_splits : int, optional
        Number of cross-validation folds (default is 5).
    min_train_size : int, optional
        Minimum number of samples in the training set (default is ~5 years).
    cost_per_100pct : float, optional
        Transaction cost for a 100% position turnover (default is 5 bps).
    annual_threshold : float, optional
        Threshold (±5%) in annualized expected returns for deciding full long/short.

    Returns
    -------
    float
        The average Sharpe ratio across all cross-validation folds
        for the hypothetical long–short strategy.
    """
    # Compute the split size for rolling expanding-window cross-validation
    split_size = (len(X) - min_train_size) // n_splits

    # Compute the number of maximum features based on kappa
    max_feats = int(kappa**2)

    # List to store Sharpe ratios for each validation fold
    sharpe_scores = []

    # Helper function to determine position based on expected annualized return
    def position_from_expected_return(ann_ret, threshold=annual_threshold):
        """
        Maps the expected annualized active return to a position in [-1, 1].
        If ann_ret > threshold, fully long = +1.
        If ann_ret < -threshold, fully short = -1.
        Else linearly scale between -1 and +1.
        """
        if ann_ret > threshold:
            return 1.0
        elif ann_ret < -threshold:
            return -1.0
        else:
            return ann_ret / threshold  # Scale linearly between -1 and +1

    # Loop over cross-validation splits
    for i in range(n_splits):
        # Define training and validation split
        train_end = min_train_size + i * split_size
        X_train_cv = X.iloc[:train_end]  # # selects the first train_end rows from X and y, creating a training dataset.
        X_val_cv   = X.iloc[train_end:]  # selectes the rows after train_end and forward to be used for out of sample validation

        y_train_cv = factor_returns.iloc[:train_end]  # Training factor returns
        y_val_cv   = factor_returns.iloc[train_end:]  # Validation factor returns

        m_train_cv = market_returns.iloc[:train_end]  # Training market returns
        m_val_cv   = market_returns.iloc[train_end:]  # Validation market returns

        # Fit Sparse Jump Model (SJM) on training data
        model = SparseJumpModel(
            n_components=2,  # Model assumes two regimes
            max_feats=max_feats,  # Limit number of features
            jump_penalty=lam,  # Regularization parameter for jumps
            cont=False,  # Discrete jump model
            max_iter=30  # Maximum iterations for fitting
        )
        model.fit(X_train_cv, y_train_cv, sort_by="cumret")  # Fit model using cumulative returns as sorting criterion

        # Get in-sample regime predictions for training data
        train_states = model.predict(X_train_cv)

        # Compute daily active returns in training set (factor return minus market return)
        train_active_ret = y_train_cv - m_train_cv

        # Compute expected annualized return for each state
        unique_states = np.unique(train_states)
        state_to_expected = {}
        for st in unique_states:
            mask_st = (train_states == st)  # Identify samples in the state
            if mask_st.sum() > 0:
                avg_daily = train_active_ret[mask_st].mean()  # Average daily return for the state
                ann_ret = avg_daily * 252.0  # Annualized return
            else:
                ann_ret = 0.0  # Default to zero if state is empty
            state_to_expected[st] = ann_ret  # Store expected return for state

        # Predict states in validation set
        val_states = model.predict(X_val_cv)

        # Compute strategy performance in validation set
        val_active_ret = y_val_cv - m_val_cv  # Compute active return in validation
        val_positions = np.zeros(len(val_states))  # Store factor exposure in [-1, 1]
        strategy_ret = np.zeros(len(val_states))  # Track daily returns (net of costs)

        prev_position = 0.0  # Initialize previous position for turnover calculation

        for t in range(len(val_states)):
            st = val_states[t]  # Get predicted state for the day
            exp_ann_ret = state_to_expected.get(st, 0.0)  # Get expected return for state
            position = position_from_expected_return(exp_ann_ret, annual_threshold)  # Determine position

            # Compute daily turnover cost
            turnover = abs(position - prev_position) * 2.0  # Turnover is doubled for long/short trades
            cost = turnover * cost_per_100pct  # Multiply by cost per 100% turnover

            # Compute daily profit and loss (PnL) net of transaction cost
            daily_pnl = position * val_active_ret.iloc[t]  # Daily PnL from active return
            daily_net = daily_pnl - cost  # Subtract transaction costs

            strategy_ret[t] = daily_net  # Store net return
            prev_position = position  # Update previous position

        # Compute Sharpe ratio for the validation period
        avg_ret = np.mean(strategy_ret)  # Mean daily return
        std_ret = np.std(strategy_ret, ddof=1)  # Standard deviation of daily returns
        if std_ret == 0:
            val_sharpe = 0.0  # Avoid division by zero
        else:
            val_sharpe = (avg_ret / std_ret) * np.sqrt(252.0)  # Annualized Sharpe ratio

        sharpe_scores.append(val_sharpe)  # Store Sharpe ratio for this fold

    return np.mean(sharpe_scores)  # Return average Sharpe ratio across folds

# Example usage in parallel cross-validation:
lambda_values = np.logspace(0.5, 2, 5)  # Generate λ values from 1 to 100
kappa_values  = np.linspace(1, np.sqrt(X_train_processed.shape[1]), 5)  # Generate κ values

# Assign market return series
market_ser = data.market_ser  # Ensure market returns are aligned with factor returns

# Perform cross-validation in parallel using joblib
results = Parallel(n_jobs=4)(
    delayed(rolling_window_cv_sjm_long_short)(
        lam, kappa, 
        X_train_processed, 
        factor_returns=data.ret_ser, 
        market_returns=market_ser
    )
    for lam in lambda_values  # Iterate over lambda values
    for kappa in kappa_values  # Iterate over kappa values
)

# Identify the best lambda and kappa combination
best_index = np.argmax(results)  # Find index of max Sharpe ratio
best_lambda = lambda_values[best_index // len(kappa_values)]  # Extract corresponding lambda
best_kappa = kappa_values[best_index % len(kappa_values)]  # Extract corresponding kappa
max_feats_best = int(best_kappa**2)  # Compute best max features

# Print optimal hyperparameters
print(f"Best Jump Penalty (λ): {best_lambda}")
print(f"Best Max Features (κ²): {max_feats_best}")


# In[28]:


# -------------------------
# Fit the Sparse Jump Model
# -------------------------


# **Final Model Training with Best (λ, κ) Values**
best_model = SparseJumpModel(n_components=2, max_feats=max_feats_best, jump_penalty=best_lambda, cont=False, max_iter=30)
best_model.fit(X_train_processed, data.ret_ser, sort_by="cumret")



# In[29]:


# **Predict and Plot Results**

# factor_data = pd.read_csv(factor_file, parse_dates=["Date"], index_col="Date")

predicted_states = best_model.predict(X_train_processed)

# print(predicted_states.head())
# print(factor_data["VLUE"].head())
# print(predicted_states.tail())
# print(factor_data["VLUE"].tail())

ax, ax2 = plot_regimes_and_cumret(predicted_states, data.ret_ser)
ax.set(title=f"Best SJM ($\\lambda$={best_lambda}, $\\kappa^2$={max_feats_best})")
plt.show()


# In[30]:


# Predict the states on the in-sample data
print("Predicted states (in-sample):")
print(predicted_states)

# Print the feature weights (sparse weights)
print("SJM Feature Weights:")
print(best_model.feat_weights)


# In[31]:


# # check constraints

# w = best_model.feat_weights ** 2
# l1 = w.sum()
# l2 = np.sqrt((w**2).sum())

# print("L1 norm =", l1)
# print("L2 norm =", l2)
# print("Expected L1 (≈ kappa):", np.sqrt(best_model.max_feats))

w_internals = best_model.w  # The internal lasso vector

l1_w = w_internals.sum()
l2_w = np.sqrt((w_internals**2).sum())

print("L1 norm of w_internals =", l1_w)
print("L2 norm of w_internals =", l2_w)
print("Expected L1 (≈ kappa) =", np.sqrt(best_model.max_feats))


# In[32]:


# print("\n\n".join(In[i] for i in range(1, len(In))))
# check constraints


# In[33]:


df = pd.DataFrame(data.ret_ser)

print(df)

print("Number of exact duplicate rows:", df.duplicated().sum())
# Inspect them:
print(df[df.duplicated(keep=False)].sort_index().head(10))


# Convert ret_ser to a DataFrame (if it isn’t already)
df = pd.DataFrame(data.ret_ser).copy()
df.columns = ['ret_ser']  # Rename the column for clarity
df = df.reset_index().rename(columns={'index': 'Date'})  # Ensure the date is a column

# Create a block identifier that increments each time the return changes
df['block'] = (df['ret_ser'] != df['ret_ser'].shift(1)).cumsum()

# Group by the block to compute the start and end dates, value, and length of each run
runs = df.groupby('block', as_index=False).agg(
    start_date=('Date', 'first'),
    end_date=('Date', 'last'),
    value=('ret_ser', 'first'),
    length=('ret_ser', 'size')
)

# Filter for runs that last more than one day (i.e. consecutive identical returns)
long_runs = runs[runs['length'] > 1]

if long_runs.empty:
    print("No consecutive identical returns found in ret_ser.")
else:
    print("Consecutive runs of identical returns in ret_ser:")
    print(long_runs)



# In[34]:


import pandas as pd

# --- Identify problematic dates in ret_ser ---

# Convert ret_ser to a DataFrame (if not already)
df_ret = pd.DataFrame(data.ret_ser).copy()
df_ret.columns = ['ret_ser']
df_ret = df_ret.reset_index().rename(columns={'index': 'Date'})

# Create a block identifier: each time the return changes, increment the block number
df_ret['block'] = (df_ret['ret_ser'] != df_ret['ret_ser'].shift(1)).cumsum()

# Group by the block to capture runs of identical returns
runs_ret = df_ret.groupby('block', as_index=False).agg(
    start_date=('Date', 'first'),
    end_date=('Date', 'last'),
    value=('ret_ser', 'first'),
    length=('ret_ser', 'size')
)

# Filter for blocks longer than 1 day (problematic periods)
problematic_runs = runs_ret[runs_ret['length'] > 1]

print("Problematic consecutive returns in ret_ser:")
print(problematic_runs)

# --- Cross-check these dates in VLUE ---

print("\nCross-checking corresponding VLUE values:")

# Loop over each problematic block
for idx, row in problematic_runs.iterrows():
    s_date = row['start_date']
    e_date = row['end_date']
    print(f"\nFor ret_ser block from {s_date.date()} to {e_date.date()} (value = {row['value']} for {row['length']} days):")
    
    # Get the VLUE values for the same date range from factor_data
    vlue_slice = factor_data["VLUE"].loc[s_date:e_date]
    print(vlue_slice)


# In[36]:


# Check if there are still zero returns in data.ret_ser
zero_return_dates = data.ret_ser[data.ret_ser == 0.0]

# Print results
if zero_return_dates.empty:
    print("No zero returns found in data.ret_ser. ✅")
else:
    print(f"Found {len(zero_return_dates)} dates with zero returns in data.ret_ser. ⚠️")
    print(zero_return_dates)


# In[ ]:




