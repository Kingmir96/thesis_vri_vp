%load_ext autoreload
%autoreload 2
from utils_dir import get_curr_dir, include_home_dir
include_home_dir()

import pandas as pd

from jumpmodels.utils import filter_date_range        # useful helpers
from jumpmodels.jump import JumpModel                 # class of JM & CJM
from jumpmodels.sparse_jump import SparseJumpModel    # class of Sparse JM

from feature import DataLoader

data = DataLoader(ticker="NDX", ver="v0").load(start_date="2007-1-1", end_date="2024-09-30")

print("Daily returns stored in `data.ret_ser`:", "-"*50, sep="\n")
print(data.ret_ser, "-"*50, sep="\n")
print("Features stored in `data.X`:", "-"*50, sep="\n")
print(data.X)

train_start, test_start = "2007-1-1", "2022-1-1"
# filter dates
X_train = filter_date_range(data.X, start_date=train_start, end_date=test_start)
X_test = filter_date_range(data.X, start_date=test_start)
# print time split
train_start, train_end = X_train.index[[0, -1]]
test_start, test_end = X_test.index[[0, -1]]
print("Training starts at:", train_start, "and ends at:", train_end)
print("Testing starts at:", test_start, "and ends at:", test_end)

# Preprocessing
from jumpmodels.preprocess import StandardScalerPD, DataClipperStd
clipper = DataClipperStd(mul=3.)
scalar = StandardScalerPD()
# fit on training data
X_train_processed = scalar.fit_transform(clipper.fit_transform(X_train))
# transform the test data
X_test_processed = scalar.transform(clipper.transform(X_test))

# set the jump penalty
jump_penalty=50.
# initlalize the JM instance
jm = JumpModel(n_components=2, jump_penalty=jump_penalty, cont=False, )

# call .fit()
jm.fit(X_train_processed, data.ret_ser, sort_by="cumret")

print("Scaled Cluster Centroids:", pd.DataFrame(jm.centers_, index=["Bull", "Bear"], columns=X_train.columns), sep="\n" + "-"*50 + "\n")

from jumpmodels.plot import plot_regimes_and_cumret, savefig_plt

ax, ax2 = plot_regimes_and_cumret(jm.labels_, data.ret_ser, n_c=2, start_date=train_start, end_date=train_end, )
ax.set(title=f"In-Sample Fitted Regimes by the JM ($\\lambda$={jump_penalty})")
savefig_plt(f"{get_curr_dir()}/plots/JM_lambd-{jump_penalty}_train.pdf")

# reset jump_penalty to zero
jump_penalty=0.
jm.set_params(jump_penalty=jump_penalty)
print("The jump penalty of the JM instance has been reset to: jm.jump_penalty =", jm.jump_penalty)

# refit
jm.fit(X_train_processed, data.ret_ser, sort_by="cumret")

# plot
ax, ax2 = plot_regimes_and_cumret(jm.labels_, data.ret_ser, n_c=2, start_date=train_start, end_date=train_end, )
ax.set(title=f"In-Sample Fitted Regimes by the JM ($\\lambda$={jump_penalty})")
savefig_plt(f"{get_curr_dir()}/plots/JM_lambd-{jump_penalty}_train.pdf")

# refit
jump_penalty=50.
jm.set_params(jump_penalty=jump_penalty).fit(X_train_processed, data.ret_ser, sort_by="cumret")
# make online inference 
labels_test_online = jm.predict_online(X_test_processed)

# plot and save
ax, ax2 = plot_regimes_and_cumret(labels_test_online, data.ret_ser, n_c=2, start_date=test_start, end_date=test_end, )
ax.set(title=f"Out-of-Sample Online Inferred Regimes by the JM ($\\lambda$={jump_penalty})")
savefig_plt(f"{get_curr_dir()}/plots/JM_lambd-{jump_penalty}_test_online.pdf")

# make inference using all test data
labels_test = jm.predict(X_test_processed)
# plot
ax, ax2 = plot_regimes_and_cumret(labels_test, data.ret_ser, n_c=2, start_date=test_start, end_date=test_end, )
_ = ax.set(title=f"Out-of-Sample Predicted Regimes by the JM Using All Test Data ($\\lambda$={jump_penalty})")

jump_penalty=600.
cjm = JumpModel(n_components=2, jump_penalty=jump_penalty, cont=True)

cjm.fit(X_train_processed, data.ret_ser, sort_by="cumret")

# plot
ax, ax2 = plot_regimes_and_cumret(cjm.proba_, data.ret_ser, n_c=2, start_date=train_start, end_date=train_end, )
ax2.set(ylabel="Regime Probability")
ax.set(title=f"In-Sample Fitted Regimes by the CJM ($\\lambda$={jump_penalty})")
savefig_plt(f"{get_curr_dir()}/plots/CJM_lambd-{jump_penalty}_train.pdf")

# online inference
proba_test_online = cjm.predict_proba_online(X_test_processed)

# plot
ax, ax2 = plot_regimes_and_cumret(proba_test_online, data.ret_ser, start_date=test_start, end_date=test_end, )
ax2.set(ylabel="Regime Probability")
ax.set(title=f"Out-of-Sample Online Inferred Regimes by the CJM ($\\lambda$={jump_penalty})")
savefig_plt(f"{get_curr_dir()}/plots/CJM_lambd-{jump_penalty}_test_online.pdf")

max_feats=3.
jump_penalty=50.
# init sjm instance
sjm = SparseJumpModel(n_components=2, max_feats=max_feats, jump_penalty=jump_penalty, )
# fit
sjm.fit(X_train_processed, ret_ser=data.ret_ser, sort_by="cumret")

print("SJM Feature Weights:", "-"*50, sjm.feat_weights, sep="\n")

# plot
ax, ax2 = plot_regimes_and_cumret(sjm.labels_, data.ret_ser, n_c=2, start_date=train_start, end_date=train_end, )
ax.set(title=f"In-Sample Fitted Regimes by the SJM ($\\lambda$={jump_penalty}, $\\kappa^2$={max_feats})")
savefig_plt(f"{get_curr_dir()}/plots/SJM_lambd-{jump_penalty}_max-feats-{max_feats}_train.pdf")

# online inference
labels_test_online_sjm = sjm.predict_online(X_test_processed)

# plot
ax, ax2 = plot_regimes_and_cumret(labels_test_online_sjm, data.ret_ser, start_date=test_start, end_date=test_end, )
ax.set(title=f"Out-of-Sample Online Inferred Regimes by the SJM ($\\lambda$={jump_penalty}, $\\kappa^2$={max_feats})")
savefig_plt(f"{get_curr_dir()}/plots/SJM_lambd-{jump_penalty}_max-feats-{max_feats}_test_online.pdf")



