import numpy as np
import pandas as pd

# ---------- inputs ----------
tenors = ["1y", "2y", "3y", "4y", "5y", "7y", "10y", "15y", "20y", "30y"]
df = pd.read_csv("swaps.csv", index_col=0, parse_dates=True)[tenors].sort_index()
UNITS_TO_BP = 100   # 100 if rates are in %, 1 if already in bp
N_PC = 3            # number of principal components to keep
HORIZON = 1         # change horizon in observations (1 = daily)
DRIVER = "5y"

# ---------- changes in bp ----------
dx = df.diff(HORIZON) * UNITS_TO_BP

# ---------- flag first observation of each month (first business day in your data) ----------
month = df.index.to_period("M")
is_first = pd.Series(~month.duplicated(), index=df.index)

# a change is contaminated if its window (the last HORIZON observations) contains a first-of-month day
# (for HORIZON = 1 this is just the first-of-month day itself)
contaminated = is_first.astype(int).rolling(HORIZON, min_periods=1).sum() > 0

n_before = dx.dropna().shape[0]
dx = dx[~contaminated].dropna()
print(f"Dropped {n_before - len(dx)} roll-affected observations, {len(dx)} remain")

# ---------- PCA on covariance (keeps bp units) ----------
cov = dx.cov().values
eigval, eigvec = np.linalg.eigh(cov)
order = eigval.argsort()[::-1]
eigval, eigvec = eigval[order], eigvec[:, order]

print("Variance explained:", np.round(eigval / eigval.sum(), 3)[:N_PC + 2])
print("Cumulative (kept): ", round(eigval[:N_PC].sum() / eigval.sum(), 3))

# ---------- PCA-filtered covariance ----------
V, L = eigvec[:, :N_PC], eigval[:N_PC]
cov_k = V @ np.diag(L) @ V.T

# beta[i, j] = expected bp move in tenor i for a +1bp move in tenor j
beta = cov_k / np.diag(cov_k)[None, :]
grid = pd.DataFrame(beta, index=tenors, columns=tenors)
grid.index.name = "responds"; grid.columns.name = "driver"

# ---------- outputs ----------
pd.set_option("display.float_format", "{:.3f}".format)
print(f"\nMove in each tenor for +1bp in {DRIVER}:")
print(grid[DRIVER].to_frame(f"per 1bp {DRIVER}"))

print("\nFull grid (columns = driver, rows = responding tenor):")
print(grid)

# optional: loadings on each PC for sanity checking (level / slope / curvature)
loadings = pd.DataFrame(V, index=tenors, columns=[f"PC{i+1}" for i in range(N_PC)])
print("\nPC loadings:\n", loadings)
