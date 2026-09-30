# pca-

import numpy as np
import pandas as pd

# ---------- inputs ----------
tenors = ["1y", "2y", "3y", "4y", "5y", "7y", "10y", "15y", "20y", "30y"]
df = pd.read_csv("swaps.csv", index_col=0, parse_dates=True)[tenors]  # rows = dates, cols = tenors
UNITS_TO_BP = 100   # 100 if rates are in %, 1 if already in bp
N_PC = 3            # number of principal components to keep
HORIZON = 1         # change horizon in observations (1 = daily)
DRIVER = "5y"

# ---------- changes in bp ----------
dx = df.diff(HORIZON).dropna() * UNITS_TO_BP

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

##pca analysis 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- inputs ----------
tenors = ["1y", "2y", "3y", "4y", "5y", "7y", "10y", "15y", "20y", "30y"]
df = pd.read_csv("swaps.csv", index_col=0, parse_dates=True)[tenors]
UNITS_TO_BP = 100   # 100 if rates in %, 1 if already bp
HORIZON = 1         # change horizon in observations
N_PC = 3            # factors to show individually

def pca_attribution(start=None, end=None, n_pc=N_PC, horizon=HORIZON, plot=True):
    """Share of each tenor's variance explained by each PC over [start, end]."""
    lv = df.loc[start:end]
    dx = lv.diff(horizon).dropna() * UNITS_TO_BP

    cov = dx.cov().values
    w, V = np.linalg.eigh(cov)
    idx = w.argsort()[::-1]
    w, V = w[idx], V[:, idx]

    # sign convention: PC1 positive on average, PC2 positive at long end, PC3 positive at belly
    for k in range(min(n_pc, len(w))):
        ref = V[:, k].sum() if k == 0 else (V[-1, k] if k == 1 else V[len(w)//2, k])
        if ref < 0:
            V[:, k] *= -1

    # contribution of PC k to var(tenor i) = V[i,k]^2 * w[k] / cov[i,i]
    contrib = (V ** 2) * w[None, :] / np.diag(cov)[:, None]
    cols = [f"PC{k+1}" for k in range(n_pc)]
    out = pd.DataFrame(contrib[:, :n_pc], index=tenors, columns=cols)
    out["Other"] = 1 - out.sum(axis=1)

    names = {0: "outright/level", 1: "curve/slope", 2: "curvature"}
    summary = pd.DataFrame({
        "factor": [names.get(k, f"PC{k+1}") for k in range(n_pc)],
        "% total var": (w[:n_pc] / w.sum() * 100).round(1),
        "vol (bp/obs)": np.sqrt(w[:n_pc]).round(2),
    }, index=cols)
    loadings = pd.DataFrame(V[:, :n_pc], index=tenors, columns=cols)

    print(f"\n=== {dx.index[0].date()} to {dx.index[-1].date()}  ({len(dx)} obs) ===")
    print("\nFactor summary:\n", summary)
    print("\nShare of each tenor's variance explained by each factor (%):")
    print((out * 100).round(1))
    print("\nLoadings (bp move per 1 std-dev factor move):")
    print((loadings * np.sqrt(w[:n_pc])).round(2))

    if plot:
        ax = (out * 100).plot(kind="bar", stacked=True, figsize=(9, 4),
                              title=f"Variance attribution {dx.index[0].date()} to {dx.index[-1].date()}")
        ax.set_ylabel("% of variance"); ax.set_ylim(0, 100)
        plt.tight_layout(); plt.show()
    return out, summary, loadings

# ---------- single period ----------
res = pca_attribution("2023-01-01", "2024-12-31")

# ---------- compare several periods side by side ----------
periods = [("2019-01-01", "2020-12-31"),
           ("2021-01-01", "2022-12-31"),
           ("2023-01-01", "2024-12-31")]
cmp = {}
for s, e in periods:
    o, sm, _ = pca_attribution(s, e, plot=False)
    cmp[f"{s[:4]}-{e[:4]}"] = sm["% total var"]
print("\nFactor importance across periods (% total var):\n", pd.DataFrame(cmp))

# ---------- rolling view: how PC1/PC2/PC3 importance evolves ----------
def rolling_importance(window=250, step=21, n_pc=N_PC):
    dx = df.diff(HORIZON).dropna() * UNITS_TO_BP
    rows = {}
    for i in range(window, len(dx) + 1, step):
        w = np.sort(np.linalg.eigvalsh(dx.iloc[i - window:i].cov().values))[::-1]
        rows[dx.index[i - 1]] = w[:n_pc] / w.sum() * 100
    return pd.DataFrame(rows, index=[f"PC{k+1}" for k in range(n_pc)]).T

# roll = rolling_importance(); roll.plot(figsize=(9, 4), title="% total variance by PC (rolling)"); plt.show()
