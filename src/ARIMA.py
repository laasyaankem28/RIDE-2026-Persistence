"""ARIMA order tuning for the RIDE 2026 inflow stations.

For each station: log-transform discharge, pick d from an ADF test, then
grid-search (p, q) in parallel and keep the order with the lowest AIC.
"""

import json
import os
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore")

DATA_DIR = "A2_Experiment data"
RESULTS_DIR = "results/arima"

# Asymmetric (p, q) grid — low AR, higher MA tends to fit streamflow residuals
# better after differencing.
ORDER_GRID = [
    (0, 1), (0, 2), (0, 3),
    (1, 0), (1, 1), (1, 2), (1, 3),
    (2, 1), (2, 2), (2, 3),
    (3, 1), (3, 2),
]

FILES = [
    "Swift_Creek_Inflow_Data_FILLED.csv",
    "Bunchgrass_Meadow_Inflow_Data_FILLED.csv",
    "Touchet_Inflow_Data_FILLED.csv",
    "Paradise_Inflow_Data_FILLED.csv",
    "Easy_Pass_Inflow_Data_FILLED.csv",
]


def choose_d(series, alpha=0.05):
    p_value = adfuller(series, autolag="AIC")[1]
    return 0 if p_value < alpha else 1


def fit_order(series, order):
    try:
        fit = ARIMA(series, order=order).fit()
        return {"order": order, "aic": fit.aic, "bic": fit.bic}
    except Exception:
        return {"order": order, "aic": np.inf, "bic": np.inf}


def select_order(series, d, n_jobs=-1):
    grid = [(p, d, q) for p, q in ORDER_GRID]
    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(fit_order)(series, order) for order in grid
    )
    best = min(results, key=lambda r: r["aic"])
    if best["aic"] == np.inf:
        return {"order": (1, d, 1), "aic": np.nan, "bic": np.nan}
    return best


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    tuned_orders = {}
    diagnostics = []

    for fname in FILES:
        station = fname.replace("_Inflow_Data_FILLED.csv", "").replace("_", " ")
        print(f"\n=== {station} ===")

        df = pd.read_csv(os.path.join(DATA_DIR, fname))
        series = np.log1p(df["DISCHRG"].values)

        d = choose_d(series)
        print(f"  ADF -> d = {d}")

        best = select_order(series, d)
        order = best["order"]
        print(f"  Best: ARIMA{order}  AIC={best['aic']:.2f}  BIC={best['bic']:.2f}")

        tuned_orders[station] = order
        diagnostics.append({
            "station": station,
            "d_selected": d,
            "p": order[0], "d": order[1], "q": order[2],
            "aic": best["aic"], "bic": best["bic"],
        })

    out_json = os.path.join(RESULTS_DIR, "tuned_orders.json")
    with open(out_json, "w") as f:
        json.dump({k: list(v) for k, v in tuned_orders.items()}, f, indent=2)
    print(f"\nTuned orders -> {out_json}")

    diag_csv = os.path.join(RESULTS_DIR, "tuning_diagnostics.csv")
    pd.DataFrame(diagnostics).to_csv(diag_csv, index=False)
    print(f"Diagnostics  -> {diag_csv}")


if __name__ == "__main__":
    main()
