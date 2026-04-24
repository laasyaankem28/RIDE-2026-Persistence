"""
ARIMA order tuning for RIDE 2026 inflow stations.

For each station, picks the best ARIMA(p,d,q) order on the full series via AIC
over a small grid, and saves the tuned per-station orders to
results/arima/tuned_orders.json so dacp_arima can reuse them without re-tuning.
"""

import json
import os
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================
DATA_DIR    = "A2_Experiment data"
RESULTS_DIR = "results/arima"

ORDER_GRID = [
    (0, 1, 1), (1, 1, 0), (1, 1, 1), (2, 1, 1), (1, 1, 2),
    (2, 1, 2), (0, 1, 2), (2, 1, 0), (3, 1, 1), (1, 1, 3),
    (2, 1, 3), (3, 1, 2),
]

FILES = [
    "Swift_Creek_Inflow_Data_FILLED.csv",
    "Bunchgrass_Meadow_Inflow_Data_FILLED.csv",
    "Touchet_Inflow_Data_FILLED.csv",
    "Paradise_Inflow_Data_FILLED.csv",
    "Easy_Pass_Inflow_Data_FILLED.csv",
]

os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# CORE
# ============================================================
def select_arima_order(series, grid):
    """Lowest-AIC order over the candidate grid; defaults to (1,1,1)."""
    best_order, best_aic = None, np.inf
    for order in grid:
        try:
            aic = ARIMA(series, order=order).fit().aic
            if aic < best_aic:
                best_order, best_aic = order, aic
        except Exception:
            continue
    return best_order if best_order is not None else (1, 1, 1)


# ============================================================
# MAIN
# ============================================================
def main():
    tuned_orders = {}

    for fname in FILES:
        station = fname.replace("_Inflow_Data_FILLED.csv", "").replace("_", " ")
        print(f"\n=== {station} ===")

        df = pd.read_csv(os.path.join(DATA_DIR, fname))
        series = pd.Series(df["DISCHRG"].values, name="DISCHRG")

        order = select_arima_order(series, ORDER_GRID)
        print(f"  selected order: {order}")
        tuned_orders[station] = order

    out_path = os.path.join(RESULTS_DIR, "tuned_orders.json")
    with open(out_path, "w") as f:
        json.dump({k: list(v) for k, v in tuned_orders.items()}, f, indent=2)
    print(f"\nTuned orders -> {out_path}")


if __name__ == "__main__":
    main()