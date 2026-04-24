import json
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================
TRAINING_SIZE = 40               # rolling window length (steps)
DATA_DIR      = "A2_Experiment data"
PLOT_DIR      = "plots/dacp"
RESULTS_DIR   = "results/dacp"
TUNED_ORDERS  = "results/arima/tuned_orders.json"
DEFAULT_ORDER = (2, 1, 2)

T_LOW   = 0.20
T_HIGH  = 0.75
T_INIT  = 0.55
K_INIT  = 0.05
EPS     = 1e-10

MU_K_HIT  = +0.01
MU_K_MISS = -0.02
MU_T_HIT  = +0.008
MU_T_MISS = -0.004

FILES = [
    "Swift_Creek_Inflow_Data_FILLED.csv",
    "Bunchgrass_Meadow_Inflow_Data_FILLED.csv",
    "Touchet_Inflow_Data_FILLED.csv",
    "Paradise_Inflow_Data_FILLED.csv",
    "Easy_Pass_Inflow_Data_FILLED.csv",
]

os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================
# CORE
# ============================================================
def load_orders():
    if os.path.exists(TUNED_ORDERS):
        with open(TUNED_ORDERS) as f:
            raw = json.load(f)
        return {k: tuple(v) for k, v in raw.items()}
    return {}


def _forecast_one(fit):
    return float(np.asarray(fit.forecast(steps=1))[0])


def run_dacp(series, order):
    """Rolling DACP+ARIMA on one series. Integer-indexed."""
    y = series.values
    n = len(y)

    t_adap = T_INIT
    k_adap = K_INIT

    preds, lowers, uppers, widths = [], [], [], []
    truths, steps_kept = [], []
    covered = []
    t_trace, k_trace = [], []

    prev_q = None

    for i in range(TRAINING_SIZE, n):
        window = y[i - TRAINING_SIZE:i]

        n_cal = max(4, int(t_adap * len(window)))
        n_fit = len(window) - n_cal
        fit_part = window[:n_fit]
        cal_part = window[n_fit:]

        try:
            fit = ARIMA(fit_part, order=order).fit()
        except Exception:
            fit = None

        if fit is not None:
            cal_preds = []
            ext = fit

            for v in cal_part:
                try:
                    yhat = _forecast_one(ext)
                except Exception:
                    yhat = float(fit_part[-1])
                cal_preds.append(yhat)

                try:
                    ext = ext.append([float(v)], refit=False)
                except Exception:
                    pass

            cal_preds = np.asarray(cal_preds, dtype=float)
            abs_resid = np.abs(cal_preds - cal_part)

            if len(abs_resid) >= 5:
                clip_hi = np.percentile(abs_resid, 95)
                abs_resid = np.clip(abs_resid, 0, clip_hi)

            q_level = float(np.clip(1.0 - k_adap, 0.05, 0.99))
            q_raw = float(np.quantile(abs_resid, q_level))

            if prev_q is None:
                q = q_raw
            else:
                q = 0.6 * prev_q + 0.4 * q_raw
            prev_q = q

            try:
                yhat_next = _forecast_one(ext)
            except Exception:
                yhat_next = float(window[-1])

        else:
            yhat_next = float(window[-1])
            q_raw = float(np.std(window))
            if prev_q is None:
                q = q_raw
            else:
                q = 0.6 * prev_q + 0.4 * q_raw
            prev_q = q

        y_true = float(y[i])
        lo = max(yhat_next - q, 0.0)
        hi = yhat_next + q

        preds.append(yhat_next)
        truths.append(y_true)
        lowers.append(lo)
        uppers.append(hi)
        widths.append(hi - lo)
        steps_kept.append(i)

        is_in = lo <= y_true <= hi
        covered.append(is_in)

        if is_in:
            k_adap = float(np.clip(k_adap + MU_K_HIT,  0.01, 0.30))
            t_adap = float(np.clip(t_adap + MU_T_HIT,  T_LOW, T_HIGH))
        else:
            k_adap = float(np.clip(k_adap + MU_K_MISS, 0.01, 0.30))
            t_adap = float(np.clip(t_adap + MU_T_MISS, T_LOW, T_HIGH))

        k_trace.append(k_adap)
        t_trace.append(t_adap)

        step = i - TRAINING_SIZE
        if step % 100 == 0:
            print(
                f"  step {step:>5}/{n - TRAINING_SIZE}   "
                f"cov={np.mean(covered):.2f}   q={q:.3f}   "
                f"k={k_adap:.3f}   t={t_adap:.3f}"
            )

    return {
        "steps":   np.array(steps_kept),
        "truth":   np.array(truths),
        "pred":    np.array(preds),
        "lower":   np.array(lowers),
        "upper":   np.array(uppers),
        "width":   np.array(widths),
        "covered": np.array(covered),
        "t_trace": np.array(t_trace),
        "k_trace": np.array(k_trace),
    }


def evaluate(r):
    mse  = float(mean_squared_error(r["truth"], r["pred"]))
    rmse = float(np.sqrt(mse))
    mae  = float(np.mean(np.abs(r["truth"] - r["pred"])))
    cov  = float(np.mean(r["covered"]))
    aw   = float(np.mean(r["width"]))
    mre  = float(np.mean(np.abs((r["truth"] - r["pred"]) /
                                (r["truth"] + EPS))))
    return {"RMSE": rmse, "MSE": mse, "MAE": mae,
            "Coverage": cov, "AvgWidth": aw, "MRE": mre}


# ============================================================
# PLOTTING
# ============================================================
COLOR_OBS  = "#1f4e79"
COLOR_PRED = "#c0504d"
COLOR_FILL = "#ffb74d"


def plot_station(station, order, r, m, out_path):
    """Five-panel diagnostic plot with integer step axis."""
    steps = r["steps"]

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(
        f"DACP + ARIMA{order}   |   training window = {TRAINING_SIZE} steps   |   {station}",
        fontsize=15, fontweight="bold", y=0.995,
    )
    gs = fig.add_gridspec(
        3, 3, hspace=0.55, wspace=0.40, height_ratios=[0.90, 1.30, 0.90]
    )
    ax_main  = fig.add_subplot(gs[0, :])
    ax_zoom  = fig.add_subplot(gs[1, :])
    ax_cov   = fig.add_subplot(gs[2, 0])
    ax_width = fig.add_subplot(gs[2, 1])
    ax_param = fig.add_subplot(gs[2, 2])

    # ── Panel 1: Main view (last 365 steps)
    n_show = min(365, len(steps))
    sl = slice(-n_show, None)

    ax_main.fill_between(steps[sl], r["lower"][sl], r["upper"][sl],
                         color=COLOR_FILL, alpha=0.45, label="DACP interval")
    ax_main.plot(steps[sl], r["truth"][sl], color=COLOR_OBS, lw=1.2, label="Observed")
    ax_main.plot(steps[sl], r["pred"][sl],  color=COLOR_PRED, lw=0.9, ls="--",
                 alpha=0.85, label="ARIMA forecast")

    miss = ~r["covered"][sl]
    if miss.any():
        ax_main.scatter(steps[sl][miss], r["truth"][sl][miss], s=10, color="black",
                        zorder=5, label="Miss")

    y_window = np.concatenate([r["truth"][sl], r["lower"][sl], r["upper"][sl]])
    ymin, ymax = y_window.min(), y_window.max()
    pad = 0.10 * (ymax - ymin if ymax > ymin else max(1.0, abs(ymax)))
    ax_main.set_ylim(ymin - pad, ymax + pad)

    ax_main.set_title(
        "One-step-ahead forecast with adaptive prediction interval (last 365 steps)",
        fontsize=11,
    )
    ax_main.set_xlabel("Step")
    ax_main.set_ylabel("Discharge")
    ax_main.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax_main.grid(alpha=0.3)

    metrics_text = (
        f"RMSE      = {m['RMSE']:.3f}\n"
        f"MAE       = {m['MAE']:.3f}\n"
        f"Coverage  = {m['Coverage']:.2%}\n"
        f"Avg width = {m['AvgWidth']:.3f}\n"
        f"MRE       = {m['MRE']:.3f}"
    )
    ax_main.text(
        0.005, 0.97, metrics_text, transform=ax_main.transAxes,
        fontsize=9, va="top", family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow",
                  ec="gray", alpha=0.95),
    )

    # ── Panel 2: Zoom (last 216 steps)
    n_zoom = min(216, len(steps))
    zsl = slice(-n_zoom, None)

    ax_zoom.fill_between(steps[zsl], r["lower"][zsl], r["upper"][zsl],
                         color=COLOR_FILL, alpha=0.45, label="DACP interval")
    ax_zoom.plot(steps[zsl], r["truth"][zsl], color=COLOR_OBS, lw=2.0, label="Observed")
    ax_zoom.plot(steps[zsl], r["pred"][zsl],  color=COLOR_PRED, lw=1.6, ls="--",
                 alpha=0.9, label="Forecast")

    miss_z = ~r["covered"][zsl]
    if miss_z.any():
        ax_zoom.scatter(steps[zsl][miss_z], r["truth"][zsl][miss_z], s=70,
                        color="black", marker="x", lw=2.0, zorder=6, label="Miss")

    ax_zoom.set_title(f"Zoom – last {n_zoom} steps", fontsize=12, fontweight="bold")
    ax_zoom.set_xlabel("Step")
    ax_zoom.set_ylabel("Discharge")
    ax_zoom.legend(fontsize=10, loc="upper right")
    ax_zoom.grid(alpha=0.3)

    # ── Panel 3: Running coverage
    running_cov = np.cumsum(r["covered"]) / np.arange(1, len(r["covered"]) + 1)
    ax_cov.plot(steps, running_cov, color=COLOR_OBS, lw=1.2)
    ax_cov.axhline(1 - K_INIT, color=COLOR_PRED, lw=1.0, ls="--",
                   label=f"target {1 - K_INIT:.0%}")
    ax_cov.set_ylim(0, 1.05)
    ax_cov.set_title("Running coverage", fontsize=11)
    ax_cov.set_xlabel("Step")
    ax_cov.set_ylabel("Fraction in interval")
    ax_cov.legend(fontsize=8)
    ax_cov.grid(alpha=0.3)

    # ── Panel 4: Interval width
    ax_width.plot(steps, r["width"], color=COLOR_OBS, lw=0.8)
    ax_width.set_title("Prediction-interval width", fontsize=11)
    ax_width.set_xlabel("Step")
    ax_width.set_ylabel("Upper − lower")
    ax_width.grid(alpha=0.3)

    # ── Panel 5: Adaptive parameters
    ax_param.plot(steps, r["k_trace"], color=COLOR_OBS, lw=1.0, label="k_adap")
    ax_param.plot(steps, r["t_trace"], color=COLOR_PRED, lw=1.0, label="t_adap")
    ax_param.set_title("Adaptive parameters", fontsize=11)
    ax_param.set_xlabel("Step")
    ax_param.set_ylabel("Value")
    ax_param.legend(fontsize=8)
    ax_param.grid(alpha=0.3)

    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_summary(summary_df, out_path):
    stations = summary_df["station"].values
    x = np.arange(len(stations))

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    fig.suptitle("DACP + ARIMA – cross-station summary",
                 fontsize=14, fontweight="bold")

    axes[0].bar(x, summary_df["RMSE"].values, color=COLOR_OBS)
    axes[0].set_title("RMSE (lower = better)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(stations, rotation=25, ha="right")
    axes[0].grid(alpha=0.3, axis="y")

    axes[1].bar(x, summary_df["Coverage"].values, color=COLOR_OBS)
    axes[1].axhline(1 - K_INIT, color=COLOR_PRED, lw=1.2, ls="--",
                    label=f"target {1 - K_INIT:.0%}")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Coverage rate")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(stations, rotation=25, ha="right")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3, axis="y")

    axes[2].bar(x, summary_df["AvgWidth"].values, color=COLOR_OBS)
    axes[2].set_title("Avg interval width (lower = tighter)")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(stations, rotation=25, ha="right")
    axes[2].grid(alpha=0.3, axis="y")

    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================
def main():
    orders = load_orders()
    if not orders:
        print(f"[warn] {TUNED_ORDERS} not found – using DEFAULT_ORDER {DEFAULT_ORDER}"
              f" for every station. Run ARIMA.py first to tune.")

    summary = []

    for fname in FILES:
        station = fname.replace("_Inflow_Data_FILLED.csv", "").replace("_", " ")
        order = orders.get(station, DEFAULT_ORDER)
        print(f"\n=== {station}   order={order} ===")

        df = pd.read_csv(os.path.join(DATA_DIR, fname))
        series = pd.Series(df["DISCHRG"].values, name="DISCHRG")

        r = run_dacp(series, order)
        m = evaluate(r)
        print(f"  RMSE={m['RMSE']:.3f}   Cov={m['Coverage']:.2%}   "
              f"Width={m['AvgWidth']:.3f}   MRE={m['MRE']:.3f}")

        out_png = os.path.join(PLOT_DIR, fname.replace(".csv", "_DACP.png"))
        plot_station(station, order, r, m, out_png)
        print(f"  -> {out_png}")

        pd.DataFrame({
            "step":      r["steps"],
            "observed":  r["truth"],
            "predicted": r["pred"],
            "lower":     r["lower"],
            "upper":     r["upper"],
            "width":     r["width"],
            "covered":   r["covered"],
            "k_adap":    r["k_trace"],
            "t_adap":    r["t_trace"],
        }).to_csv(
            os.path.join(RESULTS_DIR,
                         fname.replace(".csv", "_DACP_pointwise.csv")),
            index=False,
        )

        summary_path = os.path.join(
            RESULTS_DIR, fname.replace(".csv", "_DACP_summary.txt")
        )
        with open(summary_path, "w") as f:
            f.write(f"Station          : {station}\n")
            f.write(f"ARIMA order      : {order}\n")
            f.write(f"Training window  : {TRAINING_SIZE} steps\n")
            f.write(f"Initial k_adap   : {K_INIT}  (target coverage {1 - K_INIT:.0%})\n")
            f.write(f"Initial t_adap   : {T_INIT}\n")
            f.write(f"t_adap bounds    : [{T_LOW}, {T_HIGH}]\n")
            f.write("\n")
            f.write(f"RMSE             : {m['RMSE']:.4f}\n")
            f.write(f"MSE              : {m['MSE']:.4f}\n")
            f.write(f"MAE              : {m['MAE']:.4f}\n")
            f.write(f"Coverage rate    : {m['Coverage']:.4f}\n")
            f.write(f"Avg interval len : {m['AvgWidth']:.4f}\n")
            f.write(f"Mean rel. error  : {m['MRE']:.4f}\n")

        summary.append({"station": station, "order": str(order), **m})

    summary_df = pd.DataFrame(summary)
    summary_csv = os.path.join(RESULTS_DIR, "dacp_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    print("\n=== Cross-station summary ===")
    print(summary_df.to_string(index=False))
    print(f"-> {summary_csv}")

    plot_summary(summary_df, os.path.join(PLOT_DIR, "_summary_dacp.png"))


if __name__ == "__main__":
    main()