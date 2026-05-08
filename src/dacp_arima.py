"""DACP + ARIMA coverage-target experiment across stations and 70/80/90/95% targets."""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore")

DATA_DIR = "A2_Experiment data"
RESULTS_DIR = "results/dacp"
PLOT_DIR = "plots/dacp"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

TRAINING_SIZE = 40
N_CAL_FIXED = 5
REFIT_EVERY = 100
T_LOW, T_HIGH, T_INIT = 0.20, 0.75, 0.55
MU_T_HIT, MU_T_MISS = +0.008, -0.004
EPS = 1e-10

EWM_ALPHA = 0.15
Q_CAP_MULT = 5
Q_CAP_WINDOW = 100

# DACP equilibrium coverage = miss / (hit + miss). Setting miss = target * SCALE
# and hit = (1 - target) * SCALE makes the loop converge to the target rate.
SCALE = 0.01
COVERAGE_TARGETS = {
    "70%": {"k_init": 0.30, "hit": 0.30 * SCALE, "miss": 0.70 * SCALE},
    "80%": {"k_init": 0.20, "hit": 0.20 * SCALE, "miss": 0.80 * SCALE},
    "90%": {"k_init": 0.10, "hit": 0.10 * SCALE, "miss": 0.90 * SCALE},
    "95%": {"k_init": 0.05, "hit": 0.05 * SCALE, "miss": 0.95 * SCALE},
}

COLORS = {"70%": "#e74c3c", "80%": "#e67e22", "90%": "#2980b9", "95%": "#27ae60"}

FILES = [
    "Swift_Creek_Inflow_Data_FILLED.csv",
    "Bunchgrass_Meadow_Inflow_Data_FILLED.csv",
    "Touchet_Inflow_Data_FILLED.csv",
    "Paradise_Inflow_Data_FILLED.csv",
    "Easy_Pass_Inflow_Data_FILLED.csv",
]


def log_t(x):
    return np.log1p(x)


def inv_t(x):
    return np.expm1(x)


def choose_order(s):
    d = 0 if adfuller(s, autolag="AIC")[1] < 0.05 else 1
    return (1, d, 1)


def fc1(fit):
    return float(np.asarray(fit.forecast(steps=1))[0])


def run_dacp(series, order, k_init, hit_step, miss_step):
    raw = series.values
    y = log_t(raw)
    n = len(y)

    k_adap = k_init
    t_adap = T_INIT
    fit_base = None
    ewm_q = None
    q_history = []

    preds, lowers, uppers, widths, truths, steps_k = [], [], [], [], [], []
    covered, t_trace, k_trace = [], [], []

    for i in range(TRAINING_SIZE, n):
        w = y[i - TRAINING_SIZE : i]
        step = i - TRAINING_SIZE

        fp = w[: TRAINING_SIZE - N_CAL_FIXED]
        cp = w[TRAINING_SIZE - N_CAL_FIXED :]

        if fit_base is None or step % REFIT_EVERY == 0:
            try:
                fit_base = ARIMA(fp, order=order).fit()
            except Exception:
                fit_base = None

        # Out-of-sample calibration: roll-append through the calibration window
        # to get genuine one-step-ahead errors instead of in-sample residuals.
        if fit_base is not None:
            ext = fit_base
            cps = []
            for v in cp:
                try:
                    yh = fc1(ext)
                except Exception:
                    yh = float(fp[-1])
                cps.append(yh)
                try:
                    ext = ext.append([float(v)], refit=False)
                except Exception:
                    pass
            ar = np.abs(np.array(cps) - cp)
            try:
                yn = fc1(ext)
            except Exception:
                yn = float(w[-1])
        else:
            ar = np.array([float(np.std(w))])
            yn = float(w[-1])

        ql = float(np.clip(1.0 - k_adap, 0.01, 0.999))
        q_raw = float(np.quantile(ar, ql))

        # EWM + rolling-median cap keep a single residual spike from blowing up
        # the interval width.
        ewm_q = q_raw if ewm_q is None else EWM_ALPHA * q_raw + (1 - EWM_ALPHA) * ewm_q
        q_history.append(q_raw)
        if len(q_history) > Q_CAP_WINDOW:
            q_history.pop(0)
        q_cap = Q_CAP_MULT * np.median(q_history)
        q = min(ewm_q, q_cap)

        pred = float(inv_t(np.array([yn]))[0])
        lo = max(float(inv_t(np.array([max(yn - q, 0)]))[0]), 0.0)
        hi = float(inv_t(np.array([yn + q]))[0])

        preds.append(pred)
        truths.append(float(raw[i]))
        lowers.append(lo)
        uppers.append(hi)
        widths.append(hi - lo)
        steps_k.append(i)

        is_in = lo <= raw[i] <= hi
        covered.append(is_in)

        if is_in:
            k_adap = float(np.clip(k_adap + hit_step, 0.001, 0.999))
            t_adap = float(np.clip(t_adap + MU_T_HIT, T_LOW, T_HIGH))
        else:
            k_adap = float(np.clip(k_adap - miss_step, 0.001, 0.999))
            t_adap = float(np.clip(t_adap + MU_T_MISS, T_LOW, T_HIGH))

        k_trace.append(k_adap)
        t_trace.append(t_adap)

        if step % 300 == 0:
            print(
                f"    step {step}/{n - TRAINING_SIZE}  cov={np.mean(covered):.2%}  "
                f"k={k_adap:.4f}  t={t_adap:.3f}",
                flush=True,
            )

    return {
        "steps": np.array(steps_k),
        "truth": np.array(truths),
        "pred": np.array(preds),
        "lower": np.array(lowers),
        "upper": np.array(uppers),
        "width": np.array(widths),
        "covered": np.array(covered),
        "t_trace": np.array(t_trace),
        "k_trace": np.array(k_trace),
    }


def evaluate(r):
    rmse = float(np.sqrt(mean_squared_error(r["truth"], r["pred"])))
    mae = float(np.mean(np.abs(r["truth"] - r["pred"])))
    cov = float(np.mean(r["covered"]))
    aw = float(np.mean(r["width"]))
    mre = float(np.mean(np.abs((r["truth"] - r["pred"]) / (r["truth"] + EPS))))
    return {"RMSE": rmse, "MAE": mae, "Coverage": cov, "AvgWidth": aw, "MRE": mre}


def plot_station(station, rbt, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        f"DACP+ARIMA — Coverage Target Experiment\n{station}",
        fontsize=14, fontweight="bold",
    )

    # Forecast vs observed on the most variable 200-step window of the 95% run
    ax = axes[0, 0]
    r95 = rbt["95%"]["result"]
    WIN = 200
    truth_full = r95["truth"]
    n_full = len(truth_full)
    if n_full <= WIN:
        best_start = 0
    else:
        variances = np.array([
            np.var(truth_full[i : i + WIN])
            for i in range(0, n_full - WIN, 10)
        ])
        best_start = int(np.argmax(variances)) * 10
    sl = slice(best_start, best_start + WIN)

    ax.fill_between(
        r95["steps"][sl], r95["lower"][sl], r95["upper"][sl],
        color=COLORS["95%"], alpha=0.25, label="Interval (95%)",
    )
    ax.plot(r95["steps"][sl], r95["truth"][sl], color="#1f4e79", lw=1.5, label="Observed")
    ax.plot(r95["steps"][sl], r95["pred"][sl], color="#c0504d", lw=1.0,
            ls="--", alpha=0.85, label="Forecast")
    miss = ~r95["covered"][sl]
    if miss.any():
        ax.scatter(r95["steps"][sl][miss], r95["truth"][sl][miss],
                   s=20, color="black", zorder=5, label="Miss")
    y_vals = np.concatenate([r95["truth"][sl], r95["lower"][sl], r95["upper"][sl]])
    y_lo = np.percentile(y_vals, 1)
    y_hi = np.percentile(y_vals, 99)
    pad = 0.10 * (y_hi - y_lo) if y_hi > y_lo else 1.0
    ax.set_ylim(max(0, y_lo - pad), y_hi + pad)
    ax.set_title("Forecast vs Observed — 95% target (most variable 200-step window)", fontsize=10)
    ax.set_xlabel("Step")
    ax.set_ylabel("Discharge")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    for lbl, entry in rbt.items():
        r = entry["result"]
        k0 = entry["k_init"]
        rc = np.cumsum(r["covered"]) / np.arange(1, len(r["covered"]) + 1)
        ax.plot(r["steps"], rc, color=COLORS[lbl], lw=1.3, label=f"{lbl} (k₀={k0})")
        ax.axhline(1 - k0, color=COLORS[lbl], lw=0.8, ls=":")
    ax.set_ylim(0, 1.05)
    ax.set_title("Running Coverage (dotted = target)", fontsize=10)
    ax.set_xlabel("Step")
    ax.set_ylabel("Coverage")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    all_widths = np.concatenate([e["result"]["width"] for e in rbt.values()])
    y_cap = np.percentile(all_widths, 95)
    for lbl, entry in rbt.items():
        ax.plot(entry["result"]["steps"], entry["result"]["width"],
                color=COLORS[lbl], lw=0.8, alpha=0.85, label=lbl)
    ax.set_ylim(0, y_cap * 1.05)
    ax.set_title("Prediction Interval Width (y-axis: 95th percentile cap)", fontsize=10)
    ax.set_xlabel("Step")
    ax.set_ylabel("Width")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    tgs = list(rbt.keys())
    x = np.arange(len(tgs))
    nom = [1 - rbt[t]["k_init"] for t in tgs]
    ach = [rbt[t]["metrics"]["Coverage"] for t in tgs]
    aw = [rbt[t]["metrics"]["AvgWidth"] for t in tgs]
    ax.bar(x - 0.2, nom, 0.35, label="Target", color="#bdc3c7")
    ax.bar(x + 0.2, ach, 0.35, label="Achieved", color=[COLORS[t] for t in tgs])
    ax2 = ax.twinx()
    ax2.plot(x, aw, "k--o", lw=1.2, ms=5, label="Avg Width")
    ax2.set_ylabel("Avg Width", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(tgs)
    ax.set_ylim(0, 1.15)
    ax.set_title("Target vs Achieved Coverage  |  Avg Width", fontsize=10)
    ax.set_ylabel("Coverage")
    ax.set_xlabel("Target")
    ax.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  plot -> {out_path}")


def plot_summary_grid(all_results, out_path):
    stations = list(all_results.keys())
    targets = list(COVERAGE_TARGETS.keys())

    def mat(metric):
        m = np.zeros((len(stations), len(targets)))
        for si, st in enumerate(stations):
            for ti, tg in enumerate(targets):
                m[si, ti] = all_results[st][tg]["metrics"][metric]
        return m

    cov_m, wid_m, rmse_m = mat("Coverage"), mat("AvgWidth"), mat("RMSE")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Coverage Target Experiment — Cross-Station Summary",
                 fontsize=13, fontweight="bold")
    short = [s.replace(" ", "\n") for s in stations]

    for ax, matrix, title, fmt, cmap in zip(
        axes,
        [cov_m, wid_m, rmse_m],
        ["Achieved Coverage", "Avg Interval Width", "RMSE"],
        [".2%", ".1f", ".2f"],
        ["RdYlGn", "RdYlGn_r", "RdYlGn_r"],
    ):
        im = ax.imshow(matrix, cmap=cmap, aspect="auto",
                       vmin=matrix.min(), vmax=matrix.max())
        ax.set_xticks(range(len(targets)))
        ax.set_xticklabels(targets, fontsize=10)
        ax.set_yticks(range(len(stations)))
        ax.set_yticklabels(short, fontsize=8)
        ax.set_title(title, fontsize=11, fontweight="bold")
        for si in range(len(stations)):
            for ti in range(len(targets)):
                val = matrix[si, ti]
                rng = matrix.max() - matrix.min()
                rel = (val - matrix.min()) / rng if rng > 0 else 0.5
                col = "white" if rel < 0.25 or rel > 0.80 else "black"
                ax.text(ti, si, format(val, fmt),
                        ha="center", va="center", fontsize=8, color=col)
        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  summary grid -> {out_path}")


def plot_convergence(all_results, out_path):
    stations = list(all_results.keys())
    fig, axes = plt.subplots(len(stations), 1, figsize=(14, 3 * len(stations)))
    fig.suptitle("Running Coverage Convergence — All Stations",
                 fontsize=13, fontweight="bold")
    for ax, st in zip(axes, stations):
        for lbl, entry in all_results[st].items():
            r = entry["result"]
            k0 = entry["k_init"]
            rc = np.cumsum(r["covered"]) / np.arange(1, len(r["covered"]) + 1)
            ax.plot(r["steps"], rc, color=COLORS[lbl], lw=1.2, label=lbl)
            ax.axhline(1 - k0, color=COLORS[lbl], lw=0.7, ls=":")
        ax.set_ylim(0.4, 1.05)
        ax.set_title(st, fontsize=10, fontweight="bold")
        ax.set_ylabel("Coverage")
        ax.legend(fontsize=8, ncol=4, loc="lower right")
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("Step")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  convergence -> {out_path}")


def main():
    all_results = {}
    summary_rows = []

    for fname in FILES:
        station = fname.replace("_Inflow_Data_FILLED.csv", "").replace("_", " ")
        print(f"\n{'=' * 50}\n  {station}\n{'=' * 50}")

        df = pd.read_csv(os.path.join(DATA_DIR, fname))
        series = pd.Series(df["DISCHRG"].values, name="DISCHRG")
        order = choose_order(log_t(series.values))
        print(f"  ARIMA order: {order}")

        all_results[station] = {}

        for lbl, cfg in COVERAGE_TARGETS.items():
            k_init = cfg["k_init"]
            hit_step = cfg["hit"]
            miss_step = cfg["miss"]
            print(f"\n  --- Target {lbl}  (k_init={k_init}, "
                  f"hit={hit_step:.4f}, miss={miss_step:.4f}) ---")

            r = run_dacp(series, order, k_init, hit_step, miss_step)
            m = evaluate(r)
            print(f"  FINAL: cov={m['Coverage']:.2%}  width={m['AvgWidth']:.2f}  "
                  f"rmse={m['RMSE']:.3f}  mae={m['MAE']:.3f}")

            all_results[station][lbl] = {
                "result": r,
                "metrics": m,
                "k_init": k_init,
            }
            summary_rows.append({
                "station": station,
                "target": lbl,
                "k_init": k_init,
                **m,
            })

            pd.DataFrame({
                "step": r["steps"],
                "true_value": r["truth"],
                "prediction": r["pred"],
                "lower_bound": r["lower"],
                "upper_bound": r["upper"],
                "width": r["width"],
                "covered": r["covered"],
                "k_adap": r["k_trace"],
                "t_adap": r["t_trace"],
            }).to_csv(
                os.path.join(RESULTS_DIR,
                             f"ARIMA_{station.replace(' ', '_')}_{lbl}.csv"),
                index=False,
            )

        out_png = os.path.join(
            PLOT_DIR, f"{station.replace(' ', '_')}_coverage_experiment.png"
        )
        plot_station(station, all_results[station], out_png)

    summary_df = pd.DataFrame(summary_rows)
    csv_path = os.path.join(RESULTS_DIR, "coverage_experiment_summary.csv")
    summary_df.to_csv(csv_path, index=False)
    print(f"\n{'=' * 50}\nSummary CSV -> {csv_path}")
    print(summary_df.to_string(index=False))

    plot_summary_grid(all_results, os.path.join(PLOT_DIR, "_summary_grid.png"))
    plot_convergence(all_results, os.path.join(PLOT_DIR, "_coverage_convergence.png"))
    print("\nDone.")


if __name__ == "__main__":
    main()
