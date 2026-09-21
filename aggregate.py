# Collects every finished run in runs/ into:
#   results.csv - one row per run (config, best epoch, best eval RMSE, training time, test metrics if computed)
#   summary.csv - one row per condition (mean/std across seeds), sorted by mean eval RMSE
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent

def load_runs(runs_dir):
    rows = []
    for run_dir in sorted(runs_dir.iterdir()):
        if not (run_dir / "summary.json").exists():
            continue
        config = json.loads((run_dir / "config.json").read_text())
        summary = json.loads((run_dir / "summary.json").read_text())
        test_path = run_dir / "test.json"
        test = json.loads(test_path.read_text()) if test_path.exists() else {}

        rows.append({"run": run_dir.name,
                     "experiment": config.get("experiment", "gpu_sweep"),
                     "condition": run_dir.name.rsplit("_seed", 1)[0],
                     "hidden_layers": config["hidden_layers"],
                     "neurons_per_hidden_layer": config["neurons_per_hidden_layer"],
                     "learning_rate": config["learning_rate"],
                     "batch_size": config["batch_size"],
                     "train_size": config["train_size"],
                     "epochs": config["epochs"],
                     "seed": config["seed"],
                     "device_name": config["device_name"],
                     "best_epoch": summary["best_epoch"],
                     "best_eval_rmse": summary["best_eval_rmse"],
                     "training_time": summary["training_time"],
                     "diverged_epoch": summary["diverged_epoch"],
                     "test_mse": test.get("test_mse"),
                     "test_rmse": test.get("test_rmse")})
    return rows

def summarize(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row["experiment"], row["condition"]), []).append(row)

    summary = []
    for (experiment, condition), runs in groups.items():
        rmse = np.array([r["best_eval_rmse"] if r["best_eval_rmse"] is not None else np.nan for r in runs])
        times = np.array([r["training_time"] for r in runs])
        valid = rmse[~np.isnan(rmse)]
        summary.append({"experiment": experiment,
                        "condition": condition,
                        "n_seeds": len(runs),
                        "n_diverged": sum(r["diverged_epoch"] is not None for r in runs),
                        "eval_rmse_mean": valid.mean() if len(valid) > 0 else np.nan,
                        "eval_rmse_std": valid.std(ddof=1) if len(valid) > 1 else np.nan,
                        "time_mean": times.mean(),
                        "time_std": times.std(ddof=1) if len(times) > 1 else np.nan})
    return sorted(summary, key=lambda s: (s["experiment"], np.isnan(s["eval_rmse_mean"]), s["eval_rmse_mean"]))

def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

if __name__ == "__main__":
    runs_dir = ROOT / "runs"
    rows = load_runs(runs_dir) if runs_dir.exists() else []
    if not rows:
        raise SystemExit("no finished runs found in runs/")

    summary = summarize(rows)
    write_csv(ROOT / "results.csv", rows)
    write_csv(ROOT / "summary.csv", summary)
    print(f"wrote results.csv ({len(rows)} runs) and summary.csv ({len(summary)} conditions)\n")

    for s in summary:
        flag = "" if s["n_seeds"] == 5 else f"  <-- {s['n_seeds']}/5 seeds"
        print(f"[{s['experiment']}] {s['condition']:45s} "
              f"eval RMSE {s['eval_rmse_mean']:.3f} ± {s['eval_rmse_std']:.3f} L   "
              f"time {s['time_mean']:.1f} ± {s['time_std']:.1f} s   "
              f"diverged {s['n_diverged']}{flag}")
