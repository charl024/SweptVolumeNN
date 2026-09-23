import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent

CAPACITIES = [(h, w) for w in (64, 128, 256) for h in (1, 2, 3)]
DROPOUT = 0.1

def condition(hidden_layers, neurons):
    return f"h{hidden_layers}_w{neurons}_lr0.001_bs10000_nall"

def dropout_condition(experiment, dropout, hidden_layers, neurons, epochs):
    return f"{experiment}_d{dropout}_h{hidden_layers}_w{neurons}_lr0.001_bs10000_nall_e{epochs}"

def run_dirs(condition_name):
    return sorted((ROOT / "runs").glob(f"{condition_name}_seed*"))

def parameters(hidden_layers, neurons):
    return 14 * neurons + neurons + (hidden_layers - 1) * (neurons * neurons + neurons) + neurons + 1

def mean_std(condition_name):
    curves = [np.load(d / "history.npz")["eval_rmse"] for d in run_dirs(condition_name)]
    if not curves:
        return None, None
    curves = np.array(curves)
    return np.nanmean(curves, axis=0), np.nanstd(curves, axis=0, ddof=1)

def best_mean_std(condition_name):
    best = [json.loads((d / "summary.json").read_text())["best_eval_rmse"] for d in run_dirs(condition_name)]
    best = [b for b in best if b is not None]
    if not best:
        return None, None
    return np.mean(best), np.std(best, ddof=1) if len(best) > 1 else 0.0

def finish(title, filename, xlabel="Epoch", ylabel="Evaluation RMSE (L)"):
    plt.yscale("log")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    (ROOT / "plots").mkdir(exist_ok=True)
    plt.savefig(ROOT / "plots" / filename, dpi=150)
    plt.close()
    print(f"wrote plots/{filename}")

def curve_plot(conditions, labels, title, filename, colors=None, styles=None):
    plt.figure(figsize=(7, 4.5))
    for i, (condition_name, label) in enumerate(zip(conditions, labels)):
        mean, std = mean_std(condition_name)
        if mean is None:
            print(f"skipping {filename}: no runs found for {condition_name}")
            plt.close()
            return
        epochs = np.arange(1, len(mean) + 1)
        line, = plt.plot(epochs, mean, linewidth=0.8,
                         color=None if colors is None else colors[i],
                         linestyle="-" if styles is None else styles[i],
                         label=label)
        plt.fill_between(epochs, mean - std, mean + std, color=line.get_color(), alpha=0.2, linewidth=0)

    finish(title, filename)

def capacity_plot(title, filename):
    plt.figure(figsize=(7, 4.5))
    for label, name_of, style in [(f"no dropout", lambda h, w: condition(h, w), "-o"),
                                  (f"dropout {DROPOUT}", lambda h, w: dropout_condition("dropout_capacity", DROPOUT, h, w, 1000), "--s")]:
        points = []
        for hidden_layers, neurons in CAPACITIES:
            mean, std = best_mean_std(name_of(hidden_layers, neurons))
            if mean is not None:
                points.append((parameters(hidden_layers, neurons), mean, std))
        if not points:
            print(f"skipping {filename}: no runs found for '{label}'")
            plt.close()
            return
        params, means, stds = zip(*sorted(points))
        plt.errorbar(params, means, yerr=stds, fmt=style, markersize=4, linewidth=0.8, capsize=3, label=label)

    plt.xscale("log")
    finish(title, filename, xlabel="Parameters", ylabel="Best evaluation RMSE (L)")

if __name__ == "__main__":
    curve_plot([condition(h, 256) for h in (1, 2, 3)],
               [f"{h} hidden layer{'s' if h > 1 else ''}" for h in (1, 2, 3)],
               "Network depth (256 neurons per layer, mean +- std over 5 seeds)",
               "depth.svg")

    curve_plot([condition(3, w) for w in (32, 64, 128, 256)],
               [f"{w} neurons" for w in (32, 64, 128, 256)],
               "Network width (3 hidden layers, mean +- std over 5 seeds)",
               "width.svg")

    curve_plot([condition(3, 256), dropout_condition("dropout_standard", DROPOUT, 3, 256, 1000)],
               ["no dropout", f"dropout {DROPOUT}"],
               "Dropout on the best configuration (3x256, mean +- std over 5 seeds)",
               "dropout.svg")

    depths = [1, 2, 3]
    curve_plot([condition(h, 256) for h in depths]
               + [dropout_condition("dropout_capacity", DROPOUT, h, 256, 1000) for h in depths],
               [f"{h}x256" for h in depths] + [f"{h}x256, dropout {DROPOUT}" for h in depths],
               "Dropout by depth (256 neurons per layer, mean +- std over 5 seeds)",
               "dropout_depth.svg",
               colors=[f"C{i}" for i in range(len(depths))] * 2,
               styles=["-"] * len(depths) + ["--"] * len(depths))

    capacity_plot("Dropout across capacities (mean +- std over 5 seeds)", "dropout_capacity.svg")
