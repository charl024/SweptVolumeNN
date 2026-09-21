# Scores the 5 retained best models of ONE chosen condition on the held-out test set.
# Run only after choosing the final condition from evaluation results (summary.csv).
# Writes test.json into each run folder; re-run aggregate.py afterwards to add them to results.csv.
# Usage: python3 final_test.py --condition h2_w128_lr0.001_bs10000_nall
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from neural_network import NeuralNetwork, evaluate_network
from src.data_setup import load_data, standardize

ROOT = Path(__file__).resolve().parent

def load_model(run_dir):
    checkpoint = torch.load(run_dir / "checkpoint.pt", weights_only=True)
    if checkpoint["state_dict"] is None:
        raise SystemExit(f"{run_dir} has no checkpoint (run diverged before any valid epoch)")
    net = NeuralNetwork(14, 1, checkpoint["hidden_layers"], checkpoint["neurons_per_hidden_layer"])
    net.load_state_dict(checkpoint["state_dict"])
    s = checkpoint["stats"]
    return net, (s["x_mean"], s["x_std"], s["y_mean"], s["y_std"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score the 5 retained models of one condition on the test set")
    parser.add_argument("--condition", required=True)
    parser.add_argument("--dataset_path", default=str(ROOT / "swept_volume_data.npz"))
    args = parser.parse_args()

    run_dirs = sorted((ROOT / "runs").glob(f"{args.condition}_seed*"))
    if len(run_dirs) != 5:
        raise SystemExit(f"expected 5 seeds for {args.condition}, found {len(run_dirs)}")

    x_test_raw, y_test_raw = load_data(args.dataset_path)["testing"]

    mses, rmses = [], []
    for run_dir in run_dirs:
        net, stats = load_model(run_dir)
        x_test, y_test = standardize(x=x_test_raw, y=y_test_raw, stats=stats)
        mse, rmse = evaluate_network(net, x_test, y_test, stats)
        (run_dir / "test.json").write_text(json.dumps({"test_mse": mse, "test_rmse": rmse}, indent=4))
        mses.append(mse)
        rmses.append(rmse)
        print(f"{run_dir.name}: test MSE {mse:.3f} L^2, RMSE {rmse:.3f} L")

    print(f"\n{args.condition}: test MSE {np.mean(mses):.3f} ± {np.std(mses, ddof=1):.3f} L^2, "
          f"RMSE {np.mean(rmses):.3f} ± {np.std(rmses, ddof=1):.3f} L  (mean +- std over {len(run_dirs)} seeds)")
