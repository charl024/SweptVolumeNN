# Requirements Check: Assignment 1 (GPU track)

Last updated 2026-09-21. Checked against `docs/Assignment1_NeuralNetworks.pdf`.
Status key: ✅ done · ⚠️ partial / needs a fix · ❌ not started

---

## 1. Current code state

### Blocking bug (the script crashes right now)
- **`neural_network.py:111`:** `dataset_setup` is still called with the old arguments. Running
  `python neural_network.py --quick-test` gives
  `TypeError: dataset_setup() got an unexpected keyword argument 'batch_size'`. Fix:
  ```python
  data, stats = dataset_setup(splits=data_splits, device=config["device"], train_size=config["train_size"])
  ```

### Verified working (scratch copy with the fix above applied)
- `--quick-test` passes (evaluation RMSE ≈ 41.5 L after 2 epochs, exit code 0), leaves nothing in `runs/`,
  and works when launched from a different directory.
- A normal run writes `runs/<config>_seed<N>/` containing `config.json`, `history.npz`, `summary.json` and `checkpoint.pt`.
- A second identical run is refused unless `--overwrite` is passed.
- A forced divergence (`--learning_rate 1e6`) is recorded (`diverged_epoch`, `best_eval_rmse: null`, NaN-padded history)
  instead of crashing.
- GPU-style batching (`iterate_batches`) takes about 31 ms per epoch on CPU at batch size 10000 on the full data,
  down from about 1.6 s per epoch with `DataLoader`.

### Minor cleanups
- `neural_network.py:99`: the quick test's temporary folder is created before the data loads, so any crash leaves a
  `quick_test_*` folder in `/tmp`. The crash above left one behind. See §4 for the fix.
- `src/config.py:50`: `runs/` is relative to the directory you launch from. Anchor it to the project root, the same way
  as the dataset path: `Path(__file__).resolve().parent.parent / "runs"`.
- `neural_network.py:32`: typo "eahc".
- `README.md`: typo "atplotlib".
- `src/config.py:35,37`: the defaults `batch_size=128` and `epochs=50` don't match the GPU track. That's fine as long as the
  README and sweep script pass `--batch_size 10000 --epochs 1000` explicitly.

---

## 2. Requirements vs. current status

### Model (PDF p. 4, 8)
| Requirement | Status |
|---|---|
| File named `neural_network.py` containing class `NeuralNetwork` | ✅ |
| `__init__(in_dimension, out_dimension, hidden_layers, neurons_per_hidden_layer)` | ✅ |
| `xavier_uniform_` weights, `zeros_` biases | ✅ |
| `forward(x)`, with ReLU after every layer except the last | ✅ |
| `torch.optim.Adam`, no weight decay | ✅ |
| `nn.MSELoss()` on normalized labels | ✅ |

### Normalization (p. 3)
| Requirement | Status |
|---|---|
| Standardize features and labels using training-set mean and std | ✅ |
| Evaluation and test sets use the training stats, never their own | ✅ |
| Evaluation curves and final numbers reported in liters | ✅ (evaluation) / ❌ (final test script not written) |
| GPU: stats come from the full training set | ✅ (with `--train_size` left at `None`) |

### Seeds and reproducibility (p. 4)
| Requirement | Status |
|---|---|
| Seeds `[0, 1, 2, 3, 4]` for every condition | ⚠️ supported via `--seed`; the sweep isn't run yet |
| Seed set before the network is built | ✅ |
| One run = one configuration + one seed | ✅ |
| Training subset identical across seeds, smaller subsets nested in larger | ✅ (`SUBSET_ORDER_SEED`) |
| Seed NumPy too, if NumPy randomness is used | ✅ not needed (no NumPy randomness) |

### Per-run training procedure (p. 4, 6, 8)
| Requirement | Status |
|---|---|
| Evaluate every epoch; record evaluation RMSE per epoch | ✅ |
| Keep the best model state (`deepcopy(state_dict())`) and the best epoch | ✅ |
| Restore the best state rather than the last epoch | ✅ |
| Record training time | ✅ (training passes only; say so in the report) |
| GPU timing uses `torch.cuda.synchronize()` | ✅ |
| Record NaNs or divergence instead of crashing or silently changing settings | ✅ |
| Test set never used during training runs | ✅ |

### Command-line interface (p. 8-9)
| Requirement | Status |
|---|---|
| Arguments for train size, learning rate, architecture, batch size, seed, epochs and device | ✅ |
| `python3 neural_network.py --quick-test`: loads data, normalizes, trains briefly on a small subset, runs evaluation, exits 0 | ⚠️ works once the line-111 fix is applied |
| Quick test does **not** report test-set performance | ✅ |
| Relative paths; the dataset is found in the submission root | ✅ (`Path(__file__)`-based default) |
| Grader never has to edit the source code | ⚠️ `runs/` depends on the launch directory (see §1) |

### Saved output per run (p. 8, 11)
| Requirement | Status |
|---|---|
| Folder name that uniquely identifies the config and seed | ✅ |
| Refuse to overwrite an existing run unless `--overwrite` is passed | ✅ |
| Full config and seed (`config.json`, plus device name and torch version) | ✅ |
| Per-epoch training and evaluation history (`history.npz`) | ✅ |
| Best evaluation epoch and best evaluation RMSE (`summary.json`) | ✅ |
| Training time (`summary.json`) | ✅ |
| Best checkpoint **plus normalization stats** (`checkpoint.pt`) | ✅ |

### Experiments: GPU track (p. 5-6)
| Requirement | Status |
|---|---|
| Hidden layers `[1, 2, 3]` × neurons `[32, 64, 128, 256]` = 12 conditions | ❌ no sweep script |
| Learning rate `1e-3`, full training set, batch size 10000, Adam | ⚠️ supported; pass the flags explicitly |
| ≥ 1000 epochs per run, all 5 seeds (60 runs) | ❌ |
| Mean ± std of best evaluation RMSE and training time per condition | ❌ needs `aggregate.py` |
| ≥ 2 evaluation-RMSE-vs-epoch plots (depth, width), mean ± std across seeds | ❌ needs `plot.py` |
| Choose the final config using evaluation results only | ❌ |
| Score the 5 retained best models of that config on the test set → mean ± std of test MSE (L²) and RMSE (L) | ❌ needs `final_test.py` |
| Document the hardware used for timing | ⚠️ saved per run in `config.json`; still needs to go in the report |

### Self-directed investigation (p. 6)
| Requirement | Status |
|---|---|
| One change not already tested, plus a stated hypothesis | ❌ |
| Baseline + ≥ 1 new condition, same 5 seeds | ❌ |
| Compare on evaluation data; test only the final variant, chosen before looking at its test result | ❌ |

### 591 graduate extension (p. 6-7)
| Requirement | Status |
|---|---|
| Proposed explanation for an observed effect | ❌ |
| ≥ 3 more conditions not already required, same 5 seeds | ❌ |
| ≥ 2 more diagnostic plots (e.g. train vs. eval curves, predicted vs. true, residuals) | ❌ |
| Discuss whether the evidence supports the explanation, and what alternatives remain | ❌ |

### Support scripts (p. 9-11)
| Requirement | Status |
|---|---|
| Sweep launcher for the 60 runs (optional but practical) | ❌ |
| `aggregate.py` → `results.csv` (one row per run: experiment, condition, seed, best epoch, best evaluation RMSE, training time, test metrics if computed) | ❌ |
| `plot.py` → `plots/` | ❌ |
| `final_test.py` (loads saved checkpoints and their stats; test set only) | ❌ |

### Submission (p. 7, 9-11)
| Requirement | Status |
|---|---|
| README: name, course number, GPU track | ⚠️ course and track present; **name missing** |
| README: Python and PyTorch versions | ✅ (fix the "atplotlib" typo) |
| README: exact quick-test command | ✅ |
| README: exact single-run command | ❌ |
| README: how to aggregate runs, how to do final held-out testing | ❌ |
| README: description of the output files | ❌ |
| `requirements.txt` | ❌ |
| `runs/`, `results.csv`, history files, `plots/` in the ZIP | ❌ |
| ZIP excludes the virtual env, package caches and `swept_volume_data.npz` | ❌ (at packaging time). The `.npz` is committed to git, so build the ZIP by hand or with an explicit exclude, not from the repo as-is |
| Environment includes Gymnasium (Atari, Box2D, classic-control) for later projects | ❓ not checked in `mldev` |
| `NeuralNetworkReport.pdf`: names/course/track, intro, methodology, results + table + ≥ 2 plots, self-directed, 591 follow-up, acknowledgment of AI use | ❌ |

---

## 3. GPU notes

- **No CUDA GPU on this machine.** `torch.cuda.is_available()` returns `False`. The PyTorch build (`2.10.0+cu128`) should work on a
  CUDA machine; check `torch.cuda.is_available()` there before starting the sweep. `--device` falls back to `cpu` silently,
  but `config.json` records `device_name`, so check it after the first run.
- **Timing:** run all 60 timed runs on one machine (or identical hardware).
- **Data is kept on the device** and batched with `randperm` (`iterate_batches`). No `DataLoader` remains in the training path.
- **Checkpoints are saved on CPU,** so they load on machines without a GPU (for grading or `final_test.py`).
- **Determinism:** the same seed on the same device reproduces a run. GPU runs won't match CPU runs bit for bit; mention this
  if you compare them.

---

## 4. Fix: temporary quick-test folder left behind after a crash

**Problem:** `tempfile.mkdtemp()` runs at line 99, before anything that can fail (loading data, building the model,
training). If any of that raises an error, the script never reaches `shutil.rmtree` and the folder stays in `/tmp`.

**Fix:** decide *where* to save up front, but only *create* the temporary folder when it's about to be used, and
wrap the save-then-delete in `try`/`finally` so it's removed even if `save_run` itself fails.

At the top (line 95-99), don't create a temporary folder yet:
```python
	if config["quick_test"]:
		config["train_size"] = 1000
		config["epochs"] = 2
		config["batch_size"] = 100
		run_dir = None          # temp folder is created right before saving
	else:
		run_dir = run_directory(config)
		...
```

At the end (lines 184-196):
```python
	if config["quick_test"]:
		# exercise save_run too, but in a throwaway folder that is always cleaned up
		run_dir = Path(tempfile.mkdtemp(prefix="quick_test_"))
		try:
			save_run(run_dir, config, stats, train_losses, eval_rmses, best_epoch, best_eval_rmse, best_state, training_time, diverged_epoch)
		finally:
			shutil.rmtree(run_dir, ignore_errors=True)

		if diverged_epoch is not None:
			raise SystemExit("quick test failed: training diverged")
		print(f"quick test passed: eval RMSE {best_eval_rmse:.3f} L after {config['epochs']} epochs")
	else:
		save_run(run_dir, config, stats, train_losses, eval_rmses, best_epoch, best_eval_rmse, best_state, training_time, diverged_epoch)
		print(f"saved to {run_dir}: best eval RMSE {best_eval_rmse:.3f} L at epoch {best_epoch}")
```

Why this works: now nothing can fail between `mkdtemp` and `rmtree` except `save_run`, and `finally` covers that
case. `ignore_errors=True` stops a cleanup problem from hiding the real error.

(A simpler alternative: `with tempfile.TemporaryDirectory(prefix="quick_test_") as tmp:` around the `save_run` call
deletes the folder automatically, even when something raises an error.)

---

## 5. Suggested order of work

1. Fix `neural_network.py:111` (the blocking bug), then the temp-folder fix in §4.
2. Anchor `runs/` to the project root.
3. Write `run_sweep.sh` (3 depths × 4 widths × 5 seeds, `--learning_rate 1e-3 --batch_size 10000 --epochs 1000`).
4. On the GPU machine: check CUDA is available, run the quick test, then the sweep.
5. While it runs: write `aggregate.py` → `results.csv`, `plot.py` → `plots/`, and `final_test.py`.
6. Choose the final config from evaluation results, then run `final_test.py`.
7. Self-directed and 591 experiments (same seeds, evaluation data only until the final variant is fixed).
8. README (name, commands, aggregation/testing instructions, output files), `requirements.txt`, report, ZIP.
