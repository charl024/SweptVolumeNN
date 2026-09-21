import torch
import torch.nn as nn
import numpy as np
import math
import time
import copy
import json
import tempfile
import shutil
from pathlib import Path

from src.config import arg_parse, run_directory
from src.data_setup import load_data, dataset_setup, iterate_batches

class NeuralNetwork(nn.Module):
	def __init__(self, in_dimension, out_dimension, hidden_layers, neurons_per_hidden_layer):
		super(NeuralNetwork, self).__init__()
		layers = []

		# setup input layer
		layers.append(nn.Linear(in_dimension, neurons_per_hidden_layer))
		layers.append(nn.ReLU())
		# setup hidden layers
		for _ in range(hidden_layers - 1):
			layers.append(nn.Linear(neurons_per_hidden_layer, neurons_per_hidden_layer))
			layers.append(nn.ReLU())

		# setup output layer
		layers.append(nn.Linear(neurons_per_hidden_layer, out_dimension))
		self.network = nn.Sequential(*layers)

		# initialize weights and biases for eahc layer
		for layer in self.network:
			if (isinstance(layer, nn.Linear)):
				nn.init.xavier_uniform_(layer.weight)
				nn.init.zeros_(layer.bias)


	def forward(self, x):
		return self.network(x)

def evaluate_network(net, x, y, stats):
	_, _, y_mean, y_std = stats
	y_mean, y_std = y_mean.to(x.device), y_std.to(x.device)
	
	net.eval()
	with torch.no_grad():
		
		# denorm prediction
		output = net(x)
		pred = output * y_std + y_mean
		true = y * y_std + y_mean

		mse = ((pred - true) ** 2).mean().item()

	return mse, np.sqrt(mse)

def save_run(run_dir, config, stats, train_losses, eval_rmses, best_epoch, best_eval_rmse, best_state, training_time, diverged_epoch):
	run_dir.mkdir(parents=True, exist_ok=True)

	device = torch.device(config["device"])
	saved_config = dict(config)
	saved_config["device_name"] = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
	saved_config["torch_version"] = torch.__version__

	# json dump
	with open(run_dir/"config.json", "w") as f:
		json.dump(saved_config, f, indent=4)

	# save epoch vals
	np.savez(run_dir/"history.npz", train_loss=np.array(train_losses), eval_rmse=np.array(eval_rmses))

	# summary results
	summary =  {"best_epoch": best_epoch,
				"best_eval_rmse": best_eval_rmse if math.isfinite(best_eval_rmse) else None,
				"training_time": training_time,
				"diverged_epoch": diverged_epoch}

	with open(run_dir / "summary.json", "w") as f:
		json.dump(summary, f, indent=4)

	# save best_model
	x_mean, x_std, y_mean, y_std = stats
	checkpoint =   {"state_dict": {k: v.cpu() for k, v in best_state.items()} if best_state is not None else None,
					"stats": {"x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "y_std": y_std},
					"hidden_layers": config["hidden_layers"],
					"neurons_per_hidden_layer": config["neurons_per_hidden_layer"]}
	torch.save(checkpoint, run_dir / "checkpoint.pt")

if __name__=="__main__":
	config = arg_parse()

	# setup save directory
	# quick test check
	if config["quick_test"]:
		config["train_size"] = 1000
		config["epochs"] = 2
		config["batch_size"] = 100
		run_dir = None
	else:
		run_dir = run_directory(config)
		# directory check
		if run_dir.exists() and not config["overwrite"]:
				raise SystemExit(f"{run_dir} already exists; pass --overwrite to replace it")

	# set seeds
	torch.manual_seed(config["seed"])

	# load data
	data_splits = load_data(config["dataset_path"])
	data, stats = dataset_setup(splits=data_splits, device=config["device"], train_size=config["train_size"])

	x_train, y_train = data["training"]
	x_eval, y_eval = data["evaluation"]

	# network setup
	net = NeuralNetwork(in_dimension=14, out_dimension=1, hidden_layers=config["hidden_layers"], neurons_per_hidden_layer=config["neurons_per_hidden_layer"])
	optimizer = torch.optim.Adam(net.parameters(), lr=config["learning_rate"])
	net.to(config["device"])

	# loss function
	loss_function = nn.MSELoss()

	# train
	train_losses = []
	eval_rmses = []
	best_eval_rmse = math.inf
	best_epoch = None
	best_state = None
	diverged_epoch = None
	training_time = 0.0

	for epoch in range(1, config["epochs"] + 1):
		net.train()
		total_loss = torch.zeros((), device=config["device"])

		if torch.device(config["device"]).type == "cuda":
			torch.cuda.synchronize()

		start_time = time.perf_counter()
		
		# training loop
		for xb, yb in iterate_batches(x_train, y_train, config["batch_size"], shuffle=True):

			optimizer.zero_grad()
			output = net(xb)
			loss = loss_function(output, yb)
			loss.backward()
			optimizer.step()

			total_loss += loss.detach() * len(xb)

		if torch.device(config["device"]).type == "cuda":
			torch.cuda.synchronize()

		end_time = time.perf_counter()
		training_time += (end_time - start_time)

		train_loss = total_loss.item() / len(x_train)
		eval_mse, eval_rmse = evaluate_network(net, x_eval, y_eval, stats)

		train_losses.append(train_loss)
		eval_rmses.append(eval_rmse)

		# divergence check
		if not (math.isfinite(train_loss) and math.isfinite(eval_rmse)):
			diverged_epoch = epoch
			break

		# record best
		if eval_rmse < best_eval_rmse:
			best_eval_rmse = eval_rmse
			best_epoch = epoch
			best_state = copy.deepcopy(net.state_dict())
	
	# padding
	pad = config["epochs"] - len(train_losses)
	train_losses += [math.nan] * pad
	eval_rmses += [math.nan] * pad

	if best_state is not None:
		net.load_state_dict(best_state)
	
	# save data, configuration, and other data for plotting	
	if config["quick_test"]:

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
	

