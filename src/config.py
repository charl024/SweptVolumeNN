import argparse
import torch
from pathlib import Path

def create_config(
    dataset_path,
    experiment,
    hidden_layers,
    neurons_per_hidden_layer,
    learning_rate,
    batch_size,
    train_size,
    epochs,
    seed,
    device,
    overwrite,
    quick_test,
    dropout):
    return {"dataset_path" : dataset_path,
            "experiment" : experiment,
            "hidden_layers" : hidden_layers,
            "neurons_per_hidden_layer" : neurons_per_hidden_layer,
            "learning_rate" : learning_rate,
            "batch_size" : batch_size,
            "train_size" : train_size,
            "epochs" : epochs,
            "seed" : seed,
            "device" : device,
            "overwrite" : overwrite,
            "quick_test": quick_test,
            "dropout" : dropout}

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default=str(Path(__file__).resolve().parent.parent/"swept_volume_data.npz"))
    parser.add_argument("--experiment", type=str, default="gpu_sweep")
    parser.add_argument("--hidden_layers", type=int, default=2)
    parser.add_argument("--neurons_per_hidden_layer", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--train_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--quick-test", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.0)
    args = parser.parse_args()

    return create_config(**vars(args))

def run_directory(config):
    name = (f"{config['experiment']}_d{config['dropout']}_h{config['hidden_layers']}_w{config['neurons_per_hidden_layer']}"
            f"_lr{config['learning_rate']}_bs{config['batch_size']}"
            f"_n{config['train_size'] or 'all'}_e{config['epochs']}_seed{config['seed']}")
    return Path("runs") / name