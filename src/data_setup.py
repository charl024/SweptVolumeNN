import numpy as np
import torch

def load_data(data_path):
    data = np.load(data_path)
    splits = {}
    for name in ["training", "evaluation", "testing"]:
        x = torch.from_numpy(data[f"{name}_features"])
        y = torch.from_numpy(data[f"{name}_labels"])
        splits[name] = (x, y)
    
    return splits

def compute_stats(x, y):
    return x.mean(dim=0), x.std(dim=0), y.mean(dim=0), y.std(dim=0)

def standardize(x, y, stats):
    x_mean, x_std, y_mean, y_std = stats
    return (x - x_mean) / x_std, (y - y_mean) / y_std


SUBSET_ORDER_SEED = 0

def dataset_setup(splits, device, train_size=None):
    x_train, y_train = splits["training"]

    if train_size is not None:
        generator = torch.Generator().manual_seed(SUBSET_ORDER_SEED)
        idx = torch.randperm(len(x_train), generator=generator)[:train_size]
        x_train, y_train = x_train[idx], y_train[idx]

    # standardize training data
    stats = compute_stats(x=x_train, y=y_train)
    x_train, y_train = standardize(x=x_train, y=y_train, stats=stats)
    
    # standardize eval and test
    x_eval, y_eval = splits["evaluation"]
    x_test, y_test = splits["testing"]

    x_eval, y_eval = standardize(x=x_eval, y=y_eval, stats=stats)
    x_test, y_test = standardize(x=x_test, y=y_test, stats=stats)

    # setup data on GPU
    data = {}
    data["training"] = (x_train.to(device), y_train.to(device))
    data["evaluation"] = (x_eval.to(device), y_eval.to(device))
    data["testing"] = (x_test.to(device), y_test.to(device))

    return data, stats

def iterate_batches(x, y, batch_size, shuffle):
    # shuffle data
    if shuffle:
        order = torch.randperm(len(x), device=x.device)
        x, y = x[order], y[order]
    
    # iterate and yield data in batches
    for i in range(0, len(x), batch_size):
        yield x[i:i + batch_size], y[i:i + batch_size]

