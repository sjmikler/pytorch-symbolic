#  Copyright (c) 2022 Szymon Mikler

import argparse
import gc
import logging
import os
import socket
import sys
import time
from copy import deepcopy

import dllogger
import torch
from dllogger import DLLLoggerAlreadyInitialized, JSONStreamBackend, StdOutBackend, Verbosity

from benchmarks.bench1 import define_models
from pytorch_functional import optimize_module_calls

logging.basicConfig(level=logging.ERROR)

# garbage collector is messing with timing
gc.disable()

parser = argparse.ArgumentParser()

parser.add_argument("--n-layers", type=int, required=True)
parser.add_argument("--n-warmup", type=int, default=50)
parser.add_argument("--n-iter", type=int, default=1000)
parser.add_argument("--n-features", type=int, default=4)
parser.add_argument("--batch-size", type=int, default=4)
parser.add_argument("--model-idx", type=int, required=True)
parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--run-name", type=str, default="unknown.jsonl")
parser.add_argument("--path", type=str, default="benchmarks/logs")

args = parser.parse_args()

BASE_TAGS = ["oct20"]

os.makedirs(args.path, exist_ok=True)

N_WARMUP = args.n_warmup
N_ITER = args.n_iter
N_LAYERS = args.n_layers
N_FEATURES = args.n_features
BATCH_SIZE = args.batch_size
DEVICE = args.device

CUDA_GRAPHS = False

if DEVICE == "cpu":
    assert CUDA_GRAPHS is False

try:
    dllogger.init(
        [
            JSONStreamBackend(
                filename=os.path.join(args.path, args.run_name),
                verbosity=Verbosity.DEFAULT,
                append=True,
            ),
            StdOutBackend(verbosity=Verbosity.DEFAULT),
        ]
    )
except DLLLoggerAlreadyInitialized:
    pass


def run(model, device):
    x = torch.rand(size=(BATCH_SIZE, N_FEATURES))
    x = x.to(device)

    for _ in range(N_WARMUP):
        _ = model(x)

    if DEVICE != "cpu":
        torch.cuda.synchronize()

    # TIMING
    t0 = time.time()
    for _ in range(N_ITER):
        _ = model(x)
    if DEVICE != "cpu":
        torch.cuda.synchronize()
    td = time.time() - t0
    return td


def log(name, layers, time_per_run):
    dllogger.log(
        step={},
        data={
            "tags": name,
            "layers": layers,
            "throughput": 1 / time_per_run,
            "N_WARMUP": N_WARMUP,
            "N_ITER": N_ITER,
            "N_FEATURES": N_FEATURES,
            "BATCH_SIZE": BATCH_SIZE,
            "HOST": host_info,
        },
    )
    dllogger.flush()


if __name__ == "__main__":
    device = torch.device(DEVICE)

    sys.setrecursionlimit(10000)

    models = define_models.create_sequential_multi_layer(N_LAYERS, N_FEATURES)
    tags, model = models[args.model_idx]
    model = deepcopy(model)
    optimize_module_calls()

    if not isinstance(tags, tuple):
        tags = (tags,)

    host_info = {
        "name": socket.gethostname(),
        "device": torch.cuda.get_device_name(device) if "cuda" in DEVICE else "cpu",
    }

    td = run(model, device)
    log([*BASE_TAGS, *tags, "call_optimization"], N_LAYERS, td / N_ITER)
