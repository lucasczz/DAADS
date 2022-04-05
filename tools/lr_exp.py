import pathlib
from evaluate import test_then_train, aggregate_dataframe
import multiprocessing as mp
import numpy as np
import pandas as pd

OPTIMIZER_FNS = ["hd-sgd", "adam", "sgd"]
MODELS = ["AE", "PW-AE", "DAE"]
SEEDS = range(42, 52)
LEARNING_RATES = np.geomspace(1e-3, 0.256, 9)
DATASETS = ["covertype", "creditcard", "shuttle"]
SUBSAMPLE = 50000
N_PROCESSES = 50
SAVE_STR = "Learning Rate"

pool = mp.Pool(processes=N_PROCESSES)
runs = [
    pool.apply_async(
        test_then_train,
        kwds=dict(
            dataset=dataset,
            model=model,
            subsample=SUBSAMPLE,
            lr=lr,
            seed=seed,
            optimizer_fn=optimizer,
        ),
    )
    for optimizer in OPTIMIZER_FNS
    for dataset in DATASETS
    for model in MODELS
    for lr in LEARNING_RATES
    for seed in SEEDS
]

metrics = [run.get()[0] for run in runs]

metrics_raw = pd.DataFrame(metrics)
metrics_agg = aggregate_dataframe(
    metrics_raw, ["optimizer_fn", "dataset", "model", "lr"]
)

path = pathlib.Path(__file__).parent.parent.resolve()
metrics_raw.to_csv(f"{path}/results/{SAVE_STR}_raw.csv")
metrics_agg.to_csv(f"{path}/results/{SAVE_STR}.csv")
