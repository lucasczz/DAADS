import pandas as pd
import multiprocessing as mp
import pathlib

from evaluate import aggregate_dataframe, test_then_train

N_PROCESSES = 50
DATASETS = ["covertype", "creditcard", "shuttle"]
MODELS = ["AE", "AE", "DAE", "RRCF", "HST", "PW-AE", "xStream", "Kit-Net", "ILOF"]
SEEDS = range(42, 52)
SUBSAMPLE = 500_000
SAVE_STR = "Benchmark"

CONFIGS = {
    "AE": {"lr": 0.02, "latent_dim": 0.1},
    "DAE": {"lr": 0.02},
    "PW-AE": {"lr": 0.1},
    "OC-SVM": {},
    "HST": {"n_trees": 25, "height": 15},
}


pool = mp.Pool(processes=N_PROCESSES)
runs = [
    pool.apply_async(
        test_then_train,
        kwds=dict(
            dataset=dataset,
            model=model,
            seed=seed,
            subsample=SUBSAMPLE,
            **CONFIGS.get(model, {}),
        ),
    )
    for dataset in DATASETS
    for model in MODELS
    for seed in SEEDS
]

metrics = [run.get()[0] for run in runs]

metrics_raw = pd.DataFrame(metrics)
metrics_agg = aggregate_dataframe(metrics_raw, ["dataset", "model"])

path = pathlib.Path(__file__).parent.parent.resolve()
metrics_raw.to_csv(f"{path}/results/{SAVE_STR}_raw.csv")
metrics_agg.to_csv(f"{path}/results/{SAVE_STR}.csv")
