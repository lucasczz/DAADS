import pathlib
import numpy as np
import pandas as pd

from evaluate import aggregate_dataframe, test_then_train

DATASETS = ["covertype", "creditcard", "shuttle"]
MODELS = ["AE", "DAE", "PW-AE"]
N_TRIALS = 5
CONFIGS = [{"latent_dim": i, "lr": 0.02} for i in np.arange(0.1, 2.1, 0.1)]
SUBSAMPLE = 50_000
SAVE_STR = "Capacity"


metrics = [
    test_then_train(
        dataset=dataset,
        model=model,
        subsample=SUBSAMPLE,
        **config,
    )[0]
    for model in MODELS
    for dataset in DATASETS
    for config in CONFIGS
    for i in range(N_TRIALS)
]

metrics_raw = pd.DataFrame(metrics)
metrics_agg = aggregate_dataframe(metrics_raw, ["dataset", "model", "latent_dim"])

path = pathlib.Path(__file__).parent.parent.resolve()
metrics_raw.to_csv(f"{path}/results/{SAVE_STR}_raw.csv")
metrics_agg.to_csv(f"{path}/results/{SAVE_STR}.csv")
