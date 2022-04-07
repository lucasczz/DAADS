import pathlib
from IncrementalTorch.datasets import Covertype, Shuttle
import numpy as np
import pandas as pd
from river.datasets import CreditCard

from evaluate import aggregate_dataframe, test_then_train
from tools.benchmark_exp import SAVE_STR

DATASETS = {
    "covertype": Covertype,
    "creditcard": CreditCard,
    "shuttle": Shuttle,
}
SUBSAMPLE = 50_000
ANOM_FRACTIONS = np.arange(0.005, 0.1005, 0.005)
N_TRIALS = 5
CONFIGS = {"HST": {}, "DAE": {"lr": 0.02}, "PW-AE": {"lr": 0.1}}
SAVE_STR = "Contamination"

models = ["PW-AE", "DAE", "HST"]

metrics = []
for dataset_name, dataset in DATASETS.items():
    data = list(dataset().take(SUBSAMPLE))
    x, y = list(zip(*data))
    x, y = np.array(x), np.array(y)
    x_normal, x_anom = x[y == 0], x[y == 1]

    def set_anom_fraction(frac=0.01):
        n_anom = round(SUBSAMPLE * frac)
        x_anom_samples = x_anom[np.random.randint(len(x_anom), size=n_anom)]
        idcs = np.random.randint(0, len(x_normal), n_anom)
        idcs.sort()
        idcs = idcs[::-1]
        x_new = list(x_normal[: SUBSAMPLE - n_anom])
        y_new = list(np.zeros(len(x_new)))
        for idx, x_anom_sample in zip(idcs, x_anom_samples):
            x_new.insert(idx, x_anom_sample)
            y_new.insert(idx, 1)

        return list(zip(x_new, y_new))

    for anom_fraction in ANOM_FRACTIONS:
        for trial in range(N_TRIALS):
            data_i = set_anom_fraction(anom_fraction)
            for model in models:
                metrics_i, _ = test_then_train(
                    dataset=data_i,
                    model=model,
                    **CONFIGS[model],
                )
                metrics_i.update(
                    {"anom_fraction": anom_fraction, "dataset": dataset_name}
                )
                metrics.append(metrics_i)


metrics_raw = pd.DataFrame(metrics)
metrics_agg = aggregate_dataframe(metrics_raw, ["dataset", "model", "anom_fraction"])

path = pathlib.Path(__file__).parent.parent.resolve()
metrics_raw.to_csv(f"{path}/results/{SAVE_STR}_raw.csv")
metrics_agg.to_csv(f"{path}/results/{SAVE_STR}.csv")
