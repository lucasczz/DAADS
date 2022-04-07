import pathlib
import pandas as pd
import multiprocessing as mp
from evaluate import DATASETS, aggregate_dataframe, test_then_train, seed_everything

N_TRIALS = 5
DATASETS = ["covertype", "creditcard", "shuttle"]
SAVE_STR = "HST Baseline"
N_PROCESSES = 3

metrics = []

def eval(dataset):
    seed_everything(42)
    for trial in range(N_TRIALS):   
        metrics_i, _ = test_then_train(dataset, "HST", **{"n_trees": 25, "height": 15})
        return metrics_i

pool = mp.Pool(N_PROCESSES)
runs = [pool.apply_async(eval, args=(dataset, )) for dataset in DATASETS]
metrics = [run.get() for run in runs]
    
metrics_raw = pd.DataFrame(metrics)
metrics_agg = aggregate_dataframe(metrics_raw, ["dataset", "model"])

path = pathlib.Path(__file__).parent.parent.resolve()
metrics_raw.to_csv(f"{path}/results/{SAVE_STR}_raw.csv")
metrics_agg.to_csv(f"{path}/results/{SAVE_STR}.csv")
