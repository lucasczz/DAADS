import pathlib
import multiprocessing as mp

from evaluate import test_then_train
from evaluate import save_array

DATASETS = ["shuttle", "creditcard", "covertype"]
MODELS = ["AE-W", "AE", "DAE", "RRCF", "HST", "PW-AE", "xStream", "Kit-Net", "ILOF"]
CONFIGS = {
    "AE-W": {"lr": 0.02, "latent_dim": 1.0, "dropout": 0},
    "AE": {"lr": 0.02, "latent_dim": 0.1, "dropout": 0},
    "DAE": {"lr": 0.02},
    "PW-AE": {"lr": 0.1},
    "HST": {"n_trees": 25, "height": 15},
}
SUBSAMPLE = 20_000

PATH = pathlib.Path(__file__).parent.parent.resolve()


def get_scores(dataset, model):
    config = CONFIGS.get(model, {})
    savepath = f"{PATH}/results/scores/{model}_{dataset}"
    if model in ["AE", "DAE", "PW-AE", "AE-W"]:
        if model == "AE-W":
            model = "AE"
        for scale_scores in [True, False]:
            _, scores = test_then_train(
                dataset,
                model,
                subsample=SUBSAMPLE,
                scale_scores=scale_scores,
                seed=42,
                **config,
            )
            if not scale_scores:
                save_array(scores, savepath + "_unscaled")
            else:
                save_array(scores, savepath)

    else:
        config = CONFIGS[model]
        _, scores = test_then_train(
            dataset,
            model,
            subsample=SUBSAMPLE,
            seed=42,
            **config,
        )
        save_array(scores, savepath)


pool = mp.Pool(3)

runs = [
    pool.apply_async(get_scores, args=(dataset, model))
    for dataset in DATASETS
    for model in MODELS
]

[run.get() for run in runs]
