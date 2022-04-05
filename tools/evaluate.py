import os
import random
import sys
import warnings

import numpy as np
from IncrementalTorch.anomaly import *
from IncrementalTorch.base import AutoencoderBase
from IncrementalTorch.datasets import MNIST, Covertype, Shuttle
from river.anomaly import *
from river.datasets import HTTP, SMTP, CreditCard
from river.preprocessing import AdaptiveStandardScaler, MinMaxScaler, Normalizer
from river.feature_extraction import RBFSampler
import torch

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(script_dir))

from time import time

from tqdm import tqdm
from metrics import compute_metrics

DATASETS = {
    "covertype": Covertype,
    "creditcard": CreditCard,
    "shuttle": Shuttle,
    "mnist": MNIST,
    "smtp": SMTP,
    "http": HTTP,
}

PREPROCESSORS = {
    "minmax": MinMaxScaler,
    "standard": AdaptiveStandardScaler,
    "norm": Normalizer,
    "rbf": RBFSampler,
    "none": None,
}

POSTPROCESSORS = {
    "minmax": WindowedMinMaxScaler,
    "standard": WindowedStandardizer,
    "mean": WindowedMeanScaler,
    "standard_e": ExponentialStandardizer,
    "mean_e": ExponentialMeanScaler,
    "none": None,
}

MODELS = {
    "DAE": AutoencoderBase,
    "AE": NoDropoutAE,
    "RW-AE": RollingWindowAutoencoder,
    "PW-AE": ProbabilityWeightedAutoencoder,
    "Kit-Net": KitNet,
    "xStream": xStream,
    "RRCF": RobustRandomCutForest,
    "ILOF": ILOF,
    "OC-SVM": OneClassSVM,
    "HST": HalfSpaceTrees,
    "VAE": VariationalAutoencoder,
}


def test_then_train_batch(dataset, model, **model_kwargs):

    # Initialize postprocessor
    postprocessor = model_kwargs.pop("postprocessor", None)
    if postprocessor:
        postprocessor = WindowedStandardizer()

    # Initialize model
    model = MODELS.get(model, HalfSpaceTrees)
    model = model(**model_kwargs)

    scores, labels = [], []
    score_sum = 0
    start = time()
    for idx, data in enumerate(tqdm(dataset)):
        x, y = data
        score = [model.score_one(x_i[1].to_dict()) for x_i in x.iterrows()]
        model.learn_many(x)

        # Add results
        score_sum += np.mean(score)
        scores += score
        labels += y

    # Compute final metric scores
    perf_metrics = compute_metrics(labels, scores)

    return perf_metrics, scores


def test_then_train(
    dataset,
    model,
    subsample=50000,
    update_interv=1000,
    log_memory=False,
    preprocessor="minmax",
    postprocessor="none",
    seed=None,
    **model_kwargs,
):
    func_kwargs = dict(
        model=model,
        subsample=subsample,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        seed=seed,
        **model_kwargs,
    )

    if seed:
        seed_everything(seed)

    # Get data
    if isinstance(dataset, str):
        if dataset not in DATASETS:
            assert f"Dataset '{dataset}' could not be found."
        else:
            data = list(DATASETS[dataset]().take(subsample))
            func_kwargs["dataset"] = dataset
    else:
        data = dataset

    total_time = 0
    total_memory = 0

    # Initialize preprocessor
    try:
        _preprocessor = PREPROCESSORS[preprocessor]
        if _preprocessor:
            _preprocessor = _preprocessor()
    except KeyError:
        _preprocessor = None
        warnings.warn(f"Preprocessor '{preprocessor}' could not be found.")

    # Initialize postprocessor
    try:
        _postprocessor = POSTPROCESSORS[postprocessor]
        if _postprocessor:
            _postprocessor = _postprocessor()

    except KeyError:
        _postprocessor = None
        warnings.warn(f"Postprocessor '{postprocessor}' could not be found.")

    # Initialize model
    if isinstance(model, str):
        try:
            model = MODELS[model]
            model = model(**model_kwargs)
        except KeyError:
            warnings.warn(f"Model '{model}' could not be found.")

    scores, labels = [], []
    start = time()
    for idx, (x, y) in enumerate(tqdm(data)):

        # Preprocess input
        if _preprocessor:
            _preprocessor.learn_one(x)
            x = _preprocessor.transform_one(x)

        # Compute anomaly score
        score = model.score_learn_one(x)

        # Scale anomaly score
        if _postprocessor:
            _postprocessor.learn_one(score)
            score = _postprocessor.transform_one(score)

        # Add results
        scores.append(score)
        labels.append(y)

        # Log metrics
        if idx % update_interv == update_interv - 1:
            t = time() - start
            total_time += t
            if log_memory:
                total_memory += model._raw_memory_usage
            start = time()

    # Compute final metric scores
    total_time += time() - start

    metrics = compute_metrics(labels, scores)
    metrics["runtime"] = total_time

    if log_memory:
        total_memory += (
            model._raw_memory_usage * (len(data) % update_interv) / update_interv
        )
        metrics["avg_memory"] = total_memory / (len(data) / update_interv)

    metrics["status"] = "Completed"
    metrics.update(func_kwargs)

    return metrics, scores


def aggregate_dataframe(df, variables):
    grouped = df.groupby(variables)
    means = grouped.mean()
    stds = grouped.std()
    stds.columns = [f"{column}_std" for column in stds.columns]
    df_summary = means.join(stds, on=variables)
    return df_summary


def save_array(array, name):
    if isinstance(array, list):
        array = np.array(array)
    filename = f"{name}.npy"
    with open(filename, "wb") as f:
        np.save(f, array)


def seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
