import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from river import base


def pr_auc(labels, scores):
    precision, recall, _ = precision_recall_curve(labels, scores)
    return auc(recall, precision)


def roc_auc(labels, scores):
    fpr, tpr, _ = roc_curve(labels, scores)
    return auc(fpr, tpr)


def f1_prec_rec_thresh(labels, scores):
    precision, recall, threshold = precision_recall_curve(labels, scores)
    count = 2 * precision * recall
    denom = precision + recall
    f1 = np.divide(count, denom, out=np.zeros_like(count), where=denom != 0)
    return f1, precision, recall, threshold


def max_f1_score(labels, scores):
    f1, _, _, _ = f1_prec_rec_thresh(labels, scores)
    return np.max(f1)


def max_f1_threshold(labels, scores):
    f1, _, _, threshold = f1_prec_rec_thresh(labels, scores)
    return threshold[np.argmax(f1)]


def max_f1_precision(labels, scores):
    f1, precision, _, _ = f1_prec_rec_thresh(labels, scores)
    return precision[np.argmax(f1)]


def max_f1_recall(labels, scores):
    f1, _, recall, _ = f1_prec_rec_thresh(labels, scores)
    return recall[np.argmax(f1)]


def memory_usage(estimator: base.Base):
    return estimator._raw_memory_usage / 1000


METRICS = {
    "PR-AUC": pr_auc,
    "ROC-AUC": roc_auc,
    "Max F1": max_f1_score,
    "Max F1 Precision": max_f1_precision,
    "Max F1 Recall": max_f1_recall,
}


class MetricTracker:
    def __init__(self, update_interval=100) -> None:
        self.update_interval = update_interval
        self.step = 0
        self.metric_means = None

    def update(self, metrics):
        if self.metric_means is None:
            self.metric_means = dict.fromkeys(metrics.keys(), 0)
        for key, value in metrics.items():
            self.metric_means[key] += value
        self.step += 1
        if self.step % self.update_interval == 0:
            for key in self.metric_means.keys():
                self.metric_means[key] /= self.update_interval
            self.metric_means = dict.fromkeys(metrics.keys(), 0)


def compute_metrics(labels, scores, metrics=METRICS):
    result = {}
    for name, metric in metrics.items():
        result[name] = metric(labels, scores)
    return result
