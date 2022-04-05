from river.anomaly.ilof import ILOF
from river.anomaly import OneClassSVM, xStream, RobustRandomCutForest
from river.datasets import CreditCard
from IncrementalTorch.datasets import Covertype, Shuttle
from tqdm import tqdm
from river.preprocessing import StandardScaler, MinMaxScaler
from river.feature_extraction import RBFSampler

from sklearn.metrics import roc_auc_score

if __name__ == "__main__":
    data = list(Covertype().take(50_000))
    sampler = RBFSampler(n_components=10)
    scaler = MinMaxScaler()
    clf = RobustRandomCutForest()

    scores, labels = [], []
    for x, y in tqdm(data):
        score = clf.score_one(x)
        clf.learn_one(x)
        scores.append(score)
        labels.append(y)

    print(roc_auc_score(labels, scores))
