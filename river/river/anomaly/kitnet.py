from .kitnet_model import KitNET as kit
from river import utils, anomaly


class KitNet(anomaly.AnomalyDetector):
    """KitNET is a lightweight online anomaly detection algorithm based on an ensemble of autoencoders :cite:`mirsky2018kitsune`. This model directly uses the implementation from `KitNET-py <https://github.com/ymirsky/KitNET-py>`_.

    Args:
        num_features (int): The number of features in your input dataset.
        max_size_ae (int): The maximum size of any autoencoder in the ensemble layer (Default=10).
        grace_feature_mapping (int): The number of instances the network will learn from before producing anomaly scores (Default=None).
        grace_anomaly_detector (int): The number of instances which will be taken to learn the feature mapping. If 'None', then FM_grace_period=AM_grace_period. (Default=50000).
        learning_rate (float): The default stochastic gradient descent learning rate for all autoencoders in the KitNET instance (Default=0.1).
        hidden_ratio (float): The default ratio of hidden to visible neurons. E.g., 0.75 will cause roughly a 25% compression in the hidden layer (Default=0.75).
    """

    def __init__(
        self,
        max_size_ae=10,
        grace_feature_mapping=100,
        grace_anomaly_detector=None,
        learning_rate=0.1,
        hidden_ratio=0.75,
    ):

        self.grace_feature_mapping = grace_feature_mapping
        self.hidden_ratio = hidden_ratio
        self.learning_rate = learning_rate
        self.max_size_ae = max_size_ae
        self.grace_anomaly_detector = grace_anomaly_detector
        self.to_init = True

    def learn_one(self, x: dict) -> anomaly.AnomalyDetector:
        """Fits the model to next instance.

        Args:
            x (dict of float values): The instance to fit.

        Returns:
            object: Returns itself.
        """
        if isinstance(x, dict):
            x = utils.dict2numpy(x)
        if self.to_init:
            self._init_model(x)
        self.model.process(x)

        return self

    def _init_model(self, x):
        self.num_features = x.shape[0]
        self.model = kit.KitNET(
            self.num_features,
            self.max_size_ae,
            self.grace_feature_mapping,
            self.grace_anomaly_detector,
            self.learning_rate,
            self.hidden_ratio,
        )
        self.to_init = False

    def score_one(self, x: dict = None) -> float:
        """Scores the anomalousness of the next instance. Outputs the last score. Note that this method must be called after the fit_partial

        Args:
            x (any): Ignored.
        Returns:
            float: The anomalousness score of the last fitted instance.
        """
        if isinstance(x, dict):
            x = utils.dict2numpy(x)

        if self.to_init:
            self._init_model(x)
            return 0.0

        if self.model.v is None:
            # The feature map is not discovered (i.e., still the grace period),
            # thus, KitNet gives an error.
            return 0.0
        else:
            return self.model.execute(x)
