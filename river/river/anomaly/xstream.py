import numpy as np
from river import anomaly, utils
import mmh3

class xStream(anomaly.AnomalyDetector):
    """The xStream model for row-streaming data :cite:`xstream`. It first projects the data via streamhash projection. It then fits half space chains by reference windowing. It scores the instances using the window fitted to the reference window.

    Args:
        n_components (int): The number of components for streamhash projection (Default=100).
        n_chains (int): The number of half-space chains (Default=100).
        depth (int): The maximum depth for the chains (Default=25).
        window_size (int): The size (and the sliding length) of the reference window (Default=25).
    """

    def __init__(
            self,
            num_components=10,
            n_chains=10,
            depth=8,
            window_size=250):
            
        self.streamhash = StreamhashProjector(num_components=num_components)
        deltamax = np.ones(num_components) * 0.5
        deltamax[np.abs(deltamax) <= 0.0001] = 1.0
        self.window_size = window_size
        self.hs_chains = _HSChains(
            deltamax=deltamax,
            n_chains=n_chains,
            depth=depth)

        self.step = 0
        self.cur_window = []
        self.ref_window = None

    def learn_one(self, x: dict) -> anomaly.AnomalyDetector:
        """Fits the model to next instance.

        Args:
            x (dict of float values): The instance to fit.

        Returns:
            object: Returns itself.
        """
        x = utils.dict2numpy(x)
        self.step += 1

        x = self.streamhash.fit_transform_partial(x)

        x = x.reshape(1, -1)
        self.cur_window.append(x)

        self.hs_chains.fit(x)

        if self.step % self.window_size == 0:
            self.ref_window = self.cur_window
            self.cur_window = []
            deltamax = self._compute_deltamax()
            self.hs_chains.set_deltamax(deltamax)
            self.hs_chains.next_window()

        return self

    def score_one(self, x: dict) -> float:
        """Scores the anomalousness of the next instance.

        Args:
            x (dict of float values): The instance to fit.

        Returns:
            score (float): The anomalousness score of the input instance.
        """
        x = utils.dict2numpy(x)
        x = self.streamhash.fit_transform_partial(x)
        x = x.reshape(1, -1)
        score = self.hs_chains.score(x).flatten().item()

        return score

    def _compute_deltamax(self):
        concatenated = np.concatenate(self.ref_window, axis=0)
        mx = np.max(concatenated, axis=0)
        mn = np.min(concatenated, axis=0)

        

        deltamax = (mx - mn) / 2.0
        deltamax[np.abs(deltamax) <= 0.0001] = 1.0

        return deltamax


class _Chain:

    def __init__(self, deltamax, depth):
        k = len(deltamax)

        self.depth = depth
        self.fs = [np.random.randint(0, k) for d in range(depth)]
        self.cmsketches = [{} for i in range(depth)] * depth
        self.cmsketches_cur = [{} for i in range(depth)] * depth

        self.deltamax = deltamax  # feature ranges
        self.rand_arr = np.random.rand(k)
        self.shift = self.rand_arr * deltamax

        self.is_first_window = True

    def fit(self, x):
        prebins = np.zeros(x.shape, dtype=np.float)
        depthcount = np.zeros(len(self.deltamax), dtype=np.int)
        for depth in range(self.depth):
            f = self.fs[depth]
            depthcount[f] += 1

            if depthcount[f] == 1:
                prebins[:, f] = (x[:, f] + self.shift[f]) / self.deltamax[f]
            else:
                prebins[:, f] = 2.0 * prebins[:, f] - \
                    self.shift[f] / self.deltamax[f]

            if self.is_first_window:
                cmsketch = self.cmsketches[depth]
                for prebin in prebins:
                    l_index = tuple(np.floor(prebin).astype(np.int))
                    if l_index not in cmsketch:
                        cmsketch[l_index] = 0
                    cmsketch[l_index] += 1

                self.cmsketches[depth] = cmsketch

                self.cmsketches_cur[depth] = cmsketch

            else:
                cmsketch = self.cmsketches_cur[depth]

                for prebin in prebins:
                    l_index = tuple(np.floor(prebin).astype(np.int))
                    if l_index not in cmsketch:
                        cmsketch[l_index] = 0
                    cmsketch[l_index] += 1

                self.cmsketches_cur[depth] = cmsketch

        return self

    def bincount(self, x):
        scores = np.zeros((x.shape[0], self.depth))
        prebins = np.zeros(x.shape, dtype=np.float)
        depthcount = np.zeros(len(self.deltamax), dtype=np.int)
        for depth in range(self.depth):
            f = self.fs[depth]
            depthcount[f] += 1

            if depthcount[f] == 1:
                prebins[:, f] = (x[:, f] + self.shift[f]) / self.deltamax[f]
            else:
                prebins[:, f] = 2.0 * prebins[:, f] - \
                    self.shift[f] / self.deltamax[f]

            cmsketch = self.cmsketches[depth]
            for i, prebin in enumerate(prebins):
                l_index = tuple(np.floor(prebin).astype(np.int))
                if l_index not in cmsketch:
                    scores[i, depth] = 0.0
                else:
                    scores[i, depth] = cmsketch[l_index]

        return scores

    def score(self, x):
        # scale score logarithmically to avoid overflow:
        #    score = min_d [ log2(bincount x 2^d) = log2(bincount) + d ]
        scores = self.bincount(x)
        depths = np.array([d for d in range(1, self.depth + 1)])
        scores = np.log2(1.0 + scores) + depths  # add 1 to avoid log(0)
        return -np.min(scores, axis=1)

    def next_window(self):
        self.is_first_window = False
        self.cmsketches = self.cmsketches_cur
        self.cmsketches_cur = [{} for _ in range(self.depth)] * self.depth


class _HSChains:
    def __init__(self, deltamax, n_chains=100, depth=25):
        self.nchains = n_chains
        self.depth = depth
        self.chains = []

        for i in range(self.nchains):

            c = _Chain(deltamax=deltamax, depth=self.depth)
            self.chains.append(c)

    def score(self, x):
        scores = np.zeros(x.shape[0])
        for ch in self.chains:
            scores += ch.score(x)

        scores /= float(self.nchains)
        return scores

    def fit(self, x):
        for ch in self.chains:
            ch.fit(x)

    def next_window(self):
        for ch in self.chains:
            ch.next_window()

    def set_deltamax(self, deltamax):
        for ch in self.chains:
            ch.deltamax = deltamax
            ch.shift = ch.rand_arr * deltamax

class StreamhashProjector():
    """Streamhash projection method  from Manzoor et. al.that is similar (or equivalent) to SparseRandomProjection. :cite:`xstream` The implementation is taken from the `cmuxstream-core repository <https://github.com/cmuxstream/cmuxstream-core>`_.

        Args:
            num_components (int): The number of dimensions that the target will be projected into.
            density (float): Density parameter of the streamhash projection.
    """

    def __init__(self, num_components, density=1 / 3.0):
        self.output_dims = num_components
        self.keys = np.arange(0, num_components, 1)
        self.constant = np.sqrt(1. / density) / np.sqrt(num_components)
        self.density = density
        self.n_components = num_components

    def fit_partial(self, x):
        """Fits particular (next) timestep's features to train the projector.

        Args:
            x (np.float array of shape (n_components,)): Input feature vector.

        Returns:
            object: self.
        """
        return self

    def transform_partial(self, x):
        """Projects particular (next) timestep's vector to (possibly) lower dimensional space.

        Args:
            x (np.float array of shape (num_features,)): Input feature vector.

        Returns:
            projected_X (np.float array of shape (num_components,)): Projected feature vector.
        """
        x = x.reshape(1, -1)

        ndim = x.shape[1]

        feature_names = [str(i) for i in range(ndim)]

        R = np.array([[self._hash_string(k, f)
                       for f in feature_names]
                      for k in self.keys])

        Y = np.dot(x, R.T).squeeze()

        return Y

    def _hash_string(self, k, s):
        hash_value = int(mmh3.hash(s, signed=False, seed=k)) / (2.0 ** 32 - 1)
        s = self.density
        if hash_value <= s / 2.0:
            return -1 * self.constant
        elif hash_value <= s:
            return self.constant
        else:
            return 0

    def fit_transform_partial(self, x):
        self.fit_partial(x)
        return self.transform_partial(x)
