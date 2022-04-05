from river import anomaly, utils

class RobustRandomCutForest(anomaly.AnomalyDetector):
    """Robust Random Cut Forest model :cite:`guha2016robust`. The implementation uses `rrcf library <https://github.com/kLabUM/rrcf>`_ :cite:`bartos_2019_rrcf`.

        Args:
            num_trees (int): The number of trees.
            shingle_size (int): The shingle size (Default=4).
            tree_size (int): The tree size (Default=256).
    """

    def __init__(self, num_trees=100, shingle_size=4, tree_size=250):
        from rrcf import rrcf

        self.tree_size = tree_size
        self.shingle_size = shingle_size
        self.num_trees = num_trees

        self.forest = []
        for _ in range(self.num_trees):
            tree = rrcf.RCTree()
            self.forest.append(tree)

        self.index = 0

    def learn_one(self, x: dict) -> anomaly.AnomalyDetector:
        """Fits the model to next instance.

        Args:
            x (dict of float values): The instance to fit.

        Returns:
            object: Returns itself.
        """
        x = utils.dict2numpy(x)
        for tree in self.forest:
            if len(tree.leaves) > self.tree_size:
                tree.forget_point(self.index - self.tree_size)

            tree.insert_point(x, index=self.index)

        self.index += 1

        return self

    def score_one(self, x: dict) -> float:
        """Scores the anomalousness of the next instance.

        Args:
            X (np.float array of shape (num_features,)): The instance to score. Higher scores represent more anomalous instances whereas lower scores correspond to more normal instances.

        Returns:
            float: The anomalousness score of the input instance.
        """
        x = utils.dict2numpy(x)
        score = 0.0
        for tree in self.forest:
            if self.index > 0:
                leaf = tree.find_duplicate(x)
                if leaf is None:
                    tree.insert_point(x, index="test_point")
                    score += 1.0 * tree.codisp("test_point") / self.num_trees
                    tree.forget_point("test_point")
                else:
                    score += 1.0 * tree.codisp(leaf) / self.num_trees
            else: 
                tree.insert_point(x, index="test_point")
                score += 1.0 * tree.codisp("test_point") / self.num_trees
                tree.forget_point("test_point")
        return score

    def score_learn_one(self, x: dict) -> float:
        self.learn_one(x)
        return self.score_one(x)