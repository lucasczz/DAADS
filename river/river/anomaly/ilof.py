from cmath import sqrt
import numpy as np

from river.anomaly.base import AnomalyDetector


def safe_div(a: float, b: float, fallback: float = 0.0):
    if b == 0:
        return fallback
    else:
        return a / b


class Point:
    id = 0

    def __init__(self, coords, k_neighbors=10):
        if isinstance(coords, dict):
            coords = np.asarray(list(coords.values()))
        self.coords = coords
        self.k_neighbors = k_neighbors

        self.neighbor_dists = []
        self.neighbors = []

        self.reach_dists = {}

        self.reach_density = None
        self.lof = None

        self.id = Point.id
        Point.id += 1
        if Point.id > 1e7:
            Point.id = 0

    def __add__(self, other):
        return self.coords + other

    def __sub__(self, other):
        return self.coords - other

    def __hash__(self):
        return hash(self.id)

    def __repr__(self):
        return f"Point {self.id}"

    @property
    def kdist(self):
        return self.neighbor_dists[-1]

    def get_dist(self, other):
        if type(other) is list:
            return np.linalg.norm(self - np.asarray([p.coords for p in other]), axis=-1)
        else:
            return np.linalg.norm(self - other, axis=-1)

    def get_reach_dist(self, other_point, dist=None):
        if dist is None and other_point:
            dist = self.get_dist(other_point)
        return max(dist, other_point.kdist)

    def get_reach_density(self):
        return safe_div(
            self.k_neighbors,
            sum(self.reach_dists[neighbor] for neighbor in self.neighbors),
            fallback=1e6,
        )

    def get_lof(self):
        return safe_div(
            sum(neighbor.reach_density for neighbor in self.neighbors),
            self.k_neighbors * self.reach_density,
            fallback=1,
        )

    def get_neighbors(self, others, dists, k_neighbors=None, self_included=False):
        if k_neighbors is None:
            k_neighbors = self.k_neighbors
        if self_included:
            k_neighbors += 1
        if k_neighbors >= len(others):
            k_neighbors = len(others)
        neighbor_idcs = np.argpartition(dists, range(k_neighbors))[:k_neighbors]
        neighbors = [others[neighbor_idx] for neighbor_idx in neighbor_idcs]
        neighbor_dists = dists[neighbor_idcs].tolist()
        if self_included:
            return (neighbors[1:], neighbor_dists[1:])
        else:
            return (neighbors, neighbor_dists)

    def get_reverse_neighbors(self, others, dists=None, new_point=True):
        reverse_neighbors, reverse_neighbor_dists = [], []
        if new_point:
            for dist, other in zip(dists, others):
                if dist < other.kdist:
                    reverse_neighbors.append(other)
                    reverse_neighbor_dists.append(dist)
            return reverse_neighbors, reverse_neighbor_dists
        else:
            for other in others:
                if self in other.neighbors:
                    reverse_neighbors.append(other)
            return reverse_neighbors

    def add_neighbor(self, other, dist):
        for idx in range(self.k_neighbors - 1, -1, -1):
            if idx == 0 or dist > self.neighbor_dists[idx - 1]:
                self.neighbor_dists.insert(idx, dist)
                self.neighbors.insert(idx, other)
                self.neighbors = self.neighbors[: self.k_neighbors]
                self.neighbor_dists[: self.k_neighbors]
                self.reach_dists[other] = self.get_reach_dist(other, dist)
                return

    def remove_point(self, other):
        if isinstance(other, list):
            for p in other:
                if p in self.neighbors:
                    remove_idx = self.neighbors.index(p)
                    del self.neighbor_dists[remove_idx]
                    del self.neighbors[remove_idx]
                    del self.reach_dists[p]
        else:
            if other in self.neighbors:
                remove_idx = self.neighbors.index(other)
                del self.neighbor_dists[remove_idx]
                del self.neighbors[remove_idx]
                del self.reach_dists[other]


class ILOF(AnomalyDetector):
    def __init__(self, k_neighbors=10, window_size=250):
        self.points = []
        self.k_neighbors = k_neighbors
        self.window_size = window_size
        self.count = 0

    def insert(self, point):
        dists = point.get_dist(self.points)
        point.neighbors, point.neighbor_dists = point.get_neighbors(self.points, dists)

        for neighbor, neighbor_dist in zip(point.neighbors, point.neighbor_dists):
            point.reach_dists[neighbor] = point.get_reach_dist(
                other_point=neighbor, dist=neighbor_dist
            )

        # Update new point as neighbor to reverse nearest neighbors
        reverse_neighbors, reverse_neighbor_dists = point.get_reverse_neighbors(
            self.points, dists
        )
        for reverse_neighbor, reverse_neighbor_dist in zip(
            reverse_neighbors, reverse_neighbor_dists
        ):
            reverse_neighbor.add_neighbor(point, reverse_neighbor_dist)

        # Update reach distances of reverse nearest neighbors
        points_update_reach_density = set(reverse_neighbors)
        for point_update in reverse_neighbors:
            for neighbor in point_update.neighbors:
                if neighbor == point:
                    continue
                neighbor.reach_dists[point_update] = point_update.kdist
                if point_update in neighbor.neighbors:
                    points_update_reach_density.add(neighbor)

        # Update local reach densities of affected points
        # points_update_lof = points_update_reach_density.copy()
        for point_update in points_update_reach_density:
            point_update.reach_density = point_update.get_reach_density()
            # reverse_neighbors_update = point_update.get_reverse_neighbors(
            #     self.points, new_point=False
            # )
            # points_update_lof.update(reverse_neighbors_update)
        point.reach_density = point.get_reach_density()

        # Update LOF of affected points
        # for point_update in points_update_lof:
        #     point_update.lof = point_update.get_lof()

        point.lof = point.get_lof()
        self.points.append(point)
        return point.lof

    def delete(self, points):
        # Remove points from list of points
        if isinstance(points, Point):
            points = [points]
        for point_del in points:
            self.points.remove(point_del)

        points_update_neighbor_dists = set()

        # Find points that have selected points as neighbors
        for point_del in points:
            points_update_neighbor_dists.update(
                point_del.get_reverse_neighbors(self.points, new_point=False)
            )

        # Remove selected points from all neighbor lists
        for point_update in points_update_neighbor_dists:
            point_update.remove_point(points)

        # Fill up neighbor list of points that had removed point as neighbor
        points_update_neighbor_dists.difference_update(points)
        for point_update in points_update_neighbor_dists:
            if len(point_update.neighbors) < self.k_neighbors:
                dists = point_update.get_dist(self.points)
                (
                    point_update.neighbors,
                    point_update.neighbor_dists,
                ) = point_update.get_neighbors(self.points, dists)
                for neighbor, neighbor_dist in zip(
                    point_update.neighbors, point_update.neighbor_dists
                ):
                    if neighbor not in point_update.reach_dists:
                        point_update.reach_dists[
                            neighbor
                        ] = point_update.get_reach_dist(neighbor, neighbor_dist)

        # Update reach distances of affected points
        points_update_reach_density = points_update_neighbor_dists.copy()
        for point_update in points_update_neighbor_dists:
            for neighbor in point_update.neighbors[: self.k_neighbors - 1]:
                neighbor.reach_dists[point_update] = point_update.kdist
                if point_update in neighbor.neighbors:
                    points_update_reach_density.add(neighbor)

        # Update reach densities of affected points
        # points_update_lof = points_update_reach_density.copy()
        for point_update in points_update_reach_density:
            point_update.reach_density = point_update.get_reach_density()
            # reverse_neighbors = point_update.get_reverse_neighbors(
            #     self.points, new_point=False
            # )
            # points_update_lof.update(reverse_neighbors)

        # Recalculate LOF
        # for point_update in points_update_lof:
        #    point_update.lof = point_update.get_lof()

    def init_points(self):
        for point in self.points:
            others = self.points.copy()
            others.remove(point)
            dists = point.get_dist(others)
            sorting_idcs = np.argsort(dists)
            point.neighbors = [others[idx] for idx in sorting_idcs]
            point.neighbor_dists = [dists[idx] for idx in sorting_idcs]
        for point in self.points:
            for neighbor in point.neighbors:
                point.reach_dists[neighbor] = neighbor.kdist
        for point in self.points:
            point.reach_density = point.get_reach_density()
        for point in self.points:
            point.lof = point.get_lof()

    def score_learn_one(self, x: dict) -> float:
        return self.score_one(x)

    def score_one(self, x: dict) -> float:
        point = Point(x, self.k_neighbors)
        if len(self.points) < self.k_neighbors:
            self.points.append(point)
            lof = 1
        elif len(self.points) == self.k_neighbors:
            self.points.append(point)
            self.init_points()
            lof = point.lof
        else:
            if len(self.points) == self.window_size:
                self.delete(self.points[0])
            lof = self.insert(point)
        return lof

    def learn_one(self, x: dict) -> "AnomalyDetector":
        return self


class DILOF(AnomalyDetector):
    def __init__(self, k_neighbors=20, window_size=250):
        self.points = []
        self.k_neighbors = k_neighbors
        self.window_size = window_size
        self.count = 0

    def psi(y):
        if y > 1:
            return (y - 1) ** 2
        elif y < 0:
            return y ** 2
        else:
            return 0

    def rho(self, y, point, others):
        sorted({y * sqrt((point - other) ** 2) for other in others})[
            -int(self.window_size / 4 - self.k_neighbors + 1) :
        ]



class IndexedList:
    def __init__(self, max_entries):
        self.max_entries = max_entries
        self._entries = None
        self.n_entries = 0

    def append(self, x):
        if self.n_entries == 0:
            self._entries = np.ones((x.shape[0], self.max_entries))

        self._entries[self.n_entries] = x
        self.n_entries += 1

    def pop(self):
        if self.n_entries > 0:
            result = self._entries[0]
        self._entries = self._entries[1:]
        self._entries = np.append(self._entries, np.ones(self._entries.shape[1]))
        self.n_entries -= 1
        return result

    @property
    def entries(self):
        return self._entries[: self.n_entries]

    def remove(self, indices):
        self._entries_new = np.ones_like(self.entries)
        self._entries_new[: self._entries.shape[0] - len(indices)] = np.delete(
            self._entries, indices
        )
        self._entries = self._entries_new
        self.n_entries -= len(indices)