import pathlib

import numpy as np
import pandas as pd

from helper import get_binomial_prob, get_cdf_binomial_prob


class DOE:
    def __init__(self, n_experiments, experiment) -> None:
        self.n_experiments = n_experiments
        self.thresholds = list(range(self.n_experiments))
        self.experiment = experiment


class Experiment:
    def __init__(self) -> None:
        pass


class SuccessNode:
    def __init__(self, n_experiments, probability_of_success) -> None:
        self.probability_list = None
        self.cpt = None
        self.states = None
        self.set_cpt(
            n_experiments=n_experiments, probability_of_success=probability_of_success
        )

    def set_cpt(self, n_experiments, probability_of_success):
        self.probability_list = get_binomial_prob(
            n=n_experiments, p=probability_of_success
        )
        self.cpt = pd.DataFrame({"success": self.probability_list})
        idxlist = self.cpt.index.tolist()
        self.cpt = self.cpt.set_index([pd.Index(["n" + str(idx) for idx in idxlist])])
        self.cpt["States"] = idxlist
        self.states = self.cpt["States"]
        self.cpt.set_index("States", inplace=True)

    def get_cpt_list(self):
        return np.ravel(self.cpt.values.tolist()).tolist()

    def get_cpt_states(self):
        return self.states


class ThresholdNode:
    def __init__(self, n_experiments, threshold) -> None:
        self.cpt = None
        self.n_experiments = n_experiments
        self.threshold = threshold
        self.states = None

    def get_n_experiments(self):
        return self.n_experiments

    def get_threshold(self):
        return self.threshold

    def get_cpt_list(self):
        return np.ndarray.flatten(self.cpt.transpose().values).tolist()

    def get_cpt_states(self):
        return self.states


class MaxThresholdNode(ThresholdNode):
    """Maximum threshold applied to a node before it returns to false states

    Args:
        ThresholdNode (_type_): _description_
    """

    def __init__(self, n_experiments, threshold) -> None:
        super().__init__(n_experiments, threshold)
        self.set_cpt()

    def set_cpt(self):
        keys, values = [], []
        for i in range(self.n_experiments + 1):
            keys.append(str(i))
            if i > self.threshold:
                values.append([0, 1])
            else:
                values.append([1, 0])

        cpt_list = dict(zip(keys, values))
        self.cpt = pd.DataFrame(cpt_list)
        self.cpt["States"] = ["True", "False"]
        self.states = self.cpt["States"]
        self.cpt.set_index("States", inplace=True)


class MinThresholdNode(ThresholdNode):
    """Minimum threshold applied to a node after it returns to True states

    Args:
        ThresholdNode (_type_): _description_
    """

    def __init__(self, n_experiments, threshold) -> None:
        super().__init__(n_experiments, threshold)
        self.set_cpt()

    def set_cpt(self):
        keys, values = [], []
        for i in range(self.n_experiments + 1):
            keys.append(str(i))
            if i > self.threshold:
                values.append([1, 0])
            else:
                values.append([0, 1])
        cpt_list = dict(zip(keys, values))
        self.cpt = pd.DataFrame(cpt_list)
        self.cpt["States"] = ["True", "False"]
        self.states = self.cpt["States"]
        self.cpt.set_index("States", inplace=True)


class GoalNode(ThresholdNode):
    def __init__(self, n_children) -> None:
        self.cpt = None
        self.n_children = n_children
        self.states = None
        self.initialize()

    def initialize(self):
        keys, values = [], []
        for i in range(2**self.n_children):
            keys.append(str(i))
            if i > 0:
                values.append([1, 0])
            else:
                values.append([0, 1])
        cpt_list = dict(zip(keys, values))
        self.cpt = pd.DataFrame(cpt_list)
        self.cpt["States"] = ["True", "False"]
        self.states = self.cpt["States"]
        self.cpt.set_index("States", inplace=True)
