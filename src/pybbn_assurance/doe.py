# Author: Vishal Gattani
# Created: 2024-06-07

from typing import Any, List, Optional

import numpy as np
import pandas as pd

from pybbn_assurance.helper import get_binomial_prob


class DOE:
    def __init__(self, n_experiments: int, experiment: "Experiment") -> None:
        self.n_experiments = n_experiments
        self.thresholds = list(range(self.n_experiments))
        self.experiment = experiment


class Experiment:
    def __init__(self) -> None:
        pass


class SuccessNode:
    def __init__(
        self,
        id: int,
        name: str,
        n_experiments: int,
        probability_of_success: float,
    ) -> None:
        self.probability_list: Optional[List[float]] = None
        self.cpt: Optional[pd.DataFrame] = None
        self.states: Optional[pd.Index] = None
        self.id = id
        self.name = name
        self.child: List[int] = []
        self.parent: List[int] = []
        self.probability_of_success = probability_of_success

        self.set_cpt(n_experiments=n_experiments, probability_of_success=probability_of_success)

    def set_cpt(self, n_experiments: int, probability_of_success: float) -> None:
        self.probability_list = get_binomial_prob(n=n_experiments, p=probability_of_success)
        self.cpt = pd.DataFrame({"success": self.probability_list})
        idxlist = self.cpt.index.tolist()
        self.cpt = self.cpt.set_index([pd.Index(["n" + str(idx) for idx in idxlist])])
        self.cpt["States"] = idxlist
        self.states = self.cpt["States"]
        self.cpt.set_index("States", inplace=True)

    def get_cpt_list(self) -> List[Any]:
        return np.ravel(self.cpt.values.tolist()).tolist()

    def get_cpt_states(self) -> Optional[pd.Index]:
        return self.states


class ThresholdNode:
    def __init__(
        self,
        id: Optional[int] = None,
        name: Optional[str] = None,
        n_experiments: Optional[int] = None,
        threshold: Optional[int] = None,
    ) -> None:
        self.cpt: Optional[pd.DataFrame] = None
        self.n_experiments = n_experiments
        self.threshold = threshold
        self.states: Optional[pd.Index] = None
        self.id = id
        self.name = name
        self.child: List[int] = []
        self.parent: List[int] = []

    def get_n_experiments(self) -> Optional[int]:
        return self.n_experiments

    def get_threshold(self) -> Optional[int]:
        return self.threshold

    def get_cpt_list(self) -> List[Any]:
        return np.ndarray.flatten(self.cpt.transpose().values).tolist()

    def get_cpt_states(self) -> Optional[pd.Index]:
        return self.states


class MaxThresholdNode(ThresholdNode):
    """Maximum threshold applied to a node before it returns to false states."""

    def __init__(
        self,
        id: int,
        name: str,
        n_experiments: int,
        threshold: int,
    ) -> None:
        super().__init__(id=id, name=name, n_experiments=n_experiments, threshold=threshold)
        self.set_cpt()
        self.child = []
        self.parent = []

    def set_cpt(self) -> None:
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
    """Minimum threshold applied to a node after it returns to True states."""

    def __init__(
        self,
        id: int,
        name: str,
        n_experiments: int,
        threshold: int,
    ) -> None:
        super().__init__(id=id, name=name, n_experiments=n_experiments, threshold=threshold)
        self.set_cpt()
        self.child = []
        self.parent = []

    def set_cpt(self) -> None:
        keys, values = [], []
        for i in range(self.n_experiments + 1):
            keys.append(str(i))
            if i >= self.threshold:
                values.append([1, 0])
            else:
                values.append([0, 1])
        cpt_list = dict(zip(keys, values))
        self.cpt = pd.DataFrame(cpt_list)
        self.cpt["States"] = ["True", "False"]
        self.states = self.cpt["States"]
        self.cpt.set_index("States", inplace=True)


class GoalNode(ThresholdNode):
    def __init__(self, id: int, name: str, n_children: int) -> None:
        self.id = id
        self.name = name
        self.cpt: Optional[pd.DataFrame] = None
        self.n_children = n_children
        self.states: Optional[pd.Index] = None
        self.child: List[int] = []
        self.parent: List[int] = []
        self.initialize()

    def initialize(self) -> None:
        cpt_list = {str(i): [1, 0] if i == 0 else [0, 1] for i in range(2**self.n_children)}
        self.cpt = pd.DataFrame(cpt_list)
        self.cpt["States"] = ["True", "False"]
        self.states = self.cpt["States"]
        self.cpt.set_index("States", inplace=True)
