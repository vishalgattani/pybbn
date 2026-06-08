"""Tests for pybbn_assurance/doe.py — DOE node types and CPT shapes."""

import pytest
import numpy as np
import pandas as pd

from pybbn_assurance.doe import (
    DOE,
    Experiment,
    GoalNode,
    MaxThresholdNode,
    MinThresholdNode,
    SuccessNode,
)


class TestSuccessNode:
    """Tests for SuccessNode CPT construction."""

    def test_basic_construction(self):
        node = SuccessNode(id=1, name="test", n_experiments=5, probability_of_success=0.5)
        assert node.id == 1
        assert node.name == "test"
        assert node.probability_of_success == 0.5
        assert node.cpt is not None
        assert node.states is not None
        assert node.child == []
        assert node.parent == []

    def test_cpt_is_dataframe(self):
        node = SuccessNode(id=1, name="test", n_experiments=3, probability_of_success=0.7)
        assert isinstance(node.cpt, pd.DataFrame)

    def test_cpt_rows_equals_n_plus_one(self):
        """CPT should have n+1 rows (one per possible number of successes)."""
        node = SuccessNode(id=1, name="test", n_experiments=7, probability_of_success=0.4)
        assert len(node.cpt) == 8

    def test_cpt_column_is_success(self):
        assert "success" in [col for col in SuccessNode(0, "x", 5, 0.5).cpt.columns]

    def test_cpt_values_sum_to_one(self):
        """Sum of all CPT probability values should equal 1.0."""
        node = SuccessNode(id=1, name="test", n_experiments=10, probability_of_success=0.3)
        total = node.cpt["success"].sum()
        assert total == pytest.approx(1.0, abs=1e-10)

    def test_cpt_values_non_negative(self):
        node = SuccessNode(id=1, name="test", n_experiments=5, probability_of_success=0.5)
        assert all(v >= 0 for v in node.cpt["success"])

    def test_state_labels_are_index_values(self):
        """After set_index('States'), the index becomes integer state values."""
        node = SuccessNode(id=1, name="test", n_experiments=3, probability_of_success=0.5)
        # The final index after set_index('States') is integers
        assert all(isinstance(s, (int, np.integer)) for s in node.cpt.index)

    def test_get_cpt_list_returns_list(self):
        node = SuccessNode(id=1, name="test", n_experiments=4, probability_of_success=0.6)
        result = node.get_cpt_list()
        assert isinstance(result, list)
        assert len(result) == 5

    def test_get_cpt_states_returns_series(self):
        node = SuccessNode(id=1, name="test", n_experiments=5, probability_of_success=0.5)
        states = node.get_cpt_states()
        # Returns a pandas Series (the "States" column), not an Index
        import pandas as pd
        assert isinstance(states, pd.Series)

    def test_high_probability_favors_higher_k(self):
        """With p=0.9, the highest CPT values should be at the end."""
        node = SuccessNode(id=1, name="test", n_experiments=10, probability_of_success=0.9)
        vals = node.cpt["success"].tolist()
        # Last value should be higher than first
        assert vals[-1] > vals[0]

    def test_low_probability_favors_lower_k(self):
        """With p=0.1, the highest CPT values should be at the start."""
        node = SuccessNode(id=1, name="test", n_experiments=10, probability_of_success=0.1)
        vals = node.cpt["success"].tolist()
        assert vals[0] > vals[-1]


class TestMinThresholdNode:
    """Tests for MinThresholdNode CPT — returns True once threshold is met."""

    def test_basic_construction(self):
        node = MinThresholdNode(id=1, name="test", n_experiments=5, threshold=2)
        assert node.id == 1
        assert node.n_experiments == 5
        assert node.threshold == 2
        assert node.cpt is not None
        assert node.child == []
        assert node.parent == []

    def test_cpt_is_dataframe(self):
        node = MinThresholdNode(id=1, name="test", n_experiments=3, threshold=1)
        assert isinstance(node.cpt, pd.DataFrame)

    def test_cpt_has_true_false_states(self):
        node = MinThresholdNode(id=1, name="test", n_experiments=5, threshold=2)
        assert "True" in node.cpt.index
        assert "False" in node.cpt.index

    def test_at_or_above_threshold_is_true(self):
        """k >= threshold → [1, 0] (True=1, False=0)."""
        node = MinThresholdNode(id=1, name="test", n_experiments=5, threshold=3)
        col = "3"  # threshold column
        assert node.cpt.at["True", col] == 1
        assert node.cpt.at["False", col] == 0

    def test_below_threshold_is_false(self):
        """k < threshold → [0, 1] (True=0, False=1)."""
        node = MinThresholdNode(id=1, name="test", n_experiments=5, threshold=3)
        col = "2"  # below threshold
        assert node.cpt.at["True", col] == 0
        assert node.cpt.at["False", col] == 1

    def test_threshold_zero_all_true(self):
        """threshold=0 → every k >= 0 → all True."""
        node = MinThresholdNode(id=1, name="test", n_experiments=3, threshold=0)
        for col in ["0", "1", "2", "3"]:
            assert node.cpt.at["True", col] == 1

    def test_get_cpt_list_returns_list(self):
        node = MinThresholdNode(id=1, name="test", n_experiments=4, threshold=2)
        result = node.get_cpt_list()
        assert isinstance(result, list)

    def test_get_cpt_states_returns_series(self):
        node = MinThresholdNode(id=1, name="test", n_experiments=5, threshold=2)
        states = node.get_cpt_states()
        import pandas as pd
        assert isinstance(states, pd.Series)

    def test_getters(self):
        node = MinThresholdNode(id=1, name="test", n_experiments=5, threshold=2)
        assert node.get_n_experiments() == 5
        assert node.get_threshold() == 2

    def test_cpt_columns_match_range(self):
        node = MinThresholdNode(id=1, name="test", n_experiments=4, threshold=2)
        expected_cols = {"0", "1", "2", "3", "4"}
        assert set(node.cpt.columns) == expected_cols


class TestMaxThresholdNode:
    """Tests for MaxThresholdNode CPT — returns False once threshold is exceeded."""

    def test_basic_construction(self):
        node = MaxThresholdNode(id=1, name="test", n_experiments=5, threshold=2)
        assert node.id == 1
        assert node.n_experiments == 5
        assert node.threshold == 2
        assert node.cpt is not None
        assert node.child == []
        assert node.parent == []

    def test_cpt_is_dataframe(self):
        node = MaxThresholdNode(id=1, name="test", n_experiments=3, threshold=1)
        assert isinstance(node.cpt, pd.DataFrame)

    def test_cpt_has_true_false_states(self):
        node = MaxThresholdNode(id=1, name="test", n_experiments=5, threshold=2)
        assert "True" in node.cpt.index
        assert "False" in node.cpt.index

    def test_at_threshold_is_true(self):
        """k <= threshold → [1, 0] (True=1, False=0)."""
        node = MaxThresholdNode(id=1, name="test", n_experiments=5, threshold=3)
        col = "3"
        assert node.cpt.at["True", col] == 1
        assert node.cpt.at["False", col] == 0

    def test_above_threshold_is_false(self):
        """k > threshold → [0, 1] (True=0, False=1)."""
        node = MaxThresholdNode(id=1, name="test", n_experiments=5, threshold=3)
        col = "4"  # above threshold
        assert node.cpt.at["True", col] == 0
        assert node.cpt.at["False", col] == 1

    def test_zero_value_is_true(self):
        """k=0 is always <= threshold → True."""
        node = MaxThresholdNode(id=1, name="test", n_experiments=5, threshold=2)
        assert node.cpt.at["True", "0"] == 1

    def test_get_cpt_list_returns_list(self):
        node = MaxThresholdNode(id=1, name="test", n_experiments=4, threshold=2)
        result = node.get_cpt_list()
        assert isinstance(result, list)

    def test_get_cpt_states_returns_series(self):
        node = MaxThresholdNode(id=1, name="test", n_experiments=5, threshold=2)
        states = node.get_cpt_states()
        import pandas as pd
        assert isinstance(states, pd.Series)

    def test_cpt_columns_match_range(self):
        node = MaxThresholdNode(id=1, name="test", n_experiments=4, threshold=2)
        expected_cols = {"0", "1", "2", "3", "4"}
        assert set(node.cpt.columns) == expected_cols


class TestGoalNode:
    """Tests for GoalNode CPT — AND gate: True only when ALL children are True."""

    def test_basic_construction(self):
        node = GoalNode(id=0, name="test", n_children=2)
        assert node.id == 0
        assert node.name == "test"
        assert node.n_children == 2
        assert node.cpt is not None
        assert node.child == []
        assert node.parent == []

    def test_cpt_is_dataframe(self):
        node = GoalNode(id=0, name="test", n_children=3)
        assert isinstance(node.cpt, pd.DataFrame)

    def test_cpt_has_true_false_states(self):
        node = GoalNode(id=0, name="test", n_children=2)
        assert "True" in node.cpt.index
        assert "False" in node.cpt.index

    def test_all_children_true_yields_true(self):
        """k=0 means all children True → [1, 0]."""
        node = GoalNode(id=0, name="test", n_children=3)
        assert node.cpt.at["True", "0"] == 1
        assert node.cpt.at["False", "0"] == 0

    def test_any_child_false_yields_false(self):
        """k > 0 means at least one child is False → [0, 1]."""
        node = GoalNode(id=0, name="test", n_children=2)
        assert node.cpt.at["True", "1"] == 0
        assert node.cpt.at["False", "1"] == 1
        assert node.cpt.at["True", "2"] == 0
        assert node.cpt.at["False", "2"] == 1

    def test_cpt_rows_equal_2_to_n_children(self):
        """CPT should have 2^n_children columns."""
        node = GoalNode(id=0, name="test", n_children=3)
        # columns: "0" through "7" (8 columns) + States
        assert len(node.cpt.columns) == 2 ** 3  # 8 data columns + States handled

    def test_single_child(self):
        node = GoalNode(id=0, name="test", n_children=1)
        assert "0" in node.cpt.columns
        assert "1" in node.cpt.columns
        assert node.cpt.at["True", "0"] == 1
        assert node.cpt.at["False", "1"] == 1

    def test_get_cpt_list_returns_list(self):
        node = GoalNode(id=0, name="test", n_children=2)
        result = node.get_cpt_list()
        assert isinstance(result, list)

    def test_get_cpt_states_returns_series(self):
        node = GoalNode(id=0, name="test", n_children=2)
        states = node.get_cpt_states()
        import pandas as pd
        assert isinstance(states, pd.Series)


class TestDOE:
    """Tests for the DOE container class."""

    def test_doe_construction(self):
        exp = Experiment()
        doe = DOE(n_experiments=5, experiment=exp)
        assert doe.n_experiments == 5
        assert doe.experiment is exp

    def test_doe_thresholds_list(self):
        exp = Experiment()
        doe = DOE(n_experiments=5, experiment=exp)
        assert doe.thresholds == list(range(5))


class TestExperiment:
    """Tests for the Experiment class."""

    def test_experiment_creation(self):
        exp = Experiment()
        assert exp is not None
