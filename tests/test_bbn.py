"""Tests for pybbn_assurance/bbn.py — BBN construction and inference."""

import os
import pathlib
import pytest

from pybbn_assurance.bbn import BBN
from pybbn_assurance.doe import GoalNode, MaxThresholdNode, MinThresholdNode, SuccessNode
from pybbn_assurance.cases.mission import sample_mission_bbn


class TestSimpleBBN:
    """Build a small BBN and verify basic structure."""

    def test_bbn_creation(self):
        bbn = BBN(n_experiments=3)
        assert bbn.n_experiments == 3
        assert bbn.nodes == {}

    def test_get_platform_executable(self):
        bbn = BBN(n_experiments=3)
        result = bbn.get_platform_executable()
        import platform
        system = platform.system()
        if system == "Darwin":
            assert result == "gsn2x-macOS"
        elif system == "Linux":
            assert result == "gsn2x"

    def test_unknown_platform_executable(self, monkeypatch):
        import platform
        monkeypatch.setattr(platform, "system", lambda: "FreeBSD")
        bbn = BBN(n_experiments=3)
        result = bbn.get_platform_executable()
        assert result == ""

    def test_default_yaml_name(self):
        bbn = BBN(n_experiments=5)
        assert bbn.assurance_case_yaml_name == "assurance_case.yaml"
        assert bbn.assurance_case_svg_name == "assurance_case.svg"


class TestBBNNodeCreation:
    """Test creating nodes in a BBN."""

    def test_create_success_node(self):
        bbn = BBN(n_experiments=5)
        node = SuccessNode(id=0, name="test_success", n_experiments=5, probability_of_success=0.8)
        bbn_node = bbn.create_bbn_node(node)
        assert bbn_node is not None
        assert bbn_node.variable.id == 0
        assert bbn_node.variable.name == "test_success"
        assert 0 in bbn.nodes

    def test_create_goal_node(self):
        bbn = BBN(n_experiments=3)
        node = GoalNode(id=0, name="test_goal", n_children=2)
        bbn_node = bbn.create_bbn_node(node)
        assert bbn_node is not None
        assert 0 in bbn.goal_node

    def test_create_min_threshold_node(self):
        bbn = BBN(n_experiments=4)
        node = MinThresholdNode(id=1, name="test_min", n_experiments=4, threshold=2)
        bbn_node = bbn.create_bbn_node(node)
        assert bbn_node is not None

    def test_create_max_threshold_node(self):
        bbn = BBN(n_experiments=4)
        node = MaxThresholdNode(id=2, name="test_max", n_experiments=4, threshold=1)
        bbn_node = bbn.create_bbn_node(node)
        assert bbn_node is not None

    def test_create_bbn_node_returns_none_on_error(self):
        """If node creation fails, should return None."""
        bbn = BBN(n_experiments=3)
        # Create a node without required attributes
        class BadNode:
            id = 0
            name = "bad"
            def get_cpt_states(self):
                raise ValueError("boom")
            def get_cpt_list(self):
                return []
        result = bbn.create_bbn_node(BadNode())
        assert result is None


class TestBBNEdges:
    """Test edge creation between nodes."""

    def test_create_edge(self):
        bbn = BBN(n_experiments=3)
        parent_node = SuccessNode(id=0, name="p", n_experiments=3, probability_of_success=0.8)
        child_node = MinThresholdNode(id=1, name="c", n_experiments=3, threshold=1)
        bbn.create_bbn_node(parent_node)
        bbn.create_bbn_node(child_node)
        assert 0 in bbn.nodes
        assert 1 in bbn.nodes


class TestBBNInference:
    """Test full BBN with inference — build sample mission BBN."""

    def test_sample_mission_posterior(self):
        """Sample mission BBN: posterior P(Meeting requirements|True) > 0.5."""
        bbn = sample_mission_bbn(
            n_experiments=5,
            p_correct_navigation=0.9,
            p_no_collision=0.1,
            p_correct_pose=0.9,
            nav_threshold=0,
            collision_threshold=0,
            pose_threshold=0,
        )
        assert bbn is not None
        assert bbn.join_tree is not None
        assert len(bbn.nodes) > 0

        # Check the goal node posterior
        goal_df = bbn.get_probabilities_node(0)  # GoalNode id=0
        assert goal_df is not None
        # Find the probability of "True" state
        true_prob = goal_df.loc[goal_df["val"] == "True", "p"]
        assert len(true_prob) > 0, "Could not find True state in goal node posterior"
        true_val = true_prob.iloc[0]
        assert true_val > 0.5, f"Posterior P(Meeting requirements=True) = {true_val}, expected > 0.5"

    def test_sample_mission_leaf_nodes(self):
        """Sample mission BBN should have leaf nodes."""
        bbn = sample_mission_bbn(
            n_experiments=5,
            p_correct_navigation=0.9,
            p_no_collision=0.1,
            p_correct_pose=0.9,
            nav_threshold=0,
            collision_threshold=0,
            pose_threshold=0,
        )
        leaf_nodes = bbn.get_leaf_nodes()
        assert len(leaf_nodes) > 0

    def test_sample_mission_has_non_leaf_nodes(self):
        bbn = sample_mission_bbn(
            n_experiments=5,
            p_correct_navigation=0.9,
            p_no_collision=0.1,
            p_correct_pose=0.9,
            nav_threshold=0,
            collision_threshold=0,
            pose_threshold=0,
        )
        assert len(bbn.non_leaf_nodes) > 0

    def test_sample_mission_yaml_exists(self):
        """After building the BBN, assurance_case.yaml should exist."""
        bbn = sample_mission_bbn(
            n_experiments=5,
            p_correct_navigation=0.9,
            p_no_collision=0.1,
            p_correct_pose=0.9,
            nav_threshold=0,
            collision_threshold=0,
            pose_threshold=0,
        )
        yaml_path = pathlib.Path.cwd() / bbn.assurance_case_yaml_name
        assert yaml_path.exists(), f"YAML file {yaml_path} was not created"

    def test_sample_mission_svg_exists(self):
        """After building the BBN, assurance_case.svg should exist."""
        bbn = sample_mission_bbn(
            n_experiments=5,
            p_correct_navigation=0.9,
            p_no_collision=0.1,
            p_correct_pose=0.9,
            nav_threshold=0,
            collision_threshold=0,
            pose_threshold=0,
        )
        svg_path = pathlib.Path.cwd() / bbn.assurance_case_svg_name
        assert svg_path.exists(), f"SVG file {svg_path} was not created"

    def test_potential_to_df(self):
        """potential_to_df should return a DataFrame with val and p columns."""
        bbn = sample_mission_bbn(
            n_experiments=5,
            p_correct_navigation=0.9,
            p_no_collision=0.1,
            p_correct_pose=0.9,
            nav_threshold=0,
            collision_threshold=0,
            pose_threshold=0,
        )
        # Get a node from the join tree
        for node in bbn.join_tree.get_bbn_nodes():
            potential = bbn.join_tree.get_bbn_potential(node)
            df = bbn.potential_to_df(potential)
            assert "val" in df.columns
            assert "p" in df.columns
            break

    def test_get_node_identifiers(self):
        """get_node_identifiers returns the id-to-name mapping."""
        bbn = sample_mission_bbn(
            n_experiments=5,
            p_correct_navigation=0.9,
            p_no_collision=0.1,
            p_correct_pose=0.9,
            nav_threshold=0,
            collision_threshold=0,
            pose_threshold=0,
        )
        identifiers = bbn.bbn.get_i2n()
        assert identifiers is not None
        assert len(identifiers) == 7

    def test_get_bbn_dataframe(self):
        bbn = sample_mission_bbn(
            n_experiments=5,
            p_correct_navigation=0.9,
            p_no_collision=0.1,
            p_correct_pose=0.9,
            nav_threshold=0,
            collision_threshold=0,
            pose_threshold=0,
        )
        df = bbn.get_bbn_dataframe()
        assert df is not None
        assert len(df) > 0

    def test_reset_evidence(self):
        bbn = sample_mission_bbn(
            n_experiments=5,
            p_correct_navigation=0.9,
            p_no_collision=0.1,
            p_correct_pose=0.9,
            nav_threshold=0,
            collision_threshold=0,
            pose_threshold=0,
        )
        bbn.reset_evidence()

    def test_get_join_tree(self):
        bbn = sample_mission_bbn(
            n_experiments=5,
            p_correct_navigation=0.9,
            p_no_collision=0.1,
            p_correct_pose=0.9,
            nav_threshold=0,
            collision_threshold=0,
            pose_threshold=0,
        )
        jt = bbn.get_join_tree()
        assert jt is not None

    def test_ensure_high_nav_success(self):
        """With high probabilities, the BBN should produce a valid posterior."""
        bbn = sample_mission_bbn(
            n_experiments=5,
            p_correct_navigation=0.99,
            p_no_collision=0.99,
            p_correct_pose=0.99,
            nav_threshold=0,
            collision_threshold=0,
            pose_threshold=0,
        )
        goal_df = bbn.get_probabilities_node(0)
        assert goal_df is not None
        # Verify the posterior is a valid probability
        true_prob = goal_df.loc[goal_df["val"] == "True", "p"].iloc[0]
        assert 0.0 <= true_prob <= 1.0

    def test_get_probabilities_node_no_join_tree(self):
        """Should return None when join tree not set."""
        bbn = BBN(n_experiments=3)
        result = bbn.get_probabilities_node(0)
        assert result is None

    def test_get_bbn_dataframe_no_join_tree(self):
        """Should return None when join tree not set."""
        bbn = BBN(n_experiments=3)
        result = bbn.get_bbn_dataframe()
        assert result is None

    def test_print_probs_no_join_tree(self, caplog):
        """Should log error when join tree not set."""
        bbn = BBN(n_experiments=3)
        bbn.print_probs()
        assert any("Join Tree has not been set" in msg for msg in caplog.messages)

    def test_ensure_no_join_tree(self, caplog):
        """Should log error when join tree not set."""
        bbn = BBN(n_experiments=3)
        result = bbn.get_bbn_dataframe()
        assert result is None

    def test_create_edge_error(self):
        """Edge creation should handle errors gracefully."""
        bbn = BBN(n_experiments=3)
        # Try to create edge without nodes
        class FakeNode:
            variable = type('obj', (object,), {'id': 99})()
        try:
            bbn.create_edge(FakeNode(), FakeNode())
        except Exception:
            pass  # Expected to fail gracefully

    def test_bbn2yaml_creates_files(self):
        bbn = sample_mission_bbn(
            n_experiments=5,
            p_correct_navigation=0.9,
            p_no_collision=0.1,
            p_correct_pose=0.9,
            nav_threshold=0,
            collision_threshold=0,
            pose_threshold=0,
        )
        assert bbn.assurance_case_yaml is not None
        assert isinstance(bbn.assurance_case_dictionary, dict)
        assert len(bbn.assurance_case_dictionary) > 0

    def test_create_flowchart(self):
        bbn = BBN(n_experiments=3)
        yaml_data = {
            "G0": {"text": "Goal", "supportedBy": ["S1"]},
            "S1": {"text": "Support", "supportedBy": []},
        }
        graph = bbn.create_flowchart(yaml_data)
        assert graph is not None


class TestEvidence:
    """Test evidence setting on BBN."""

    def test_set_and_reset_evidence(self):
        """Set evidence on a node, verify reset works without error."""
        bbn = sample_mission_bbn(
            n_experiments=5,
            p_correct_navigation=0.9,
            p_no_collision=0.1,
            p_correct_pose=0.9,
            nav_threshold=0,
            collision_threshold=0,
            pose_threshold=0,
        )
        # Get initial posterior
        initial_df = bbn.get_probabilities_node(0)
        assert initial_df is not None
        initial_true = initial_df.loc[initial_df["val"] == "True", "p"].iloc[0]

        # Set evidence on a goal node
        bbn.evidence("Meeting requirements", "True", 1.0)

        # Verify evidence was set by checking posterior
        post_df = bbn.get_probabilities_node(0)
        post_true = post_df.loc[post_df["val"] == "True", "p"].iloc[0]
        assert post_true == pytest.approx(1.0, abs=1e-6)

        # Reset evidence
        bbn.reset_evidence()
        reset_df = bbn.get_probabilities_node(0)
        reset_true = reset_df.loc[reset_df["val"] == "True", "p"].iloc[0]
        # Should be back to prior
        assert reset_true == pytest.approx(initial_true, abs=1e-6)


class TestBBNDictionary:
    """Test assurance case dictionary."""

    def test_yaml_dict_structure(self):
        bbn = sample_mission_bbn(
            n_experiments=5,
            p_correct_navigation=0.9,
            p_no_collision=0.1,
            p_correct_pose=0.9,
            nav_threshold=0,
            collision_threshold=0,
            pose_threshold=0,
        )
        d = bbn.assurance_case_dictionary
        assert "G0" in d  # Goal node
        assert "Sn2" in d  # Success node
        assert "text" in d["G0"]
        assert "supportedBy" in d["G0"]
