"""Tests for pybbn_assurance/cases/mission.py — sample mission BBN."""

import pathlib

from pybbn_assurance.cases.mission import (
    n_experiments,
    p_correct_navigation,
    p_correct_pose,
    p_no_collision,
    sample_mission_bbn,
)
from pybbn_assurance.doe import GoalNode, SuccessNode, MinThresholdNode, MaxThresholdNode


def test_sample_mission_bbn():
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
    assert len(bbn.nodes) > 0


def test_sample_mission_has_goal_node():
    bbn = sample_mission_bbn(
        n_experiments=5,
        p_correct_navigation=0.9,
        p_no_collision=0.1,
        p_correct_pose=0.9,
        nav_threshold=0,
        collision_threshold=0,
        pose_threshold=0,
    )
    # GoalNode is id=0
    assert 0 in bbn.goal_node
    assert isinstance(bbn.nodes[0], GoalNode)


def test_sample_mission_has_success_nodes():
    bbn = sample_mission_bbn(
        n_experiments=5,
        p_correct_navigation=0.9,
        p_no_collision=0.1,
        p_correct_pose=0.9,
        nav_threshold=0,
        collision_threshold=0,
        pose_threshold=0,
    )
    # SuccessNodes are ids 2, 4, 6
    for sid in [2, 4, 6]:
        assert sid in bbn.nodes
        assert isinstance(bbn.nodes[sid], SuccessNode)


def test_sample_mission_has_threshold_nodes():
    bbn = sample_mission_bbn(
        n_experiments=5,
        p_correct_navigation=0.9,
        p_no_collision=0.1,
        p_correct_pose=0.9,
        nav_threshold=0,
        collision_threshold=0,
        pose_threshold=0,
    )
    # MinThresholdNodes: 1, 5; MaxThresholdNode: 3
    for tid in [1, 5]:
        assert tid in bbn.nodes
        assert isinstance(bbn.nodes[tid], MinThresholdNode)
    assert 3 in bbn.nodes
    assert isinstance(bbn.nodes[3], MaxThresholdNode)


def test_sample_mission_default_params():
    """Default module-level params should be reasonable."""
    assert n_experiments == 5
    assert 0.0 <= p_correct_navigation <= 1.0
    assert 0.0 <= p_no_collision <= 1.0
    assert 0.0 <= p_correct_pose <= 1.0


def test_sample_mission_join_tree_is_set():
    bbn = sample_mission_bbn(
        n_experiments=5,
        p_correct_navigation=0.9,
        p_no_collision=0.1,
        p_correct_pose=0.9,
        nav_threshold=0,
        collision_threshold=0,
        pose_threshold=0,
    )
    assert bbn.join_tree is not None


def test_sample_mission_yaml_generated():
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
    assert yaml_path.exists()


def test_sample_mission_png_generated():
    bbn = sample_mission_bbn(
        n_experiments=5,
        p_correct_navigation=0.9,
        p_no_collision=0.1,
        p_correct_pose=0.9,
        nav_threshold=0,
        collision_threshold=0,
        pose_threshold=0,
    )
    png_path = pathlib.Path.cwd() / f"{bbn.assurance_case_name}.png"
    assert png_path.exists()


def test_sample_mission_node_count():
    """Sample mission should have exactly 7 nodes (1 goal + 3 threshold + 3 success)."""
    bbn = sample_mission_bbn(
        n_experiments=5,
        p_correct_navigation=0.9,
        p_no_collision=0.1,
        p_correct_pose=0.9,
        nav_threshold=0,
        collision_threshold=0,
        pose_threshold=0,
    )
    assert len(bbn.nodes) == 7


def test_sample_mission_dataframe():
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
    assert "True" in df.columns
    assert "False" in df.columns
