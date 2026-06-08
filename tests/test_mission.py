# Author: Vishal Gattani
# Created: 2024-06-07

from pybbn_assurance.cases.mission import (
    n_experiments,
    p_correct_navigation,
    p_correct_pose,
    p_no_collision,
    sample_mission_bbn,
)
from pybbn_assurance.logger import logger


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
    logger.info("Test passed: sample_mission_bbn creates a valid BBN")
