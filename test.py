import sys

import numpy as np
import pandas as pd

from bbn import BBN
from doe import GoalNode, MaxThresholdNode, MinThresholdNode, SuccessNode
from helper import get_binomial_prob, get_cdf_binomial_prob
from logger import logger

bbn = BBN()

mission_success = bbn.create_bbn_node(0, "Meeting requirements", GoalNode(n_children=3))
mission_all_waypoints = bbn.create_bbn_node(
    id=1,
    name=f"Robot reached all waypoints by traversing atleast {MinThresholdNode(n_experiments=3,threshold=1).get_threshold()} times on navigable terrain",
    node_type=MinThresholdNode(n_experiments=3, threshold=1),
)
mission_times_navigable_terrain = bbn.create_bbn_node(
    2,
    "Number of times navigable terrain traversed over unwanted terrain",
    node_type=SuccessNode(n_experiments=3, probability_of_success=0),
)

mission_no_collision = bbn.create_bbn_node(
    3,
    f"Robot collided at max {MaxThresholdNode(n_experiments=3,threshold=0).get_threshold()}",
    MaxThresholdNode(n_experiments=3, threshold=0),
)

mission_times_collision = bbn.create_bbn_node(
    4,
    "Number of times robot not collide",
    SuccessNode(n_experiments=3, probability_of_success=1),
)

mission_pose_in_threshold = bbn.create_bbn_node(
    5,
    f"Robot pose within threshold atleast {MinThresholdNode(n_experiments=3,threshold=0).get_threshold()}",
    MinThresholdNode(n_experiments=3, threshold=0),
)

mission_times_pose_within_threshold = bbn.create_bbn_node(
    6,
    "Number of times robot pose within threshold",
    SuccessNode(n_experiments=3, probability_of_success=1),
)


bbn.create_edge(mission_times_navigable_terrain, mission_all_waypoints)
bbn.create_edge(mission_all_waypoints, mission_success)
bbn.create_edge(mission_times_collision, mission_no_collision)
bbn.create_edge(mission_no_collision, mission_success)
bbn.create_edge(mission_times_pose_within_threshold, mission_pose_in_threshold)
bbn.create_edge(mission_pose_in_threshold, mission_success)

bbn.set_join_tree()
bbn.get_leaf_nodes()
