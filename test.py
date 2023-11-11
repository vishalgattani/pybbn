import subprocess
import sys

import numpy as np
import pandas as pd

from bbn import BBN
from doe import GoalNode, MaxThresholdNode, MinThresholdNode, SuccessNode
from gui import App
from logger import logger

n_experiments = 5

bbn = BBN(n_experiments=n_experiments)

mission_success = bbn.create_bbn_node(GoalNode(0, "Meeting requirements", n_children=3))
mission_all_waypoints = bbn.create_bbn_node(
    node_type=MinThresholdNode(
        id=1,
        name=f"Robot Nav Terrain under Threshold",
        n_experiments=bbn.n_experiments,
        threshold=0,
    )
)
mission_times_navigable_terrain = bbn.create_bbn_node(
    node_type=SuccessNode(
        2,
        "Number of times navigable terrain traversed over wanted terrain",
        n_experiments=bbn.n_experiments,
        probability_of_success=0.9,
    ),
)

mission_no_collision = bbn.create_bbn_node(
    MaxThresholdNode(
        3,
        f"Robot Collision under Threshold",
        n_experiments=bbn.n_experiments,
        threshold=0,
    )
)

mission_times_collision = bbn.create_bbn_node(
    SuccessNode(
        4,
        "Number of times robot not collide",
        n_experiments=bbn.n_experiments,
        probability_of_success=0.1,
    )
)

mission_pose_in_threshold = bbn.create_bbn_node(
    MinThresholdNode(
        5, f"Robot Pose under Threshold", n_experiments=bbn.n_experiments, threshold=0
    )
)

mission_times_pose_within_threshold = bbn.create_bbn_node(
    SuccessNode(
        6,
        "Number of times robot pose within region",
        n_experiments=bbn.n_experiments,
        probability_of_success=0.9,
    )
)

bbn.create_edge(mission_times_navigable_terrain, mission_all_waypoints)
bbn.create_edge(mission_all_waypoints, mission_success)
bbn.create_edge(mission_times_collision, mission_no_collision)
bbn.create_edge(mission_no_collision, mission_success)
bbn.create_edge(mission_times_pose_within_threshold, mission_pose_in_threshold)
bbn.create_edge(mission_pose_in_threshold, mission_success)
bbn.set_join_tree()
bbn.get_leaf_nodes()
# data = bbn.get_bbn_dataframe()

app = App(bbn=bbn)
app.mainloop()
