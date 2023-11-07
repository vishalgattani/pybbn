import numpy as np
import pandas as pd

from bbn import BBN
from helper import get_binomial_prob, get_cdf_binomial_prob
from logger import logger

n_runs = 5
threshold_num_collision = 1
p_nav = 0.9
threshold_num_nav = 3
prob_pose_within_threshold = 0.9
threshold_num_pose_within_threshold = 2

p_collision = 0.1
p_no_collision = 1 - p_collision
prob_no_collision_lst = get_binomial_prob(n_runs, 1 - p_no_collision)
prob_no_collision_cpt = pd.DataFrame({"success": prob_no_collision_lst})
idxlist = prob_no_collision_cpt.index.tolist()
prob_no_collision_cpt = prob_no_collision_cpt.set_index(
    [pd.Index(["n" + str(idx) for idx in idxlist])]
)

mission_no_collision_keys = []
mission_no_collision_values = []

for i in range(n_runs + 1):
    mission_no_collision_keys.append(str(i))
    # if threshold is small, then more collisions are not allowed, higher threshold means we are okay if the robot collides
    if i > threshold_num_collision:
        # false state
        mission_no_collision_values.append([0, 1])
    else:
        # true state
        mission_no_collision_values.append([1, 0])

prob_collision_lst = dict(zip(mission_no_collision_keys, mission_no_collision_values))
prob_collision = pd.DataFrame(prob_collision_lst)
prob_collision["States"] = ["True", "False"]
prob_collision.set_index("States", inplace=True)


prob_navterrain_better_lst = get_binomial_prob(n_runs, p_nav)
navterrain_better = pd.DataFrame({"success": prob_navterrain_better_lst})
idxlist = navterrain_better.index.tolist()
prob_navterrain_better = navterrain_better.set_index(
    [pd.Index(["n" + str(idx) for idx in idxlist])]
)

# display(prob_navterrain_better)

nav_keys = []
nav_vals = []
for i in range(n_runs + 1):
    nav_keys.append(str(i))
    # if threshold is small, then more collisions are allowed, higher threshold means we would like the robot to not traverse over bad terrain
    if i >= threshold_num_nav:
        nav_vals.append([1, 0])
    else:
        nav_vals.append([0, 1])

nav_dict = dict(zip(nav_keys, nav_vals))
all_waypoints_cpt = pd.DataFrame(nav_dict)
all_waypoints_cpt["States"] = ["True", "False"]
all_waypoints_cpt.set_index("States", inplace=True)
# display(all_waypoints_cpt)


prob_pose_within_threshold_lst = get_binomial_prob(n_runs, prob_pose_within_threshold)
prob_pose_within_threshold_cpt = pd.DataFrame(
    {"success": prob_pose_within_threshold_lst}
)
idxlist = prob_pose_within_threshold_cpt.index.tolist()
prob_pose_within_threshold_cpt = prob_pose_within_threshold_cpt.set_index(
    [pd.Index(["n" + str(idx) for idx in idxlist])]
)
# display(prob_pose_within_threshold_cpt)

mission_pose_within_threshold_keys = []
mission_pose_within_threshold_values = []
for i in range(n_runs + 1):
    mission_pose_within_threshold_keys.append(str(i))
    if i >= threshold_num_pose_within_threshold:
        mission_pose_within_threshold_values.append([1, 0])
    else:
        mission_pose_within_threshold_values.append([0, 1])

prob_pose_lst = dict(
    zip(mission_pose_within_threshold_keys, mission_pose_within_threshold_values)
)
prob_pose = pd.DataFrame(prob_pose_lst)
prob_pose["States"] = ["True", "False"]
prob_pose.set_index("States", inplace=True)


bbn = BBN()
mission_success = bbn.create_bbn_node(
    0,
    "Meeting requirements",
    ["True", "False"],
    [
        1,
        0,  # ttt
        0,
        1,  # ttf
        0,
        1,  # tft
        0,
        1,  # ftt
        0,
        1,  # tff
        0,
        1,  # ftf
        0,
        1,  # fft
        0,
        1,
    ],
)  # fff
mission_all_waypoints = bbn.create_bbn_node(
    1,
    "Robot reached all waypoints by traversing atleast "
    + str(threshold_num_nav)
    + " times on navigable terrain",
    all_waypoints_cpt.index.values.tolist(),
    np.ndarray.flatten(all_waypoints_cpt.transpose().values).tolist(),
)


mission_times_navigable_terrain = bbn.create_bbn_node(
    2,
    "Number of times navigable terrain traversed over unwanted terrain",
    [str(i) for i in range(n_runs + 1)],
    prob_navterrain_better_lst,
)


mission_no_collision = bbn.create_bbn_node(
    3,
    "Robot collided at max " + str(threshold_num_collision),
    prob_collision.index.values.tolist(),
    np.ndarray.flatten(prob_collision.transpose().values).tolist(),
)


mission_times_collision = bbn.create_bbn_node(
    4,
    "Number of times robot collided",
    [str(i) for i in range(n_runs + 1)],
    prob_no_collision_lst,
)

mission_pose_in_threshold = bbn.create_bbn_node(
    5,
    "Robot pose within threshold atleast " + str(threshold_num_pose_within_threshold),
    prob_pose.index.values.tolist(),
    np.ndarray.flatten(prob_pose.transpose().values).tolist(),
)

mission_times_pose_within_threshold = bbn.create_bbn_node(
    6,
    "Number of times robot pose within threshold",
    [str(i) for i in range(n_runs + 1)],
    prob_pose_within_threshold_lst,
)

bbn.create_edge(mission_times_navigable_terrain, mission_all_waypoints)
bbn.create_edge(mission_all_waypoints, mission_success)
bbn.create_edge(mission_times_collision, mission_no_collision)
bbn.create_edge(mission_no_collision, mission_success)
bbn.create_edge(mission_times_pose_within_threshold, mission_pose_in_threshold)
bbn.create_edge(mission_pose_in_threshold, mission_success)

bbn.set_join_tree()
bbn.get_leaf_nodes()

mission_df = bbn.print_probs_node(0)
mission_df.p = mission_df.p.round(4)

waypoint_with_terrain_df = bbn.print_probs_node(1)
waypoint_with_terrain_df.p = waypoint_with_terrain_df.p.round(4)

nwaypoint_with_terrain_df = bbn.print_probs_node(2)
nwaypoint_with_terrain_df.p = waypoint_with_terrain_df.p.round(4)

collision_df = bbn.print_probs_node(3)
collision_df.p = collision_df.p.round(4)

ncollision_df = bbn.print_probs_node(4)
ncollision_df.p = ncollision_df.p.round(4)

pose_df = bbn.print_probs_node(5)
pose_df.p = pose_df.p.round(4)

npose_df = bbn.print_probs_node(6)
npose_df.p = pose_df.p.round(4)

simulation_data = []

data = [
    n_runs,
    p_no_collision,
    prob_pose_within_threshold,
    p_nav,
    threshold_num_collision,
    threshold_num_nav,
    threshold_num_pose_within_threshold,
    mission_df.p[0],
    mission_df.p[1],
    waypoint_with_terrain_df.p[0],
    waypoint_with_terrain_df.p[1],
    collision_df.p[0],
    collision_df.p[1],
    pose_df.p[0],
    pose_df.p[1],
]
simulation_data.append(data)


df = pd.DataFrame(
    simulation_data,
    columns=[
        "n_runs",
        "prob_no_collision",
        "prob_pose_within_threshold",
        "p_nav",
        "threshold_num_collision",
        "threshold_num_nav",
        "threshold_num_pose_within_threshold",
        "mission_true",
        "mission_false",
        "waypoint_true",
        "waypoint_false",
        "collision_true",
        "collision_false",
        "pose_true",
        "pose_false",
    ],
)

logger.debug(
    f"I want atleast {threshold_num_nav}/{n_runs} times to be on navigable terrain!"
)
logger.debug(
    f"I want atleast {threshold_num_pose_within_threshold}/{n_runs} times to be within accepted limits of goal region!"
)
logger.debug(
    f"I am okay with {threshold_num_collision}/{n_runs} instances of collision incidents!"
)

logger.debug(f"{df.transpose()}")
