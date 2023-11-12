from bbn import BBN
from doe import GoalNode, MaxThresholdNode, MinThresholdNode, SuccessNode
from logger import logger

n_experiments = 5
p_correct_navigation = 0.9
p_no_collision = 0.1
p_correct_pose = 0.9


def sample_mission_bbn(
    n_experiments,
    p_correct_navigation,
    p_no_collision,
    p_correct_pose,
    nav_threshold=0,
    collision_threshold=0,
    pose_threshold=0,
):
    bbn = BBN(n_experiments=n_experiments)
    mission_success = bbn.create_bbn_node(
        GoalNode(0, "Meeting requirements", n_children=3)
    )
    mission_all_waypoints = bbn.create_bbn_node(
        node_type=MinThresholdNode(
            id=1,
            name=f"Robot Nav Terrain under Threshold",
            n_experiments=bbn.n_experiments,
            threshold=nav_threshold,
        )
    )
    mission_times_navigable_terrain = bbn.create_bbn_node(
        node_type=SuccessNode(
            2,
            "P(robot on navigable terrain)",
            n_experiments=bbn.n_experiments,
            probability_of_success=p_correct_navigation,
        ),
    )
    mission_no_collision = bbn.create_bbn_node(
        MaxThresholdNode(
            3,
            f"Robot Collision under Threshold",
            n_experiments=bbn.n_experiments,
            threshold=collision_threshold,
        )
    )
    mission_times_collision = bbn.create_bbn_node(
        SuccessNode(
            4,
            "P(robot not collide)",
            n_experiments=bbn.n_experiments,
            probability_of_success=p_no_collision,
        )
    )
    mission_pose_in_threshold = bbn.create_bbn_node(
        MinThresholdNode(
            5,
            f"Robot Pose under Threshold",
            n_experiments=bbn.n_experiments,
            threshold=pose_threshold,
        )
    )
    mission_times_pose_within_threshold = bbn.create_bbn_node(
        SuccessNode(
            6,
            "P(robot pose within region)",
            n_experiments=bbn.n_experiments,
            probability_of_success=p_correct_pose,
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
    return bbn
