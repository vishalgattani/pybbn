from fastapi import APIRouter

from api.models import BBNParams, BBNResult, NodeProbability
from pybbn_assurance.cases.mission import sample_mission_bbn

router = APIRouter()


@router.post("/bbn/compute", response_model=BBNResult)
def compute_bbn(params: BBNParams):
    # Build and compute the BBN
    bbn = sample_mission_bbn(
        n_experiments=5,
        p_correct_navigation=params.p_correct_navigation,
        p_no_collision=params.p_no_collision,
        p_correct_pose=params.p_correct_pose,
        nav_threshold=params.nav_threshold,
        collision_threshold=params.collision_threshold,
        pose_threshold=params.pose_threshold,
    )

    # Collect node probabilities
    nodes = []
    for node_id, node_name in bbn.bbn.get_i2n().items():
        df = bbn.get_probabilities_node(node_id)
        if df is not None and len(df) >= 2:
            nodes.append(
                NodeProbability(
                    name=node_name,
                    p_true=round(float(df.p.iloc[0]), 4),
                    p_false=round(float(df.p.iloc[1]), 4),
                )
            )

    # Get the assurance case SVG
    svg = bbn.get_assurance_case_svg()

    return BBNResult(nodes=nodes, assurance_case_svg=svg)
