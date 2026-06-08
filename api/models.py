from pydantic import BaseModel, Field


class BBNParams(BaseModel):
    p_correct_navigation: float = Field(default=0.9, ge=0.0, le=1.0)
    p_no_collision: float = Field(default=0.1, ge=0.0, le=1.0)
    p_correct_pose: float = Field(default=0.9, ge=0.0, le=1.0)
    nav_threshold: int = Field(default=0, ge=0)
    collision_threshold: int = Field(default=0, ge=0)
    pose_threshold: int = Field(default=0, ge=0)


class NodeProbability(BaseModel):
    name: str
    p_true: float
    p_false: float


class BBNResult(BaseModel):
    nodes: list[NodeProbability]
    assurance_case_svg: str
