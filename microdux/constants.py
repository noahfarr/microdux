import re
from pathlib import Path

XMLS = Path(__file__).parent / "xmls"

WALK = XMLS / "robot_walk.xml"
ALL_COLLISIONS = XMLS / "robot_allcollisions.xml"
ROLLERS = XMLS / "robot_allcollisions_rollers.xml"
WALK_BACKLASH = XMLS / "robot_walk_backlash.xml"
BACKLASH = XMLS / "robot_allcollisions_backlash.xml"
ROLLERS_BACKLASH = XMLS / "robot_allcollisions_rollers_backlash.xml"
BALL = XMLS / "ball.xml"

VARIANTS = {
    "walk": WALK,
    "allcollisions": ALL_COLLISIONS,
    "rollers": ROLLERS,
    "walk_backlash": WALK_BACKLASH,
    "backlash": BACKLASH,
    "rollers_backlash": ROLLERS_BACKLASH,
}

HOME = {
    r".*hip_yaw.*": 0.0,
    r".*left_hip_roll.*": -0.0873,
    r".*right_hip_roll.*": 0.0873,
    r".*left_hip_pitch.*": -0.4579,
    r".*right_hip_pitch.*": 0.4579,
    r".*left_knee.*": -0.0049,
    r".*right_knee.*": 0.0049,
    r".*left_ankle.*": 0.4530,
    r".*right_ankle.*": -0.4530,
    r".*neck_pitch.*": 0.3491,
    r".*head_pitch.*": 0.3491,
    r".*head_yaw.*": 0.0,
    r".*head_roll.*": 0.0,
}

SIT = {
    r".*hip_yaw.*": 0.0,
    r".*left_hip_roll.*": 0.0,
    r".*right_hip_roll.*": 0.0,
    r".*left_hip_pitch.*": -0.4079,
    r".*right_hip_pitch.*": 0.4079,
    r".*left_knee.*": 1.35,
    r".*right_knee.*": -1.35,
    r".*left_ankle.*": 0.0,
    r".*right_ankle.*": 0.0,
    r".*neck_pitch.*": 0.3491,
    r".*head_pitch.*": 0.3491,
    r".*head_yaw.*": 0.0,
    r".*head_roll.*": 0.0,
}

SIT_Z = 0.060

SERVO = r"^(?!passive_).*"

FEET_SITES = ("left_foot", "right_foot")
IMU_SITE = "imu"
TRUNK = "trunk_base"
ROOT_JOINT = "trunk_base_freejoint"
FOOT_GEOMS = ("left_foot_collision", "right_foot_collision")

ROLLER_HEIGHT = (0.1335, 0.1435)

HEAD_JOINTS = ("neck_pitch", "head_pitch", "head_yaw", "head_roll")
HEAD_COMMAND = 4
BODY_COMMAND = 6
HIP_JOINTS = ("left_hip_roll", "right_hip_roll", "left_hip_yaw", "right_hip_yaw")
KNEE_JOINTS = ("left_knee", "right_knee")

BALL_FREE_JOINT = "ball_free"
MOUTH_SITE = "mouth_tip"
JAW_BODY = "jaw_soft"
NECK_BODY = "neck"

NOMINAL_HEIGHT = 0.095


def matches(pattern: str, name: str) -> bool:
    return re.fullmatch(pattern, name) is not None


def resolve(pose: dict[str, float], names) -> list[float]:
    values = []
    for name in names:
        hit = [v for k, v in pose.items() if matches(k, name)]
        if not hit:
            raise KeyError(f"no pose entry matches joint {name!r}")
        values.append(hit[-1])
    return values
