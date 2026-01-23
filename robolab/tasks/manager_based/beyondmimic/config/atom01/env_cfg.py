
import os

from robolab.assets.robots import ATOM01_CFG
from robolab.tasks.manager_based.beyondmimic.tracking_env_cfg import BeyondMimicEnvCfg

from isaaclab.utils import configclass


@configclass
class Atom01BeyondMimicEnvCfg(BeyondMimicEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = ATOM01_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.commands.motion.motion_file = f"{os.path.dirname(__file__)}/motion/yundong1.npz"
        # self.commands.motion.motion_file = f"{os.path.dirname(__file__)}/motion/G1_gangnam_style_V01.bvh_60hz.npz"
        self.commands.motion.anchor_body_name = [
            "torso_link",

        ]
        self.commands.motion.body_names = [
            'torso_link', 
            'left_thigh_roll_link', 
            'right_thigh_roll_link', 
            'left_arm_roll_link', 
            'right_arm_roll_link', 
            'left_knee_link', 
            'right_knee_link', 
            'left_elbow_pitch_link', 
            'right_elbow_pitch_link', 
            'left_ankle_roll_link', 
            'right_ankle_roll_link', 
        ]

        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None

        self.episode_length_s = 20.0
