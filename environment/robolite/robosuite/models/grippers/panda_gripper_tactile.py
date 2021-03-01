"""
Gripper for Franka's Panda (has two fingers).
"""
import numpy as np
from robosuite.utils.mjcf_utils import xml_path_completion
from robosuite.models.grippers.panda_gripper import PandaGripper


class PandaGripperTactile(PandaGripper):
    """
    Gripper for Franka's Panda (has two fingers) with tactile sensors attached.
    """

    def __init__(self):
        super().__init__(xml_path_completion("grippers/panda_gripper_tactile.xml"))
