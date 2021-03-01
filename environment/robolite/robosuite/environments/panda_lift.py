from collections import OrderedDict
import numpy as np

from robosuite.utils.transform_utils import convert_quat
from robosuite.environments.panda import PandaEnv

from robosuite.models.arenas import TableArena
from robosuite.models.objects import FullyFrictionalBoxObject
from robosuite.models.robots import Panda
from robosuite.models.tasks import TableTopTask, UniformRandomSampler


class PandaLift(PandaEnv):
    dof = 8

    """
    This class corresponds to the lifting task for the Panda robot arm.
    """
    
    parameters_spec = {
        **PandaEnv.parameters_spec,
        'table_size_0': [0.7, 0.9],
        'table_size_1': [0.7, 0.9],
        'table_size_2': [0.7, 0.9],
        'table_friction_0': [0.4, 1.6],
        'table_friction_1': [0.0025, 0.0075],
        'table_friction_2': [0.00005, 0.00015],
        'boxobject_friction_0': [0.4, 1.6],
        # 'boxobject_friction_1': [0.0025, 0.0075],    # fixed this to zero
        'boxobject_friction_2': [0.00005, 0.00015],
        'boxobject_density_1000': [0.6, 1.4],
    }

    def reset_props(self,
                    table_size_0=0.8, table_size_1=0.8, table_size_2=0.8,
                    table_friction_0=1.0, table_friction_1=0.005, table_friction_2=0.0001,
                    boxobject_size_0=0.020, boxobject_size_1=0.020, boxobject_size_2=0.020,
                    boxobject_friction_0=1.0, boxobject_friction_1=0.0, boxobject_friction_2=0.0001,
                    boxobject_density_1000=0.1,
                    **kwargs):
        
        self.table_full_size = (table_size_0, table_size_1, table_size_2)
        self.table_friction = (table_friction_0, table_friction_1, table_friction_2)
        self.boxobject_size = (boxobject_size_0, boxobject_size_1, boxobject_size_2)
        self.boxobject_friction = (boxobject_friction_0, boxobject_friction_1, boxobject_friction_2)
        self.boxobject_density = boxobject_density_1000 * 1000.
        super().reset_props(**kwargs)

    def __init__(self,
                 use_object_obs=True,
                 reward_shaping=True,
                 placement_initializer=None,
                 object_obs_process=True,
                 **kwargs):
        """
        Args:

            use_object_obs (bool): if True, include object (cube) information in
                the observation.

            reward_shaping (bool): if True, use dense rewards.

            placement_initializer (ObjectPositionSampler instance): if provided, will
                be used to place objects on every reset, else a UniformRandomSampler
                is used by default.

            object_obs_process (bool): if True, process the object observation to get a task_state.
                Setting this to False is useful when some transformation (eg. noise) need to be done to object observation raw data prior to the processing.
        """
        
        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # reward configuration
        self.reward_shaping = reward_shaping

        # object placement initializer
        if placement_initializer:
            self.placement_initializer = placement_initializer
        else:
            self.placement_initializer = UniformRandomSampler(
                x_range=[-0.03, 0.03],
                y_range=[-0.03, 0.03],
                ensure_object_boundary_in_range=False,
                z_rotation=None,
            )

        # for first initialization
        self.table_full_size = (0.8, 0.8, 0.8)
        self.table_friction = (1.0, 0.005, 0.0001)
        self.boxobject_size = (0.02, 0.02, 0.02)
        self.boxobject_friction = (1.0, 0.005, 0.0001)
        self.boxobject_density = 100.

        self.object_obs_process = object_obs_process

        super().__init__(gripper_visualization=True, **kwargs)

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()
        self.mujoco_robot.set_base_xpos([0, 0, 0])

        # load model for table top workspace
        self.mujoco_arena = TableArena(
            table_full_size=self.table_full_size, table_friction=self.table_friction
        )
        if self.use_indicator_object:
            self.mujoco_arena.add_pos_indicator()

        # The panda robot has a pedestal, we want to align it with the table
        self.mujoco_arena.set_origin([0.16 + self.table_full_size[0] / 2, 0, 0])

        # initialize objects of interest
        # in original robosuite, a simple domain randomization is included in BoxObject implementation, and called here. We choose to discard that implementation.
        cube = FullyFrictionalBoxObject(
            size=self.boxobject_size,
            friction=self.boxobject_friction,
            density=self.boxobject_density,
            rgba=[1, 0, 0, 1],
        )
        self.mujoco_objects = OrderedDict([("cube", cube)])

        # task includes arena, robot, and objects of interest
        self.model = TableTopTask(
            self.mujoco_arena,
            self.mujoco_robot,
            self.mujoco_objects,
            initializer=self.placement_initializer,
        )
        self.model.place_objects()

    def _get_reference(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._get_reference()
        self.cube_body_id = self.sim.model.body_name2id("cube")
        self.l_finger_geom_ids = [
            self.sim.model.geom_name2id(x) for x in self.gripper.left_finger_geoms
        ]
        self.r_finger_geom_ids = [
            self.sim.model.geom_name2id(x) for x in self.gripper.right_finger_geoms
        ]
        self.cube_geom_id = self.sim.model.geom_name2id("cube")

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        # reset positions of objects
        self.model.place_objects()

        # reset joint positions
        init_pos = self.mujoco_robot.init_qpos
        init_pos += np.random.randn(init_pos.shape[0]) * 0.02
        self.sim.data.qpos[self._ref_joint_pos_indexes] = np.array(init_pos)

    def reward(self, action=None):
        """
        Reward function for the task.

        The dense reward has three components.

            Reaching: in [0, 1], to encourage the arm to reach the cube
            Grasping: in {0, 0.25}, non-zero if arm is grasping the cube
            Lifting: in {0, 1}, non-zero if arm has lifted the cube

        The sparse reward only consists of the lifting component.

        Args:
            action (np array): unused for this task

        Returns:
            reward (float): the reward
        """
        reward = 0.

        # sparse completion reward
        if self._check_success():
            reward = 1.0

        # use a shaping reward
        if self.reward_shaping:

            # reaching reward
            cube_pos = self.sim.data.body_xpos[self.cube_body_id]
            gripper_site_pos = self.sim.data.site_xpos[self.eef_site_id]
            dist = np.linalg.norm(gripper_site_pos - cube_pos)
            reaching_reward = 1 - np.tanh(10.0 * dist)
            reward += reaching_reward

            # grasping reward
            touch_left_finger = False
            touch_right_finger = False
            for i in range(self.sim.data.ncon):
                c = self.sim.data.contact[i]
                if c.geom1 in self.l_finger_geom_ids and c.geom2 == self.cube_geom_id:
                    touch_left_finger = True
                if c.geom1 == self.cube_geom_id and c.geom2 in self.l_finger_geom_ids:
                    touch_left_finger = True
                if c.geom1 in self.r_finger_geom_ids and c.geom2 == self.cube_geom_id:
                    touch_right_finger = True
                if c.geom1 == self.cube_geom_id and c.geom2 in self.r_finger_geom_ids:
                    touch_right_finger = True
            if touch_left_finger and touch_right_finger:
                reward += 0.25

        return reward

    def put_raw_object_obs(self, di):
        di['cube_pos'] = np.array(self.sim.data.body_xpos[self.cube_body_id])
        di['cube_quat'] = convert_quat(
            np.array(self.sim.data.body_xquat[self.cube_body_id]), to="xyzw"
        )
        di['gripper_site_pos'] = np.array(self.sim.data.site_xpos[self.eef_site_id])

    def process_object_obs(self, di):
        gripper_to_cube = di['gripper_site_pos'] - di['cube_pos']
        task_state = np.concatenate([di['cube_pos'], di['cube_quat'], gripper_to_cube])
        di['task_state'] = task_state

    def _get_observation(self):
        """
        Returns an OrderedDict containing observations [(name_string, np.array), ...].

        Important keys:
            robot-state: contains robot-centric information.
            object-state: requires @self.use_object_obs to be True.
                contains object-centric information.
            image: requires @self.use_camera_obs to be True.
                contains a rendered frame from the simulation.
            depth: requires @self.use_camera_obs and @self.camera_depth to be True.
                contains a rendered depth map from the simulation
        """
        di = super()._get_observation()
        # camera observations
        if self.use_camera_obs:
            camera_obs = self.sim.render(
                camera_name=self.camera_name,
                width=self.camera_width,
                height=self.camera_height,
                depth=self.camera_depth,
            )
            if self.camera_depth:
                di["image"], di["depth"] = camera_obs
            else:
                di["image"] = camera_obs

        # low-level object information
        if self.use_object_obs:
            self.put_raw_object_obs(di)
            if self.object_obs_process:
                self.process_object_obs(di)

        return di

    def _check_contact(self):
        """
        Returns True if gripper is in contact with an object.
        """
        collision = False
        for contact in self.sim.data.contact[: self.sim.data.ncon]:
            if (
                self.sim.model.geom_id2name(contact.geom1)
                in self.gripper.contact_geoms()
                or self.sim.model.geom_id2name(contact.geom2)
                in self.gripper.contact_geoms()
            ):
                collision = True
                break
        return collision

    def _check_success(self):
        """
        Returns True if task has been completed.
        """
        cube_height = self.sim.data.body_xpos[self.cube_body_id][2]
        table_height = self.table_full_size[2]

        # cube is higher than the table top above a margin
        return cube_height > table_height + 0.04

    def _gripper_visualization(self):
        """
        Do any needed visualization here. Overrides superclass implementations.
        """

        # color the gripper site appropriately based on distance to cube
        if self.gripper_visualization:
            # get distance to cube
            cube_site_id = self.sim.model.site_name2id("cube")
            dist = np.sum(
                np.square(
                    self.sim.data.site_xpos[cube_site_id]
                    - self.sim.data.get_site_xpos("grip_site")
                )
            )

            # set RGBA for the EEF site here
            max_dist = 0.1
            scaled = (1.0 - min(dist / max_dist, 1.)) ** 15
            rgba = np.zeros(4)
            rgba[0] = 1 - scaled
            rgba[1] = scaled
            rgba[3] = 0.5

            self.sim.model.site_rgba[self.eef_site_id] = rgba
