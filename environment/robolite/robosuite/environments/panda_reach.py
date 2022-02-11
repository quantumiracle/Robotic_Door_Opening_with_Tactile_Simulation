from collections import OrderedDict
import numpy as np
import copy

from robosuite.utils.transform_utils import convert_quat
from robosuite.environments.panda import PandaEnv

from robosuite.utils import transform_utils as T

from robosuite.models.arenas import TableArena
from robosuite.models.objects import CylinderObject
from robosuite.models.robots import Panda
from robosuite.models.tasks import TableTopTask, UniformRandomSampler

from robosuite.class_wrappers import change_dof

class PandaReach(change_dof(PandaEnv, 7, 8)): # don't need to control a gripper

    """
    This class corresponds to the reaching task for the Panda robot arm.
    """
    
    parameters_spec = {
        **PandaEnv.parameters_spec,
        'table_size_0': [0.7, 0.9],
        'table_size_1': [0.7, 0.9],
        'table_size_2': [0.7, 0.9],
    }
    
    def reset_props(self,
                    table_size_0=0.8, table_size_1=0.8, table_size_2=0.8,
                    **kwargs):
        
        self.table_full_size = (table_size_0, table_size_1, table_size_2)
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
                x_range=[-0.1, 0.1],
                y_range=[-0.1, 0.1],
                ensure_object_boundary_in_range=False,
                z_rotation=None,
            )

        # for first initialization
        self.table_full_size = (0.8, 0.8, 0.8)
        self.table_friction = (1.0, 0.005, 0.0001)

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
        goal = CylinderObject(
            size=[0.03, 0.001],
            rgba=[0, 1, 0, 1],
        )
        self.mujoco_goal = goal
        
        self.mujoco_objects = OrderedDict([("goal", goal)])

        # task includes arena, robot, and objects of interest
        self.model = TableTopTask(
            self.mujoco_arena,
            self.mujoco_robot,
            self.mujoco_objects,
            initializer=self.placement_initializer,
            visual_objects=['goal'],
        )
        self.model.place_objects()

    def _get_reference(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._get_reference()
        self.l_finger_geom_ids = [
            self.sim.model.geom_name2id(x) for x in self.gripper.left_finger_geoms
        ]
        self.r_finger_geom_ids = [
            self.sim.model.geom_name2id(x) for x in self.gripper.right_finger_geoms
        ]

        # gripper ids
        self.goal_body_id = self.sim.model.body_name2id('goal')
        self.goal_site_id = self.sim.model.site_name2id('goal')

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()
        self.sim.forward()

        # reset positions of objects
        self.model.place_objects()

        # reset joint positions
        init_pos = self.mujoco_robot.init_qpos
        init_pos += np.random.randn(init_pos.shape[0]) * 0.02
        self.sim.data.qpos[self._ref_joint_pos_indexes] = np.array(init_pos)

        # shut the gripper
        self.sim.data.qpos[self._ref_joint_gripper_actuator_indexes] = np.array([0., -0.])

        # set other reference attributes
        eef_rot_in_world = self.sim.data.get_body_xmat("right_hand").reshape((3, 3))
        self.world_rot_in_eef = copy.deepcopy(eef_rot_in_world.T)  # TODO inspect on this: should we set a golden reference other than a initial position?

    # reward function from sawyer_push
    def reward(self, action=None):
        """
        Reward function for the task.

        The dense reward has three components.

            Reaching: in [-inf, 0], to encourage the arm to reach the object
            Goal Distance: in [-inf, 0] the distance between the pushed object and the goal
            Safety reward in [-inf, 0], -1 for every joint that is at its limit.

        The sparse reward only receives a {0,1} upon reaching the goal

        Args:
            action (np array): The action taken in that timestep

        Returns:
            reward (float): the reward
            previously in robosuite-extra, when dense reward is used, the return value will be a dictionary. but we removed that feature.
        """
        reward = 0.

        # sparse completion reward
        if not self.reward_shaping and self._check_success():
            reward = 1.0

        # use a dense reward
        if self.reward_shaping:
            # max joint angles reward
            joint_limits = self._joint_ranges
            current_joint_pos = self._joint_positions

            hitting_limits_reward = - int(any([(x < joint_limits[i, 0] + 0.05 or x > joint_limits[i, 1] - 0.05) for i, x in
                                              enumerate(current_joint_pos)]))

            reward += hitting_limits_reward

            # reaching reward
            goal_pos = self.sim.data.site_xpos[self.goal_site_id]
            gripper_site_pos = self.sim.data.site_xpos[self.eef_site_id]
            dist = np.linalg.norm(gripper_site_pos - goal_pos)
            reaching_reward = -0.3 * dist
            reward += reaching_reward

            # Success Reward
            success = self._check_success()
            if (success):
                reward += 0.1

            # print('grippersitepos', gripper_site_pos,
            #       'goalpos', goal_pos,
            #       'jointangles', hitting_limits_reward,
            #       'reaching', reaching_reward,
            #       'success', success)

            unstable = reward < -2.5

            # # Return all three types of rewards
            # reward = {"reward": reward, "reaching_distance": -10 * reaching_reward,
            #           "goal_distance": - goal_distance_reward,
            #           "hitting_limits_reward": hitting_limits_reward,
            #           "unstable":unstable}

        return reward
    
    def _check_success(self):
        """
        Returns True if task has been completed.
        """
        gripper_site_pos = self.sim.data.site_xpos[self.eef_site_id]
        goal_pos = self.sim.data.site_xpos[self.goal_site_id]

        dist = np.linalg.norm(goal_pos - gripper_site_pos)
        goal_horizontal_radius = self.model.mujoco_objects['goal'].get_horizontal_radius()

        # gripper centre is within the goal radius
        # print(gripper_site_pos, goal_pos)
        return dist < goal_horizontal_radius

    def step(self, action):
        """ explicitly shut the gripper """
        joined_action = np.append(action, [1.])
        return super().step(joined_action)

    def world2eef(self, world):
        return self.world_rot_in_eef.dot(world)

    def put_raw_object_obs(self, di):
        # Extract position and velocity of the eef
        eef_pos_in_world = self.sim.data.get_body_xpos("right_hand")
        eef_xvelp_in_world = self.sim.data.get_body_xvelp("right_hand")

        # Get the goal position in the world
        goal_site_pos_in_world = np.array(self.sim.data.site_xpos[self.goal_site_id])

        # Record observations into a dictionary
        di['goal_pos_in_world'] = goal_site_pos_in_world
        di['eef_pos_in_world'] = eef_pos_in_world
        di['eef_vel_in_world'] = eef_xvelp_in_world

    def process_object_obs(self, di):
        eef_to_goal_in_world = di['goal_pos_in_world'] - di['eef_pos_in_world']
        eef_to_goal_in_eef = self.world2eef(eef_to_goal_in_world)

        eef_xvelp_in_eef = self.world2eef(di['eef_vel_in_world'])

        task_state = np.concatenate([eef_to_goal_in_world, eef_to_goal_in_eef,
                                     di['eef_vel_in_world'], eef_xvelp_in_eef])

        di['task_state'] = task_state

    def _get_observation(self):
        """
        Returns an OrderedDict containing observations [(name_string, np.array), ...].

        Important keys:
            gripper_to_object : The x-y component of the gripper to object distance
            object_to_goal : The x-y component of the object-to-goal distance
            object_z_rot : the roation of the object around an axis sticking out the table

            object_xvelp: x-y linear velocity of the object
            gripper_xvelp: x-y linear velocity of the gripper


            task-state : a concatenation of all the above.
        """
        # di = super()._get_observation()  # joint angles & vel, which we don't need.
        di = OrderedDict()

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

    def _check_contact_with(self, object):
        """
        Returns True if gripper is in contact with an object.
        """
        collision = False
        for contact in self.sim.data.contact[: self.sim.data.ncon]:
            if (
                    (self.sim.model.geom_id2name(contact.geom1) in self.gripper.contact_geoms()
                     and contact.geom2 == self.sim.model.geom_name2id(object))

                    or (self.sim.model.geom_id2name(contact.geom2) in self.gripper.contact_geoms()
                        and contact.geom1 == self.sim.model.geom_name2id(object))
            ):
                collision = True
                break
        return collision

    def _gripper_visualization(self):
        """
        Do any needed visualization here. Overrides superclass implementations.
        """

        # color the gripper site appropriately based on distance to cube
        if self.gripper_visualization:
            # get distance to goal
            dist = np.sum(
                np.square(
                    self.sim.data.site_xpos[self.goal_site_id]
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
