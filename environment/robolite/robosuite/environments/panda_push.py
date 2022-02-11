from collections import OrderedDict
import numpy as np
import copy

from robosuite.utils.transform_utils import convert_quat, quat2mat, mat2euler
from robosuite.environments.panda import PandaEnv

from robosuite.utils import transform_utils as T

from robosuite.models.arenas import TableArena
from robosuite.models.objects import FullyFrictionalBoxObject, CylinderObject
from robosuite.models.robots import Panda
from robosuite.models.tasks import TableTopTask, UniformRandomSamplerObjectSpecific

from robosuite.class_wrappers import change_dof

# https://stackoverflow.com/a/13849249/11815215

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

class PandaPush(change_dof(PandaEnv, 7, 8)): # don't need to control a gripper

    """
    This class corresponds to the pushing task for the Panda robot arm.
    """
    minimal_offset = 1e-5
    parameters_spec = {
        **PandaEnv.parameters_spec,
        'table_size_0': [0.8, 0.8+minimal_offset],
        'table_size_1': [0.8, 0.8+minimal_offset],
        'table_size_2': [0.903, 0.903+minimal_offset],
        'table_friction_0': [0.0, minimal_offset],
        'table_friction_1': [0.0, minimal_offset],
        'table_friction_2': [0.0, minimal_offset],
        'boxobject_size_0': [0.025, 0.05], # default [0.0298, 0.0302]
        'boxobject_size_1': [0.025, 0.05],
        'boxobject_size_2': [0.025, 0.05],
        'boxobject_friction_0': [0.1, 0.6],  # tangential, default [0.1, 0.4]
        'boxobject_friction_1': [0.0, 0.2],  # torsional, default [0.0, 0.02]
        'boxobject_friction_2': [0.00005, 0.00015],  # rolling
        'boxobject_density_1000': [0.6, 1.4],
    }
    
    def reset_props(self,
                    table_size_0=0.8, table_size_1=0.8, table_size_2=0.903,  # z-position of table surface is lower than robot base by 10 mm in our real settting
                    table_friction_0=0., table_friction_1=0., table_friction_2=0.,
                    boxobject_size_0=0.030, boxobject_size_1=0.030, boxobject_size_2=0.030,
                    boxobject_friction_0=0.25, boxobject_friction_1=0.01, boxobject_friction_2=0.0001,
                    boxobject_density_1000=1.,
                    **kwargs):
        
        self.table_full_size = (table_size_0, table_size_1, table_size_2)
        self.table_friction = (table_friction_0, table_friction_1, table_friction_2)
        self.boxobject_size = (boxobject_size_0, boxobject_size_1, boxobject_size_2)
        self.boxobject_friction = (boxobject_friction_0, boxobject_friction_1, boxobject_friction_2)
        self.boxobject_density = boxobject_density_1000 * 1000.
        super().reset_props(**kwargs)  # keep the same order as parameters_spec, so put this before self.params_dict.udpate()
        self.params_dict.update({
                            'table_size_0': table_size_0, 
                            'table_size_1': table_size_1,
                            'table_size_2': table_size_2,
                            'table_friction_0': table_friction_0,
                            'table_friction_1': table_friction_1,
                            'table_friction_2': table_friction_2,
                            'boxobject_size_0': boxobject_size_0,
                            'boxobject_size_1': boxobject_size_1,
                            'boxobject_size_2': boxobject_size_2,
                            'boxobject_friction_0': boxobject_friction_0,
                            'boxobject_friction_1': boxobject_friction_1,
                            'boxobject_friction_2': boxobject_friction_2,
                            'boxobject_density_1000': boxobject_density_1000,
                            })

    def __init__(self,
                 use_object_obs=True,
                 reward_shaping=True,
                 placement_initializer=None,
                 object_obs_process=True,
                 face_downwards=True,
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

        # whether force the gripper to face downwards
        self.face_downwards = face_downwards

        object_ini_area = [0.275, 0.15]  # length: x, width: y
        goal_pos_area = [0.225, 0.15]
        # object placement initializer
        if placement_initializer:
            self.placement_initializer = placement_initializer
        else:
            self.placement_initializer = UniformRandomSamplerObjectSpecific(
                # a good simulation pose
                # x_ranges=[[-0.03, -0.02], [0.09, 0.1]], # the position here is the table base, but self.sim.data.site_xpos[self.goal_site_id] gives the position of goal in robot base
                # y_ranges=[[-0.05, -0.04], [-0.05, -0.04]],

                # matched with real pose
                # x_ranges=[[-0.5, -0.6], [-0.5, -0.6]],
                # y_ranges=[[0.4, 0.45], [0.55, 0.6]],

                # x_ranges=[[-object_ini_area[0]/2, object_ini_area[0]/2], [-goal_pos_area[0]/2, goal_pos_area[0]/2]],
                # y_ranges=[[-0.206-object_ini_area[1]/2, -0.206+object_ini_area[1]/2], [0.044-goal_pos_area[1]/2, 0.044+goal_pos_area[1]/2]],

                # with goal fixed
                x_ranges=[[-object_ini_area[0]/4, object_ini_area[0]/4], [-0.001, 0.001]],
                y_ranges=[[-0.206-object_ini_area[1]/4, -0.206+object_ini_area[1]/4], [0.043, 0.045]],

                ensure_object_boundary_in_range=False,
                z_rotation=None,
            )
            
        # for first initialization
        self.table_full_size = (0.8, 0.8, 0.8)
        self.table_friction = (0., 0., 0.)
        self.boxobject_size = (0.02, 0.02, 0.02)
        self.boxobject_friction = (0.001, 0.001, 0.0001)
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
        # self.mujoco_arena.set_origin([0.16 + self.table_full_size[0] / 2, 0, 0])  # good simulation setting
        self.mujoco_arena.set_origin([0, 0.7, 0])  # match with reality

        # initialize objects of interest
        # in original robosuite, a simple domain randomization is included in BoxObject implementation, and called here. We choose to discard that implementation.
        cube = FullyFrictionalBoxObject(
            size=self.boxobject_size,
            friction=self.boxobject_friction,
            density=self.boxobject_density,
            rgba=[1, 0, 0, 1],
        )
        self.mujoco_cube = cube

        goal = CylinderObject(
            size=[0.03, 0.001],
            rgba=[0, 1, 0, 1],
        )
        self.mujoco_goal = goal
        
        self.mujoco_objects = OrderedDict([("cube", cube), ("goal", goal)])

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
        self.cube_body_id = self.sim.model.body_name2id("cube")
        self.l_finger_geom_ids = [
            self.sim.model.geom_name2id(x) for x in self.gripper.left_finger_geoms
        ]
        self.r_finger_geom_ids = [
            self.sim.model.geom_name2id(x) for x in self.gripper.right_finger_geoms
        ]
        self.cube_geom_id = self.sim.model.geom_name2id("cube")

        # ids
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
        init_pos += np.random.randn(init_pos.shape[0]) * 0.02  # a good simulation pose
        # init_pos = [1.57103,0.144537, 0 ,-2.97299, 0, 3.11744,0.785635]  # matched with real pose
        init_pos = [1.58992658e+00,  2.97199045e-01,  1.89987854e-05, -2.84390673e+00, 1.73368590e-03,  3.13953742e+00,  8.02213039e-01] # a position preventing stuck
        
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
        reach_multi = 0.4
        goal_multi = 1.

        reward = 0.

        # sparse completion reward
        if not self.reward_shaping and self._check_success():
            reward = 1.0

        # use a dense reward
        if self.reward_shaping:
            object_pos = self.sim.data.body_xpos[self.cube_body_id]

            # max joint angles reward
            joint_limits = self._joint_ranges
            current_joint_pos = self._joint_positions

            # hitting_limits_reward = - int(any([(x < joint_limits[i, 0] + 0.05 or x > joint_limits[i, 1] - 0.05) for i, x in
            #                                   enumerate(current_joint_pos)]))

            hitting_limits_reward = 0.
            # if hitting_limits_reward:
            #     print('joint limit reward: ', hitting_limits_reward)

            reward += hitting_limits_reward

            # reaching reward
            gripper_site_pos = self.sim.data.site_xpos[self.eef_site_id]
            dist = np.linalg.norm(gripper_site_pos - object_pos)
            reaching_reward = 1-np.tanh(10.*dist)
            reward += reach_multi * reaching_reward

            # Success Reward
            success = self._check_success()
            if (success):
                reward += 0.1

            # goal distance reward
            goal_pos = self.sim.data.site_xpos[self.goal_site_id]

            dist = np.linalg.norm(goal_pos - object_pos)
            goal_distance_reward = 1-np.tanh(10.*dist)
            reward += goal_multi * goal_distance_reward

            # punish when there is a line of object--gripper--goal
            # angle_g_o_g = angle_between(gripper_site_pos - object_pos,
            #                             goal_pos - object_pos)
            # if not success and angle_g_o_g < np.pi / 2.:
            #     reward += -0.03 - 0.02 * (np.pi / 2. - angle_g_o_g)

            # print('grippersitepos', gripper_site_pos,
            #       'objpos', object_pos,
            #       'jointangles', hitting_limits_reward,
            #       'reaching', reaching_reward,
            #       'success', success,
            #       'goaldist', goal_distance_reward)

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
        object_pos = self.sim.data.body_xpos[self.cube_body_id]
        goal_pos = self.sim.data.site_xpos[self.goal_site_id]

        dist = np.linalg.norm(goal_pos - object_pos)
        goal_horizontal_radius = self.model.mujoco_objects['goal'].get_horizontal_radius()

        # object centre is within the goal radius
        return dist < goal_horizontal_radius

    def step(self, action):
        """ explicitly shut the gripper """
        joined_action = np.append(action, [1.])
        obs, reward, done, info = super().step(joined_action)

        if self.face_downwards: # keep the gripper facing downwards (sometimes it's not due to the imperfect IK)
            ori_threshold = 0.1  # threshold of orientation error for setting done=True
            if np.min([np.linalg.norm(mat2euler(self._right_hand_orn) - np.array([-np.pi, 0., 0.])),
                np.linalg.norm(mat2euler(self._right_hand_orn) - np.array([np.pi, 0., 0.]))]) > ori_threshold:
                done = True
        
        # success case
        if self._check_success():
            done = True
        return obs, reward, done, info

    def world2eef(self, world):
        return self.world_rot_in_eef.dot(world)

    def put_raw_object_obs(self, di):
        # Extract position and velocity of the eef
        eef_pos_in_world = self.sim.data.get_body_xpos("right_hand")
        eef_xvelp_in_world = self.sim.data.get_body_xvelp("right_hand")

        # Get the position, velocity, rotation  and rotational velocity of the object in the world frame
        object_pos_in_world = self.sim.data.body_xpos[self.cube_body_id]
        object_xvelp_in_world = self.sim.data.get_body_xvelp('cube')
        object_rot_in_world = self.sim.data.get_body_xmat('cube')
        
        # Get the z-angle with respect to the reference position and do sin-cosine encoding
        # world_rotation_in_reference = np.array([[0., 1., 0., ], [-1., 0., 0., ], [0., 0., 1., ]])
        # object_rotation_in_ref = world_rotation_in_reference.dot(object_rot_in_world)
        # object_euler_in_ref = T.mat2euler(object_rotation_in_ref)
        # z_angle = object_euler_in_ref[2]
        
        object_quat = convert_quat(self.sim.data.body_xquat[self.cube_body_id], to='xyzw')
        
        # Get the goal position in the world
        goal_site_pos_in_world = np.array(self.sim.data.site_xpos[self.goal_site_id])

        # Record observations into a dictionary
        di['goal_pos_in_world'] = goal_site_pos_in_world
        di['eef_pos_in_world'] = eef_pos_in_world
        di['eef_vel_in_world'] = eef_xvelp_in_world
        di['object_pos_in_world'] = object_pos_in_world
        di['object_vel_in_world'] = object_xvelp_in_world
        # di["z_angle"] = np.array([z_angle])
        di['object_quat'] = object_quat

    def process_object_obs(self, di):
        # z_angle = di['z_angle']
        # sine_cosine = np.array([np.sin(8*z_angle), np.cos(8*z_angle)]).reshape((2,))

        eef_to_object_in_world = di['object_pos_in_world'] - di['eef_pos_in_world']
        # eef_to_object_in_eef = self.world2eef(eef_to_object_in_world)

        object_to_goal_in_world = di['goal_pos_in_world'] - di['object_pos_in_world']
        # object_to_goal_in_eef = self.world2eef(object_to_goal_in_world)

        
        # processed state
        # task_state = np.concatenate([eef_to_object_in_world,
        #                              object_to_goal_in_world,
        #                              di['eef_vel_in_world'],
        #                              di['object_vel_in_world'],
        #                              di['object_quat']])

        object_euler = mat2euler(quat2mat(di['object_quat']))

        # raw state for convenience of real-world experiments
        task_state = np.concatenate([
                                    # di['eef_pos_in_world'],
                                    # di['eef_vel_in_world'],
                                    eef_to_object_in_world,
                                    object_to_goal_in_world,
                                    di['eef_vel_in_world'],
                                    object_euler])  

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
