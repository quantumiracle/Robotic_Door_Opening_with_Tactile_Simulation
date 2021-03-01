from collections import OrderedDict
import numpy as np
import copy

from robosuite.utils.transform_utils import convert_quat
from robosuite.environments.panda import PandaEnv
from gym.envs.robotics.rotations import quat2euler, euler2quat, mat2euler, quat_mul, quat_conjugate

from robosuite.utils import transform_utils as T

from robosuite.models.arenas import TableCabinetArena
from robosuite.models.objects import FullyFrictionalBoxObject, CylinderObject
from robosuite.models.robots import Panda
from robosuite.models.tasks import TableTopTask, UniformRandomSamplerObjectSpecific

from robosuite.class_wrappers import change_dof
import matplotlib.pyplot as plt

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

def sin_cos_encoding(arr):
    """ Encode an array of angle value to correspongding Sines and Cosines, avoiding value jump in 2PI measure like from PI to -PI. """
    return np.concatenate((np.sin(arr), np.cos(arr)))

class PandaOpenDoor(change_dof(PandaEnv, 8, 8)): # keep the dimension to control the gripper; better not to remove change_dof
    """
    This class corresponds to the pushing task for the Panda robot arm.
    """
    minimal_offset = 1e-5
    parameters_spec = {
        **PandaEnv.parameters_spec,
        'knob_friction': [0.8, 1.],  # a smaller range
        # 'knob_friction': [0.2, 1.], # the friction of gripper pads are 1, setting knob friction is easier
        'hinge_stiffness': [0.1, 0.8], # a smaller range
        # 'hinge_stiffness': [0.1, 3.],  # the stiffness value affects significantly on door behaviour, general range in 0-100
        'hinge_damping': [0.1, 0.3],
        'hinge_frictionloss': [0., 1.,],
        'door_mass': [50, 150],  # the door mass does not affect too much in this task
        'knob_mass': [2, 10],
        'table_size_0': [0.8, 0.8+minimal_offset],
        'table_size_1': [1.8, 1.8+minimal_offset],
        'table_size_2': [0.9, 0.9+minimal_offset],
        'table_position_offset_x': [-0.05, 0.05], # randomization of table position in x-axis
        'table_position_offset_y': [-0.05, 0.05], # randomization of table position in y-axis
    }
    
    def reset_props(self,
                    knob_friction = 0.8,
                    hinge_stiffness = 0.1,
                    hinge_damping = 0.1,
                    hinge_frictionloss = 0.1,
                    door_mass = 100., 
                    knob_mass = 5.,
                    table_size_0=0.8, table_size_1=1.8, table_size_2=0.9,
                    table_position_offset_x = 0.05,
                    table_position_offset_y = 0.05, 
                    **kwargs):
        
        self.konb_friction = knob_friction
        self.hinge_stiffness = hinge_stiffness
        self.hinge_damping =  hinge_damping
        self.hinge_frictionloss = hinge_frictionloss
        self.door_mass = door_mass
        self.knob_mass = knob_mass
        self.table_full_size = (table_size_0, table_size_1, table_size_2)
        self.table_position_offset = np.array([table_position_offset_x, table_position_offset_y, 0.])
        super().reset_props(**kwargs)
        self.params_dict.update({
            'knob_friction': knob_friction,
            'hinge_stiffness': hinge_stiffness,
            'hinge_damping': hinge_damping,
            'hinge_frictionloss': hinge_frictionloss,
            'door_mass': door_mass,
            'knob_mass': knob_mass,
            'table_size_0': table_size_0,
            'table_size_1': table_size_1,
            'table_size_2': table_size_2,
            'table_position_offset_x': table_position_offset_x,
            'table_position_offset_y': table_position_offset_y,
        })

    def __init__(self,
                 use_object_obs=True,
                 use_tactile=False,
                 full_obs=False,
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
        
        # whether to use tactile information
        self.use_tactile = use_tactile

        # whether to use full observation information
        self.full_obs = full_obs

        # reward configuration
        self.reward_shaping = reward_shaping

        # object placement initializer
        if placement_initializer:
            self.placement_initializer = placement_initializer
        else:
            # self.placement_initializer = UniformRandomSampler(
            #     x_range=[-0.1, 0.1],
            #     y_range=[-0.1, 0.1],0.05
            #     ensure_object_boundary_in_range=False,
            #     z_rotation=None,
            # )
            self.placement_initializer = UniformRandomSamplerObjectSpecific(
                x_ranges=[[-0.03, -0.02], [0.09, 0.1]],
                y_ranges=[[-0.05, -0.04], [-0.05, -0.04]],
                ensure_object_boundary_in_range=False,
                z_rotation=None,
            )
            

        # for first initialization, before reset parameters and load model, so the values do not matter
        self.table_full_size = (0.8, 0.8, 0.8)
        self.table_friction = (0., 0.005, 0.0001)
        self.konb_friction = 0.8
        self.hinge_stiffness = 0.1
        self.hinge_damping =  0.1
        self.hinge_frictionloss = 0.1
        self.table_position_offset = np.array([0., 0., 0.])

        self.door_mass = 100.
        self.knob_mass = 5.

        self.object_obs_process = object_obs_process
        self.grasp_state = False

        super().__init__(gripper_visualization=True, **kwargs)

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()
        self.mujoco_robot.set_base_xpos([0, 0, 0])

        # load model for table top workspace
        self.mujoco_arena = TableCabinetArena(
            table_full_size=self.table_full_size, table_friction=self.table_friction
        )
        if self.use_indicator_object:
            self.mujoco_arena.add_pos_indicator()

        # Set the table position with certain randomness in x- and y-axis for better sim2real,
        # note that this is not observation noise, but different env settings,
        central_pos = np.array([-0.84, 0.5, -0.03])  # match with reality
        central_pos = central_pos + self.table_position_offset
        self.mujoco_arena.set_origin(central_pos) # the vector is the relative distance from tabel top center to the robot base
        
        self.mujoco_objects = None

        # task includes arena, robot, and objects of interest
        self.model = TableTopTask(
            self.mujoco_arena,
            self.mujoco_robot,
            self.mujoco_objects,
            initializer=self.placement_initializer,
            visual_objects=[],
        )
        if self.mujoco_objects is not None:
            self.model.place_objects()

        # set the chosen parameter values
        # Note: set values here rather than in reset_props() since the model is reloaded after reset_props()
        self.mujoco_arena.knob_geom.set('friction', str(self.konb_friction)+' 0 0')  # only set the sliding friction (the first dim)
        self.mujoco_arena.door_hinge.set('stiffness', str(self.hinge_stiffness))
        self.mujoco_arena.door_hinge.set('damping', str(self.hinge_damping))
        self.mujoco_arena.door_hinge.set('frictionloss', str(self.hinge_frictionloss))
        self.mujoco_arena.door_inertial.set('mass', str(self.door_mass))
        self.mujoco_arena.knob_link_inertial.set('mass', str(self.knob_mass))

        # set robot grippers friction to be very small, so that the knob friction matters
        self.robot_hand = self.mujoco_robot.root.find(".//body[@name='{}']".format("right_hand")).find("./body[@name='right_gripper']")
        self.robot_hand.find("./body[@name='leftfinger']").find("./body[@name='finger_joint1_tip']").find("./geom[@name='finger1_tip_collision']").set('friction', '0.001 0 0')
        self.robot_hand.find("./body[@name='rightfinger']").find("./body[@name='finger_joint2_tip']").find("./geom[@name='finger2_tip_collision']").set('friction', '0.001 0 0')

    def _get_reference(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._get_reference()
        # self.cube_body_id = self.sim.model.body_name2id("cube")
        self.l_finger_geom_ids = [
            self.sim.model.geom_name2id(x) for x in self.gripper.left_finger_geoms
        ]
        self.r_finger_geom_ids = [
            self.sim.model.geom_name2id(x) for x in self.gripper.right_finger_geoms
        ]
        self.knob_geom_id = self.sim.model.geom_name2id("cabinet_knob")

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()
        self.grasp_state = False
        self.sim.forward()

        # reset positions of objects
        if self.mujoco_objects is not None:
            self.model.place_objects()

        # reset joint positions
        # self.sim.data.qpos[self._ref_joint_pos_indexes] =  [-2.38552629,  0.11408278, -0.43481802, -1.64875619,  1.77681087,  3.37056892, -0.8571096 ]
        # self.sim.data.qpos[self._ref_joint_pos_indexes] =  [-2.73830829,  0.23346824, -0.09714798, -1.63363,  1.66059114,  3.52977957, -0.83828194] # a closer position
        scale = 0.
        noise = np.random.uniform(-scale, scale, 7)
        ini_pos = np.array([-2.04294938, 0.18509384, -0.89699324, -1.75267233,  1.64237899,  3.33180868, -0.70387438])  # a position far from limits and close to knob
        self.sim.data.qpos[self._ref_joint_pos_indexes] = ini_pos + noise
        # for test
        # self.sim.data.qpos[self._ref_joint_pos_indexes] =  [1.58992658e+00,  2.97199045e-01,  1.89987854e-05, -2.84390673e+00, 1.73368590e-03,  3.13953742e+00,  8.02213039e-01]
        # open the gripper
        self.sim.data.ctrl[-2:] = np.array([0.04, -0.04])  # panda gripper finger joint range is -0.04~0.04

        # set other reference attributes
        eef_rot_in_world = self.sim.data.get_body_xmat("right_hand").reshape((3, 3))
        self.world_rot_in_eef = copy.deepcopy(eef_rot_in_world.T)  # TODO inspect on this: should we set a golden reference other than a initial position?

    def get_gripper_state(self,):
        return abs(self.sim.data.qpos[-1])  # last joint is the gripper

    def reward(self, action=None):
        """
        Reward function for the task.
        Args:
            action (np array): The action taken in that timestep

        Returns:
            reward (float): the reward
            previously in robosuite-extra, when dense reward is used, the return value will be a dictionary. but we removed that feature.
        """
        # print contact information
        # for i in range(self.sim.data.ncon):  # total number of contact: env.sim.data.ncon
        #     c = self.sim.data.contact[i]
        #     print('Contact {}: {} and {}'.format(i, self.sim.model.geom_id2name(c.geom1), self.sim.model.geom_id2name(c.geom2)))
        # self.ee_ori = quat2euler(mat2quat(self._right_hand_orn))

        open_multi = 5.
        dis_multi = 0.4
        ori_multi = 0.05
        grasp_multi = 1.
        tac_multi = 0.01
        force_multi = 0.1

        reward = 0.
        self.door_open_angle = abs(self.sim.data.get_joint_qpos("hinge0"))

        # door open angle reward
        reward_door_open = 0.
        # If the gripper is nearly closed, ignore the reward for door opening; fully closed is about 0.001
        # However, the self.get_gripper_state() can be inaccurate sometimes, so deprecate this approach.
        # if self.get_gripper_state() > 0.002:  
        #     reward_door_open += self.door_open_angle
        if self.grasp_state:  # only count for the door opening reward when the knob is grasped by the robot
            reward_door_open += self.door_open_angle

        # distance and orientation rewards for reaching
        reward_dist = 0.
        reward_ori = 0.
        if self.door_open_angle < 0.02:
            # A distance reward: minimize the distance between the gripper and door konb when the door is almost closed 
            reward_dist = -1. - np.tanh(np.linalg.norm(self.get_hand2knob_dist_vec())) # ensure this is penalty, such that openning door is always encouraged

            # An orientation reward: make the orientation of gripper horizontal (better for knob grasping) when the door is almost closed 
            fingerEulerDesired =  [0, 0, -np.pi/2]  # horizontal gesture for gripper
            finger_ori = self.get_finger_ori()
            ori_diff = sin_cos_encoding(fingerEulerDesired) - sin_cos_encoding(finger_ori)  # use sin_cos_encoding to avoid value jump in 2PI measure
            reward_ori = -1. - np.tanh(np.linalg.norm(ori_diff))  # ensure this is penalty, such that openning door is always encouraged

        # grasping reward
        touch_left_finger = False
        touch_right_finger = False
        reward_grasp = 0.
        for i in range(self.sim.data.ncon):
            c = self.sim.data.contact[i]
            if c.geom1 in self.l_finger_geom_ids and c.geom2 == self.knob_geom_id:
                touch_left_finger = True
            if c.geom1 == self.knob_geom_id and c.geom2 in self.l_finger_geom_ids:
                touch_left_finger = True
            if c.geom1 in self.r_finger_geom_ids and c.geom2 == self.knob_geom_id:
                touch_right_finger = True
            if c.geom1 == self.knob_geom_id and c.geom2 in self.r_finger_geom_ids:
                touch_right_finger = True
        if touch_left_finger and touch_right_finger and self.get_gripper_state()>0.005: # the grasping detection here (when True) not only requires the knob to be grasped by the gripper, but also in a good gesture
            self.grasp_state = True
            reward_grasp += 0.1
        else:
            self.grasp_state = False

        # an additional reward for providing more tactile signals
        if self.use_tactile and self.door_open_angle > 0.02 and self.get_gripper_state()>0.005:  # only when door is open and gripper is not fully closed (contact with itself)
            reward_tactile = np.sum(self._get_tactile_singals()) 
        else:
            reward_tactile = 0.

        # additional reward for minimizing force
        ee_force = np.abs(self.sim.data.get_sensor('force_ee'))
        reward_force = np.tanh(1./(ee_force + 1e-5))
        # print(reward_door_open, reward_dist, reward_ori, reward_grasp, reward_tactile)
        # a summary of reward values
        reward = open_multi*reward_door_open + dis_multi*reward_dist + ori_multi*reward_ori + grasp_multi*reward_grasp + tac_multi*reward_tactile  
        # reward = open_multi*reward_door_open + grasp_multi*reward_grasp + tac_multi*reward_tactile + force_multi*reward_force # only a open-door policy

        # print('force: ', self.sim.data.get_sensor('force_ee'))  # Gives one value
        # print('torque: ', self.sim.data.get_sensor('torque_ee'))  # Gives one value
        # print(self.sim.data.sensordata[:6])
        # print(self.sim.data.sensordata[7::3]) # Gives array of all sensorvalues: force tactile
        # print(self._get_tactile_singals())
        # print(self.sim.data.sensordata[6:]) # Gives array of all sensorvalues: touch tactile

        self.done = self._check_success()

        return reward
    
    def _check_success(self):
        """
        Returns True if task has been completed.
        """

        if self.door_open_angle >= 1.55: # 1.57 ~ PI/2
            return True
        else:
            return False

    def _get_tactile_singals(self, contact_threshold=1e-3, Binary=True):
        """
        Get the tactile signals from sensors. 

        Params:        # print(self._get_tactile_singals())

            contact_threshold: the threshold of normal contact force for being contact;
            Binary: whether using binary representation of contact info or not.

        Return: 
            An array of (30,) for two sensor arries on pads.

        """
        tactile_force = self.sim.data.sensordata[7::3]  # 3d force to 1d: only use the perpendicular one
        if Binary:
            binary_tactile = np.where(np.abs(tactile_force)>contact_threshold, 1, 0)  # change to binary: if absolute value > threthold, then 1 otherwise 0
            return binary_tactile
        else:
            return tactile_force

    def step(self, action):
        return super().step(action)

    def world2eef(self, world):
        return self.world_rot_in_eef.dot(world)

    def _joint_limit(self,):
        """ 
        Joint position limits (rad) of real robot, reference: 
        https://frankaemika.github.io/docs/control_parameters.html#control-parameters-specifications 
        """
        q_limit_max = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
        q_limit_min = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])

        return q_limit_max, q_limit_min

    def check_joint_limit(self, threshold=0.03):
        """
        Check whether current joint position is close to the joint limits.
        """
        curr_q = self.sim.data.qpos[self._ref_joint_pos_indexes]
        for i, (q, q_min, q_max) in enumerate(zip(curr_q, *self._joint_limit())):
            if np.min([np.abs(q-q_min), np.abs(q_max-q)]) < threshold:
                print("Current {}-th joint position in {} is close to joint limits with threshold {}.".format(i+1, curr_q, threshold))


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
            eef_pos_in_world = self.sim.data.get_body_xpos("right_hand")
            eef_xvelp_in_world = self.sim.data.get_body_xvelp("right_hand")
            di['eef_pos_in_world'] = eef_pos_in_world  # dim=3
            di['eef_vel_in_world'] = eef_xvelp_in_world  # dim=3
            di['joint_pos_in_world'] = self.sim.data.qpos[self._ref_joint_pos_indexes]  # dim=7
            di['joint_vel_in_world'] = self.sim.data.qvel[self._ref_joint_pos_indexes]  # dim=7
            # di['finger_knob_dist'] = self.get_hand2knob_dist_vec()  # dim=3, not used in reality due to the uncertain position of hand_visual
            di['knob_pos_in_world'] = self.get_knob_pos() # dim=3, position of center of the knob
            di['knob_pos_to_eef'] = di['knob_pos_in_world'] - di['eef_pos_in_world']   # dim=3, position of center of the knob relative to eef
            di['door_hinge_angle'] = [self.sim.data.get_joint_qpos("hinge0")]  # dim=1
            di['gripper_width'] = [self.get_gripper_state()]  # dim=1

            # self.check_joint_limit()

            if self.full_obs: # dim=25
                task_state = np.concatenate([
                                        di['eef_pos_in_world'], 
                                        di['eef_vel_in_world'], 
                                        di['joint_pos_in_world'],
                                        di['joint_vel_in_world'],
                                        di['gripper_width'],
                                        # di['finger_knob_dist'],
                                        # di['knob_pos_in_world'],  
                                        di['knob_pos_to_eef'],
                                        di['door_hinge_angle'],
                                    ])

            else:
                # FK, dim=12
                task_state = np.concatenate([ 
                                        di['joint_pos_in_world'],
                                        di['gripper_width'],
                                        di['knob_pos_to_eef'],
                                        di['door_hinge_angle'],
                                    ])
                # IK, dim=8
                # task_state = np.concatenate([ 
                #                         di['eef_pos_in_world'],
                #                         di['gripper_width'],
                #                         di['knob_pos_to_eef'],
                #                         di['door_hinge_angle'],
                #                     ])
        di['task_state_no_tactile'] = task_state
        if self.use_tactile:
            di['tactile'] = self._get_tactile_singals()
            task_state = np.concatenate((task_state, di['tactile']))
        
        di['task_state'] = task_state

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
        if self.gripper_visualization:
            rgba = np.zeros(4)

            self.sim.model.site_rgba[self.eef_site_id] = rgba


    def get_finger_ori(self):
        finger_rel_quat = self.sim.data.get_body_xquat("rightfinger")
        hand_quat = self.sim.data.get_body_xquat("right_hand")
        finger_world_quat = quat_mul(finger_rel_quat, hand_quat)  # TODO: which one at first?
        return quat2euler(finger_world_quat)

    def get_hand_pos(self):
        return self.sim.data.get_geom_xpos("hand_visual")

    def get_knob_pos(self):
        knob_pos = self.sim.data.get_geom_xpos("center_cabinet_knob")
        return knob_pos

    def get_hand2knob_dist_vec(self):
        return self.get_hand_pos() - self.get_knob_pos()
