from collections import OrderedDict
import numpy as np

import robosuite.utils.transform_utils as T
from robosuite.environments import MujocoEnv

from robosuite.models.grippers import gripper_factory
from robosuite.models.robots import Panda


class PandaEnv(MujocoEnv):
    """Initializes a Panda robot environment."""

    parameters_spec = {
        'link1_mass': [2., 3.],
        'link2_mass': [2., 3.],
        'link3_mass': [2., 3.],
        'link4_mass': [2., 3.],
        'link5_mass': [2., 3.],
        'link6_mass': [1.4, 1.6],
        'link7_mass': [0.4, 0.6],
        'joint1_damping': [0.06, 0.14],
        'joint2_damping': [0.06, 0.14],
        'joint3_damping': [0.06, 0.14],
        'joint4_damping': [0.06, 0.14],
        'joint5_damping': [0.06, 0.14],
        'joint6_damping': [0.006, 0.014],
        'joint7_damping': [0.006, 0.014],
        'joint1_armature': [0.0, 0.5],    # armature default 0
        'joint2_armature': [0.0, 0.5],    # armature default 0
        'joint3_armature': [0.0, 0.5],    # armature default 0
        'joint4_armature': [0.0, 0.5],    # armature default 0
        'joint5_armature': [0.0, 0.5],    # armature default 0
        'joint6_armature': [0.0, 0.5],    # armature default 0
        'joint7_armature': [0.0, 0.5],    # armature default 0
        # [TODO] is joint*_frictionloss necessary?
        'actuator_velocity_joint1_kv': [30.0, 50.0],
        'actuator_velocity_joint2_kv': [30.0, 50.0],
        'actuator_velocity_joint3_kv': [30.0, 50.0],
        'actuator_velocity_joint4_kv': [30.0, 50.0],
        'actuator_velocity_joint5_kv': [30.0, 50.0],
        'actuator_velocity_joint6_kv': [30.0, 50.0],
        'actuator_velocity_joint7_kv': [30.0, 50.0],
        'actuator_position_finger_joint1_kp_1000000': [0.6, 1.4],    # will be multiplied by 1e6
        'actuator_position_finger_joint2_kp_1000000': [0.6, 1.4],
    }

    def __init__(
        self,
        gripper_type="PandaGripper",
        gripper_visualization=False,
        use_indicator_object=False,
        has_renderer=False,
        has_offscreen_renderer=False,
        render_collision_mesh=False,
        render_visual_mesh=True,
        control_freq=20,
        horizon=1000,
        ignore_done=False,
        use_camera_obs=False,
        camera_name="frontview",
        camera_height=256,
        camera_width=256,
        camera_depth=False,
    ):
        """
        Args:
            gripper_type (str): type of gripper, used to instantiate
                gripper models from gripper factory.

            gripper_visualization (bool): True if using gripper visualization.
                Useful for teleoperation.

            use_indicator_object (bool): if True, sets up an indicator object that
                is useful for debugging.

            has_renderer (bool): If true, render the simulation state in
                a viewer instead of headless mode.

            has_offscreen_renderer (bool): True if using off-screen rendering.

            render_collision_mesh (bool): True if rendering collision meshes
                in camera. False otherwise.

            render_visual_mesh (bool): True if rendering visual meshes
                in camera. False otherwise.

            control_freq (float): how many control signals to receive
                in every second. This sets the amount of simulation time
                that passes between every action input.

            horizon (int): Every episode lasts for exactly @horizon timesteps.

            ignore_done (bool): True if never terminating the environment (ignore @horizon).

            use_camera_obs (bool): if True, every observation includes a
                rendered image.

            camera_name (str): name of camera to be rendered. Must be
                set if @use_camera_obs is True.

            camera_height (int): height of camera frame.

            camera_width (int): width of camera frame.

            camera_depth (bool): True if rendering RGB-D, and RGB otherwise.
        """

        self.has_gripper = gripper_type is not None
        self.gripper_type = gripper_type
        self.gripper_visualization = gripper_visualization
        self.use_indicator_object = use_indicator_object
        self.params_dict = {}
        super().__init__(
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            use_camera_obs=use_camera_obs,
            camera_name=camera_name,
            camera_height=camera_height,
            camera_width=camera_width,
            camera_depth=camera_depth,
        )

    def reset_props(self, **kwargs):
        parameters_defaults = {
            # 'link1_mass': 3.0,
            # 'link2_mass': 3.0,
            # 'link3_mass': 2.0,
            # 'link4_mass': 2.0,
            # 'link5_mass': 2.0,
            # 'link6_mass': 1.5,
            # 'link7_mass': 0.5,
            # new mass refer to: https://github.com/mkrizmancic/franka_gazebo/blob/master/robots/panda_arm.xacro
            'link1_mass': 2.74,
            'link2_mass': 2.74,
            'link3_mass': 2.38,
            'link4_mass': 2.38,
            'link5_mass': 2.74,
            'link6_mass': 1.55,
            'link7_mass': 0.54,
            'joint1_damping': 0.1,
            'joint2_damping': 0.1,
            'joint3_damping': 0.1,
            'joint4_damping': 0.1,
            'joint5_damping': 0.1,
            'joint6_damping': 0.01,
            'joint7_damping': 0.01,
            'joint1_armature': 0.0,
            'joint2_armature': 0.0,
            'joint3_armature': 0.0,
            'joint4_armature': 0.0,
            'joint5_armature': 0.0,
            'joint6_armature': 0.0,
            'joint7_armature': 0.0,
            # [TODO] is joint*_frictionloss necessary?
            'actuator_velocity_joint1_kv': 40.0,
            'actuator_velocity_joint2_kv': 40.0,
            'actuator_velocity_joint3_kv': 40.0,
            'actuator_velocity_joint4_kv': 40.0,
            'actuator_velocity_joint5_kv': 40.0,
            'actuator_velocity_joint6_kv': 40.0,
            'actuator_velocity_joint7_kv': 40.0,
            'actuator_position_finger_joint1_kp_1000000': 1.0,    # will be multiplied by 1e6
            'actuator_position_finger_joint2_kp_1000000': 1.0,
        }

        if not all(key in parameters_defaults for key in kwargs):
            for key in kwargs:
                if key not in parameters_defaults:
                    print(key, '<- here')

            assert(False)  # GZZ: if an error is triggered here, then you must have passed in an invalid parameter keyword to reset(). please check that.
            
        self.params_dict.update(parameters_defaults, **kwargs)  # if same key is assigned value several times, the last value counts, therefore parameters_defaults may be overwritten by **kwargs in DR

    def _load_model(self):
        """
        Loads robot and optionally add grippers.
        """
        super()._load_model()
        self.mujoco_robot = Panda()
        if self.has_gripper:
            self.gripper = gripper_factory(self.gripper_type)
            if not self.gripper_visualization:
                self.gripper.hide_visualization()
            self.mujoco_robot.add_gripper("right_hand", self.gripper)

        if bool(self.params_dict): # if it is not empty
            params_dict = self.params_dict.copy()
            for link in self.mujoco_robot._link_body:
                lie = self.mujoco_robot.root.find(".//body[@name='{}']".format(link)).find("./inertial[@mass]")
                # <inertial pos="0 0 -0.07" mass="3" diaginertia="0.3 0.3 0.3" />
                lie.set('mass', str(params_dict['{}_mass'.format(link)]))
                
            for joint in self.mujoco_robot._joints:
                je = self.mujoco_robot.root.find(".//joint[@name='{}']".format(joint))
                # <joint name="joint1" pos="0 0 0" axis="0 0 1" limited="true" range="-2.8973 2.8973" damping="0.1"/>
                je.set('damping', str(params_dict['{}_damping'.format(joint)]))
                je.set('armature', str(params_dict['{}_armature'.format(joint)]))

                avje = self.mujoco_robot.root.find(".//velocity[@joint='{}']".format(joint))
                # <velocity ctrllimited="true" ctrlrange="-2.1750 2.1750" joint="joint1" kv="40.0" name="vel_right_j1"/>
                avje.set('kv', str(params_dict['actuator_velocity_{}_kv'.format(joint)]))

            for gripper_joint in self.gripper_joints:
                gpvje = self.mujoco_robot.root.find(".//position[@joint='{}']".format(gripper_joint))
                # <position ctrllimited="true" ctrlrange="0.0 0.04" forcelimited="true" forcerange="-20 20" joint="finger_joint1" kp="1000000" name="gripper_joint1" />
                gpvje.set('kp', str(params_dict['actuator_position_{}_kp_1000000'.format(gripper_joint)] * 1000000.0))

    def _reset_internal(self):
        """
        Sets initial pose of arm and grippers.
        """
        super()._reset_internal()

        self.sim.data.qpos[self._ref_joint_pos_indexes] = self.mujoco_robot.init_qpos


        if self.has_gripper:
            self.sim.data.qpos[
                self._ref_joint_gripper_actuator_indexes
            ] = self.gripper.init_qpos

    def _get_reference(self):
        """
        Sets up necessary reference for robots, grippers, and objects.
        """
        super()._get_reference()

        # indices for joints in qpos, qvel
        self.robot_joints = list(self.mujoco_robot.joints)
        # print( self.sim.model.get_joint_qpos_addr('joint0'))
        self._ref_joint_pos_indexes = [
            self.sim.model.get_joint_qpos_addr(x) for x in self.robot_joints
        ]
        self._ref_joint_vel_indexes = [
            self.sim.model.get_joint_qvel_addr(x) for x in self.robot_joints
        ]

        if self.use_indicator_object:
            ind_qpos = self.sim.model.get_joint_qpos_addr("pos_indicator")
            self._ref_indicator_pos_low, self._ref_indicator_pos_high = ind_qpos

            ind_qvel = self.sim.model.get_joint_qvel_addr("pos_indicator")
            self._ref_indicator_vel_low, self._ref_indicator_vel_high = ind_qvel

            self.indicator_id = self.sim.model.body_name2id("pos_indicator")

        # indices for grippers in qpos, qvel
        if self.has_gripper:
            self.gripper_joints = list(self.gripper.joints)
            self._ref_gripper_joint_pos_indexes = [
                self.sim.model.get_joint_qpos_addr(x) for x in self.gripper_joints
            ]
            self._ref_gripper_joint_vel_indexes = [
                self.sim.model.get_joint_qvel_addr(x) for x in self.gripper_joints
            ]

        # indices for joint pos actuation, joint vel actuation, gripper actuation
        self._ref_joint_pos_actuator_indexes = [
            self.sim.model.actuator_name2id(actuator)
            for actuator in self.sim.model.actuator_names
            if actuator.startswith("pos")
        ]

        self._ref_joint_vel_actuator_indexes = [
            self.sim.model.actuator_name2id(actuator)
            for actuator in self.sim.model.actuator_names
            if actuator.startswith("vel")
        ]

        if self.has_gripper:
            # self._ref_joint_gripper_actuator_indexes = [
            #     self.sim.model.actuator_name2id(actuator)
            #     for actuator in self.sim.model.actuator_names
            #     if actuator.startswith("gripper")
            # ]

            ''' Fix the bug for above. If there are joints before the robot and gripper, for example 3 joints
            before 7 joints of the robot and 2 joints of the gripper, then only the modified one below works correct.
            The reason is that the joints on the robot and gripper are also called actuators, but those 3 joints may 
            not be actuators, therefore the counting in actuators will not consider the 3, and causing an index mismatch.
            '''  
            self._ref_joint_gripper_actuator_indexes =[self.sim.model.joint_name2id(joint)
                for joint in self.sim.model.joint_names
                if joint.startswith("finger")]

        # IDs of sites for gripper visualization
        self.eef_site_id = self.sim.model.site_name2id("grip_site")
        self.eef_cylinder_id = self.sim.model.site_name2id("grip_site_cylinder")

    def move_indicator(self, pos):
        """
        Sets 3d position of indicator object to @pos.
        """
        if self.use_indicator_object:
            index = self._ref_indicator_pos_low
            self.sim.data.qpos[index : index + 3] = pos

    def _pre_action(self, action, rescale=False):
        """
        Overrides the superclass method to actuate the robot with the
        passed joint velocities and gripper control.

        Args:
            action (numpy array): The control to apply to the robot. The first
                @self.mujoco_robot.dof dimensions should be the desired
                normalized joint velocities and if the robot has
                a gripper, the next @self.gripper.dof dimensions should be
                actuation controls for the gripper.
        """

        # clip actions into valid range
        assert len(action) == self.dof, "environment got invalid action dimension"
        low, high = self.action_spec
        action = np.clip(action, low, high)

        if self.has_gripper:
            arm_action = action[: self.mujoco_robot.dof]
            gripper_action_in = action[
                self.mujoco_robot.dof : self.mujoco_robot.dof + self.gripper.dof
            ]
            gripper_action_actual = self.gripper.format_action(gripper_action_in)
            action = np.concatenate([arm_action, gripper_action_actual])

        if rescale:
            # for sim-only usage: rescale normalized action to control ranges
            ctrl_range = self.sim.model.actuator_ctrlrange
            bias = 0.5 * (ctrl_range[:, 1] + ctrl_range[:, 0])
            weight = 0.5 * (ctrl_range[:, 1] - ctrl_range[:, 0])
            bias[-2:] = 2*[0.]  # modified: the bias for gripper shoule be 0. when using velocity control, zero action corresponds to zero velocity
            applied_action = bias + weight * action

        else:
            # for sim2real usage: do not rescale to keep the input action, this ensures the actions from policy to be straightforwardly applied in reality
            applied_action = action  

        # Two gripper control modes:
        # 1. Control the gripper with action directly being position
        # self.sim.data.ctrl[:] = applied_action  # comment the bias[-2:]=2*[0.] above to set zero action correspond to half of position range.

        # 2. Control the gripper with action being position change of fingers
        self.sim.data.ctrl[:-2] = applied_action[:-2]
        self.sim.data.ctrl[-2:] += applied_action[-2:]  # velocity range for gripper equals to weight, i.e. half of position range for each gripper finger

        # gravity compensation
        self.sim.data.qfrc_applied[
            self._ref_joint_vel_indexes
        ] = self.sim.data.qfrc_bias[self._ref_joint_vel_indexes]

        if self.use_indicator_object:
            self.sim.data.qfrc_applied[
                self._ref_indicator_vel_low : self._ref_indicator_vel_high
            ] = self.sim.data.qfrc_bias[
                self._ref_indicator_vel_low : self._ref_indicator_vel_high
            ]

    def _post_action(self, action):
        """
        (Optional) does gripper visualization after actions.
        """
        ret = super()._post_action(action)
        self._gripper_visualization()
        return ret

    def _get_observation(self):
        """
        Returns an OrderedDict containing observations [(name_string, np.array), ...].

        Important keys:
            robot-state: contains robot-centric information.
        """

        di = super()._get_observation()
        # proprioceptive features
        di["joint_pos"] = np.array(
            [self.sim.data.qpos[x] for x in self._ref_joint_pos_indexes]
        )
        di["joint_vel"] = np.array(
            [self.sim.data.qvel[x] for x in self._ref_joint_vel_indexes]
        )

        robot_states = [
            np.sin(di["joint_pos"]),
            np.cos(di["joint_pos"]),
            di["joint_vel"],
        ]

        if self.has_gripper:
            di["gripper_qpos"] = np.array(
                [self.sim.data.qpos[x] for x in self._ref_gripper_joint_pos_indexes]
            )
            di["gripper_qvel"] = np.array(
                [self.sim.data.qvel[x] for x in self._ref_gripper_joint_vel_indexes]
            )

            di["eef_pos"] = np.array(self.sim.data.site_xpos[self.eef_site_id])
            di["eef_quat"] = T.convert_quat(
                self.sim.data.get_body_xquat("right_hand"), to="xyzw"
            )

            # add in gripper information
            robot_states.extend([di["gripper_qpos"], di["eef_pos"], di["eef_quat"]])

        di["robot-state"] = np.concatenate(robot_states)
        return di

    @property
    def action_spec(self):
        """
        Action lower/upper limits per dimension.
        """
        low = np.ones(self.dof, dtype=np.float32) * -1.
        high = np.ones(self.dof, dtype=np.float32) * 1.
        return low, high

    @property
    def dof(self):
        """
        Returns the DoF of the robot (with grippers).
        """
        dof = self.mujoco_robot.dof
        if self.has_gripper:
            dof += self.gripper.dof
        return dof

    def pose_in_base(self, pose_in_world):
        """
        A helper function that takes in a pose in world frame and returns that pose in  the
        the base frame.
        """
        base_pos_in_world = self.sim.data.get_body_xpos("base")
        base_rot_in_world = self.sim.data.get_body_xmat("base").reshape((3, 3))
        base_pose_in_world = T.make_pose(base_pos_in_world, base_rot_in_world)
        world_pose_in_base = T.pose_inv(base_pose_in_world)

        pose_in_base = T.pose_in_A_to_pose_in_B(pose_in_world, world_pose_in_base)
        return pose_in_base

    def pose_in_base_from_name(self, name):
        """
        A helper function that takes in a named data field and returns the pose
        of that object in the base frame.
        """

        pos_in_world = self.sim.data.get_body_xpos(name)
        rot_in_world = self.sim.data.get_body_xmat(name).reshape((3, 3))
        pose_in_world = T.make_pose(pos_in_world, rot_in_world)

        base_pos_in_world = self.sim.data.get_body_xpos("base")
        base_rot_in_world = self.sim.data.get_body_xmat("base").reshape((3, 3))
        base_pose_in_world = T.make_pose(base_pos_in_world, base_rot_in_world)
        world_pose_in_base = T.pose_inv(base_pose_in_world)

        pose_in_base = T.pose_in_A_to_pose_in_B(pose_in_world, world_pose_in_base)
        return pose_in_base

    def set_robot_joint_positions(self, jpos):
        """
        Helper method to force robot joint positions to the passed values.
        """
        self.sim.data.qpos[self._ref_joint_pos_indexes] = jpos
        self.sim.forward()

    @property
    def _right_hand_joint_cartesian_pose(self):
        """
        Returns the cartesian pose of the last robot joint in base frame of robot.
        """
        return self.pose_in_base_from_name("right_l6")

    @property
    def _right_hand_pose(self):
        """
        Returns eef pose in base frame of robot.
        """
        return self.pose_in_base_from_name("right_hand")

    @property
    def _right_hand_quat(self):
        """
        Returns eef quaternion in base frame of robot.
        """
        return T.mat2quat(self._right_hand_orn)

    @property
    def _right_hand_total_velocity(self):
        """
        Returns the total eef velocity (linear + angular) in the base frame
        as a numpy array of shape (6,)
        """

        # Use jacobian to translate joint velocities to end effector velocities.
        Jp = self.sim.data.get_body_jacp("right_hand").reshape((3, -1))
        Jp_joint = Jp[:, self._ref_joint_vel_indexes]

        Jr = self.sim.data.get_body_jacr("right_hand").reshape((3, -1))
        Jr_joint = Jr[:, self._ref_joint_vel_indexes]

        eef_lin_vel = Jp_joint.dot(self._joint_velocities)
        eef_rot_vel = Jr_joint.dot(self._joint_velocities)
        return np.concatenate([eef_lin_vel, eef_rot_vel])

    @property
    def _right_hand_pos(self):
        """
        Returns position of eef in base frame of robot.
        """
        eef_pose_in_base = self._right_hand_pose
        return eef_pose_in_base[:3, 3]

    @property
    def _right_hand_orn(self):
        """
        Returns orientation of eef in base frame of robot as a rotation matrix.
        """
        eef_pose_in_base = self._right_hand_pose
        return eef_pose_in_base[:3, :3]

    @property
    def _right_hand_vel(self):
        """
        Returns velocity of eef in base frame of robot.
        """
        return self._right_hand_total_velocity[:3]

    @property
    def _right_hand_ang_vel(self):
        """
        Returns angular velocity of eef in base frame of robot.
        """
        return self._right_hand_total_velocity[3:]

    @property
    def _joint_positions(self):
        """
        Returns a numpy array of joint positions.
        Panda robots have 7 joints and positions are in rotation angles.
        """
        return self.sim.data.qpos[self._ref_joint_pos_indexes]

    @property
    def _joint_ranges(self):
        return self.sim.model.jnt_range[self._ref_joint_pos_indexes]

    @property
    def _joint_velocities(self):
        """
        Returns a numpy array of joint velocities.
        Panda robots have 7 joints and velocities are angular velocities.
        """
        return self.sim.data.qvel[self._ref_joint_vel_indexes]

    def _gripper_visualization(self):
        """
        Do any needed visualization here.
        """

        # By default, don't do any coloring.
        self.sim.model.site_rgba[self.eef_site_id] = [0., 0., 0., 0.]

    def _check_contact(self):
        """
        Returns True if the gripper is in contact with another object.
        """
        return False
