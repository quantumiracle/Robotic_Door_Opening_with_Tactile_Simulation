"""
Panda inverse kinematics using PyKDL, core by Zihan Ding
Currently only position control is supported. (that means we have to pass a fixed pose for the end effector)
When 3-d space is accomodated, the dof is 3. When 2-d space is accomodated, a separate fixed z coordinate should be provided.
If limit_range is provided as [[xl, xr], [yl, yr]], the ik wrapper will try hard to limit the robot arm within the plane range.
"""

import numpy as np
from . import change_dof
from mujoco_py import functions
from gym.envs.robotics.rotations import quat2euler, euler2quat, mat2euler, mat2quat, quat_mul, quat_conjugate

def mul_quat(quat1, quat2):
    quat1 = np.array(quat1)
    quat2 = np.array(quat2)
    res = np.array([1., 0., 0., 0.])
    functions.mju_mulQuat(res, quat1, quat2)
    return res

def from_matrix(matrix):

    is_single = False
    matrix = np.asarray(matrix, dtype=float)

    if matrix.ndim not in [2, 3] or matrix.shape[-2:] != (3, 3):
        raise ValueError("Expected `matrix` to have shape (3, 3) or "
                        "(N, 3, 3), got {}".format(matrix.shape))

    # If a single matrix is given, convert it to 3D 1 x 3 x 3 matrix but
    # set self._single to True so that we can return appropriate objects in
    # the `to_...` methods
    if matrix.shape == (3, 3):
        matrix = matrix.reshape((1, 3, 3))
        is_single = True

    num_rotations = matrix.shape[0]

    decision_matrix = np.empty((num_rotations, 4))
    decision_matrix[:, :3] = matrix.diagonal(axis1=1, axis2=2)
    decision_matrix[:, -1] = decision_matrix[:, :3].sum(axis=1)
    choices = decision_matrix.argmax(axis=1)

    quat = np.empty((num_rotations, 4))

    ind = np.nonzero(choices != 3)[0]
    i = choices[ind]
    j = (i + 1) % 3
    k = (j + 1) % 3

    quat[ind, i] = 1 - decision_matrix[ind, -1] + 2 * matrix[ind, i, i]
    quat[ind, j] = matrix[ind, j, i] + matrix[ind, i, j]
    quat[ind, k] = matrix[ind, k, i] + matrix[ind, i, k]
    quat[ind, 3] = matrix[ind, k, j] - matrix[ind, j, k]

    ind = np.nonzero(choices == 3)[0]
    quat[ind, 0] = matrix[ind, 2, 1] - matrix[ind, 1, 2]
    quat[ind, 1] = matrix[ind, 0, 2] - matrix[ind, 2, 0]
    quat[ind, 2] = matrix[ind, 1, 0] - matrix[ind, 0, 1]
    quat[ind, 3] = 1 + decision_matrix[ind, -1]

    quat /= np.linalg.norm(quat, axis=1)[:, None]

    if is_single:
        return quat[0]
        # return cls(quat[0], normalize=False, copy=False)
    else:
        # return cls(quat, normalize=False, copy=False)
        return quat

def trans2posquat(tform):
    pos = (tform[0:3, 3]).transpose()
    # a = R.identity(3)
    # R.from_rotvec()
    # r = R.from_matrix(tform[0:3, 0:3])
    # quat = r.as_quat()
    # quat = np.hstack((quat[3], quat[0:3]))
    # print('--------------')
    # print(quat)
    quat = from_matrix(tform[0:3, 0:3])
    quat = np.hstack((quat[3], quat[0:3]))
    # print(quat)
    return pos, quat

def quat2vel(quat, dt):
    quat = np.array(quat)
    res = np.array([0., 0., 0.])
    functions.mju_quat2Vel(res, quat, dt)
    return res

def conj_quat(quat):
    # w x y z
    quat = np.array(quat)
    res = np.array([1., 0., 0., 0.])
    functions.mju_negQuat(res, quat)
    return res

def panda_ik_simple_wrapper(Env, rotation = False, fix_z=None, gripper=False, max_action=1., pose_mat=None, limit_range=None, modify_dof=True):  # TODO limit_range
    ik_dof = 2 if fix_z is not None else 3
        
    if Env.dof > 7 and not gripper:  # when has the freedom of gripper but not intend to use it
        real_dof = Env.dof - 1
    else:
        real_dof = Env.dof

    if rotation is True:   # if allowing rotation control for EE, add 3 dims of euler for EE orientation
        ik_dof += 3
    if modify_dof: # only in very special case, this is set to be False, e.g. a FK control task but sometimes using IK
        wrapping_dof = (real_dof - 7) + ik_dof  # dof after IK wrapper, e.g. 7 -> 3, 8 -> 4
    else:
        wrapping_dof = real_dof


    class PandaIK(change_dof(Env, wrapping_dof)):
        parameters_spec = {
            **Env.parameters_spec,
            'pandaik_z_proportional_gain': [1.5, 2.5],
        } if fix_z is not None else Env.parameters_spec

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.z_prop_gain = 2.0
            self._set_reference_right_hand_orn()

        def _set_reference_right_hand_orn(self):
            if pose_mat is not None:
                self.reference_right_hand_orn = pose_mat
            else:
                self.reference_right_hand_orn = self._right_hand_orn.copy()

        def reset(self, **kwargs):
            ret = super().reset(**kwargs)
            self._set_reference_right_hand_orn()

            # added, move to initial position
            # print(Env.dof)
            # move_to_init_action = np.zeros(wrapping_dof)
            # move_to_init_action[2]  = -1
            # self.step(move_to_init_action)

            return ret

        def reset_props(self, pandaik_z_proportional_gain=2.0, **kwargs):
            self.z_prop_gain = pandaik_z_proportional_gain
            super().reset_props(**kwargs)

        def jac_geom(self, geom_name, first_joint_idx):
            jacp = self.sim.data.get_geom_jacp(geom_name)  # 3-dimensional position
            jacr = self.sim.data.get_geom_jacr(geom_name)  # 3-dimensional orientation
            jacp = jacp.reshape(3, -1)  
            jacr = jacr.reshape(3, -1)
            # print(np.vstack((jacp, jacr)))
            return np.vstack((jacp[:, first_joint_idx:first_joint_idx+7], jacr[:, first_joint_idx:first_joint_idx+7]))

        def get_geom_posquat(self, name):
            rot = self.sim.data.get_geom_xmat(name)
            tform = np.eye(4)
            tform[0:3, 0:3] = rot
            tform[0:3, 3] = self.sim.data.get_geom_xpos(name).transpose()
            pos, quat = trans2posquat(tform)
            return np.hstack((pos, quat))

        def step(self, action_all, ignore_IK=False):
            """
            @brief
                Take the input of all action, return a processed action.

                The first ik_dof dims of action are for EE control, which will be transformed into IK actions on 7 joints: 

                * If fixed_z, the first two dimensions are for x- and y-position;
                    if not, the first three dimensions are for x-, y- and z-position.

                * If rotation is True, the rest three dimensions (Euler) in ik_dof are for orientation control;
                    if it is False, the rotation is not controlled by action but trying to keep the same as the current;
                    if it is neither True or False, it is a list/array containing the specified rotation angles, which will be fixed.

                The other dimensions of input action are not for EE control, but for like gripper control, 
                which remains the same (no need for IK transformation).

            """
            if ignore_IK:
                return  super().step(action_all)
            else:
                try:
                    assert(action_all.shape == ((real_dof - 7) + ik_dof, ))
                except:
                    print('Action Shape Error')
                # print(action_all)
                action = action_all[:ik_dof]  # the first 2 or 3 dims are position, and the last 3 dims are orientation (euler)
                action_other = action_all[ik_dof:]  # gripper control, etc
                action = np.clip(action, np.ones(ik_dof) * -1, np.ones(ik_dof) * 1) * max_action   # action range: [-max_action, max_action]
                # ee_curr = self.get_geom_posquat("hand_visual")  # does not give a fully correct control: correct for position and x- and y-rotation, wrong for z-rotation
                curr_quat = mat2quat(self._right_hand_orn)
                ee_curr = np.concatenate([self._right_hand_pos, curr_quat])  # gives correct control for both opsition and orientation        

                if fix_z is not None:
                    z_error = fix_z - ee_curr[2]
                    action = np.concatenate([action, [5.*z_error]])  # add action to z-axis to construct a complete position action vector since input is of 2 dims: set a large proportional gain (5.) to quickly move to fix_z
                
                if rotation is False: # not control the rotation
                    ee_tget = np.concatenate([action+ee_curr[:-4], curr_quat])
                    # below gives same result as above
                    # action = np.concatenate([action, [0.,0.,0.]])  # add zero rotations to three axis to construct a complete action vector   
                    # ee_tget = action + np.concatenate([ee_curr[:-4], quat2euler(ee_curr[-4:])])
                    # ee_tget = np.concatenate([ee_tget[:-3], euler2quat(ee_tget[-3:])])  # change last 3 dims for orientation from euler to quaternion
                elif rotation is True: # with rotation control/action (3 dim)
                    current_euler = quat2euler(ee_curr[-4:])  # change last 3 dims for orientation from quaternion to euler
                    ee_tget = action + np.concatenate([ee_curr[:-4], current_euler])
                    ee_tget = np.concatenate([ee_tget[:-3], euler2quat(ee_tget[-3:])])  # change last 3 dims for orientation from euler to quaternion
                else:  # with a given fixed rotation, 3-dim euler
                    ee_tget = np.concatenate([action+ee_curr[:-4], euler2quat(rotation)])  # target orientation: gripper facing downwards
                first_joint_idx = self._ref_joint_gripper_actuator_indexes[0] - 7 # in different envs the first joint idx may be different

                ee_jac = self.jac_geom("hand_visual", first_joint_idx)  # get the jacobian w.r.t. a geom with its name
                # vel = np.hstack(((ee_tget[:3] - ee_curr[:3])/5.,    # divided by 5 to generate small action, but will affect the velocity in simulation
                vel = np.hstack(((ee_tget[:3] - ee_curr[:3]),  
                                    quat2vel(mul_quat(ee_tget[-4:], conj_quat(ee_curr[-4:])), 1)))    # difference of quaternions is calculated with  M^(-1)^T * N: inverse transpose is conjugate 
                qvel = np.matmul(np.linalg.pinv(ee_jac), vel.transpose())
                # print(ee_jac, ee_tget)
                
                final_action = np.concatenate([np.asarray(qvel).squeeze(), action_other])
                return super().step(final_action)

    return PandaIK
