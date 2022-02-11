"""
Panda inverse kinematics using PyKDL, core by Zihan Ding
Currently only position control is supported. (that means we have to pass a fixed pose for the end effector)
When 3-d space is accomodated, the dof is 3. When 2-d space is accomodated, a separate fixed z coordinate should be provided.
If limit_range is provided as [[xl, xr], [yl, yr]], the ik wrapper will try hard to limit the robot arm within the plane range.
"""

import numpy as np
from . import change_dof
from robosuite.kdl.panda_eef_velocity_controller import PandaEEFVelocityController

def panda_ik_wrapper(Env, fix_z=None, max_action=0.1, pose_mat=None, limit_range=None):
    ik_dof = 2 if fix_z is not None else 3
    wrapping_dof = (Env.dof - 7) + ik_dof
    
    class PandaIK(change_dof(Env, wrapping_dof)):
        parameters_spec = {
            **Env.parameters_spec,
            'pandaik_z_proportional_gain': [1.5, 2.5],
        } if fix_z is not None else Env.parameters_spec

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.controller = PandaEEFVelocityController()
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
            return ret

        def reset_props(self, pandaik_z_proportional_gain=2.0, **kwargs):
            self.z_prop_gain = pandaik_z_proportional_gain
            super().reset_props(**kwargs)

        # action range: [-1, 1]
        def step(self, action_all):
            assert(action_all.shape == (self.dof, ))
            action = action_all[:ik_dof]
            action_other = action_all[ik_dof:]
            
            current_pos = self._right_hand_pos
            current_pos_eef = self.reference_right_hand_orn.dot(current_pos)
            current_orn = self._right_hand_orn
            reference_orn = self.reference_right_hand_orn
            current_joint_angles = np.array(self._joint_positions)
            
            action = np.clip(action, np.ones(ik_dof) * -1, np.ones(ik_dof) * 1) * max_action
            if fix_z is not None:
                z_error = current_pos[2] - fix_z
                # print('fix_z info', current_pos, z_error)
                action = np.append(action, [z_error * self.z_prop_gain])
            if limit_range is not None:
                # print('limit range info', current_pos, action)
                action = action.copy()
                action[1] *= -1.   # GZZ: it's strange, but the action on the second dimension have reversed effect.
                for k in range(2):
                    if current_pos[k] < limit_range[k][0] and action[k] < 0:
                        action[k] = 0
                    if current_pos[k] > limit_range[k][1] and action[k] > 0:
                        action[k] = 0
                # print('limit range info after', action)
                action[1] *= -1.
                
            orn_diff = reference_orn.dot(current_orn.T)
            orn_diff_twice = orn_diff.dot(orn_diff).dot(orn_diff)

            xyz_vel_base = reference_orn.T.dot(action)
            pose_matrix = np.zeros((4, 4))
            pose_matrix[:3, :3] = orn_diff_twice
            pose_matrix[:3, 3] = xyz_vel_base

            joint_vel = self.controller.compute_joint_velocities_for_endpoint_velocity(pose_matrix, current_joint_angles)
            final_action = np.concatenate([np.asarray(joint_vel).squeeze(), action_other])

            return super().step(final_action)

    return PandaIK
