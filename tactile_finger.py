import numpy as np
from robosuite.utils.mjcf_utils import xml_path_completion
from robosuite.models.grippers.gripper import Gripper
from robosuite.models.objects import FullyFrictionalBoxObject, BoxObject
from robosuite.environments import MujocoEnv
from robosuite.models.tasks import Task
from collections import OrderedDict
from robosuite.utils.mjcf_utils import new_joint, array_to_string


class TactileFinger(Gripper):

    def __init__(self):
        super().__init__(xml_path_completion("grippers/tactile_finger.xml"))

    @property
    def visualization_sites(self):
        return ["grip_site", "grip_site_cylinder"]

    def contact_geoms(self):
        return [
            "hand_collision",
            "finger1_collision",
            "finger1_tip_collision",
        ]

    @property
    def left_finger_geoms(self):
        return [
            "finger1_tip_collision",
        ]


class Model(Task):
    def __init__(self, finger, objects):
        super().__init__()
        self.merge(finger)
        self.merge_objects(objects)

    def merge_objects(self, mujoco_objects):
        """Adds physical objects to the MJCF model."""
        self.mujoco_objects = mujoco_objects
        self.objects = []  # xml manifestation
        self.targets = []  # xml manifestation
        self.max_horizontal_radius = 0

        for obj_name, obj_mjcf in mujoco_objects.items():
            self.merge_asset(obj_mjcf)
            # Load object
            obj = obj_mjcf.get_collision(name=obj_name, site=True)
            obj.append(new_joint(name=obj_name, type="free"))

            self.objects.append(obj)
            self.worldbody.append(obj)

            self.max_horizontal_radius = max(
                self.max_horizontal_radius, obj_mjcf.get_horizontal_radius()
            )

    def place_objects(self, distance):
        for i in range(len(self.objects)):
            self.objects[i].set("pos", array_to_string(
                np.array([0, 0.0292+0.0001*distance, 0.008])))  # 0.0128, 0.0256, 0.035 (0.006)


class TactileTestEnv(MujocoEnv):
    def __init__(self,
                 has_renderer=False,
                 has_offscreen_renderer=False,
                 render_collision_mesh=False,
                 render_visual_mesh=True,
                 control_freq=2,
                 horizon=1000,
                 ignore_done=True,
                 use_camera_obs=False,
                 camera_name="frontview",
                 camera_height=256,
                 camera_width=256,
                 camera_depth=False,
                 # task specific
                 distance=128,
                 weight=100,
                 ):
        self.distance = distance
        self.weight = weight

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

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()
        # half of length is the definition of size for box
        size_x, size_y, size_z = 0.01289/2, 0.0128/2, 0.01/2
        density = (self.weight+2.1)/1000/((size_x * size_y * size_z)
                                          * (2**3))  # weight of the base is 2.1 grams
        print("Cube Density: {}".format(density))

        cube = BoxObject(
            size=(size_x, size_y, size_z),
            friction=1.,
            # 61881.788 kg/(m^3), calculated with: (100+2.1)/1000 kg = 0.01289 * 0.0128 * height * density
            density=density,
            rgba=[0.8, 0, 0, 0.9],
        )

        self.mujoco_objects = OrderedDict([("cube", cube)])

        finger = TactileFinger()

        self.model = Model(finger, self.mujoco_objects)
        self.model.place_objects(self.distance)


if __name__ == '__main__':
    import time
    import numpy as np
    import os
    seed = 3
    np.random.seed(seed)

    # DISTANCE = [128, 256, 350]
    DISTANCE = [350, 222, 128]  # match with real measurement

    WEIGHTS = [50, 100, 200, 500]
    PATH = './data/tactile_sim3/'

    def print_contact_info(env):
        for i in range(env.sim.data.ncon):  # total number of contact: env.sim.data.ncon
            c = env.sim.data.contact[i]
            print('Contact {}: {} and {}'.format(i, env.sim.model.geom_id2name(
                c.geom1), env.sim.model.geom_id2name(c.geom2)))

    for w in WEIGHTS:
        for d in DISTANCE:
            env = TactileTestEnv(distance=d, weight=w)
            env.reset()
            env.model_timestep = 1  # slot down the simulation, default as 0.002
            # This will change the gravity in simulation. Ref: https://github.com/openai/mujoco-py/issues/459
            env.sim.model.opt.gravity[:] = [0, 0, -9.81]
            print(env.model_timestep, env.sim.model.opt.gravity)
            step = 0

            log = []
            while(step < 1000):
                print_contact_info(env)
                # Gives array of all sensorvalues: force tactile
                print(env.sim.data.sensordata[1::3])
                log.append(env.sim.data.sensordata[1::3])
                env.step(0)
                # env.render()
                step += 1
            path = PATH+'sim_{}_{}'.format(str(w), str(d))
            os.makedirs(path, exist_ok=True)
            np.save(path, log)
