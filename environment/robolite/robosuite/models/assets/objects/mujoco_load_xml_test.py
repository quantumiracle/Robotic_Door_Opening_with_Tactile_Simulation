import mujoco_py
from mujoco_py import load_model_from_path, MjSim, MjViewer
import os
# model = load_model_from_path('./cabinet.xml')
model = load_model_from_path('./soft_cuboid.xml')

sim = mujoco_py.MjSim(model)

viewer = MjViewer(sim)
t = 0
while True:
    t += 1
    sim.step()
    viewer.render()

