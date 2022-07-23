import numpy as np
import mujoco_py as mp

model = mp.load_model_from_path('./jk5.xml')
sim = mp.MjSim(model)
viewer = mp.MjViewer(sim)

qpos_init = np.array([0, 0, 0, 30, 0, 0]) / 180 * np.pi
while True:
    for i in range(6):
        sim.data.qpos[i] = qpos_init[i]
    print(sim.data.qpos)
    sim.step()
    viewer.render()

