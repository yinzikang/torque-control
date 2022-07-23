import mujoco_py as mp
import numpy as np
from func import rbt
import PyKDL as kdl
import csv

PI = 3.1415926

t = kdl.JntSpaceInertiaMatrix()
model = mp.load_model_from_path('./jk5.xml')
sim = mp.MjSim(model)
viewer = mp.MjViewer(sim)

initial_qpos = {
    'joint1': 0,
    'joint2': -30,
    'joint3': 120,
    'joint4': 0,
    'joint5': -90,
    'joint6': 0,
}

for name, value in initial_qpos.items():
    sim.data.set_joint_qpos(name, value * PI / 180)
sim.forward()

qvel_index = [0,1,2,3,4,5]
eef_name = "ee"
desired_pos = np.array(sim.data.get_site_xpos(eef_name).copy())
desired_ori = np.array(sim.data.site_xmat[sim.model.site_name2id(eef_name)].reshape([3, 3]).copy())
# desired_ori = np.array([[0,-1,0], [-1,0,0], [0,0,-1]])

robot = rbt(sim)

K = [1000,1000,100,1000,1000,1000] # 速度
K = np.array(K, dtype=np.float32)
D = [1000,1000,100,1000,1000,1000] # 位置
D = np.array(D, dtype=np.float32)


def orientation_error(desired, current):
    rc1 = current[0:3, 0]
    rc2 = current[0:3, 1]
    rc3 = current[0:3, 2]
    rd1 = desired[0:3, 0]
    rd2 = desired[0:3, 1]
    rd3 = desired[0:3, 2]
    error = 0.5 * (np.cross(rc1, rd1) + np.cross(rc2, rd2) + np.cross(rc3, rd3))
    return error


buffer = []
headers = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
for i in range(50000):
    id = sim.model.body_name2id('link6')
    sim.data.xfrc_applied[id][2] = -10
    # if 500 <= i < 1000:
    #     id = sim.model.body_name2id('link6')
    #     sim.data.xfrc_applied[id][2] = -50
    # else:
    #     id = sim.model.body_name2id('link6')
    #     sim.data.xfrc_applied[id][0] = 0
    M = robot.mass_matrix()
    # M2 = robot.mass_matrix2()
    qd = np.array(sim.data.qvel[:])
    # Cq1 = np.add(robot.coriolis(), robot.gravity_torque())
    # Cq2 = sim.data.qfrc_bias[:]
    # print(Cq1)
    # print(Cq2)
    # print(np.linalg.norm(Cq1 - Cq2))
    J = robot.jacobian()
    # J2 = robot.jacobian2()
    J_inv = np.linalg.inv(J)
    # Jd = robot.jacobian_dot()
    Jd2 = robot.jacobian_dot2()
    Md = robot.mass_desired()
    Md_inv = np.linalg.inv(Md)
    x_pos = np.array(sim.data.get_site_xpos(eef_name))
    x_mat = np.array(sim.data.site_xmat[sim.model.site_name2id(eef_name)].reshape([3, 3]))
    x_pos_vel = np.array(sim.data.site_xvelp[sim.model.site_name2id(eef_name)])
    x_ori_vel = np.array(sim.data.site_xvelr[sim.model.site_name2id(eef_name)])

    tau = sim.data.qfrc_bias[:]

    coef = np.dot(np.dot(M, J_inv), Md_inv)
    # coef = J.transpose()
    xd_error = np.concatenate([-x_pos_vel, -x_ori_vel])
    sum = np.multiply(K, xd_error)

    pos_error = desired_pos - x_pos
    ori_error = orientation_error(desired_ori, x_mat)
    x_error = np.concatenate([pos_error, ori_error])
    # print(x_error)
    sum += np.multiply(D, x_error)

    sum -= np.dot(np.dot(Md, Jd2), qd)
    tau += np.dot(coef, np.dot(Md, sum))

    sim.data.ctrl[:] = tau
    sim.step()
    buffer.append([sim.data.qpos[i] for i in range(6)])
    viewer.render()

    print(sim.data.sensordata)
    # print(sim.data.body_xpos[sim.model.body_name2id('tip')])

with open('ipd_K100_D100.csv', 'w') as f:
    f_csv = csv.writer(f)
    f_csv.writerow(headers)
    for buf in buffer:
        f_csv.writerow(buf)


