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
    'joint2': 0,
    'joint3': 70,
    'joint4': 20,
    'joint5': -90,
    'joint6': 0,
}

for name, value in initial_qpos.items():
    sim.data.set_joint_qpos(name, value * PI / 180)
sim.forward()

qvel_index = [0, 1, 2, 3, 4, 5]
eef_name = "ee"
desired_pos = np.array(sim.data.get_site_xpos(eef_name).copy())
desired_ori = np.array(
    sim.data.site_xmat[sim.model.site_name2id(eef_name)].reshape([3, 3]).copy())
# desired_ori = np.array([[0,-1,0], [-1,0,0], [0,0,-1]])

robot = rbt(sim)

# 设定阻抗参数矩阵
Md = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)  # 质量
Md = np.diag(Md)

D = np.array([100, 100, 10, 100, 100, 100], dtype=np.float32)   # 阻尼
D = np.diag(D*100)

K = np.array([1000, 1000, 100, 1000, 1000, 1000], dtype=np.float32)   # 刚度
K = np.diag(K)

Md_inv = np.linalg.inv(Md)


def orientation_error(desired, current):
    rc1 = current[0:3, 0]
    rc2 = current[0:3, 1]
    rc3 = current[0:3, 2]
    rd1 = desired[0:3, 0]
    rd2 = desired[0:3, 1]
    rd3 = desired[0:3, 2]
    error = 0.5 * (np.cross(rc1, rd1) +
                   np.cross(rc2, rd2) + np.cross(rc3, rd3))
    return error


buffer = []
headers = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']

for i in range(9000):
    if 500 <= i < 5000:
        id = sim.model.body_name2id('tool')
        sim.data.xfrc_applied[id] = [0, 0, -100, 0, 0, 0]
    else:
        id = sim.model.body_name2id('tool')
        sim.data.xfrc_applied[id] = [0, 0, 0, 0, 0, 0]

    Fext = sim.data.xfrc_applied[id]

    qd = np.array(sim.data.qvel[:])

    # 获取机器人质量矩阵
    M = robot.mass_matrix()
    # M_inv = np.linalg.inv(M)
    J = robot.jacobian()
    J_transpose = np.transpose(J)
    # J2 = robot.jacobian2()
    J_inv = np.linalg.inv(J)
    J_inv_transpose = np.transpose(J_inv)
    # Jd = robot.jacobian_dot()
    Jd2 = robot.jacobian_dot2()

    x_pos = np.array(sim.data.get_site_xpos(eef_name))
    x_mat = np.array(
        sim.data.site_xmat[sim.model.site_name2id(eef_name)].reshape([3, 3]))
    x_pos_vel = np.array(sim.data.site_xvelp[sim.model.site_name2id(eef_name)])
    x_ori_vel = np.array(sim.data.site_xvelr[sim.model.site_name2id(eef_name)])

    '''======== 计算位置、速度误差 ========'''
    # 速度误差
    xd_error = np.concatenate([-x_pos_vel, -x_ori_vel])
    # 位置误差
    pos_error = desired_pos - x_pos
    ori_error = orientation_error(desired_ori, x_mat)
    x_error = np.concatenate([pos_error, ori_error])

    '''======== 计算力矩 ========'''
    # 重力和科氏力补偿
    tau = sim.data.qfrc_bias[:]
    # 质量相关项
    coef1 = np.dot(np.dot(M, J_inv), Md_inv)
    tmp1 = np.dot(K, xd_error) + np.dot(D, x_error) - \
        np.dot(np.dot(Md, Jd2), qd)
    Lambda = robot.mass_desired()
    tau += np.dot(coef1, tmp1)
    # # 外力相关项
    # # coef2 = J_transpose - np.dot(np.dot(M, J_inv), Md_inv)
    # # tau += np.dot(coef2, Fext)

    sim.data.ctrl[:] = tau
    sim.step()
    # buffer.append([sim.data.qpos[i] for i in range(6)])
    buffer.append(x_pos[2])
    viewer.render()

# with open('ipd_K100_D100.csv', 'w') as f:
#     f_csv = csv.writer(f)
#     f_csv.writerow(headers)
#     for buf in buffer:
#         f_csv.writerow(buf)
