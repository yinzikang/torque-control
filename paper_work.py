import mujoco_py as mp
import numpy as np
from funcmini import rbt
import PyKDL as kdl
import csv
import matplotlib.pyplot as plt

PI = 3.1415926


def orientation_error(desired, current):
    rc1 = current[0:3, 0]
    rc2 = current[0:3, 1]
    rc3 = current[0:3, 2]
    rd1 = desired[0:3, 0]
    rd2 = desired[0:3, 1]
    rd3 = desired[0:3, 2]
    error = 0.5 * (np.cross(rc1, rd1) + np.cross(rc2, rd2) + np.cross(rc3, rd3))
    return error


model = mp.load_model_from_path('./jk5mini.xml')
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

duration = 4000
eef_name = "ee"
# desired_pos = np.concatenate((np.linspace(-0.3, -0.75, duration).reshape(duration, 1),
#                               -0.1135 * np.ones((duration, 1), dtype=float),
#                               0.05 * np.ones((duration, 1), dtype=float)),
#                              axis=1)
desired_pos = np.concatenate((-0.45 * np.ones((duration, 1), dtype=float),
                              -0.1135 * np.ones((duration, 1), dtype=float),
                              0.05 * np.ones((duration, 1), dtype=float)),
                             axis=1)
# desired_pos = np.array(sim.data.get_site_xpos(eef_name).copy())
desired_ori = np.array(sim.data.site_xmat[sim.model.site_name2id(eef_name)].reshape([3, 3]).copy())
# desired_pos_vel = np.array([-0.4 / duration, 0, 0])
desired_pos_vel = np.array([0, 0, 0])
desired_ori_vel = np.array([0, 0, 0])
desired_pos_acc = np.array([0, 0, 0])
desired_ori_acc = np.array([0, 0, 0])
desired_acc = np.concatenate([desired_pos_acc, desired_ori_acc])

h_old = sim.data.qfrc_bias[:]
tau_old = sim.data.qfrc_bias[:]
qdd_old = np.array([0, 0, 0, 0, 0, 0])
fext_old = np.array([0, 0, 0, 0, 0, 0])

update_rate = 0
omega = 0
F_desire = np.array([0, 0, -20, 0, 0, 0])

B_x = [500, 500, 40, 500, 500, 500]  # 速度
# B_x = [500, 500, 400, 4000, 4000, 4000]
B_x = np.array(B_x, dtype=np.float32)
# K_x = [500, 500, 500, 5000, 5000, 5000]  # 位置
K_x = [300, 300, 100, 1000, 1000, 1000]
K_x = np.array(K_x, dtype=np.float32)
M_x = np.eye(6)
M_x = np.array(M_x, dtype=np.float32)
M_x_inv = np.linalg.inv(M_x)

robot = rbt(sim)
buffer = []
ee_pos_buffer = []
fext_buffer = []
headers = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
id = sim.model.body_name2id('tip')

# J = robot.jacobian()
# J_inv = np.linalg.inv(J)
# J_T_inv = np.linalg.inv(J.T)
# D_q = robot.mass_matrix()
# D_x = np.dot(J_T_inv, np.dot(D_q, J_inv))

for step in range(duration):
    J = robot.jacobian()
    J_inv = np.linalg.inv(J)
    J_T_inv = np.linalg.inv(J.T)
    Jd2 = robot.jacobian_dot2()

    # 机器人状态
    q = np.array(sim.data.qpos[:])
    qd = np.array(sim.data.qvel[:])
    qdd = np.array(sim.data.qacc[:])
    # print("qdd",qdd)
    x_pos = np.array(sim.data.get_site_xpos(eef_name))
    # print(x_pos[2])
    x_ori = np.array(sim.data.site_xmat[sim.model.site_name2id(eef_name)].reshape([3, 3]))
    x_pos_vel = np.array(sim.data.site_xvelp[sim.model.site_name2id(eef_name)])
    x_ori_vel = np.array(sim.data.site_xvelr[sim.model.site_name2id(eef_name)])
    x_error = np.concatenate([desired_pos[step] - x_pos, orientation_error(desired_ori, x_ori)])
    # x_error = np.concatenate([desired_pos - x_pos, orientation_error(desired_ori, x_ori)])
    xd_error = np.concatenate([desired_pos_vel - x_pos_vel, desired_ori_vel - x_ori_vel])
    xdd_error = np.concatenate([desired_pos_acc, desired_ori_acc])
    fext = np.array(sim.data.sensordata[:])
    # pos_error = desired_pos[step] - x_pos
    pos_error = desired_pos - x_pos
    # ori_error = orientation_error(desired_ori, x_ori)
    # print(x_pos[2])
    ee_pos_buffer.append([x_pos[i] for i in range(3)])
    fext_buffer.append(fext)

    # 最简单的力矩设置
    # D_q = robot.mass_matrix()
    # tau = np.dot(D_q, xdd_error) + sim.data.qfrc_bias[:]
    # D_q = robot.mass_matrix()
    # D_x = np.dot(J_T_inv, np.dot(D_q, J_inv))
    # h_x = np.dot(J_T_inv, sim.data.qfrc_bias[:]) - np.dot(np.dot(D_x, Jd2), qd)
    # F = np.dot(D_x, xdd_error) + h_x
    # tau = np.dot(J.T, F)
    # tau = sim.data.qfrc_bias[:]

    # 师兄的阻抗控制力矩计算
    # D_q = robot.mass_matrix()
    # # D_q = sim.data.qM
    # Md_x = robot.mass_desired()
    # Md_x_inv = np.linalg.inv(Md_x)
    # tau = sim.data.qfrc_bias[:]
    # coef = np.dot(D_q, J_inv)
    # # coef = np.dot(np.dot(M_x, J_inv), Md_x_inv)
    # sum = np.multiply(B_x, xd_error)
    # sum += np.multiply(K_x, x_error)
    # sum -= np.dot(np.dot(Md_x, Jd2), qd)
    # tau += np.dot(coef, sum)
    # # tau += np.dot(coef, np.dot(Md_x, sum))

    # paper中不稳定的阻抗控制力跟踪, M_x, B_x, K_x
    # D_q = robot.mass_matrix()
    # D_x = np.dot(J_T_inv, np.dot(D_q, J_inv))
    # h_x = np.dot(J_T_inv, sim.data.qfrc_bias[:]) - np.dot(np.dot(D_x, Jd2), qd)
    # # T = np.multiply(B_x, xd_error) + np.multiply(K_x, x_error) + F_desire + fext
    # T = np.multiply(B_x, xd_error) + np.multiply(K_x, x_error) + F_desire
    # V = desired_acc + np.dot(M_x_inv, T)
    #
    # F = np.dot(D_x, V) + h_x + fext
    # tau = np.dot(J.T, F)

    # paper中不稳定的力跟踪阻抗控制算法, M_x, B_x, K_x，different control parameter in different occasions
    D_q = robot.mass_matrix()
    D_x = np.dot(J_T_inv, np.dot(D_q, J_inv))
    h_x = np.dot(J_T_inv, sim.data.qfrc_bias[:]) - np.dot(np.dot(D_x, Jd2), qd)
    # h_x = np.dot(J_T_inv, sim.data.qfrc_bias[:])
    if np.all(fext == 0):
        print("free space")
        K_x = [300, 300, 100, 1000, 1000, 1000]
        T = np.multiply(B_x, xd_error) + np.multiply(K_x, x_error) + F_desire
    else:
        print("contact space", fext[2])
        K_x = [100, 100, 0, 1000, 1000, 1000]
        T = np.multiply(B_x, xd_error) + np.multiply(K_x, x_error) + F_desire + fext
    V = xdd_error + np.dot(M_x_inv, T)
    F = np.dot(D_x, V) + h_x - fext
    tau = np.dot(J.T, F)

    # paper中稳定的力跟踪阻抗控制算法, M_x, B_x, K_x, import robust control
    # D_q = robot.mass_matrix()
    # D_x = np.dot(J_T_inv, np.dot(D_q, J_inv))
    # if np.all(fext == 0):
    #     print("free space")
    #     # K_x = [500, 500, 50, 5000, 5000, 5000]
    #     K_x = [300, 300, 100, 100, 100, 100]
    #     T = np.multiply(B_x, xd_error) + np.multiply(K_x, x_error) + F_desire
    #     # T[2] = B_x[2] * xd_error[2] + B_x[2] * omega + F_desire[2]
    # else:
    #     print("contact space", fext[2])
    #     K_x = [100, 100, 0, 100, 100, 100]
    #     # K_x = [100, 100, 0, 300, 300, 300]
    #     # omega += update_rate * (F_desire[2] - fext_old[2]) / B_x
    #     T = np.multiply(B_x, xd_error) + np.multiply(K_x, x_error) + F_desire + fext
    #     # T[2] = B_x[2] * xd_error[2] + B_x[2] * omega + F_desire[2] + fext[2]
    #     # T = np.multiply(B_x + omega, xd_error) + np.multiply(K_x, x_error) + F_desire - fext
    # V = xdd_error + np.dot(M_x_inv, T)
    # U = np.dot(J_inv, V - np.dot(Jd2, qd))
    # tau = np.dot(D_q, U) + tau_old - np.dot(D_q, qdd_old) - np.dot(J.T, fext) + np.dot(J.T, fext_old)

    # tau = np.dot(D_q, U) + tau_old - np.dot(D_q, qdd_old)

    # h_old = tau - np.dot(D_q, qdd) - np.dot(J.T, fext)
    # tau_old = tau
    # qdd_old = qdd
    # fext_old = fext

    # if step > 3000:
    #     F_desire = np.array([0, 0, -10, 0, 0, 0])

    sim.data.ctrl[:] = tau
    sim.step()
    buffer.append([fext[2]])
    # buffer.append([x_pos[2]])
    viewer.render()

# plt.plot(range(duration), buffer)
# plt.show()
ee_pos_buffer = np.array(ee_pos_buffer)
fext_buffer = np.array(fext_buffer)

# plt.title("X pos")
# plt.xlabel("time/ms")
# plt.ylabel("pos/m")
# plt.plot(np.arange(0, duration - 2), ee_pos_buffer[1:duration - 1, 0])
# plt.show()
# plt.title("Y pos")
# plt.xlabel("time/ms")
# plt.ylabel("pos/m")
# plt.plot(np.arange(0, duration - 2), ee_pos_buffer[1:duration - 1, 1])
# plt.show()
# plt.title("Z pos")
# plt.xlabel("time/ms")
# plt.ylabel("pos/m")
# plt.plot(np.arange(0, duration - 2), ee_pos_buffer[1:duration - 1, 2])
# plt.show()
#
plt.title("X force")
plt.xlabel("time/ms")
plt.ylabel("force/N")
plt.plot(np.arange(0, duration - 2), fext_buffer[1:duration - 1, 0])
plt.show()
plt.title("Y force")
plt.xlabel("time/ms")
plt.ylabel("force/N")
plt.plot(np.arange(0, duration - 2), fext_buffer[1:duration - 1, 1])
plt.show()
plt.title("Z force")
plt.xlabel("time/ms")
plt.ylabel("force/N")
plt.plot(np.arange(0, duration - 2), fext_buffer[1:duration - 1, 2])
plt.show()
#
# plt.title("X force VS pos")
# plt.xlabel("pos/m")
# plt.ylabel("force/N")
# plt.plot(ee_pos_buffer[1:duration - 1, 0], fext_buffer[1:duration - 1, 0])
# plt.show()
# plt.title("Y force VS pos")
# plt.xlabel("pos/m")
# plt.ylabel("force/N")
# plt.plot(ee_pos_buffer[1:duration - 1, 1], fext_buffer[1:duration - 1, 1])
# plt.show()
# plt.title("Z force VS pos")
# plt.xlabel("pos/m")
# plt.ylabel("force/N")
# plt.plot(ee_pos_buffer[1:duration - 1, 2], fext_buffer[1:duration - 1, 2])
# plt.show()
