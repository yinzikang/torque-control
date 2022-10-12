import matplotlib.pyplot as plt
import mujoco_py as mp
import numpy as np
from mujoco_py.generated import const
from funcmini import rbt


def orientation_error(desired, current):
    rc1 = current[0:3, 0]
    rc2 = current[0:3, 1]
    rc3 = current[0:3, 2]
    rd1 = desired[0:3, 0]
    rd2 = desired[0:3, 1]
    rd3 = desired[0:3, 2]
    error = 0.5 * (np.cross(rc1, rd1) + np.cross(rc2, rd2) + np.cross(rc3, rd3))
    return error


# robot model
model = mp.load_model_from_path('./jk5_table.xml')
sim = mp.MjSim(model)
robot = rbt(sim)
headers = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
id = sim.model.body_name2id('tip')
eef_name = "ee"

viewer = mp.MjViewer(sim)

# robot init
initial_qpos = {
    'joint1': 0,
    'joint2': -30,
    'joint3': 120,
    'joint4': 0,
    'joint5': -90,
    'joint6': 0,
}
for name, value in initial_qpos.items():
    sim.data.set_joint_qpos(name, value * const.PI / 180)
sim.forward()
h_old = np.array(sim.data.qfrc_bias[:])
tau_old = np.array(sim.data.qfrc_bias[:])
qdd_old = np.array([0, 0, 0, 0, 0, 0], dtype=np.float32)
fext_old = np.array([0, 0, 0, 0, 0, 0], dtype=np.float32)

# 期望轨迹
duration = 5000
# 移动
desired_pos = np.concatenate((np.linspace(-0.3, -0.75, duration).reshape(duration, 1),
                              -0.1135 * np.ones((duration, 1), dtype=float),
                              0.05 * np.ones((duration, 1), dtype=float)),
                             axis=1)

desired_ori = np.array(sim.data.site_xmat[sim.model.site_name2id(eef_name)].reshape([3, 3]).copy())
desired_pos_vel = np.array([-0.4 / duration, 0, 0])
# 定点
# desired_pos = np.concatenate((-0.45 * np.ones((duration, 1), dtype=float),
#                               -0.1135 * np.ones((duration, 1), dtype=float),
#                               0.1 * np.ones((duration, 1), dtype=float)),
#                              axis=1)
# desired_pos = np.array(sim.data.get_site_xpos(eef_name).copy())
# desired_pos_vel = np.array([0, 0, 0], dtype=np.float32)
desired_ori_vel = np.array([0, 0, 0], dtype=np.float32)
desired_pos_acc = np.array([0, 0, 0], dtype=np.float32)
desired_ori_acc = np.array([0, 0, 0], dtype=np.float32)
desired_acc = np.concatenate([desired_pos_acc, desired_ori_acc])

# 期望状态
F_desire = np.array([0, 0, 10, 0, 0, 0], dtype=np.float32)  # [0, 0, 10, 0, 0, 0]
# B_x = np.array([500, 500, 40, 500, 500, 500], dtype=np.float32)
# K_x = np.array([300, 300, 100, 100, 100, 100], dtype=np.float32)
B_x = np.array([500, 500, 40, 500, 500, 500], dtype=np.float32)
K_x = np.array([500, 500, 50, 1000, 1000, 1000], dtype=np.float32)
M_x = np.array(np.eye(6), dtype=np.float32)
M_x_inv = np.linalg.inv(M_x)

# 自适应与鲁棒参数
update_rate = 0.01
omega = 0.0

# buffer
ee_pos_buffer = []
fext_buffer = []

control_mode = 6
for step in range(duration):
    J = robot.jacobian()
    J_inv = np.linalg.inv(J)
    J_T_inv = np.linalg.inv(J.T)
    Jd2 = robot.jacobian_dot2()  # jacob的一介导数

    # 机器人状态
    q = np.array(sim.data.qpos[:])
    qd = np.array(sim.data.qvel[:])
    qdd = np.array(sim.data.qacc[:])
    # print("qdd", qdd)
    x_pos = np.array(sim.data.get_site_xpos(eef_name))
    # print(x_pos[2])
    x_ori = np.array(sim.data.site_xmat[sim.model.site_name2id(eef_name)].reshape([3, 3]))
    x_pos_vel = np.array(sim.data.site_xvelp[sim.model.site_name2id(eef_name)])
    x_ori_vel = np.array(sim.data.site_xvelr[sim.model.site_name2id(eef_name)])
    x_acc = np.dot(Jd2, qd) + np.dot(J, qdd)
    x_error = np.concatenate([desired_pos[step] - x_pos, orientation_error(desired_ori, x_ori)])
    xd_error = np.concatenate([desired_pos_vel - x_pos_vel, desired_ori_vel - x_ori_vel])
    xdd_error = np.concatenate([desired_pos_acc, desired_ori_acc]) - x_acc  # 未验证
    fext = np.array(sim.data.sensordata[:])
    pos_error = desired_pos[step] - x_pos
    ori_error = orientation_error(desired_ori, x_ori)

    ee_pos_buffer.append([x_pos[i] for i in range(3)])
    fext_buffer.append(fext)

    D_q = robot.mass_matrix()
    D_x = np.dot(J_T_inv, np.dot(D_q, J_inv))

    if control_mode == 1:
        # 静止
        tau = sim.data.qfrc_bias[:]

    elif control_mode == 2:
        # 静止
        h_x = np.dot(J_T_inv, sim.data.qfrc_bias[:]) - np.dot(np.dot(D_x, Jd2), qd)
        F = np.dot(D_x, xdd_error) + h_x
        tau = np.dot(J.T, F)

    elif control_mode == 3:
        # 师兄的阻抗控制力矩计算, Md_x, B_x, K_x
        # 两种coef和tau, way 1 是openai，相较于paper，tau计算不太一样，
        D_q = robot.mass_matrix()
        Md_x = robot.mass_desired()
        Md_x_inv = np.linalg.inv(Md_x)
        tau = sim.data.qfrc_bias[:]
        coef = np.dot(D_q, J_inv)  # way 1
        # coef = np.dot(np.dot(D_q, J_inv), Md_x_inv) # way 2
        sum = np.multiply(B_x, xd_error)
        sum += np.multiply(K_x, x_error)
        sum -= np.dot(np.dot(Md_x, Jd2), qd)
        tau += np.dot(coef, sum)  # way 1
        # tau += np.dot(coef, np.dot(Md_x, sum)) # way 2

    elif control_mode == 4:
        # paper中不稳定的阻抗控制力跟踪, M_x, B_x, K_x
        # T = np.multiply(B_x, xd_error) + np.multiply(K_x, x_error) - F_desire + fext
        # F = np.dot(D_x, V) + h_x - fext
        # abrove really works
        h_x = np.dot(J_T_inv, sim.data.qfrc_bias[:]) - np.dot(np.dot(D_x, Jd2), qd)
        T = np.multiply(B_x, xd_error) + np.multiply(K_x, x_error) - F_desire + fext
        print("contact space", fext[2])
        V = desired_acc + np.dot(M_x_inv, T)
        F = np.dot(D_x, V) + h_x - fext
        tau = np.dot(J.T, F)

    elif control_mode == 5:
        # paper中不稳定的力跟踪阻抗控制算法, M_x, B_x, K_x，different control parameter in different occasions
        # the symbol of fext is different from the paper
        # so it should be plus while the desire force turns to be minus if it is 0 0 0 20 0 0
        D_q = robot.mass_matrix()
        D_x = np.dot(J_T_inv, np.dot(D_q, J_inv))
        h_x = np.dot(J_T_inv, sim.data.qfrc_bias[:]) - np.dot(np.dot(D_x, Jd2), qd)
        # h_x = np.dot(J_T_inv, sim.data.qfrc_bias[:])
        if np.all(fext == 0):
            print("free space")
            K_x = np.array([500, 500, 50, 1000, 1000, 1000], dtype=np.float32)
            T = np.multiply(B_x, xd_error) + np.multiply(K_x, x_error) - F_desire
        else:
            print("contact space", fext[2])
            K_x = np.array([100, 100, 0, 1000, 1000, 1000], dtype=np.float32)
            T = np.multiply(B_x, xd_error) + np.multiply(K_x, x_error) - F_desire + fext
        V = desired_acc + np.dot(M_x_inv, T)
        F = np.dot(D_x, V) + h_x - fext
        tau = np.dot(J.T, F)

    else:
        # paper中稳定的力跟踪阻抗控制算法, M_x, B_x, K_x, import robust control and adaptive control
        # 若h_old=h_x = np.dot(J_T_inv, sim.data.qfrc_bias[:]) - np.dot(np.dot(D_x, Jd2), qd),则与上一方法完全相同
        if np.all(fext == 0):
            print("free space")
            K_x = np.array([500, 500, 50, 1000, 1000, 1000], dtype=np.float32)
            T = np.multiply(B_x, xd_error) + np.multiply(K_x, x_error) - F_desire
            T[2] = B_x[2] * xd_error[2] + B_x[2] * omega - F_desire[2]  # adaptive control
        else:
            print("contact space", fext[2])
            K_x = np.array([100, 100, 0, 1000, 1000, 1000], dtype=np.float32)
            omega += update_rate * (fext[2] - F_desire[2]) / B_x[2]  # adaptive control
            T = np.multiply(B_x, xd_error) + np.multiply(K_x, x_error) - F_desire + fext
            T[2] = B_x[2] * xd_error[2] + B_x[2] * omega - F_desire[2] + fext[2]  # adaptive control
        V = desired_acc + np.dot(M_x_inv, T)
        U = np.dot(J_inv, V - np.dot(Jd2, qd))

        h_old = tau_old - np.dot(D_q, qdd) + np.dot(J.T, fext)
        # h_old = tau_old - np.dot(D_q, qdd_old) + np.dot(J.T, fext) # this won't work
        tau = np.dot(D_q, U) + h_old - np.dot(J.T, fext)
        tau_old = tau
        qdd_old = qdd

    if step > 3000:
        F_desire = np.array([0, 0, 20, 0, 0, 0])

    sim.data.ctrl[:] = tau
    sim.step()
    sim.forward()

    # render
    viewer.render()
    if len(ee_pos_buffer) == 1000:
        pass
    for i in range(max(0, len(ee_pos_buffer) - 900), len(ee_pos_buffer) - 1):
        viewer.add_marker(pos=ee_pos_buffer[i],
                          type=const.GEOM_SPHERE,
                          size=(0.01, 0.01, 0.01),
                          label='',
                          rgba=(1, 1, 0, 1))

ee_pos_buffer = np.array(ee_pos_buffer)
fext_buffer = np.array(fext_buffer)

# plt.title("X force")
# plt.xlabel("time/ms")
# plt.ylabel("force/N")
# plt.plot(np.arange(0, duration - 2), fext_buffer[1:duration - 1, 0])
# plt.show()
# plt.title("Y force")
# plt.xlabel("time/ms")
# plt.ylabel("force/N")
# plt.plot(np.arange(0, duration - 2), fext_buffer[1:duration - 1, 1])
# plt.show()
plt.title("Z force")
plt.xlabel("time/ms")
plt.ylabel("force/N")
plt.plot(np.arange(0, duration - 2), fext_buffer[1:duration - 1, 2])
plt.show()
