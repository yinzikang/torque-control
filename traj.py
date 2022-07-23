import roboticstoolbox as rtb
import PyKDL as kdl
import math
import numpy as np

PI = 3.1415926

chain = kdl.Chain()
chain.addSegment(
    kdl.Segment("link1", kdl.Joint(kdl.Vector(0, 0, 0.069), kdl.Vector(0, 0, 1), kdl.Joint.JointType(0)),
                kdl.Frame(kdl.Rotation(1, 0, 0, 0, 1, 0, 0, 0, 1), kdl.Vector(0, 0, 0.069)),
                kdl.RigidBodyInertia(4.27, kdl.Vector(-0.000038005, -0.0024889, 0.054452),
                                     kdl.RotationalInertia(0.0340068, 0.0340237, 0.00804672, -5.0973e-06,
                                                           2.9246e-05, 0.00154238))))
chain.addSegment(
    kdl.Segment("link2", kdl.Joint(kdl.Vector(0, 0, 0.073), kdl.Vector(0, -1, 0), kdl.Joint.JointType(0)),
                kdl.Frame(kdl.Rotation(1, 0, 0, 0, 0, -1, 0, 1, 0), kdl.Vector(0, 0, 0.073)),
                kdl.RigidBodyInertia(10.1, kdl.Vector(0, 0.21252, 0.12053),
                                     kdl.RotationalInertia(0.771684, 1.16388, 1.32438, -5.9634e-05,
                                                           0.258717, -0.258717))))
chain.addSegment(
    kdl.Segment("link3", kdl.Joint(kdl.Vector(0, 0.425, 0), kdl.Vector(0, 0, 1), kdl.Joint.JointType(0)),
                kdl.Frame(kdl.Rotation(0, -1, 0, 1, 0, 0, 0, 0, 1), kdl.Vector(0, 0.425, 0)),
                kdl.RigidBodyInertia(2.61, kdl.Vector(0.27205, 1.9634e-05, 0.023591),
                                     kdl.RotationalInertia(0.00627892, 0.45698, 0.453824, 2.74907e-06,
                                                           0.000484699, -1.39783e-06))))
chain.addSegment(
    kdl.Segment("link4", kdl.Joint(kdl.Vector(0.395, 0, 0), kdl.Vector(0, 0, 1), kdl.Joint.JointType(0)),
                kdl.Frame(kdl.Rotation(1, 0, 0, 0, 1, 0, 0, 0, 1), kdl.Vector(0.395, 0, 0)),
                kdl.RigidBodyInertia(1.45, kdl.Vector(3.0707e-05, -0.015152, 0.11185),
                                     kdl.RotationalInertia(0.0385748, 0.0374742, 0.0022715, -1.1633e-06,
                                                           -9.50907e-06, 0.00496695))))
chain.addSegment(
    kdl.Segment("link5", kdl.Joint(kdl.Vector(0, 0, 0.1135), kdl.Vector(1, 0, 0), kdl.Joint.JointType(0)),
                kdl.Frame(kdl.Rotation(0, 0, 1, -1, 0, 0, 0, -1, 0), kdl.Vector(0, 0, 0.1135)),
                kdl.RigidBodyInertia(1.45, kdl.Vector(-3.0707e-05, 0.015152, 0.099848),
                                     kdl.RotationalInertia(0.0165655, 0.0278416, 0.0022718, -1.77049e-06,
                                                           4.43911e-06, -0.00444154))))
chain.addSegment(
    kdl.Segment("link6", kdl.Joint(kdl.Vector(0, 0, 0.1015), kdl.Vector(0, -1, 0), kdl.Joint.JointType(0)),
                kdl.Frame(kdl.Rotation(1, 0, 0, 0, 0, -1, 0, 1, 0), kdl.Vector(0, 0, 0.1015)),
                kdl.RigidBodyInertia(0.21, kdl.Vector(0, 0.00058691, 0.0072051),
                                     kdl.RotationalInertia(0.0023707, 0.00229868, 0.000194624, 0, 0,
                                                           -1.71489e-05))))

initial_qpos = [0, -30, 120, 0, -90, 0]
q_pre = kdl.JntArray(6)
q_cur = kdl.JntArray(6)
for i in range(6):
    q_cur[i] = initial_qpos[i] * PI / 180
    q_pre[i] = q_cur[i]
fk_solver = kdl.ChainFkSolverPos_recursive(chain)
center = kdl.Frame()
radius = 0.05
center.p[0] -= 0.05
fk_solver.JntToCart(q_cur, center)
ik_solver = kdl.ChainIkSolverPos_LMA(chain)

viapoints = []
count = 300
for i in range(count):
    point = center.__copy__()
    point.p[0] += radius * math.cos(i / count * 2 * PI)
    point.p[1] += radius * math.sin(i / count * 2 * PI)
    ik_solver.CartToJnt(q_pre, point, q_cur)
    q_pre = q_cur
    viapoints.append([q_cur[0], q_cur[1], q_cur[2], q_cur[3], q_cur[4], q_cur[5]])
viapoints = np.array(viapoints).reshape(-1, 6)
traj = rtb.mstraj(viapoints, 0.01, 0.1, 0.01)


