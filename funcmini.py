import mujoco_py as mp
import numpy as np
import PyKDL as kdl
import time

class rbt():
    def __init__(self, sim):
        self.sim = sim
        self.qvel_index = [0, 1, 2, 3, 4, 5]
        self.eef_name = 'ee'
        self.last_jacobian = self.jacobian()
        self.timer = time.time()
        self.chain = kdl.Chain()
        self.chain.addSegment(
            kdl.Segment("link1", kdl.Joint(kdl.Vector(0, 0, 0.069), kdl.Vector(0, 0, 1), kdl.Joint.JointType(0)),
                        kdl.Frame(kdl.Rotation(1, 0, 0, 0, 1, 0, 0, 0, 1), kdl.Vector(0, 0, 0.069)),
                        kdl.RigidBodyInertia(4.27, kdl.Vector(-0.000038005, -0.0024889, 0.054452),
                                             kdl.RotationalInertia(0.0340068, 0.0340237, 0.00804672, -5.0973e-06,
                                                                   2.9246e-05, 0.00154238))))
        self.chain.addSegment(
            kdl.Segment("link2", kdl.Joint(kdl.Vector(0, 0, 0.073), kdl.Vector(0, -1, 0), kdl.Joint.JointType(0)),
                        kdl.Frame(kdl.Rotation(1, 0, 0, 0, 0, -1, 0, 1, 0), kdl.Vector(0, 0, 0.073)),
                        kdl.RigidBodyInertia(10.1, kdl.Vector(0, 0.21252, 0.12053),
                                             kdl.RotationalInertia(0.771684, 1.16388, 1.32438, -5.9634e-05,
                                                                   0.258717, -0.258717))))
        self.chain.addSegment(
            kdl.Segment("link3", kdl.Joint(kdl.Vector(0, 0.425, 0), kdl.Vector(0, 0, 1), kdl.Joint.JointType(0)),
                        kdl.Frame(kdl.Rotation(0, -1, 0, 1, 0, 0, 0, 0, 1), kdl.Vector(0, 0.425, 0)),
                        kdl.RigidBodyInertia(2.61, kdl.Vector(0.27205, 1.9634e-05, 0.023591),
                                             kdl.RotationalInertia(0.00627892, 0.45698, 0.453824, 2.74907e-06,
                                                                   0.000484699, -1.39783e-06))))
        self.chain.addSegment(
            kdl.Segment("link4", kdl.Joint(kdl.Vector(0.395, 0, 0), kdl.Vector(0, 0, 1), kdl.Joint.JointType(0)),
                        kdl.Frame(kdl.Rotation(1, 0, 0, 0, 1, 0, 0, 0, 1), kdl.Vector(0.395, 0, 0)),
                        kdl.RigidBodyInertia(1.45, kdl.Vector(3.0707e-05, -0.015152, 0.11185),
                                             kdl.RotationalInertia(0.0385748, 0.0374742, 0.0022715, -1.1633e-06,
                                                                   -9.50907e-06, 0.00496695))))
        self.chain.addSegment(
            kdl.Segment("link5", kdl.Joint(kdl.Vector(0, 0, 0.1135), kdl.Vector(1, 0, 0), kdl.Joint.JointType(0)),
                        kdl.Frame(kdl.Rotation(0, 0, 1, -1, 0, 0, 0, -1, 0), kdl.Vector(0, 0, 0.1135)),
                        kdl.RigidBodyInertia(1.45, kdl.Vector(-3.0707e-05, 0.015152, 0.099848),
                                             kdl.RotationalInertia(0.0165655, 0.0278416, 0.0022718, -1.77049e-06,
                                                                   4.43911e-06, -0.00444154))))
        # self.chain.addSegment(
        #     kdl.Segment("link6", kdl.Joint(kdl.Vector(0, 0, 0.1015), kdl.Vector(0, -1, 0), kdl.Joint.JointType(0)),
        #                 kdl.Frame(kdl.Rotation(1, 0, 0, 0, 0, -1, 0, 1, 0), kdl.Vector(0, 0, 0.1015)),
        #                 kdl.RigidBodyInertia(0.21, kdl.Vector(0, 0.00058691, 0.0072051),
        #                                      kdl.RotationalInertia(0.0023707, 0.00229868, 0.000194624, 0, 0,
        #                                                            -1.71489e-05))))
        self.chain.addSegment(
            kdl.Segment("link6", kdl.Joint(kdl.Vector(0, 0, 0.1015), kdl.Vector(0, -1, 0), kdl.Joint.JointType(0)),
                        kdl.Frame(kdl.Rotation(1, 0, 0, 0, 0, -1, 0, 1, 0), kdl.Vector(0, 0, 0.1015)),
                        kdl.RigidBodyInertia(0.634, kdl.Vector(0, 0, 0.0968),
                                             kdl.RotationalInertia(0.0052, 0.0052, 0.0041))))
        self.jac_solver = kdl.ChainJntToJacSolver(self.chain)
        self.jacdot_solver = kdl.ChainJntToJacDotSolver(self.chain)
        self.gravity = kdl.Vector(0, 0, -9.81)
        self.dyn_params = kdl.ChainDynParam(self.chain, self.gravity)


    def mass_matrix(self): # mujoco
        mass_matrix = np.ndarray(shape=(len(self.sim.data.qvel) ** 2,), dtype=np.float64, order='C')
        mp.cymj._mj_fullM(self.sim.model, mass_matrix, self.sim.data.qM)
        mass_matrix = np.reshape(mass_matrix, (len(self.sim.data.qvel), len(self.sim.data.qvel)))
        mass_matrix = mass_matrix[self.qvel_index, :][:, self.qvel_index]
        return mass_matrix

    def mass_matrix2(self): # kdl
        mass_matrix = [[0] * 6 for i in range(6)]
        input = kdl.JntArray(6)
        for i in range(6):
            input[i] = self.sim.data.qpos[i]
        output = kdl.JntSpaceInertiaMatrix(6)
        self.dyn_params.JntToMass(input, output)
        for i in range(6):
            for j in range(6):
                mass_matrix[i][j] = output.__getitem__((i, j))
        return mass_matrix

    def jacobian(self): # mujoco
        J_pos = np.array(self.sim.data.get_site_jacp(self.eef_name).reshape((3, -1))[:, self.qvel_index])
        J_ori = np.array(self.sim.data.get_site_jacr(self.eef_name).reshape((3, -1))[:, self.qvel_index])
        J_full = np.array(np.vstack([J_pos, J_ori]))
        return J_full

    def jacobian2(self): # kdl
        input = kdl.JntArray(6)
        for i in range(6):
            input[i] = self.sim.data.qpos[i]
        output = kdl.Jacobian(6)
        self.jac_solver.JntToJac(input, output)
        ret = [[0] * 6 for i in range(6)]
        for i in range(6):
            for j in range(6):
                ret[i][j] = output.__getitem__((i, j))
        return ret

    def mass_desired(self):
        mass_matrix_inv = np.linalg.inv(self.mass_matrix())
        mass_desired_inv = np.dot(np.dot(self.jacobian(), mass_matrix_inv), self.jacobian().transpose())
        return np.linalg.inv(mass_desired_inv)

    def jacobian_dot(self): # time-diff
        cur_jacobian = self.jacobian()
        dt = time.time() - self.timer
        self.timer += dt
        jacobian_d = np.subtract(cur_jacobian, self.last_jacobian)
        jacobian_d = jacobian_d / (dt + 0.0001)
        self.last_jacobian = cur_jacobian
        return jacobian_d

    def jacobian_dot2(self): # kdl
        input_q = kdl.JntArray(6)
        input_qd = kdl.JntArray(6)
        for i in range(6):
            input_q[i] = self.sim.data.qpos[i]
            input_qd[i] = self.sim.data.qvel[i]
        input = kdl.JntArrayVel(input_q, input_qd)
        output = kdl.Jacobian(6)
        self.jacdot_solver.JntToJacDot(input, output)
        ret = [[0] * 6 for i in range(6)]
        for i in range(6):
            for j in range(6):
                ret[i][j] = output.__getitem__((i, j))
        return ret

    def coriolis(self): # coriolis torque C(q, qd)*qd kdl
        input_q = kdl.JntArray(6)
        input_qd = kdl.JntArray(6)
        for i in range(6):
            input_q[i] = self.sim.data.qpos[i]
            input_qd[i] = self.sim.data.qvel[i]
        output = kdl.JntArray(6)
        self.dyn_params.JntToCoriolis(input_q, input_qd, output)
        ret = [output[i] for i in range(6)]
        return ret

    def gravity_torque(self): # kdl
        input_q = kdl.JntArray(6)
        for i in range(6):
            input_q[i] = self.sim.data.qpos[i]
        output = kdl.JntArray(6)
        self.dyn_params.JntToGravity(input_q, output)
        ret = [output[i] for i in range(6)]
        return ret


