import numpy as np


class vehicle_SS:
    def __init__(self, dt):
        # State space matrices
        self.A_c = np.matrix([[0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]])

        self.B_c = np.matrix([[0, 0], [0, 0], [1, 0], [0, 1]])

        self.C_c = np.matrix([[1, 0, 0, 0], [0, 1, 0, 0]])

        self.D_c = np.matrix([[0, 0], [0, 0]])

        self.A = np.eye(4) + self.A_c * dt
        self.B = self.B_c
        self.C = self.C_c
        self.D = self.D_c

    def CalculateNextStep(self, x, u):
        x_next = self.A.dot(x.T) + self.B.dot(u.T)
        return x_next
