import numpy as np


class vehicle_SS:
    def __init__(self, dt):
        # State space matrices
        self.A_c = np.array([[0, 0], [0, 0]])

        self.B_c = np.array([[1, 0], [0, 1]])

        self.C_c = np.array([[1, 0], [0, 1]])

        self.D_c = np.array([[0, 0], [0, 0]])

        self.A = np.eye(2) + self.A_c * dt
        self.B = self.B_c * dt
        self.C = self.C_c
        self.D = self.D_c

    def calculate_next_step(self, x, u):
        x_next = self.A.dot(x.T) + self.B.dot(u.T)
        return x_next
