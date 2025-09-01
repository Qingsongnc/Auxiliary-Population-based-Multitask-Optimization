import numpy as np
import scipy.io as sio
import os

"""
# The import format
from RealWorld.PKACP.tasks import PKACP
"""

class Kinematic:
    def __init__(self, A, L, Low=0, High=1, n=20):
        self.Dimension = n
        self.L = L
        self.Low = -A / self.Dimension
        self.High = A / self.Dimension

    def decode(self, X):
        X1 = self.Low + (self.High - self.Low) * X[:self.Dimension]
        return X1.reshape(self.Dimension) * np.pi

    def function(self, X, **kwargs):
        temp = self.decode(X)
        target = 0.5 * np.ones(2)

        p = np.append(temp, 0)
        joint_xy = []
        mat = np.matrix(np.identity(4))
        lengths = np.ones(self.Dimension) * self.L / self.Dimension
        lengths = np.concatenate(([0], lengths))
        for i in range(0, self.Dimension + 1):
            m = [[np.cos(p[i]), -np.sin(p[i]), 0, lengths[i]],
                 [np.sin(p[i]), np.cos(p[i]), 0, 0],
                 [0, 0, 1, 0],
                 [0, 0, 0, 1]]
            mat = mat * np.matrix(m)
            v = mat * np.matrix([0, 0, 0, 1]).transpose()
            joint_xy += [v[0:2].A.flatten()]

        return np.linalg.norm(joint_xy[self.Dimension] - target)

    def Info(self):
        return 'PKACP ' + str(self.Dimension)

def mat2python(filename):
    path = os.path.abspath(os.path.dirname(__file__))
    file = os.path.join(path, filename)
    data = sio.loadmat(file)
    return data['task_para']


def PKACP(dim, T=2):
    Tasks = []
    filename = 'cvt_d' + str(dim) + '_nt' + str(T) + '.mat'
    for t in range(T):
        params = mat2python(filename)
        Task = Kinematic(params[t, 0], params[t, 1], n=dim)
        Tasks.append(Task)
    return Tasks
