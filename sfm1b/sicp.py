# find similarity transform between two point clouds
# with known correspondences

import numpy as np
from utils import skew, v2s
from scipy.linalg import solve

from .bundle_adjustment import BA

def sicp(p: np.ndarray, z: np.ndarray, initial_guess=np.eye(4), n_iters=10, damping=1.0, threshold=1.0):
    chi_stats = []
    X = initial_guess

    for iter in range(n_iters):
        H = np.zeros((7,7))
        b = np.zeros(7)
        chi2 = 0.0
        for i in range(p.shape[0]):

            e, J = error_jacobian(X, p[i], z[i])
            chi2 += e.T@e

            if (e.T@e) > threshold:
                pass# e *= threshold/np.linalg.norm(e)
            
            H += J.T@J
            b += J.T@e
        chi_stats.append(chi2)
        H += np.eye(7)*damping
        dx = np.linalg.solve(H,-b)
        X = v2s(dx)@X

    return X, chi_stats
    


def error_jacobian(X: np.ndarray, p: np.ndarray, z: np.ndarray):
    J = np.zeros((3,7))

    t = X[:3,3]
    R = X[:3,:3]
    s = X[3,3]
    """
    print(X)
    print(t)
    print(R)
    print(s)
    """
    pred = (R@p + t)/s

    error = pred - z

    J[:3,:3] = np.eye(3)
    J[:3,3:6] = skew(-pred)
    J[:3,6:] = pred.reshape(-1,1)

    return error, J