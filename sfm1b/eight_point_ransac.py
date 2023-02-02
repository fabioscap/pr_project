# NOT USED, THERE IS NO BAD ASSOCIATION

import numpy as np
from utils import eight_point, eight_point_LS
import random, math

# extract essential matrix from pairs of correspondences 
# using RANSAC
def ransac(p1: np.ndarray, 
           p2: np.ndarray, 
           threshold=5e-3, # in/outlier gating threshold
           prob=0.99, # to extract number of iterations
           max_iters=100,
           refine=False
        ):     
        assert p1.shape == p2.shape
        n_points = p1.shape[0]

        # inlier ratio
        w = 0.3

        # compute the number of iterations given prob
        n_iters = math.ceil(math.log(1-prob) / math.log(1-w**8))
        # it is often a huge number, compare it with the required number with 5 point alg.
        # math.ceil(math.log(1-prob) / math.log(1-w**5))
        # so I added another parameter to cap the iterations
        n_iters = min(n_iters, max_iters)
        # best model 
        E = None
        best_inliers = []
        best_score = None

        for iter in range(n_iters):
            E_, error, inliers = _one_iter(p1, p2, n_points, threshold)
            score = _eval_model(error, inliers)
            if best_score == None or score > best_score:
                E = E_
                best_inliers = inliers
                best_score = score
        if refine:
            E = eight_point_LS(p1[inliers,:], p2[inliers,:])
        
        return E, best_inliers


def _one_iter(p1,p2,n_points,threshold)->tuple[np.ndarray, float, list]:

    inliers = []
    # cumulative error
    error = np.zeros(1, dtype=p1.dtype)
    
    # sample a set of 8 correspondences
    idxes = random.sample(range(n_points), k=8)

    E = eight_point(p1[idxes,:], p2[idxes,:])

    for idx in range(n_points):
        # compute the error
        err = p1[idx,:]@E@p2[idx,:]
        assert err.shape == ()

        if err*err < threshold:
            inliers.append(idx)
        error += err

    return E, float(error), inliers

def _eval_model(error, inliers):
    # define a function to evaluate a proposal
    # naive approach: count inliers
    return len(inliers)

# wrapper around opencv implementation
import cv2
def ransac_opencv(p1: np.ndarray, 
           p2: np.ndarray, 
           threshold=5e-3, # in/outlier gating threshold
           prob=0.99, # to extract number of iterations
           max_iters=100,
           ):

    # cv2 wants 2D points, divide by z
    p1_norm = p1 / np.vstack((p1[:,2],p1[:,2],p1[:,2])).T
    p2_norm = p2 / np.vstack((p2[:,2],p2[:,2],p1[:,2])).T
    # cv2 returns the transposed with respect to my convention
    # so I swap the inputs
    E, _inliers = cv2.findEssentialMat(p2_norm[:,:2],p1_norm[:,:2], 
                                      method=cv2.RANSAC, threshold=threshold, 
                                      prob=prob, maxIters=max_iters)
    # cv2 returns a boolean array mask, extract the indexes from that
    idxes = np.arange(p1.shape[0])
    inliers = [i for i in idxes if _inliers[i] == 1]

    return E, inliers

