# find pairs between images, then extract their 
# essential matrix. From the matrix get the relative translation
# (rotation already done in 1a) 
from dataset import Dataset
from utils import decompose_E, eight_point_LS
import numpy as np

import cv2

class Pair():
    def __init__(self, i, 
                       j, 
                       features, 
                       d:Dataset, 
                       # no wrong associations
                       E_proc = eight_point_LS 
                       ):
        self.i = i
        self.j = j
        self.features = features

        self.E_proc = E_proc
        self.d = d

        self.E, self.inliers = self._compute_E()

        # don't care for R
        _,_,self.t_ij = cv2.decomposeEssentialMat(self.E)
        self.t_ij = self.t_ij.reshape(-1)
        # _, tij_mine = decompose_E(self.E) # my implementation is wrong

    def _compute_E(self):
        p1 = np.zeros((len(self.features),3))
        p2 = np.zeros((len(self.features),3))
    
        # fill this with the observed directions
        p = 0
        for feature in self.features:
            p1[p:] = self.d.get_direction(self.i,feature)
            p2[p:] = self.d.get_direction(self.j,feature)
            p+=1

        return self.E_proc(p1, p2)

    def __repr__(self) -> str:
        return f"({self.i},{self.j})"


def find_pairs(dataset: Dataset, 
               min_overlap = 40, # for 2 cameras to become a pair
               ):
    pairs = []
    # iterate over all cameras to find pairs
    for i in range(dataset.n_cameras):
        for j in range(i+1, dataset.n_cameras):
            overlap = dataset.feature_overlap(i,j)
            if len(overlap) >= min_overlap:
                # print(f"found pair {i}, {j}")
                p = Pair(i,j, overlap, dataset)
                pairs.append(p)

    return pairs


