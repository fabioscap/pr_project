from dataset import Dataset
from utils import skew, lines_intersection
from .find_pairs import Pair
import numpy as np

# initialize the translation by doing least squares
# with the error function Ri'(ti-tj) x t_ij
# it is a linear constraint

# find the solution in the null space of H
class TranslationInit():

    def __init__(self, dataset: Dataset, 
                       pairs: list[Pair]):
        self.dataset = dataset
        self.pairs = pairs

        self.n = self.dataset.n_cameras
        self.H = np.zeros((3*self.n,3*self.n), dtype=dataset.dtype)

        self.t = np.zeros(3*self.n)
        
        self._build_H_b()
        self._solve()
        sign = self._adjust_scaling_sign()
        self.t *= sign
        self._update_poses()

    def _build_H_b(self):
        
        # each pair provides a measurement t_ij
        for pair in self.pairs:
            _, R_i = self.dataset.get_camera_pose(pair.i)
            s_p = skew(pair.t_ij)
            
            J_p = np.zeros((3,3*self.n))

            J_i = s_p@R_i.T
            J_j = -J_i

            # only the ith and jth block are non zero
            J_p[:,3*pair.i:3*pair.i+3] = J_i
            J_p[:,3*pair.j:3*pair.j+3] = J_j
            
            self.H += J_p.T@J_p
        
    def _solve(self):
        # H should have 4 singular values close to zero
        # one for the scale ambiguity, the others for the global translation
        # _,S,_ = np.linalg.svd(self.H,hermitian=True)
        # print(S[-6:])

        # solve 3 degrees by fixing the first camera in (0,0,0)
        self.H_ = self.H[3:,3:]
        self.t_ = self.t[3:]
        
        _, _, Vt = np.linalg.svd(self.H_)
        self.t_ = Vt[-1,:] # eigvector of the smallest singular value

        self.t[3:] =   self.t_*100 # better convergence on sicp

        return self.t

    def _update_poses(self):
        # update the poses on the dataset with this initial estimate
        for id in range(self.n):
            self.dataset.set_camera_pose(id, t=self.t[3*id:3*id+3])
        

    def _adjust_scaling_sign(self, i=0, j=1):
        # attempt to find the sign of the scaling factor

        # intersect correspondences and find the sign of the parameter s
        # which corresponds to the intersection
        # it makes sense that the parameter s should always be positive

        _, Ri = self.dataset.get_camera_pose(i)
        _, Rj = self.dataset.get_camera_pose(j)

        ti = self.t[3*i:3*i+3]
        tj = self.t[3*j:3*j+3]

        overlap = self.dataset.feature_overlap(i,j)

        # there should be a clear winner
        votes_for_plus = 0
        votes_for_minus = 0
        discordant = 0

        for l_idx in overlap:
            di = self.dataset.get_direction(i, l_idx)
            dj = self.dataset.get_direction(j, l_idx)

            
            s_i, s_j = lines_intersection(ti, Ri, di, tj, Rj, dj)
            if s_i >= 0 and s_j >= 0:
                votes_for_plus += 1
            elif s_i < 0 and s_j < 0:
                votes_for_minus +=1
            else:
                discordant += 1

        if votes_for_plus > votes_for_minus: return 1
        elif votes_for_minus > votes_for_plus: return -1
        else: return 1 # what happened

