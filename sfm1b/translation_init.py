from dataset import Dataset
from utils import skew
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

        # TODO research why do I have to take the minus / how to choose the sign
        # if I take the plus everything is flipped with respect to gt
        # scale it by 100 so sicp converges better and you can see the landmarks overlapping with the gt
        self.t[3:] = - self.t_* 100

        return self.t

    def _update_poses(self):
        # update the poses on the dataset with this initial estimate
        for id in range(self.n):
            self.dataset.set_camera_pose(id, t=self.t[3*id:3*id+3])
        

