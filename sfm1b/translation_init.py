from utils import Dataset, skew
from .find_pairs import Pair
import numpy as np

# initialize the translation by doing least squares
# with the error function Ri'(ti-tj) x t_ij
class TranslationInit():

    def __init__(self, dataset: Dataset, 
                       pairs: list[Pair]):
        self.dataset = dataset
        self.pairs = pairs

        self.n = self.dataset.n_cameras
        self.H = np.zeros((3*self.n,3*self.n), dtype=dataset.dtype)

        self._build_H()
        self._solve()
        self._update_poses()

    def _build_H(self):
        
        # each pair provides a measurement t_ij
        for pair in self.pairs:
            _, R_i = self.dataset.get_pose(pair.i)
            s_p = skew(pair.t_ij)
            
            J_p = np.zeros((3,3*self.n))

            J_i = s_p@R_i.T
            J_j = -J_i

            J_p[:,3*pair.i:3*pair.i+3] = J_i
            J_p[:,3*pair.j:3*pair.j+3] = J_j
            
            self.H += J_p.T@J_p
        
    def _solve(self):
        # H should have 4 singular values close to zero
        # one for the scale ambiguity, the others for the global translation
        # U,S,Vt = np.linalg.svd(self.H,hermitian=True)
        # print(S)

        # solve 3 degrees by fixing the first camera in (0,0,0)
        self.H_fixed = self.H[3:,3:]
        #U,S,Vt = np.linalg.svd(self.H_fixed,hermitian=True)
        #print(S)

        # how to solve scale ambiguity?
        # 1) add a constraint t_ij = Ri'(t_j-t_i)
        # 2) enforce ||t_j-t_i|| = k impose a global scale
        # those choices make a b vector appear

        # 3) choose the eigenvector corresponding to the
        # unique zero singular value left

        U,S,Vt = np.linalg.svd(self.H_fixed,hermitian=True)
        self.t = np.concatenate((np.zeros(shape=(3,)),Vt[-1,:])) # eigvector of the smallest singular value

        # t is a very long vector of norm 1 so it's elements are very small
        # scale it a bit
        self.t /= np.min(np.abs(self.t[3:]))
    
    def _update_poses(self):
        # update the poses on the dataset with this initial estimate
        for id in range(self.n):
            self.dataset.set_pose(id, t=self.t[3*id:3*id+3])
        

