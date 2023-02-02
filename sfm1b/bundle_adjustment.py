# test with automatic differentiation
# the quickest way to me is to use torch
from utils import v2t, t2v, skew
from dataset import Dataset
import numpy as np

import scipy.sparse

# perform bundle adjustment
# estimate the position of the landmarks
# and the poses of the camera starting
# from an initial guess provided by the previous steps 
class BA():
    # the state are all the camera poses SE(3) and 
    # all the landmark positions R^3

    # the objective to minimize is the reprojection error
    # d_cl - norm(X_c * X_l) (chordal distance with directions)

    def __init__(self, d: Dataset, 
                       n_iters: int,
                       damping= 1.5):
        self.d = d

        self.chi_stats = []

        self.n_iters = n_iters

        self.Xc = np.zeros((d.n_cameras, 4,4), dtype=d.dtype)
        self.Xl = np.zeros((d.n_landmarks,3), dtype=d.dtype)

        self.state_dim = 6*d.n_cameras+3*d.n_landmarks

        self.landmark_map = {}
        self.landmark_map_inverse = np.zeros(self.d.n_landmarks, dtype=np.uint8)

        self.damping = damping

        self._pre()

    def _pre(self):

        # build the state to id vector for landmarks
        # no need to do it for cameras
        for i, l_i in zip(range(self.d.n_landmarks), self.d.landmark_poses.keys()):
            self.landmark_map[l_i] = i
            self.landmark_map_inverse[i] = l_i
            self.Xl[i] = self.d.get_landmark_pose(l_i)

        for camera_id in range(self.d.n_cameras):
            # store the inverse (I can copy jacobians from my notes)
            t_i, R_i = self.d.get_camera_pose(camera_id)
            self.Xc[camera_id][:3,:3] = R_i.T
            self.Xc[camera_id][:3,3] = -R_i.T@t_i
            self.Xc[camera_id][3,3] = 1.0
            
        
    def _build_H_b(self):
    
        chi2 = []
        inliers = 0
        H = np.zeros((self.state_dim, self.state_dim), dtype=self.d.dtype)
        b = np.zeros((self.state_dim), dtype=H.dtype)

        for camera_id in range(self.d.n_cameras):
            for landmark_seen in self.d.get_direction(camera_id):
                landmark_id = self.landmark_map[landmark_seen]

                z = self.d.get_direction(camera_id,landmark_seen)

                e, Jc, Jl = self._error_jacobian(z, camera_id, landmark_id)
                
                chi2.append( e.T@e )

                e, weight, inlier = self._robustifier(e)

                inliers += inlier

                Hcc = weight*Jc.T@Jc
                Hcl = weight*Jc.T@Jl
                Hll = weight*Jl.T@Jl
                bc  = weight*Jc.T@e
                bl  = weight*Jl.T@e

                cam_state = self._cam_to_state(camera_id)
                landmark_state = self._landmark_to_state(landmark_id)

                H[cam_state:cam_state+self.d.c_pose_size,
                  cam_state:cam_state+self.d.c_pose_size]+=Hcc

                H[cam_state:cam_state+self.d.c_pose_size,
                  landmark_state:landmark_state+3]+=Hcl

                H[landmark_state:landmark_state+3,
                  landmark_state:landmark_state+3]+=Hll

                H[landmark_state:landmark_state+3,
                  cam_state:cam_state+self.d.c_pose_size]+=Hcl.T

                b[cam_state:cam_state+self.d.c_pose_size]+=bc
                b[landmark_state:landmark_state+3]+=bl

        # add damping
        H += np.eye(self.state_dim)*self.damping
        
        return H, b, chi2, inliers,

    def _solve(self):
        for iter in range(self.n_iters):
            
            delta_x = np.zeros(self.state_dim)
            H, b, chi2, inliers= self._build_H_b()
            chi2 = np.array(chi2)
            print(f"{iter}, {inliers}, {chi2.sum()}, {chi2.mean()}+-{chi2.std()}", end="\r")
            self.chi_stats.append(sum(chi2))

            # block the first camera pose
            H_ = H[self.d.c_pose_size:, self.d.c_pose_size:]
            b_ = b[self.d.c_pose_size:]

            # TODO do all the sparse things to optimize
            delta_x[self.d.c_pose_size:] = np.linalg.solve(H_, -b_)
            
            self._box_plus(self.Xc, self.Xl, delta_x)

        self._update_poses()

    
    # to fill the jacobians you need the position of the given
    # element in the state vector
    def _cam_to_state(self, i):
        return 6*i
    def _landmark_to_state(self, i):
        return 6*self.d.n_cameras + 3*i

    def _error_jacobian(self, z, camera_id, landmark_id):

        X_c = self.Xc[camera_id]
        X_l = self.Xl[landmark_id]

        landmark_in_camera = X_c[:3,:3]@X_l+X_c[:3,3]
        projection = landmark_in_camera / np.linalg.norm(landmark_in_camera)

        error = projection - z

        J_norm = self._J_norm(landmark_in_camera)

        J_icp = self._J_icp(landmark_in_camera)

        J_l = X_c[:3,:3]

        return error, J_norm@J_icp, J_norm@J_l

    def _robustifier(self, e):
       
        threshold = 0.2
        inlier = True
        weight = 1.0

        if (e.T@e) > threshold:
            e *= threshold/np.linalg.norm(e)
            inlier = False
            weight = 0

        return e, weight, inlier

    def _box_plus(self, Xc, Xl, dx):
        for cam in range(self.d.n_cameras):
            state = self._cam_to_state(cam)
            dx_cam = dx[state:state+self.d.c_pose_size]
            T = v2t(dx_cam)
            Xc[cam] = T@Xc[cam]
        for landmark in range(self.d.n_landmarks):
            state = self._landmark_to_state(landmark)
            dx_landmark = dx[state:state+3]
            Xl[landmark] += dx_landmark
    
    def _J_norm(self, p):
        norm = np.linalg.norm(p)
        outer = np.outer(p,p)

        J_norm = np.eye(3) - outer / (norm**2)
        J_norm /= norm
        
        return J_norm
    
    def _J_icp(self, p):

        J_icp = np.zeros((3,6))
        J_icp[:,:3] = np.eye(3)
        J_icp[:,3:] = -skew(p)

        return J_icp

    def _update_poses(self):
        for camera_id in range(self.d.n_cameras):
            # fetch the pose vector from Xc
            X_c = self.Xc[camera_id]
            # need to remeber I stored the inverse...
            Rinv = X_c[:3,:3]
            tinv = X_c[:3,3]

            self.d.set_camera_pose(camera_id, -Rinv.T@tinv, Rinv.T)

        for landmark_id in range(self.d.n_landmarks):
            X_l = self.Xl[landmark_id]
            landmark_id = self.landmark_map_inverse[landmark_id]

            self.d.set_landmark_pose(landmark_id, X_l)


