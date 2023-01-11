# test with automatic differentiation
# the quickest way to me is to use torch
from utils import Dataset, v2t, t2v, skew
import numpy as np

# perform bundle adjustment
# estimate the position of the landmarks
# and the poses of the camera starting
# from an initial guess provided by the previous steps 
class BA():
    # the state are all the camera poses SE(3) and 
    # all the landmark positions R^3

    # the objective to minimize is the reprojection error
    # x_cl - norm(X_c * X_l) (chordal distance with directions)

    # cameras id are in order so no need for map
    # landmarks need a map from id to state

    def __init__(self, d: Dataset, 
                       n_iters: int,
                       damping= 10.0):
        self.d = d

        self.chi_stats = []

        self.n_iters = n_iters

        self.Xc = np.zeros((d.n_cameras, 4,4), dtype=d.dtype)
        self.Xl = np.zeros((d.n_landmarks,3), dtype=d.dtype)

        self.state_dim = 6*d.n_cameras+3*d.n_landmarks
        self.delta_x = np.zeros(self.state_dim)

        self.landmark_map = {}
        self.landmark_map_inverse = np.zeros(self.d.n_landmarks, dtype=np.uint8)

        self.H = np.zeros((self.state_dim, self.state_dim), dtype=self.d.dtype)
        self.b = np.zeros((self.state_dim), dtype=self.H.dtype)

        self.damping = damping

    def _pre(self):

        # build the state to id vector for landmarks
        # no need to do it for cameras
        for i, l_i in zip(range(self.d.n_landmarks), self.d.landmark_poses.keys()):
            self.landmark_map[l_i] = i
            self.landmark_map_inverse[i] = l_i
            # TODO replace with triangulated points
            self.Xl[i] = self.d.landmark_poses_gt[l_i]

        
        for camera_id in range(self.d.n_cameras):
            # store the inverse
            t_i, R_i = self.d.get_camera_pose(camera_id, gt=True)
            self.Xc[camera_id][:3,:3] = R_i.T
            self.Xc[camera_id][:3,3] = -R_i.T@t_i
            self.Xc[3,3] = 1.0

        
    def _build_H_b(self):
        chi2 = 0
        self.H = np.zeros((self.state_dim, self.state_dim), dtype=self.d.dtype)
        self.b = np.zeros((self.state_dim), dtype=self.H.dtype)

        for camera_id in range(self.d.n_cameras):
            for landmark_seen in self.d.observed_keypoints[camera_id]:
                landmark_id = self.landmark_map[landmark_seen]

                z = self.d.observed_keypoints[camera_id][landmark_seen]

                e, Jc, Jl = self._error_jacobian(z, camera_id, landmark_id)

                chi2 += e.T@e

                e, weight = self._robustifier(e)

                Hrr = weight*Jc.T@Jc
                Hrl = weight*Jc.T@Jl
                Hll = weight*Jl.T@Jl
                br  = weight*Jc.T@e
                bl  = weight*Jl.T@e

                cam_state = self._cam_to_state(camera_id)
                landmark_state = self._landmark_to_state(landmark_id)

                self.H[cam_state:cam_state+self.d.pose_size,
                       cam_state:cam_state+self.d.pose_size]+=Hrr

                self.H[cam_state:cam_state+self.d.pose_size,
                landmark_state:landmark_state+3]+=Hrl

                self.H[landmark_state:landmark_state+3,
                landmark_state:landmark_state+3]+=Hll

                self.H[landmark_state:landmark_state+3,
                cam_state:cam_state+self.d.pose_size]+=Hrl.T

                self.b[cam_state:cam_state+self.d.pose_size]+=br
                self.b[landmark_state:landmark_state+3]+=bl

        # add damping
        self.H += np.eye(self.state_dim)*self.damping
        
        self.chi_stats.append(chi2)

    def _solve(self):
        
        for iter in range(self.n_iters):
            print(iter)
            self.delta_x = np.zeros(self.state_dim)
            self._build_H_b()

            # block the first camera pose
            H_ = self.H[self.d.pose_size:, self.d.pose_size:]
            b_ = self.b[self.d.pose_size:]
            
            #print(self.H_.shape)
            #print("rank:",np.linalg.matrix_rank(self.H_))

            # TODO do all the sparse things to optimize
            self.delta_x[self.d.pose_size:] = np.linalg.solve(H_, -b_)

            self._box_plus(self.Xc, self.Xl, self.delta_x)

    
    # to fill the jacobians you need the position of the given
    # element in the state vector
    def _cam_to_state(self, i):
        return 6*i
    def _landmark_to_state(self, i):
        return 6*self.d.n_cameras + 3*i

    def _error_jacobian(self, z, camera_id, landmark_id):

        X_c = self.Xc[camera_id]
        X_l = self.Xl[landmark_id]

        landmark_in_camera = X_c[:3,:3]@X_l+self.Xc[camera_id][:3,3]
        projection = landmark_in_camera / np.linalg.norm(landmark_in_camera)

        # chordal distance error
        error = projection - z

        J_norm = self._J_norm(landmark_in_camera)

        J_icp = self._J_icp(landmark_in_camera)

        J_l = X_c[:3,:3]

        # J = np.zeros((3,self.state_dim))

        # camera_state = self._cam_to_state(camera_id)
        # landmark_state = self._landmark_to_state(landmark_id)

        # J[:,camera_state:camera_state+6] = J_norm@J_icp
        # J[:,landmark_state:landmark_state+3] = J_norm@J_l

        return error, J_norm@J_icp, J_norm@J_l

    # TODO do
    def _robustifier(self, e):
        return e, 1.0

    def _box_plus(self, Xc, Xl, dx):
        for cam in range(self.d.n_cameras):
            state = self._cam_to_state(cam)
            dx_cam = dx[state:state+self.d.pose_size]
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
        