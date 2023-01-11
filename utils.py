import numpy as np
from scipy.spatial.transform import Rotation as R
import os, math, random

class Dataset():

    pose_size = 6
    dtype = np.float32

    angles_seq = "xyz"

    def __init__(self, camera_path: str, landmark_path: str, max_cameras=100):
        self.camera_path = camera_path
        self.landmark_path = landmark_path
        self.max_cameras = max_cameras

        self.camera_poses = np.zeros((max_cameras, Dataset.pose_size), 
                                      dtype=Dataset.dtype)
        self.camera_poses_gt = np.zeros_like(self.camera_poses)
    
        self.landmark_poses = {}
        self.landmark_poses_gt = {}

        # keep track of how many cameras you have
        self.last_camera_idx = -1
        # keep track of how many cameras you have
        self.n_landmarks = 0

        # for each camera add a dict {kpoint_id->direction}
        self.observed_keypoints = {}

        self._load()

    def _load(self):

        with open(self.camera_path, 'r') as dataset_file:
            line = Dataset._preprocess_line(dataset_file.readline())

            while line != []:
                if line[0] == "KF:":
                    # it's a camera
                    self._add_new_camera(line)
                elif line[0] == "F:":
                    # it's a landmark (observed by the last added camera)
                    self._add_keypoint_to_camera(self.last_camera_idx, line)
                else:
                    raise Exception("First value must be a camera [KF] or a keypoint [F].")

                line = Dataset._preprocess_line(dataset_file.readline())
        
        with open(self.landmark_path, 'r') as landmark_file:
            line = Dataset._preprocess_line(landmark_file.readline())

            while line != []:
                assert line[0] == "L:"
                self.n_landmarks += 1
                landmark_id = int(line[1])
                landmark_pose = np.array(line[2:], dtype=Dataset.dtype)
                self.landmark_poses_gt[landmark_id] = landmark_pose
                self.landmark_poses[landmark_id] = np.zeros(3, dtype=Dataset.dtype)
                line = Dataset._preprocess_line(landmark_file.readline())


    @staticmethod
    def _preprocess_line(line):
        # split on spaces
        line = line.split(' ')
        # remove additional extra spaces
        line = [el for el in line if el != '']

        return line

    @property
    def n_cameras(self):
        return self.last_camera_idx+1

    def _add_new_camera(self, camera_line):     
        # increment the number of cameras registered
        self.last_camera_idx += 1

        if self.n_cameras >= self.max_cameras:
            # TODO resize the arrays
            return

        camera_id = int(camera_line[1])
        gt = camera_line[2:2+Dataset.pose_size]
        est = camera_line[2+Dataset.pose_size:]

        # camera idxes are in order and contiguous
        assert camera_id == self.last_camera_idx 
        
        self.camera_poses[camera_id,:] = est
        self.camera_poses_gt[camera_id, :] = gt 

        # initialize an empty dictionary to store the observed keypoints
        self.observed_keypoints[camera_id] = {}

        return camera_id
        

    def _add_keypoint_to_camera(self, idx, line):
        # register the observed keypoint in the relative camera dictionary
        camera_dict = self.observed_keypoints[idx]

        kpoint_id = int(line[2])
        direction = np.array(line[3:], dtype=Dataset.dtype)
        camera_dict[kpoint_id] = direction

        return kpoint_id

    def get_camera_pose(self, i, gt=False)->tuple[np.ndarray, np.ndarray]:
        if gt:
            t = self.camera_poses_gt[i,:3]
            rot = self.camera_poses_gt[i,3:Dataset.pose_size]
        else:
            t = self.camera_poses[i,:3]
            rot = self.camera_poses[i,3:Dataset.pose_size]

        return (t, R.from_euler(Dataset.angles_seq, rot, degrees=False).as_matrix())
    
    def get_landmark_pose(self, i, gt=False)->np.ndarray:
        if gt:
            return self.landmark_poses_gt[i]
        else:
            return self.landmark_poses[i]

        return (t, R.from_euler(Dataset.angles_seq, rot, degrees=False).as_matrix())

    def set_camera_pose(self, i, t=None, rot=None):
        if t is not None:
            self.camera_poses[i,:3] = t
        if rot is not None:
            self.camera_poses[i,3:Dataset.pose_size] = R.from_matrix(rot).as_euler(Dataset.angles_seq)


    def feature_overlap(self, i, j): # i,j are camera indexes
        # https://stackoverflow.com/questions/18554012/intersecting-two-dictionaries
        overlapping_features = self.observed_keypoints[i].keys() & self.observed_keypoints[j].keys()
        return overlapping_features

# estimate the essential matrix between two cameras
# given 8 correspondences
def eight_point( points_i: np.ndarray, points_j: np.ndarray):
    assert points_i.shape == (8,3)
    assert points_j.shape == (8,3)

    return eight_point_LS(points_i, points_j)

def eight_point_LS( points_i: np.ndarray, points_j: np.ndarray):
    H = np.zeros((9,9))
    for p in range(points_i.shape[0]):
        pi = points_i[p,:] 
        pj = points_j[p,:] 
        A =  np.array([
            pi[0]*pj[0], pi[0]*pj[1], pi[0]*pj[2], 
            pi[1]*pj[0], pi[1]*pj[1], pi[1]*pj[2], 
            pi[2]*pj[0], pi[2]*pj[1], pi[2]*pj[2]
        ])
        H += np.outer(A,A)

    _, _, Vt = np.linalg.svd(H, full_matrices=True, hermitian=True)
    e = Vt[-1,:] # eigvector of the smallest singular value

    E = e.reshape((3,3))
    E/=E[2,2]
    # TODO enforce singular values are (s, s, 0)

    return E, None

def decompose_E(E):
    assert(E.shape == (3,3))

    _, S, Vt = np.linalg.svd(E, full_matrices=True)

    # rearranging matrix
    W = np.array([
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)
    skew_t = Vt.T@np.diag(S)@W.T@Vt

    t = unskew(skew_t)

    if t[2] < 0:
        t*=-1.0
    t/= np.linalg.norm(t)
     
    # no need to extract R also
    return None, t

def skew(t):
    return np.array([
        [0.0,  -t[2], t[1]],
        [t[2],  0.0, -t[0]],
        [-t[1], t[0], 0.0]
    ], dtype=np.float32)

def unskew(s):
    return np.array([s[2,1],s[0,2],s[1,0]])

def t2v(T: np.ndarray)->np.ndarray:
    assert T.shape==(4,4)

    v = np.zeros(6, dtype=T.dtype)

    v[:3] = T[:3,3]
    v[3:] = R.from_matrix(T[:3,:3]).as_euler(Dataset.angles_seq)

    return v

def v2t(v: np.ndarray)->np.ndarray:
    assert v.reshape(-1).shape == (6,)

    T = np.eye(4)
    T[:3,3] = v[:3]
    T[:3,:3] = R.from_euler(Dataset.angles_seq, v[3:]).as_matrix()

    return T


def generate_fake_data(path, n_cameras, n_landmarks):
    # generate a dataset containing noise free data
    bounds_x = (-10,10)
    bounds_y = (-10,10)
    bounds_z = (0,3)

    # scatter landmarks
    landmarks = np.random.rand(n_landmarks, 3)

    landmarks[:,0] = landmarks[:,0]*(bounds_x[1]-bounds_x[0]) + bounds_x[0]
    landmarks[:,1] = landmarks[:,1]*(bounds_y[1]-bounds_y[0]) + bounds_y[0]
    landmarks[:,2] = landmarks[:,2]*(bounds_z[1]-bounds_z[0]) + bounds_z[0]


    landmarks_path = os.path.join(path,"GT_landmarks.txt")
    with open(landmarks_path,"w") as file:
        for i in range(n_landmarks):
            line = f"L: {i} {landmarks[i,0]:6f} {landmarks[i,1]:6f} {landmarks[i,2]:6f}\n"
            file.write(line)

    cameras = np.random.rand(n_cameras, 6)

    cameras[:,0] = cameras[:,0]*(bounds_x[1]-bounds_x[0]) + bounds_x[0]
    cameras[:,1] = cameras[:,1]*(bounds_y[1]-bounds_y[0]) + bounds_y[0]
    cameras[:,2] = cameras[:,2]*(bounds_z[1]-bounds_z[0]) + bounds_z[0]

    cameras[:,3] = cameras[:,3]*(2*math.pi) - math.pi
    cameras[:,4] = cameras[:,4]*(2*math.pi) - math.pi
    cameras[:,5] = cameras[:,5]*(2*math.pi) - math.pi

    observed_keypoints = {}

    for camera_id in range(n_cameras):
        # each camera observes some landmarks
        observed_keypoints[camera_id] = {}
        n_observed = random.randint(0,n_landmarks//2) + n_landmarks//2

        observed_landmarks_ids = random.sample(range(n_landmarks), k=n_observed)

        for landmark_id in observed_landmarks_ids:
            # global landmark position
            x_l = landmarks[landmark_id]

            # camera pose
            t_c = cameras[camera_id][:3]
            R_c = R.from_euler(Dataset.angles_seq, cameras[camera_id][3:]).as_matrix()

            landmark_in_camera = R_c.T@(x_l - t_c)
            landmark_in_camera /= np.linalg.norm(landmark_in_camera)
            observed_keypoints[camera_id][landmark_id] = landmark_in_camera
    
    cameras_path = os.path.join(path,"dataset.txt")
    with open(cameras_path,"w") as file:
        for i in range(n_cameras):
            line = f"KF: {i} {cameras[i,0]:6f} {cameras[i,1]:6f} {cameras[i,2]:6f} {cameras[i,3]:6f} {cameras[i,4]:6f} {cameras[i,5]:6f} \
{cameras[i,0]:6f} {cameras[i,1]:6f} {cameras[i,2]:6f} {cameras[i,3]:6f} {cameras[i,4]:6f} {cameras[i,5]:6f}\n"
            file.write(line)
            n = 0
            for j in observed_keypoints[i].keys():
                line = f"F: {n} {j} {observed_keypoints[i][j][0]:6f} {observed_keypoints[i][j][1]:6f} {observed_keypoints[i][j][2]:6f}\n"
                file.write(line)
                n+= 1


