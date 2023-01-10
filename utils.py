import numpy as np
from scipy.spatial.transform import Rotation as R

class Dataset():

    pose_size = 6
    dtype = np.float32

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
        self.n_cameras = -1
        # keep track of how many cameras you have
        self.n_landmarks = -1

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
                    self._add_keypoint_to_camera(self.n_cameras, line)
                else:
                    raise Exception("First value must be a camera [KF] or a keypoint [F].")

                line = Dataset._preprocess_line(dataset_file.readline())
        
        with open(self.landmark_path, 'r') as landmark_file:
            line = Dataset._preprocess_line(landmark_file.readline())

            while line != []:
                assert line[0] == "L:"
                landmark_id = line[1]
                landmark_pose = np.array(line[2:], dtype=Dataset.dtype)
                self.landmark_poses_gt[landmark_id] = landmark_pose
                line = Dataset._preprocess_line(landmark_file.readline())


    @staticmethod
    def _preprocess_line(line):
        # split on spaces
        line = line.split(' ')
        # remove additional extra spaces
        line = [el for el in line if el != '']

        return line

    def _add_new_camera(self, camera_line):
        # increment the number of cameras registered
        self.n_cameras += 1

        if self.n_cameras >= self.max_cameras:
            # TODO resize the arrays
            return

        camera_id = int(camera_line[1])
        gt = camera_line[2:2+Dataset.pose_size]
        est = camera_line[2+Dataset.pose_size:]

        # camera idxes are in order and contiguous
        assert camera_id == self.n_cameras 
        
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

    def get_pose(self, i, gt=False)->tuple[np.ndarray, np.ndarray]:
        if gt:
            t = self.camera_poses_gt[i,:3]
            rot = self.camera_poses_gt[i,3:Dataset.pose_size]
        else:
            t = self.camera_poses[i,:3]
            rot = self.camera_poses[i,3:Dataset.pose_size]

        return (t, R.from_euler("xyz", rot, degrees=False).as_matrix())

    def set_pose(self, i, t=None, rot=None):
        if t is not None:
            self.camera_poses[i,:3] = t
        if rot is not None:
            self.camera_poses[i,3:Dataset.pose_size] = R.from_matrix(rot).as_euler("xyz")


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