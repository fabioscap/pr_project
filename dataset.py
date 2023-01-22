import numpy as np
from utils import complete_quaternion, v2t
from scipy.spatial.transform import Rotation as R

class Dataset():

    dtype = np.float32
    c_pose_size = 6
    l_pose_size = 3
    # we use directions
    d_size = 3

    # where to pick the coordinates in the file
    c_gt_slice = slice(2,2+c_pose_size)
    c_es_slice = slice(2+c_pose_size,2+2*c_pose_size)

    l_slice = slice(2,2+l_pose_size)
    d_slice = slice(3,3+d_size)

    def __init__(self, camera_path, landmark_path, ground_truth=False) -> None:
        self.camera_path = camera_path
        self.landmark_path = landmark_path

        self.gt = ground_truth
        if self.gt:
            self.c_slice = Dataset.c_gt_slice
        else:
            self.c_slice = Dataset.c_es_slice

        self.n_cameras = 0
        self.n_landmarks = 0

        # camera in world
        self.camera_poses = {}

        self.landmark_poses = {}

        self.directions = {}

        self._load()

    def get_camera_pose(self, i, **kwargs)->tuple[np.ndarray, np.ndarray]:
        t = self.camera_poses[i][:3,3]
        r = self.camera_poses[i][:3,:3]

        return (t, r)
    
    def get_landmark_pose(self, i, **kwargs)->np.ndarray:
        return self.landmark_poses[i]

    def set_camera_pose(self, i, t=None, rot=None):
        if self.gt: return 
        if t is not None:
            self.camera_poses[i][:3,3] = t
        if rot is not None:
            self.camera_poses[i][:3,:3] = rot

    def set_landmark_pose(self, i, t):
        if self.gt: return
        self.landmark_poses[i] = t

    def get_direction(self, c_idx, l_idx=None):
        if l_idx is None: return self.directions[c_idx]
        else: return self.directions[c_idx].get(l_idx, None)

    def feature_overlap(self, i, j): # i,j are camera indexes
        # https://stackoverflow.com/questions/18554012/intersecting-two-dictionaries
        overlapping_features = self.directions[i].keys() & self.directions[j].keys()
        return overlapping_features


    # fill the arrays
    def _load(self):    
        with open(self.landmark_path, 'r') as landmark_file:
            line = Dataset._preprocess_line(landmark_file.readline())
            while line != []:
                assert line[0] == "L:"
                self._add_new_landmark(line)
                line = Dataset._preprocess_line(landmark_file.readline())


        with open(self.camera_path, 'r') as dataset_file:
            # useful to assign directions to cameras
            c_idx = None

            line = Dataset._preprocess_line(dataset_file.readline())

            while line != []:
                if line[0] == "KF:":
                    # it's a camera
                    c_idx = self._add_new_camera(line)
                elif line[0] == "F:":
                    # it's a landmark (observed by the last added camera)
                    self._add_new_direction(line,c_idx)
                else:
                    raise Exception("First value must be a camera [KF] or a keypoint [F].")

                line = Dataset._preprocess_line(dataset_file.readline())


    @staticmethod
    def _preprocess_line(line):
        # split on spaces
        line = line.split(' ')
        # remove additional extra spaces
        line = [el for el in line if el != '']

        return line
    

    def _add_new_landmark(self, line):
        landmark_id = int(line[1])
        if self.gt:
            landmark_pose = np.array(line[self.l_slice], dtype=Dataset.dtype)
        else:
            # initialize the estimate to zeros
            landmark_pose = np.zeros(3, dtype=Dataset.dtype)

        self.landmark_poses[landmark_id] = landmark_pose            
        self.n_landmarks+=1


    def _add_new_camera(self, line) -> int:
        camera_id = int(line[1])

        pose_vector = np.array(line[self.c_slice], dtype=Dataset.dtype)

        # store the expanded pose in a 4x4 matrix
        camera_t = pose_vector[:3]
        quat = complete_quaternion(pose_vector[3:])
        camera_r = R.from_quat(quat).as_matrix()

        camera_pose = np.zeros((4,4),dtype=Dataset.dtype)
        camera_pose[:3,:3] = camera_r
        camera_pose[:3,3]  = camera_t

        self.camera_poses[camera_id] = camera_pose

        # initialize an empty dict to store the directions seen by this camera
        self.directions[camera_id] = {}

        self.n_cameras += 1

        return camera_id
    
    def _add_new_direction(self, line, c_idx):
        # register the observed keypoint in the relative camera dictionary
        camera_dict = self.directions[c_idx]

        l_id = int(line[2])
        direction = np.array(line[3:], dtype=Dataset.dtype)
        camera_dict[l_id] = direction