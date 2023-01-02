import numpy as np

class Dataset():

    pose_size = 6
    dtype = np.float32

    def __init__(self, path: str, max_cameras=100):
        self.path = path
        self.max_cameras = max_cameras
        
        self.camera_poses = np.zeros((max_cameras, Dataset.pose_size), 
                                      dtype=Dataset.dtype)
        self.camera_poses_gt = np.zeros_like(self.camera_poses)
        
        # keep track of how many cameras you have
        self.n_cameras = -1

        # for each camera append a dict {kpoint_id->direction}
        self.observed_keypoints = {}

        self._load()

    def _load(self):

        with open(self.path, 'r') as dataset_file:
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
        self.camera_dict = self.observed_keypoints[idx]

        kpoint_id = int(line[2])
        direction = np.array(line[3:], dtype=Dataset.dtype)
        self.camera_dict[kpoint_id] = direction

        return kpoint_id

 

    def feature_overlap(self, i, j):
        # https://stackoverflow.com/questions/18554012/intersecting-two-dictionaries
        overlapping_features = self.observed_keypoints[i].keys() & self.observed_keypoints[j].keys()
        return len(overlapping_features)
