import numpy as np

class Dataset():

    pose_size = 6
    pose_dtype = np.float32

    def __init__(self, path: str, n_cameras=100):
        self._path = path
        self._n_cameras = n_cameras
        
        self._camera_poses = np.zeros((n_cameras, Dataset.pose_size), 
                                      dtype=Dataset.pose_dtype)
        self._camera_poses_gt = np.zeros_like(self._camera_poses)
        self._camera_idx = 0

        self._load()



    def _load(self):

        with open(self._path, 'r') as dataset_file:
            line = Dataset._preprocess_line(dataset_file.readline())

            # store all the lines in the file relative to a certain camera
            camera_kp_lines = []
        
            while line != []:
                if line[0] == "KF:":
                    # it's a camera
                    self._add_new_camera(line, camera_kp_lines)
                    camera_kp_lines = []
                elif line[0] == "F:":
                    # it's a landmark (observed by the last added camera)
                    camera_kp_lines.append(line)
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

    def _add_new_camera(self, camera_line, keypoint_lines):
        id = int(camera_line[1])
        gt = camera_line[2:2+6]
        est = camera_line[2+6:]

        # camera idxes are in order and contiguous
        assert id == self._camera_idx 
        
        self._camera_poses[id,:] = est
        self._camera_poses_gt[id, :] = gt

        self._camera_idx += 1
        