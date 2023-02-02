from dataset import Dataset, gen_output
from sfm1b.translation_init import TranslationInit
from sfm1b.find_pairs import find_pairs
from sfm1b.bundle_adjustment import BA
from sfm1b.triangulation import triangulate_landmarks
from test import eval_landmarks, eval_solutions


path = "./1B-CameraSFM/dataset.txt"
landmark_path = "./1B-CameraSFM/GT_landmarks.txt"
output_path = "./output/"

d = Dataset(path, landmark_path, ground_truth=False)
d_gt = Dataset(path, landmark_path, ground_truth=True)


pairs = find_pairs(d, min_overlap=25)
t = TranslationInit(d, pairs)
triangulate_landmarks(d)

b = BA(d, n_iters=10, damping=1)
b._solve()

eval_solutions(d,d_gt)
eval_landmarks(d,d_gt)

gen_output(d,d_gt, output_path)

d_= Dataset("./output/dataset.txt", "./output/landmarks.txt", ground_truth=False)
d_gt_ = Dataset("./output/dataset.txt", "./output/landmarks.txt", ground_truth=True)

eval_solutions(d,d_gt)
eval_solutions(d_,d_gt_)

# sanity checks
import numpy as np
for c in range(d.n_cameras):
    pose = d.get_camera_pose(c)
    pose_ = d_.get_camera_pose(c)
    pose_gt = d_gt.get_camera_pose(c)
    pose_gt_ = d_gt_.get_camera_pose(c)

    assert np.all(np.abs(pose[0] - pose_[0]) < 1e-4) 
    assert np.all(np.abs(pose[1] - pose_[1]) < 1e-4)

    assert np.all(np.abs(pose_gt[0] - pose_gt_[0]) < 1e-4)
    assert np.all(np.abs(pose_gt[1] - pose_gt_[1]) < 1e-4)