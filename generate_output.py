from dataset import Dataset, gen_output
from sfm1b.translation_init import TranslationInit
from sfm1b.find_pairs import find_pairs
from sfm1b.bundle_adjustment import BA
from sfm1b.triangulation import triangulate_landmarks
from test import eval_landmarks, eval_solutions
import numpy as np


path = "./1B-CameraSFM/dataset.txt"
landmark_path = "./1B-CameraSFM/GT_landmarks.txt"
output_path = "./output/"

d = Dataset(path, landmark_path, ground_truth=False)
d_gt = Dataset(path, landmark_path, ground_truth=True)


pairs = find_pairs(d, min_overlap=33)
t = TranslationInit(d, pairs)

triangulate_landmarks(d)

b = BA(d, n_iters=6, damping=0.5)
b._solve()
"""
import matplotlib.pyplot as plt
plt.plot(b.chi_stats)
plt.show()
"""

rotation_errors, translation_ratio = eval_solutions(d,d_gt)
landmark_rmse = eval_landmarks(d, d_gt)

with open("./output/stats.txt","w") as file:
   file.write(f"rotation errors: {rotation_errors.mean()}+-{rotation_errors.std()}\n")
   file.write(f"translation ratios: {np.mean(translation_ratio, axis=0)} +- {np.std(translation_ratio, axis=0)}\n")
   file.write(f"landmark rmse: {landmark_rmse}\n")
gen_output(d,d_gt, output_path)

