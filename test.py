from utils import Dataset, eight_point
from sfm1b.translation_init import TranslationInit
import numpy as np

def test_dataset(d):
    assert d.n_cameras == 98

    (tr,rot) = d.get_pose(65, gt=True)

def test_eight_point(d: Dataset):
    i = 11
    j = 12
    # id of overlapping features
    features = list(d.feature_overlap(i,j))


    p1 = np.zeros((len(features),3))
    p2 = np.zeros((len(features),3))
    
    # fill this with the observed directions
    for p in range(len(features)):
        p1[p:] = d.observed_keypoints[i][features[p]]
        p2[p:] = d.observed_keypoints[j][features[p]]

    import cv2
    # cv2 wants 2D points, divide by z
    p1_norm = p1 / np.vstack((p1[:,2],p1[:,2],p1[:,2])).T
    p2_norm = p2 / np.vstack((p2[:,2],p2[:,2],p1[:,2])).T
    E, inliers = cv2.findEssentialMat(p1_norm[:,:2],p2_norm[:,:2], method=cv2.RANSAC, threshold=1e-3, prob=0.999)

    # remove outliers
    idxes = [i for i in range(len(features)) if inliers[i]]

    p1_8 = p1[idxes][0:8,:]
    p2_8 = p2[idxes][0:8,:]
    E_ = eight_point(p1_8, p2_8)

    for i in idxes:
        # both should be low for inliers
        print(p2[i,:]@E@p1[i,:], p1[i,:]@E_@p2[i,:])

if __name__ == "__main__":
    path = "./1B-CameraSFM/dataset.txt"
    d = Dataset(path)
    test_dataset(d)
    test_eight_point(d)