from utils import Dataset, eight_point, decompose_E, skew
from sfm1b.translation_init import TranslationInit
from sfm1b.find_pairs import find_pairs
from sfm1b.eight_point_ransac import ransac
import numpy as np

def test_dataset(d):
    assert d.n_cameras == 98

    (tr,rot) = d.get_pose(65, gt=True)

def test_eight_point(d: Dataset):
    i = 11
    j = 12
    # id of overlapping features
    features = list(d.feature_overlap(i,j))
    features.sort()


    p1 = np.zeros((len(features),3))
    p2 = np.zeros((len(features),3))
    
    # fill this with the observed directions
    for p in range(len(features)):
        p1[p:] = d.observed_keypoints[i][features[p]]
        p2[p:] = d.observed_keypoints[j][features[p]]

    E_ = eight_point(p1[0:8,:], p2[0:8,:])
    E_/=E_[2,2]

    for i in range(len(features)):
        print(p1[i,:]@E_@p2[i,:])

def test_ransac(d: Dataset):
    i = 0
    j = 12
    # id of overlapping features
    features = list(d.feature_overlap(i,j))
    features.sort()
    p1 = np.zeros((len(features),3))
    p2 = np.zeros((len(features),3))
    
    # fill this with the observed directions
    for p in range(len(features)):
        p1[p:] = d.observed_keypoints[i][features[p]]
        p2[p:] = d.observed_keypoints[j][features[p]]
    E_, inliers = ransac(p1, p2, threshold=1e-3)
    
    print(len(inliers))
    print(E_)

    #import cv2
    # cv2 wants 2D points, divide by z
    #p1_norm = p1 / np.vstack((p1[:,2],p1[:,2],p1[:,2])).T
    #p2_norm = p2 / np.vstack((p2[:,2],p2[:,2],p1[:,2])).T
    #E, inliers = cv2.findEssentialMat(p1_norm[:,:2],p2_norm[:,:2], method=cv2.RANSAC, threshold=5e-3, prob=0.999)

    
    for i in range(len(features)):
        #print(f"me, {(p1[i,:]@E_@p2[i,:]):5f}") 
        #print("opencv,",(p2[i,:]@E@p1[i,:]))
        pass   

def eval_solutions(d: Dataset):
    rotation_errors = []
    translation_ratio = []
    for i in range(d.n_cameras):
        for j in range(i+1, d.n_cameras):
            t_i, R_i = d.get_pose(i)
            t_j, R_j = d.get_pose(j)
            t_i_gt, R_i_gt = d.get_pose(i,gt=True)
            t_j_gt, R_j_gt = d.get_pose(j,gt=True)

            R_delta = R_i.T @ R_j
            R_delta_gt = R_i_gt.T @ R_j_gt

            rotation_error = np.trace(np.eye(3) - R_delta.T @ R_delta_gt)
            # rotation_error_gt = np.trace(np.eye(3) - R_delta_gt.T @ R_delta_gt)
            rotation_errors.append(rotation_error)

            t_delta = R_i.T @ (t_j-t_i)
            norm = np.linalg.norm(t_delta)
            if norm != 0.0:
                t_delta /= np.linalg.norm(t_delta)
            
            t_delta_gt = R_i_gt.T @ (t_j_gt-t_i_gt)
            norm = np.linalg.norm(t_delta_gt)
            if norm != 0.0:
                t_delta_gt /= np.linalg.norm(t_delta_gt)

            ratio = (t_delta_gt / t_delta)
            translation_ratio.append(ratio)
            
            

    rotation_errors = np.array(rotation_errors)
    # numbers should be zero
    print(f"rotation error: {np.mean(rotation_errors)} +- {np.std(rotation_errors)}")
    translation_ratio = np.array(translation_ratio)
    # numbers should be equal with low std
    print(f"translation ratios: {np.mean(translation_ratio, axis=0)} +- {np.std(translation_ratio, axis=0)}")
    

if __name__ == "__main__":
    path = "./1B-CameraSFM/dataset.txt"
    landmark_path = "./1B-CameraSFM/GT_landmarks.txt"
    BA_path = "./1B-CameraSFM/input_BA.txt" 
    d = Dataset(path, landmark_path)
    #test_ransac(d)
    pairs = find_pairs(d, min_overlap=35)
    t = TranslationInit(d,pairs)

    eval_solutions(d)
