from utils import eight_point, decompose_E, skew, v2t, t2v, generate_fake_data, v2s
from dataset import Dataset
import viz
from sfm1b.translation_init import TranslationInit
from sfm1b.find_pairs import find_pairs
from sfm1b.eight_point_ransac import ransac, ransac_opencv
from sfm1b.bundle_adjustment import BA
from sfm1b.triangulation import triangulate_landmarks, triangulate_lines
from sfm1b.sicp import sicp
import numpy as np

def test_dataset(d: Dataset):
    assert d.n_cameras == 98

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

def test_epipolar(d: Dataset):
    
    i = 0
    j = 1

    features = list(d.feature_overlap(i,j))
    t_i, R_i = d.get_camera_pose(i)
    t_j, R_j = d.get_camera_pose(j)

    for feature in features:
        # both cameras see this
        # compute the distance from rays
        
        # transform both rays in world frame
        d_i = R_i@d.directions[i][feature]
        d_j = R_j@d.directions[j][feature]
        
        # block 15 slide 3

        delta_D = np.stack((d_i, -d_j), axis=-1)
        assert delta_D.shape == (3,2)

        delta_p = t_i-t_j 

        s_star = - np.linalg.inv(delta_D.T@delta_D) @ delta_D.T @ delta_p

        # evaluate the lines at those points and compute the difference
        p_i = t_i + d_i * s_star[0]
        p_j = t_j + d_j * s_star[1]

        err = p_i - p_j

        print(err.T@err)


def test_ransac(d: Dataset):
    i = 0
    j = 1
    # id of overlapping features
    features = list(d.feature_overlap(i,j))
    features.sort()
    p1 = np.zeros((len(features),3))
    p2 = np.zeros((len(features),3))
    
    # fill this with the observed directions
    for p in range(len(features)):
        p1[p:] = d.observed_keypoints[i][features[p]]
        p2[p:] = d.observed_keypoints[j][features[p]]
    E_, inliers = ransac_opencv(p1, p2, threshold=1e-3)
    print(f"inlier ratio in pair {i}-{j} {len(inliers) / len(features)}")
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
            t_i, R_i = d.get_camera_pose(i)
            t_j, R_j = d.get_camera_pose(j)
            t_i_gt, R_i_gt = d.get_camera_pose(i,gt=True)
            t_j_gt, R_j_gt = d.get_camera_pose(j,gt=True)

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
 
def test_triangulation():
    # two lines
    t1 = np.zeros(3)
    t2 = np.array([1.0,1.0,1.0])

    d1 = np.array([1.0, 1.0,1.0])
    d1 /= np.linalg.norm(d1)

    d2 = np.array([0,1.0,0.0])
    d2 /= np.linalg.norm(d2)

    l1 = (d1,t1)
    l2 = (d2,t2)

    lines = [l2,l1]

    point = triangulate_lines(lines)
    print(point)

def test_sicp():
    n = 1000
    noise = 0.0001
    n_iters = 100

    p1 = np.random.rand(n,3)

    v = np.random.rand(7)
    S = v2s(v)

    p1_homog = np.hstack((p1, np.ones((p1.shape[0], 1), dtype=p1.dtype)))
    p2_homog = (S@p1_homog.T).T
    
    p2 = p2_homog[:,:-1] / p2_homog[:,-1].reshape(-1,1)
    p2 += np.random.rand(*p2.shape)*noise

    v_guess = v + np.array([-2,0.1,-0.2,-0.1,0.1,0.0,0.01])
    S_guess = v2s(v_guess)

    X, chi_stats = sicp(p1,p2, n_iters=n_iters, damping=500, initial_guess=S)

    import matplotlib.pyplot as plt
    plt.plot(range(n_iters),chi_stats)
    plt.show()
    

if __name__ == "__main__":
    path = "./1B-CameraSFM/dataset.txt"
    landmark_path = "./1B-CameraSFM/GT_landmarks.txt"
    BA_path = "./1B-CameraSFM/input_BA.txt" 
    generate_fake_data("./fake_data", 10, 40, 0.)
    d = Dataset("./fake_data/dataset.txt", "./fake_data/GT_landmarks.txt", ground_truth=True) 

    # d = Dataset(path, landmark_path, ground_truth=False)
