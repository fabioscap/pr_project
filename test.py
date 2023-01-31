from utils import eight_point, decompose_E, skew, v2t, t2v, generate_fake_data, v2s
from dataset import Dataset
import viz
from sfm1b.translation_init import TranslationInit
from sfm1b.find_pairs import find_pairs
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
        p1[p:] = d.get_direction(i,features[p])
        p2[p:] = d.get_direction(j,features[p])

    E_ = eight_point(p1[0:8,:], p2[0:8,:])
    E_/=E_[2,2]

    for i in range(len(features)):
        print(p1[i,:]@E_@p2[i,:])

def test_epipolar(d: Dataset):
    
    errs = []

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

        errs.append(err.T@err)
    errs = np.array(errs)
    print(f"{errs.mean()} +- {errs.std()}")

def eval_solutions(d: Dataset, d_gt: Dataset):
    rotation_errors = []
    translation_ratio = []
    for i in range(d.n_cameras):
        for j in range(i+1, d.n_cameras):
            t_i, R_i = d.get_camera_pose(i)
            t_j, R_j = d.get_camera_pose(j)
            t_i_gt, R_i_gt = d_gt.get_camera_pose(i)
            t_j_gt, R_j_gt = d_gt.get_camera_pose(j)

            R_delta = R_i.T @ R_j
            R_delta_gt = R_i_gt.T @ R_j_gt

            rotation_error = np.trace(np.eye(3) - R_delta.T @ R_delta_gt)

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
    np.random.seed(1834913)
    n = 1000
    noise = 0.
    n_iters = 100

    p1 = np.random.rand(n,3)*10

    v = np.array([20.0, 1.0, -10.0, 1.57, 0.3, -1.6, -1])
    S = v2s(v)
    print(S)
    p1_homog = np.hstack((p1, np.ones((p1.shape[0], 1), dtype=p1.dtype)))
    p2_homog = (S@p1_homog.T).T
    
    p2 = p2_homog[:,:-1] * p2_homog[:,-1].reshape(-1,1)
    p2 += (np.random.rand(*p2.shape)-0.5)*noise

    v_guess = v + np.array([-2,0.1,-0.2,-0.1,0.1,0.0,0.01])
    S_guess = v2s(v_guess)

    X, chi_stats = sicp(p1,p2, n_iters=n_iters, damping=10, threshold=2.0)
    np.set_printoptions(suppress=True)
    print(X)
    import matplotlib.pyplot as plt
    plt.plot(range(n_iters),chi_stats)
    plt.yscale("log")
    plt.show()
    

def eval_landmarks(d: Dataset, d_gt: Dataset):
    gtposes = np.zeros((d_gt.n_landmarks,3))
    poses = np.zeros((d.n_landmarks,3))
    i=0
    for gtpose,pose in zip(d_gt.landmark_poses.values(), d.landmark_poses.values()):
        gtposes[i,:] = gtpose
        poses[i,:] = pose
        i+=1
    
    X, chi_stats = sicp(poses,gtposes, n_iters=1000, damping=100, threshold=1)

    t = X[:3,3]
    R = X[:3,:3]
    s = X[3,3]

    tf_poses = ((R@poses.T).T + t)*s

    rmse = 0.0

    deltas = gtposes - tf_poses

    for l in range(d.n_landmarks):
        rmse += np.dot(deltas[l], deltas[l])

    rmse = np.sqrt(rmse.sum() / d.n_landmarks)

    print(f"rmse: {rmse}")
    return rmse

if __name__ == "__main__":
    path = "./1B-CameraSFM/dataset.txt"
    landmark_path = "./1B-CameraSFM/GT_landmarks.txt"
    BA_path = "./1B-CameraSFM/input_BA.txt" 
    
    # generate_fake_data("./fake_data", 99, 86, 0.)

    dataset = "true" # "true" "fake" "ba"
    if dataset == "fake":
        d_gt = Dataset("./fake_data/dataset.txt", "./fake_data/GT_landmarks.txt", ground_truth=True) 
        d = Dataset("./fake_data/dataset.txt", "./fake_data/GT_landmarks.txt", ground_truth=False) 
    elif dataset == "true":
        d = Dataset(path, landmark_path, ground_truth=False)
        d_gt = Dataset(path, landmark_path, ground_truth=True)
    elif dataset == "ba":
        d = Dataset(BA_path, landmark_path, ground_truth=False)
        d_gt = Dataset(BA_path, landmark_path, ground_truth=True)


    # eval_solutions(d,d_gt)
    # eval_landmarks(d,d_gt)

    pairs = find_pairs(d, min_overlap=30)
    # viz.visualize_landmarks(d,d_gt)

    t = TranslationInit(d, pairs)
    
    # eval_solutions(d,d_gt)
    triangulate_landmarks(d)
    
    eval_solutions(d,d_gt)
    eval_landmarks(d,d_gt)

    # viz.visualize_landmarks(d,d_gt,lines=True)
    b = BA(d, n_iters=4, damping=2)
    b._solve()
    
    # viz.visualize_H(b._build_H_b()[0])

    eval_solutions(d,d_gt)
    eval_landmarks(d,d_gt)

    viz.plot_dataset(d,d_gt)
    
    
