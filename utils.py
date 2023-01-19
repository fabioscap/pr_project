import numpy as np
from scipy.spatial.transform import Rotation as R
import os, math, random

# estimate the essential matrix between two cameras
# given 8 correspondences
def eight_point( points_i: np.ndarray, points_j: np.ndarray):
    assert points_i.shape == (8,3)
    assert points_j.shape == (8,3)

    return eight_point_LS(points_i, points_j)[0]

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

def t2v(T: np.ndarray)->np.ndarray:
    assert T.shape==(4,4)

    v = np.zeros(6, dtype=T.dtype)

    v[:3] = T[:3,3]
    v[3:] = R.from_matrix(T[:3,:3]).as_quat()[:-1] # spare the last component

    return v

def v2t(v: np.ndarray)->np.ndarray:
    assert v.reshape(-1).shape == (6,)

    T = np.eye(4)
    T[:3,3] = v[:3]

    quat = complete_quaternion(v[3:])

    T[:3,:3] = R.from_quat(quat).as_matrix()

    return T


def generate_fake_data(path, n_cameras, n_landmarks, observation_noise=0.01):
    # generate a dataset containing noise free data
    bounds_x = (-10,10)
    bounds_y = (-10,10)
    bounds_z = (0,10)

    # scatter landmarks
    landmarks = np.random.rand(n_landmarks, 3)

    landmarks[:,0] = landmarks[:,0]*(bounds_x[1]-bounds_x[0]) + bounds_x[0]
    landmarks[:,1] = landmarks[:,1]*(bounds_y[1]-bounds_y[0]) + bounds_y[0]
    landmarks[:,2] = landmarks[:,2]*(bounds_z[1]-bounds_z[0]) + bounds_z[0]


    landmarks_path = os.path.join(path,"GT_landmarks.txt")
    with open(landmarks_path,"w") as file:
        for i in range(n_landmarks):
            line = f"L: {i} {landmarks[i,0]} {landmarks[i,1]} {landmarks[i,2]}\n"
            file.write(line)

    cameras = np.random.rand(n_cameras, 6)

    cameras[:,0] = cameras[:,0]*(bounds_x[1]-bounds_x[0]) + bounds_x[0]
    cameras[:,1] = cameras[:,1]*(bounds_y[1]-bounds_y[0]) + bounds_y[0]
    cameras[:,2] = cameras[:,2]*(bounds_z[1]-bounds_z[0]) + bounds_z[0]

    cameras[:,3] = cameras[:,3]*(2*math.pi) - math.pi
    cameras[:,4] = cameras[:,4]*(2*math.pi) - math.pi
    cameras[:,5] = cameras[:,5]*(2*math.pi) - math.pi

    observed_keypoints = {}

    for camera_id in range(n_cameras):
        # each camera observes some landmarks
        observed_keypoints[camera_id] = {}
        n_observed = random.randint(0,n_landmarks//3) + n_landmarks//3

        observed_landmarks_ids = random.sample(range(n_landmarks), k=n_observed)

        for landmark_id in observed_landmarks_ids:
            # global landmark position
            x_l = landmarks[landmark_id]

            # camera pose
            t_c = cameras[camera_id][:3]
            R_c = R.from_euler(Dataset.angles_seq, cameras[camera_id][3:]).as_matrix()

            landmark_in_camera = R_c.T@(x_l - t_c)
            landmark_in_camera /= np.linalg.norm(landmark_in_camera)

            # add some noise
            landmark_in_camera += np.random.randn(3)*observation_noise

            observed_keypoints[camera_id][landmark_id] = landmark_in_camera
    
    cameras_path = os.path.join(path,"dataset.txt")
    with open(cameras_path,"w") as file:
        for i in range(n_cameras):
            line = f"KF: {i} {cameras[i,0]} {cameras[i,1]} {cameras[i,2]} {cameras[i,3]} {cameras[i,4]} {cameras[i,5]} \
{cameras[i,0]} {cameras[i,1]} {cameras[i,2]} {cameras[i,3]} {cameras[i,4]} {cameras[i,5]}\n"
            file.write(line)
            n = 0
            for j in observed_keypoints[i].keys():
                line = f"F: {n} {j} {observed_keypoints[i][j][0]} {observed_keypoints[i][j][1]} {observed_keypoints[i][j][2]}\n"
                file.write(line)
                n+= 1


def v2s(v: np.ndarray)->np.ndarray:
    assert v.reshape(-1).shape == (7,)

    T = v2t(v[:-1])
    T[3,3] = np.exp(-v[-1])
    return T

def complete_quaternion(q:np.ndarray)->np.ndarray:
    out = np.zeros(4, dtype=q.dtype)
    out[:-1] = q.copy()
    out[-1] = 1 - np.linalg.norm(q)

    return out