from dataset import Dataset
from sfm1b.find_pairs import Pair
from sfm1b.sicp import sicp
import matplotlib.pyplot as plt
import numpy as np
plot_path = "./plots"
data_path = "./1B-CameraSFM/dataset.txt"

def plot_overlap(d:Dataset):
    overlaps = []
    for i in range(d.n_cameras):
        for j in range(i+1,d.n_cameras):
            overlap = len(d.feature_overlap(i,j))
            overlaps.append(overlap)
    plt.hist(overlaps, bins=10)
    plt.title("frequency of overlaps between any two images")
    plt.xlabel("number of overlaps")
    plt.ylabel("occurrences")
    plt.savefig(f"{plot_path}/overlap.png")

def plot_pairs(d: Dataset, pairs: list[Pair]):
    pair_matrix = np.zeros((d.n_cameras,d.n_cameras), dtype=np.uint8)

    for pair in pairs:
        i = pair.i
        j = pair.j

        pair_matrix[i][j] = 255
        pair_matrix[j][i] = 255
    
    plt.matshow(pair_matrix)
    plt.savefig(f"{plot_path}/pairs.png")

def visualize_landmarks(d: Dataset, d_gt:Dataset, lines=True):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D


    # TODO
    assert d.n_landmarks == d_gt.n_landmarks
    gtposes = np.zeros((d_gt.n_landmarks,3))
    poses = np.zeros((d.n_landmarks,3))
    i=0
    for gtpose,pose in zip(d_gt.landmark_poses.values(), d.landmark_poses.values()):
        gtposes[i,:] = gtpose
        poses[i,:] = pose

        i+=1
    sicp_plot(poses,gtposes,lines)
    plt.show()

def sicp_plot(p1,p2, lines=True):
    X, chi_stats = sicp(p1,p2, n_iters=100, damping=100, threshold=1)
    ax = plt.axes(projection='3d')
    print(X)
    t = X[:3,3]
    R = X[:3,:3]
    s = X[3,3]

    print(chi_stats[-1])

    tf_p1 = ((R@p1.T).T + t)*s

    ax.scatter(tf_p1[:,0],tf_p1[:,1],tf_p1[:,2], c="green")
    ax.scatter(p2[:,0],p2[:,1],p2[:,2], c="red")
    if lines:
        for i in range(p1.shape[0]):
            print(f"GT:{p2[i]} ES:{tf_p1[i]} OR:{p1[i]}")
            xs = [tf_p1[i,0], p2[i,0]]
            ys = [tf_p1[i,1], p2[i,1]]
            zs = [tf_p1[i,2], p2[i,2]]

            ax.plot(xs,ys,zs, c="black")



def visualize_H(H: np.ndarray, filename="H"):
    import matplotlib.pyplot as plt
    H_to_show = np.where(H>0, 1,0)
    plt.matshow(H_to_show)
    plt.savefig(f"{plot_path}/{filename}.png")
    plt.close()


def plot_dataset(d: Dataset, d_gt: Dataset):
    ax = plt.axes(projection='3d')


    plt.axis("equal")
    plt.xlim((-10,10))
    
    # find transform between landmarks 
    gtposes = np.zeros((d_gt.n_landmarks,3))
    poses = np.zeros((d.n_landmarks,3))
    i=0
    for gtpose,pose in zip(d_gt.landmark_poses.values(), d.landmark_poses.values()):
        gtposes[i,:] = gtpose
        poses[i,:] = pose

        i+=1
    X, chi_stats = sicp(poses,gtposes, n_iters=100, damping=100, threshold=1)

    tp = X[:3,3]
    Rp = X[:3,:3]
    sp = X[3,3]

    tf_p1 = ((Rp@poses.T).T + tp)*sp

    ax.scatter(tf_p1[:,0],tf_p1[:,1],tf_p1[:,2], c="green")
    ax.scatter(gtposes[:,0],gtposes[:,1],gtposes[:,2], c="red", alpha=0.3)
    for idx in range(d.n_cameras):
        t, R = d.get_camera_pose(idx)
        camera_pose_tf = np.zeros((4,4))
        camera_pose_tf[:3,:3] = Rp@R
        camera_pose_tf[:3,3]  = Rp@t +tp
        camera_pose_tf[3,3]   = sp
        plot_camera_at(camera_pose_tf, ax, color="green")

        t_gt, R_gt = d_gt.get_camera_pose(idx)
        camera_pose_gt = np.zeros((4,4))
        camera_pose_gt[:3,:3] = R_gt
        camera_pose_gt[:3,3]  = t_gt
        camera_pose_gt[3,3]   = 1
        plot_camera_at(camera_pose_gt, ax, color="red", alpha=0.3)

        xs = [sp*camera_pose_tf[0,3], camera_pose_gt[0,3]]
        ys = [sp*camera_pose_tf[1,3], camera_pose_gt[1,3]]
        zs = [sp*camera_pose_tf[2,3], camera_pose_gt[2,3]]
        plt.xlim((-10,10))
        plt.axis("equal")
        ax.plot(xs,ys,zs, c="black")
        

    plt.xlim((-10,10))
    plt.axis("equal")

    plt.show()


def plot_camera_at(T, plt, color="blue", alpha=1.0):
    print(T)
    frame = np.eye(3)
    R = T[:3,:3]
    # directions
    x = R@frame[:,0]
    y = R@frame[:,1]
    z = R@frame[:,2]

    t = T[3,3]*T[:3,3] # where is camera
    
    plt.quiver(t[0],t[1],t[2], x,y,z, color=color, alpha=alpha)


if __name__ == "__main__":
    plot_pairs()