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

def visualize_landmarks(d: Dataset):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    ax = plt.axes(projection='3d')

    # TODO
    gtposes = np.zeros((d.n_landmarks,3))
    poses = np.zeros_like(gtposes)
    i=0
    for gtpose,pose in zip(d.landmark_poses_gt.values(), d.landmark_poses.values()):
        gtposes[i,:] = gtpose
        poses[i,:] = pose

        i+=1

    # TODO move this in another function
    # maybe this takes as input two point clouds instead of the whole dataset
    X, _ = sicp(gtposes, poses, n_iters=10, damping=10)
    s = X[3,3]
    print("SCALE", s)

    poses_homog = np.hstack((poses, np.ones((poses.shape[0], 1), dtype=poses.dtype)))
    tf_poses_homog = (np.linalg.inv(X)@poses_homog.T).T

    tf_poses = tf_poses_homog[:,:-1] /  tf_poses_homog[:,-1].reshape(-1,1)

    ax.scatter(gtposes[:,0],gtposes[:,1],gtposes[:,2], c="red")
    ax.scatter(tf_poses[:,0],tf_poses[:,1],tf_poses[:,2], c="green")
    plt.show()

def visualize_H(H: np.ndarray, filename="H"):
    import matplotlib.pyplot as plt
    H_to_show = np.where(H>0, 1,0)
    plt.matshow(H_to_show)
    plt.savefig(f"{plot_path}/{filename}.png")


if __name__ == "__main__":
    plot_pairs()