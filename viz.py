from utils import Dataset
import matplotlib.pyplot as plt
import numpy as np
plot_path = "./plots"
data_path = "./1B-CameraSFM/dataset.txt"

def plot_pairs():
    d = Dataset(data_path)

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

def visualize_dataset(d: Dataset):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    ax = plt.axes(projection='3d')

    # TODO

def visualize_H(H: np.ndarray, filename="H"):
    import matplotlib.pyplot as plt
    H_to_show = np.where(H>0, 1,0)
    plt.matshow(H_to_show)
    plt.savefig(f"{plot_path}/{filename}.png")


if __name__ == "__main__":
    plot_pairs()