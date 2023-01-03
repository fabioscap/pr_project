from utils import Dataset
import matplotlib.pyplot as plt
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


if __name__ == "__main__":
    plot_pairs()