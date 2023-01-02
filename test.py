from utils import Dataset

def test_dataset():
    path = "./1B-CameraSFM/dataset.txt"
    d = Dataset(path)

    print(d._camera_idx)
    


if __name__ == "__main__":
    test_dataset()