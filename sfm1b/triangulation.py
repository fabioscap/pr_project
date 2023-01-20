from dataset import Dataset
import numpy as np

def triangulate_landmarks(d: Dataset):
    for idx in d.landmark_poses.keys():
        lines = []
        for camera in range(d.n_cameras):
            if idx in d.get_direction(camera):
                ti, Ri = d.get_camera_pose(camera)
                di = d.get_direction(camera,idx)

                n = Ri@di
                lines.append((n, ti))

        landmark_pose = triangulate_lines(lines)
        d.set_landmark_pose(idx,landmark_pose)

        # valid only with fake dataset
        # print(f"error: {np.linalg.norm(d.get_landmark_pose(idx,gt=True)-landmark_pose)}")
        print(landmark_pose)


# https://silo.tips/download/least-squares-intersection-of-lines
def triangulate_lines(lines):
    dim = lines[0][0].shape[0]

    H = np.zeros((dim,dim))
    b = np.zeros(dim)

    for line in lines:
        n = line[0]
        ti = line[1]
        # print(n,ti)
        proj = np.eye(dim) - np.outer(n,n)
        H += proj
        b += proj@ti
    
    return np.linalg.solve(H,b)