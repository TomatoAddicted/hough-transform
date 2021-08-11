import matplotlib.pyplot as plt
import numpy as np
import torch
import matplotlib.pyplot as plt

import pims


def test():
    print("start")
    vid = load_video('vid.mp4')
    print("video loaded")
    seg_frame = get_segmentation(vid[-1])
    print("frame segmented")
    plt.imshow(seg_frame, cmap="gray")
    plt.show()


def load_video(file):
    v = pims.Video(file)
    return np.array(v)


def get_segmentation(img, color=[0, 255, 9]):  # color of segmentations from simulation is [0, 255, 9]
    seg_img = np.zeros(img.shape)
    seg_img[img == color] = 1
    seg_img[np.abs(img - color) <= 50] = 1  # segmentation array has 3 identical channels
    return seg_img[:, :, 0] * seg_img[:, :, 1] * seg_img[:, :, 2]  # only return one of them

def get_segmentation_video(file):
    vid = load_video(file)
    seg_vid = torch.zeros(vid.shape)
    for i, frame in enumerate(vid):
        seg_vid[i] = get_segmentation(frame)
    return seg_vid


if __name__ == "__main__":
    test()