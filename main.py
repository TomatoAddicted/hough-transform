from model.layers import Hough_Transform, AccumulationSpace  #, plot_hist
from data.labels import load_video, get_segmentation
import cv2 as cv
import numpy as np
import time
import os
import torch
import matplotlib.pyplot as plt

# Press Umschalt+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

"""Nothing to see here: under construction
def train_treshold(acc_space, annotations, FPR, tol = 0.001, thr_init = 0):
    n_annots = len(annotations)
    vote_annots = torch.zeros(n_annots)
    for i in range(n_annots):
        vote_annots = acc_space[annotations[i]]  # using values of annotation as indicies for the acc_space



    thr = thr_init

    if get_FPR(acc_spaces, annotations, thr) <= FPR and get_FPR(acc_spaces, annotations, thr) >= FPR - tol:
        # perfect threshold found
        return thr
    elif get_FPR(acc_spaces, annotations, thr) <= FPR:
        # threshold is too large (we drop too many relevant lines)

    elif get_FPR(acc_spaces, annotations, thr) >= FPR - tol:
        # threshold is too small (we should drop more relevant lines)

def get_FPR(acc_spaces, annotaions, thr):
    fpr = 0.05

    return FPR

def apply_threshold(acc_space, thr):
    # returns thresholded acc_space
    # TODO: implement some kind of threshold
    return acc_space 
    
"""


def plot_lines():
    print("Version 1.0.0.2")
    #root = "C:/Users/HP/Documents/Uni/10.Semester/Pallet_detection/data/"
    root = "data/"
    #root = "/data/pallet/"
    #filename = "pallet.jpg"
    filename = "edge.jpg"
    #filename = "triangles.jpg"
    #filename = "hallway.jpg"
    #filename = "hall1.jpg"
    path = os.path.join(root, filename)
    img = cv.imread(path)
    rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    gray = cv.cvtColor(rgb_img, cv.COLOR_RGB2GRAY)
    gray = shrink_h(gray, 128)
    """
    mask = np.zeros(gray.shape) # only selecting right doorframe
    mask[:, :] = True
    mask = torch.Tensor(mask)
    plt.imshow(mask)
    plt.show()
    mask = (mask > 0)
    """
    plt.imshow(gray, cmap="gray")
    plt.show()

    start = time.time()

    HT = Hough_Transform(input_dim=gray.shape, output_dim=(200, 200), h_plane=7, w_plane=7, origin=(64, 64))

    acc_space = HT.forward(gray,
                           #mask=mask,
                           visualize=False)
    acc_space.plot_values()


    radius = 5

    indices_local_maxima = np.array(acc_space.get_local_maxima(r=radius))

    end = time.time()
    print(f"extracting local maxima took {int(end - start)} seconds")

    local_maxima_values = acc_space.values[indices_local_maxima[:, 0], indices_local_maxima[:, 1]] # acc_space.values[indices_local_maxima]

    print(local_maxima_values)

    threshold = 1

    indices_lines = indices_local_maxima[local_maxima_values >= threshold]

    theta_lines = acc_space.get_value_theta(indices_lines[:, 0])
    rho_lines = acc_space.get_value_rho(indices_lines[:, 1])

    acc_space.plot_values("Hough Space " + filename)

    print(theta_lines)
    print(rho_lines)

    y = [0, img.shape[1] - 1] # y value boundaries (we only need two points per line)
    x = np.zeros((len(theta_lines), 2))
    for i in range(len(x)):
        for j in range(len(y)):
            # x-coordinates are calculated by polar coordinates of the lines
            x[i, j] = (rho_lines[i] - y[j] * np.sin(theta_lines[i]) ) / np.cos(theta_lines[i])
            # rho = x* cos(theta) + y*sin(theta)
        plt.plot(x[i], y, linewidth = 4)

    plt.imshow(gray, cmap = "gray")
    plt.title(f"Threshold: {threshold}, radius: {radius}")
    plt.show()







def test_HT_threshold_study():
    """
    path = "C:/Users/HP/Documents/Uni/10.Semester/Pallet_detection/data/pallet.jpg"
    img = cv.imread(path)
    rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    gray = cv.cvtColor(rgb_img, cv.COLOR_RGB2GRAY)
    gray = shrink_h(gray, 64)
    """
    img_dim = (100, 177)  # same ratio as 1080x1920 (simulation data size)


    path_vid = "data/vid.mp4"
    vid = load_video(path_vid)  # loads vid in RGB (hopefully)
    img = vid[-1]
    img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    img = cv.resize(img, (img_dim[1], img_dim[0]))  # cv needs the dims switched

    path_seg = "data/vid.mp4"
    seg_vid_raw = load_video(path_seg)
    seg_raw = seg_vid_raw[-1]
    mask = get_segmentation(seg_raw)
    mask = cv.resize(mask, (img_dim[1], img_dim[0]))  # cv needs the dims switched

    pal = mask * img

    mask = torch.Tensor(mask)
    mask = (mask > 0)


    plt.imshow(img, cmap="gray")
    plt.show()
    plt.imshow(pal, cmap="gray")
    plt.show()

    HT = Hough_Transform(input_dim=img_dim, output_dim=(200, 200))

    acc_space_img = HT.forward(img)
    #acc_space_pal_cropped = HT.forward(pal)
    acc_space_pal_masked = HT.forward(img, mask=mask)

    acc_space_img.plot_values("Hough Space whole Image")
    #acc_space_pal_cropped.plot_values("Hough Space cropped pallet")
    acc_space_pal_masked.plot_values("Hough Space masked pallet")


    x_hist_img, y_hist_img = acc_space_img.plot_hist(clip_hist=200, log_y=True, show=False)
    #x_hist_pal_cropped, y_hist_pal_cropped = acc_space_pal_cropped.plot_hist(clip_hist=200, log_y=True, show=False)
    x_hist_pal_masked, y_hist_pal_masked = acc_space_pal_masked.plot_hist(clip_hist=200, log_y=True, show=False)

    plt.plot(x_hist_img, y_hist_img, label="image")
    #plt.plot(x_hist_pal_cropped, y_hist_pal_cropped, label="pallet cropped")
    plt.plot(x_hist_pal_masked, y_hist_pal_masked, label="pallet masked")
    plt.legend()
    plt.title("Histogram of Accumulation Space")
    plt.show()

    list_thr_img, n_maxima_img = acc_space_img.get_n_maxima_plot(steps=50, r=5, log_y=True, show=False)
    #list_thr_pal_cropped, n_maxima_pal_cropped = acc_space_pal_cropped.get_n_maxima_plot(steps=50, r=5, log_y=True, show=False)
    list_thr_pal_masked, n_maxima_pal_masked = acc_space_pal_masked.get_n_maxima_plot(steps=50, r=5, log_y=True, show=False)

    plt.plot(list_thr_img, n_maxima_img, label="image")
    #plt.plot(list_thr_pal_cropped, n_maxima_pal_cropped, label="pallet cropped")
    plt.plot(list_thr_pal_masked, n_maxima_pal_masked, label="pallet masked")
    plt.title("Amount of remaining maxima after applying threshold")
    plt.xlabel("Threshold")
    plt.ylabel("log(N)")
    plt.legend()
    plt.show()

    """
    start = time.time()
    HT = Hough_Transform(input_dim=gray.shape, output_dim=(200, 200))
    acc_space = HT.forward(gray, visualize=True)
    end = time.time()
    print(end - start)
    """

def testing_noise_estimator():
    root = "data/"
    filename = "hallway.jpg"
    path = os.path.join(root, filename)
    img = cv.imread(path)
    rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    gray = cv.cvtColor(rgb_img, cv.COLOR_RGB2GRAY)
    gray = shrink_h(gray, 32)

    estimator_plot(gray)



def estimator_plot(image):
    sigma_range = np.arange(0, 15)
    var_range = sigma_range**2
    var_estimated = np.zeros(sigma_range.shape)
    HT = Hough_Transform(input_dim=image.shape, output_dim=(200, 200), h_plane=3, w_plane=3)
    for i, sigma in enumerate(sigma_range):
        noise = np.random.normal(loc=0, scale=sigma, size=image.shape)
        noisy_image = np.clip(image + noise, a_min=0, a_max=255)
        var_estimated[i] = HT.approximate_plane(noisy_image, visualize=False)[-1]

    plt.plot(sigma_range, var_estimated, label="estimated")
    plt.plot(sigma_range, var_range, label="correct")
    plt.xlabel("sigma of applied noise")
    plt.ylabel("estimated variance")
    plt.legend()
    plt.show()


def shrink_h(img, h_des):
    # shrinking the images to a desired height, while keeping the side ratio (w / h) constant
    h = img.shape[0]
    w = img.shape[1]
    w_des = int(h_des * w / h)  # desired width
    return cv.resize(img, (w_des, h_des))

if __name__ == "__main__":
    #test_HT_threshold_study()
    plot_lines()
    #testing_noise_estimator()