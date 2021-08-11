import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats.mvn import mvnun


class Hough_Transform(nn.Module):
    """
    performs hough transform on input images.
    returns the accumulated hough space
    Parameters:
        input_dim : dimensions of input image
        output_dim : size of hough space. Determines bin size for voting.
        h_plane, w_plane : dimensions of area to fit plane around each pixel
        thr_var : Threshold for variance. Each vote surpassing the threshold is considered insane and denied voting
    """

    def __init__(self,
                 input_dim=(64, 64),
                 output_dim=(100, 100),
                 h_plane=3,
                 w_plane=3,
                 origin=(0, 0),
                 thr_var=100):

        self.input_dim = input_dim
        self.H = input_dim[0]
        self.W = input_dim[1]
        self.output_dim = output_dim
        self.h_plane = h_plane
        self.w_plane = w_plane
        self.origin = origin
        self.thr_var = thr_var

        # defining value ranges for hough space:
        self.theta_min = - np.pi / 2
        self.theta_max = np.pi / 2
        self.rho_min = - np.sqrt(max(origin[0], self.H - origin[0])**2 + max(origin[1], self.W - origin[1])**2)
        self.rho_max = np.sqrt(max(origin[0], self.H - origin[0])**2 + max(origin[1], self.W - origin[1])**2)



    def forward(self, img, mask=None, visualize = False):

        alpha, beta, gamma, var_alpha, var_beta, covar_alpha_beta, noise_var = self.approximate_plane(img, visualize=visualize)



        theta, rho, cov_matrix = self.approximate_polar_coordinates(img, alpha, beta, var_alpha, var_beta,
                                                                    covar_alpha_beta)

        acc_space = self.voting(theta, rho, cov_matrix, mask)

        if visualize:
            #plot_hist(cov_matrix[:, :, 0, 0], title="variance of theta")
            #plot_hist(cov_matrix[:, :, 1, 1], title="variance of rho")
            #plot_hist(cov_matrix[:, :, 0, 1], clip_val=500, clip_hist=100, title="covariance")
            ax = sns.heatmap(theta, linewidth=0)
            plt.title("theta")
            plt.show()
            ax = sns.heatmap(rho, linewidth=0)
            plt.title("rho")
            plt.show()
            self.plot_variance_images(cov_matrix)
            acc_space.plot_values()
            #self.plot_hist(acc_space.values, clip_hist=1000, title="Histopgram Hough Space")
            #acc_space.get_n_maxima_plot(steps=50, r=7)


        return acc_space

    def plot_variance_images(self, cov_matrix):
        var_theta_clipped = torch.clamp(cov_matrix[:, :, 0, 0], max=3) # 5
        var_rho_clipped = torch.clamp(cov_matrix[:, :, 1, 1], max=200) # 600
        covar_theta_rho_clipped = torch.clamp(torch.abs(cov_matrix[:, :, 0, 1]), max=50)
        ax = sns.heatmap(var_theta_clipped, linewidth=0)
        #plt.imshow(var_theta_clipped, cmap="gray")  # var theta
        plt.title("Variance Theta")
        plt.show()
        #plt.imshow(var_rho_clipped, cmap="gray")  # var rho
        ax = sns.heatmap(var_rho_clipped, linewidth=0)
        plt.title("Variance Rho")
        plt.show()
        #plt.imshow(covar_theta_rho_clipped, cmap="gray")  # covar theta rho
        ax = sns.heatmap(covar_theta_rho_clipped, linewidth=0)
        plt.title("Covariance Theta and Rho")
        plt.show()


    def approximate_plane(self, img, visualize = False):
        """
        approximates gradient for  each position of an array by a plane. dimensions of the plane are given by
        self.h_plane, self.w_plane
        :param img: source image
        :return: alpha: array of slopes of planes in x-direction
        :return: beta: array of slopes of planes in y-direction
        :return: var_alpha: array of variances of alpha (uncertainty)
        :return: var_beta:  array of variances of beta (uncertainty)
        :return: covar_alpha_beta: array of covariances of alpha and beta (joint uncertainty)
        """
        alpha = np.zeros(img.shape)
        beta = np.zeros(img.shape)
        gamma = np.zeros(img.shape)

        sum_x_squared = np.zeros(img.shape)
        sum_y_squared = np.zeros(img.shape)
        sum_xy = np.zeros(img.shape)

        delta_xi_min = - (self.h_plane // 2)  # -1
        delta_xi_max = (self.h_plane // 2) + 1  # 2
        delta_yi_min = - (self.w_plane // 2)  # -1
        delta_yi_max = (self.w_plane // 2) + 1  # 2

        for hi in range(self.H):
            for wi in range(self.W):
                for delta_x in range(delta_xi_min, delta_xi_max):  # deltax: local position {-1, 0, 1}
                    xi = max(min(hi + delta_x, self.H - 1), 0)  # xi: global position e.g. {19, 20, 21}
                    for delta_y in range(delta_yi_min, delta_yi_max):
                        yi = max(min(wi + delta_y, self.W - 1), 0)
                        alpha[hi, wi] += delta_x * img[xi, yi]
                        sum_x_squared[hi, wi] += delta_x ** 2
                        beta[hi, wi] += delta_y * img[xi, yi]
                        sum_y_squared[hi, wi] += delta_y ** 2
                        gamma[hi, wi] += img[xi, yi]
                        sum_xy[hi, wi] += delta_x * delta_y

        alpha = alpha / sum_x_squared + 0.000001  # adding a small epsilon to prevent dividing by zero
        beta = beta / sum_y_squared + 0.000001
        gamma = gamma / (self.h_plane * self.w_plane)

        """
        Additionally estimates the uncertainty of the approximated plane by calculating variances for the parameters
        """

        local_noise_var = np.zeros(img.shape)  # first calculate local var for each position
        epsilon_squared = np.zeros(img.shape)  # required to get variance

        for hi in range(self.H):
            for wi in range(self.W):
                for delta_x in range(delta_xi_min, delta_xi_max):  # deltax: local position {-1, 0, 1}
                    xi = max(min(hi + delta_x, self.H - 1), 0)  # xi: global position e.g. {19, 20, 21}
                    for delta_y in range(delta_yi_min, delta_yi_max):
                        yi = max(min(wi + delta_y, self.W - 1), 0)
                        epsilon_squared[hi, wi] += (img[xi, wi] - alpha[hi, wi] * delta_y - beta[hi, wi] * delta_x -
                                                    gamma[hi, wi]) ** 2

        local_noise_var = epsilon_squared / (self.h_plane * self.w_plane - 2)

        # regular average:
        #noise_var = np.sum(local_noise_var, axis=(0, 1)) / (self.H * self.W)

        # average weighted by 1/gradient
        gradient = np.sqrt(np.abs(alpha)**2 + np.abs(beta)**2)
        weights = (1/gradient) / np.sum(1/gradient, axis=(0, 1))
        noise_var = np.average(local_noise_var, axis=(0, 1), weights=weights)


        var_alpha = noise_var / sum_x_squared
        var_beta = noise_var / sum_y_squared
        covar_alpha_beta = noise_var * sum_xy / (sum_x_squared * sum_y_squared)

        if visualize:
            ax = sns.heatmap(np.log(np.abs(alpha) + 0.0001), linewidth=0)
            plt.title("alpha")
            plt.show()
            ax = sns.heatmap(np.log(var_alpha), linewidth=0)
            plt.title("var_alpha")
            plt.show()
            ax = sns.heatmap(np.log(np.abs(beta) + 0.0001), linewidth=0)
            plt.title("beta")
            plt.show()
            ax = sns.heatmap(np.log(var_beta), linewidth=0)
            plt.title("var_beta")
            plt.show()
            ax = sns.heatmap(gamma, linewidth=0)
            plt.title("gamma")
            plt.show()
            ax = sns.heatmap(gradient, linewidth=0)
            plt.title("gradient")
            plt.show()
            ax = sns.heatmap(local_noise_var, linewidth=0)
            #plt.imshow(local_noise_var)
            plt.title("local_noise_var")
            plt.show()
            print(f"Noise Variance = {noise_var}")

        return alpha, beta, gamma, var_alpha, var_beta, covar_alpha_beta, noise_var

    def approximate_polar_coordinates(self, img, alpha, beta, var_alpha, var_beta, covar_alpha_beta):
        """
        given an approximated plane for each position in an image and given uncertainty, approximates the polar
        coordinates for the line best representing each pixels gradient. Furthermore estimates the uncertainty of the
        resulting parameters.
        All used arrays share the images dimensions
        :param img: source image
        :param alpha: array of slopes of planes in x-direction
        :param beta: array of slopes of planes in y-direction
        :param var_alpha: array of variances of alpha (uncertainty)
        :param var_beta:  array of variances of beta (uncertainty)
        :param covar_alpha_beta: array of covariances of alpha and beta (joint uncertainty)
        :return: theta: array of polar angles theta of the most likely lines
        :return: rho: array of distances of the lines to origin
        :return: cov_matrix: array of the respective covariance matricies. Holds a matrix for each pixel
        """
        theta = np.arctan(alpha / beta)

        rho = np.zeros(img.shape)
        for hi in range(self.H):
            for wi in range(self.W):
                rho[hi, wi] = self.relative_w(wi) * np.cos(theta[hi, wi]) + self.relative_h(hi) * np.sin(theta[hi, wi])

        k = np.zeros(img.shape)
        for hi in range(self.H):
            for wi in range(self.W):
                k[hi, wi] = self.relative_w(wi) * np.cos(theta[hi, wi]) - self.relative_h(hi) * np.sin(theta[hi, wi])


        """ax = sns.heatmap(k, linewidth=0)
        plt.title("k")
        plt.show()"""

        var_p = 0  # made up value, replace with something reasonable

        var_theta = (var_alpha / (beta ** 2) + (alpha ** 2) * (var_beta / (beta ** 4)) - 2 * (
                    alpha / (beta ** 3)) * covar_alpha_beta) * np.cos(theta) ** 4
        var_rho = k ** 2 * var_theta + var_p
        covar_theta_rho = k * var_theta

        # fusing them into the covariance matrix
        cov_matrix = torch.zeros((self.H, self.W, 2, 2))
        cov_matrix[:, :, 0, 0] = torch.Tensor(var_theta)
        cov_matrix[:, :, 0, 1] = torch.Tensor(covar_theta_rho)
        cov_matrix[:, :, 1, 0] = torch.Tensor(covar_theta_rho)
        cov_matrix[:, :, 1, 1] = torch.Tensor(var_rho)

        return theta, rho, cov_matrix

    def relative_h(self, h):
        """Transforms image coordinate h to coordinate relative to origin"""
        return h - self.origin[0]

    def relative_w(self, w):
        """Transforms image coordinate w to coordinate relative to origin"""
        return w - self.origin[1]

    def voting(self, theta, rho, cov_matrix, mask=None):
        # dropping votes with absurdly high uncertainty, as they would break the voting system while contributing
        # margin values
        var_theta = cov_matrix[:, :, 0, 0]
        var_rho = cov_matrix[:, :, 1, 1]
        sane_votes = (var_theta <= self.thr_var) * (var_rho <= self.thr_var)
        if mask is not None:
            mask = sane_votes * mask  # combine the sane_votes mask with the custom mask
        else:
            mask = sane_votes  # use only sane vote mask

        # Setting up hough space:
        acc_space = AccumulationSpace(self.output_dim,
                                      theta[mask], rho[mask],
                                      var_theta[mask], var_rho[mask],
                                      theta_min=self.theta_min, theta_max=self.theta_max,
                                      rho_min=self.rho_min, rho_max=self.rho_max)

        """
        Bayesian Voting: Each pixel of the image with sane variances gives a vote. 
        Each vote is shaped gaussian with the given uncertainty (cov_matrix).
        The vote for each bin is calculated by integrating the gaussian over the bin
        All votes are added up in the accumulated hough space (acc_space)
        """

        # creating a vector containing theta and rho for each element (makes code below look more clean)
        mean_vector = torch.Tensor(np.concatenate((np.expand_dims(theta, axis=2), np.expand_dims(rho, axis=2)), axis=2))

        # TODO: Performance!
        for hi in range(self.H):
            for wi in range(self.W):
                if mask[hi, wi]:
                    # getting boundaries for a three sigma radius to only vote for relevant positions (performance)
                    theta_interval = 3 * acc_space.theta_std  # 99% confidence interval
                    rho_interval = 3 * acc_space.rho_std  # 99% confidence interval
                    theta_lower_bound = max(min(theta[hi,wi] - theta_interval, acc_space.theta_max), acc_space.theta_min)
                    theta_upper_bound = max(min(theta[hi,wi] + theta_interval, acc_space.theta_max), acc_space.theta_min)
                    theta_lower_i = acc_space.get_index_theta(theta_lower_bound)
                    theta_upper_i = acc_space.get_index_theta(theta_upper_bound)
                    rho_lower_bound = max(min(rho[hi,wi] - rho_interval, acc_space.rho_max), acc_space.rho_min)
                    rho_upper_bound = max(min(rho[hi,wi] + rho_interval, acc_space.rho_max), acc_space.rho_min)
                    rho_lower_i = acc_space.get_index_rho(rho_lower_bound)
                    rho_upper_i = acc_space.get_index_rho(rho_upper_bound)

                    theta_ix = np.arange(theta_lower_i, theta_upper_i)
                    rho_ix = np.arange(rho_lower_i, rho_upper_i)
                    """
                    theta_value = acc_space.get_value_theta(theta_ix)
                    rho_value = acc_space.get_value_rho(rho_ix)
                    theta_bin_lower = theta_value - 0.5 * acc_space.theta_binsize
                    theta_bin_upper = theta_value + 0.5 * acc_space.theta_binsize
                    rho_bin_lower = rho_value - 0.5 * acc_space.rho_binsize
                    rho_bin_upper = rho_value + 0.5 * acc_space.rho_binsize
                    """
                    for theta_i in range(theta_lower_i, theta_upper_i + 1):
                        for rho_i in range(rho_lower_i, rho_upper_i + 1):

                            theta_value = acc_space.get_value_theta(theta_i)
                            rho_value = acc_space.get_value_rho(rho_i)
                            theta_bin_lower = theta_value - 0.5 * acc_space.theta_binsize
                            theta_bin_upper = theta_value + 0.5 * acc_space.theta_binsize
                            rho_bin_lower = rho_value - 0.5 * acc_space.rho_binsize
                            rho_bin_upper = rho_value + 0.5 * acc_space.rho_binsize

                            vote = mvnun(np.array([theta_bin_lower, rho_bin_lower]),
                                         np.array([theta_bin_upper, rho_bin_upper]),
                                         mean_vector[hi, wi], cov_matrix[hi, wi])[0]
                            acc_space.values[theta_i, rho_i] += vote

        return acc_space

class AccumulationSpace():
    def __init__(self, dims, theta, rho, var_theta, var_rho,
                 theta_min=None,
                 theta_max=None,
                 rho_min=None,
                 rho_max=None):
        self.dims = dims
        self.votes = np.zeros(dims)
        # defining ranges and stepsizes for accumulation space
        self.theta_std = np.average(np.sqrt(var_theta))
        self.rho_std = np.average(np.sqrt(var_rho))
        if theta_min is None:
            self.theta_min = np.min(theta) - self.theta_std
        else:
            self.theta_min = theta_min
            assert self.theta_min <= np.min(theta)
        if theta_max is None:
            self.theta_max = np.max(theta) + self.theta_std
        else:
            self.theta_max = theta_max
            assert self.theta_max >= np.max(theta)
        if rho_min is None:
            self.rho_min = np.min(rho) - self.rho_std
        else:
            self.rho_min = rho_min
            assert self.rho_min <= np.min(rho)
        if rho_max is None:
            self.rho_max = np.max(rho) + self.rho_std
        else:
            self.rho_max = rho_max
            assert self.rho_max >= np.max(rho)

        self.theta_binsize = (self.theta_max - self.theta_min) / (dims[0] - 1)
        self.rho_binsize = (self.rho_max - self.rho_min) / (dims[1] - 1)

        self.values = torch.zeros(dims)

    def get_index_theta(self, value, round="floor"):
        if round == "floor":
            return int(np.floor((value - self.theta_min) / self.theta_binsize))
        if round == "ceil":
            return int(np.ceil((value - self.theta_min) / self.theta_binsize))

    def get_value_theta(self, index):
        return index * self.theta_binsize + self.theta_min

    def get_index_rho(self, value, round="floor"):
        if round == "floor":
            return int(np.floor((value - self.rho_min) / self.rho_binsize))
        if round == "ceil":
            return int(np.ceil((value - self.rho_min) / self.rho_binsize))

    def get_value_rho(self, index):
        return index * self.rho_binsize + self.rho_min


    def plot_hist(self, clip_val=255, clip_hist=300, title="Histogram Hough Space", log_y=False, show=True):
        values = torch.clamp(self.values, min=-clip_val, max=clip_val)
        hist = torch.histc(values, bins=100)
        if log_y:
            hist = torch.log(hist)
        else:
            hist = torch.clamp(hist, max=clip_hist)
        x_values = torch.linspace(start=torch.min(values), end=torch.max(values), steps=100)
        if show:
            plt.plot(x_values, hist)
            plt.xlabel("value")
            plt.ylabel("log(N)")
            plt.title(title)
            plt.show()
        return x_values, hist

    def plot_values(self, title="Hough Space"):
        #print("dimensions: ", self.theta_min, self.theta_max, self.rho_min, self.rho_max)
        #plt.imshow(self.values)
        ax = sns.heatmap(self.values, linewidth=0)
        plt.title(title)
        plt.yticks(np.linspace(0, self.dims[0], 10), np.around(np.linspace(self.theta_min, self.theta_max, 10), decimals=1))
        plt.xticks(np.linspace(0, self.dims[0], 10), np.around(np.linspace(self.rho_min, self.rho_max, 10), decimals=0))
        plt.ylabel("Theta")
        plt.xlabel("Rho")
        plt.show()

    def get_local_maxima(self, r=10):
        """
        #:param r: radius to check for local maxima
        #:return: returns list of indices for all local maxima
        """
        indices_loc_max = []
        for theta_i in range(self.dims[0]):
            for rho_i in range(self.dims[1]):

                max_near_val = torch.max(self.values[max(0, theta_i - r): min(self.dims[0] - 1, theta_i + r + 1),
                                         max(0, rho_i - r): min(self.dims[1] - 1, rho_i + r + 1)])
                #print(acc_space[theta_lines[i], rho_lines[i]], max_near_val)
                if self.values[theta_i, rho_i] >= max_near_val and self.values[theta_i, rho_i] > 0:
                    indices_loc_max.append((theta_i, rho_i))
        return indices_loc_max

    def get_n_maxima_plot(self, steps = 50, r=10, log_x=False, log_y=False, show=True):
        min_vote = torch.min(self.values)
        max_vote = torch.max(self.values)
        inddices_loc_max = self.get_local_maxima(r=r)
        thr_list = torch.linspace(min_vote, max_vote, steps)
        n_maxima = torch.zeros(steps)
        for i, thr in enumerate(thr_list):
            for pos in inddices_loc_max:
                if self.values[pos] >= thr:
                    n_maxima[i] += 1
        if log_x:
            thr_list = torch.log(thr_list)
        if log_y:
            n_maxima = torch.log(n_maxima)
        if show:
            plt.plot(thr_list, n_maxima)
            plt.title("Amount of maxima left after applying threshold (r = " + str(r) + ")")
            plt.xlabel("log(Threshold)" if log_x else "Threshold")
            plt.ylabel("log(N)" if log_y else "N")
            plt.show()
        return thr_list, n_maxima



def plot_hist(values, clip_val=255, clip_hist=300, title="histogram", log_y=False, plot_type="line", show=True):
    values = torch.clamp(values, min=-clip_val, max=clip_val)
    hist = torch.histc(values, bins=100)
    if log_y:
        hist = torch.log(hist)
    else:
        hist = torch.clamp(hist, max=clip_hist)
    x_values = torch.linspace(start=torch.min(values), end=torch.max(values), steps=100)
    if plot_type == "bar":
        plt.bar(x_values, hist)
    else:
        plt.plot(x_values, hist)
    plt.xlabel("value")
    plt.ylabel("log(N)")
    plt.title(title)
    if show:
        plt.show()
    return x_values, hist
