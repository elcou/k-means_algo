# FILTERS.py
#
# Quantization of an image : Uses the k-means algorithm to color-quantize the image to only k colors.
#
# Autor: Elodie Couturier
# ######################################################################################################################

from PIL import Image
import numpy as np
import os
from operator import itemgetter
import pickle


class FILTER:

    def __init__(self, image_path):

        self.filename, self.file_extension = os.path.splitext(image_path)
        self.im = Image.open(image_path)
        self.width, self.height = self.im.size
        self.nb_max = pow(2, self.im.bits) - 1  # Value max of a pixel
        self.savename = None

    # ------------------------------------------------------------------------------------------------------------------
    # Training of the algorithm k-means used to regroup every pixels in k clusters
    # The mean values of each cluster after the training is saved in a txt file
    def k_means_training(self, k, max_iter):

        pixels = np.array(self.im.getdata())
        n = len(pixels)

        # Initialize the clusters
        cluster_index = [0] * n
        clusters_prev = list()
        clusters_new = list()
        for i in range(0, k):
            clusters_prev.append([0])
            clusters_new.append([1])

        nb_changes = k

        # Initialize the means to k randomly chosen pixel values
        ind = np.random.randint(0, n - 1, size=k)
        means = list(itemgetter(*ind)(pixels))

        sq_dists = list()

        i = 0
        while (i < max_iter) and (nb_changes > 0):

            # Attach each pixel to one of the k clusters
            for i_pixel in range(0, n):
                # dist = norm(pixels - means)^2
                # No need to square up the result because we don't need the real value of dist,
                # only the position of its min value
                dists = list(np.linalg.norm(np.subtract(pixels[i_pixel], means), axis=1))
                sq_dists.append(min(dists))
                cluster_index[i_pixel] = dists.index(min(dists))

            nb_changes = k

            # Update the new means of the clusters
            for i_k in range(0, k):

                clusters_new[i_k] = itemgetter(*[item for item, value in enumerate(cluster_index) if value == i_k])(pixels)
                means[i_k] = list(1/len(clusters_new[i_k]) * np.array(clusters_new[i_k]).sum(axis=0))

                # Update nb_changes. If none of the clusters change, the algorithm has converged and can stop.
                if len(clusters_prev[i_k]) == len(clusters_new[i_k]):
                    if clusters_prev[i_k] == clusters_new[i_k]:
                        nb_changes = nb_changes - 1

            clusters_prev = clusters_new

            i = i + 1

        # Save the means to use for the quantization.
        # One set of means can only be used on the same image it has been trained on.
        with open(self.filename + '_kmeans_' + str(k) + '.txt', "wb") as file:
            pickle.dump(means, file, protocol=pickle.HIGHEST_PROTOCOL)

        self.quantization(k, max_iter)

    # ------------------------------------------------------------------------------------------------------------------
    # Color quantization.
    # Uses the means learned with the k-means algorithm to classify each pixel to one of the k clusters
    def quantization(self, k, max_iter):

        # If the training has already been done, just load the means from the txt file.
        # Otherwise do the training
        try:
            with open(self.filename + '_kmeans_' + str(k) + '.txt', "rb") as file:
                means = pickle.load(file)
        except FileNotFoundError:
            self.k_means_training(k, max_iter)

        pixels = np.array(self.im.getdata())
        n = len(pixels)

        # Assign every pixel to one of the k clusters
        # Change the value of the pixel to the mean of its corresponding cluster
        for i in range(0, n):
            dists = list(np.linalg.norm(np.subtract(pixels[i], means), axis=1))
            pos = dists.index(min(dists))

            pixels[i] = np.array(means[pos])

        pixels = pixels.reshape(self.height, self.width, 3)
        pixels = np.array(pixels, dtype='int8')

        # Save the quantized image
        self.savename = self.filename + '_quantized-' + str(k) + '_colors' + self.file_extension
        Image.fromarray(pixels, mode='RGB').save(self.savename)

