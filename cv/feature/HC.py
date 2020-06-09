from feature.extractor import Extractor
from imageio import imread
import numpy as np
import itertools


class HC(Extractor):
    def __init__(self, slices, n_bin, h_type='global', d_type='euler'):
        self.h_type = h_type
        self.d_type = d_type
        self.slices = slices
        self.n_bin = n_bin
        self.bins = np.linspace(0, 255, self.n_bin + 1, endpoint=True)

    def hist(self, resource):
        image = imread(resource, pilmode='RGB')
        height, width, channel = image.shape
        if self.h_type == 'global':
            hist = self._compute_hist(image, channel)
        elif self.h_type == 'region':
            hist = np.zeros((self.slices, self.slices, self.n_bin ** channel))
            width_slices = np.linspace(0, width, self.slices + 1, endpoint=True).astype(int)
            height_slices = np.linspace(0, height, self.slices + 1, endpoint=True).astype(int)
            for h in range(len(height_slices) - 1):
                for w in range(len(width_slices) - 1):
                    region = image[height_slices[h]: height_slices[h + 1], width_slices[w]: width_slices[w + 1]]
                    hist[h, w] = self._compute_hist(region, channel)
        else:
            raise Exception("不支持的特征类型")

        hist /= np.sum(hist)
        return hist.flatten()

    def _compute_hist(self, input, channel):
        img = input.copy()
        bins_idx = {key: idx for idx, key in
                    enumerate(itertools.product(np.arange(self.n_bin), repeat=channel))}
        hist = np.zeros(self.bins ** channel)

        for idx in range(len(self.bins) - 1):
            img[(input >= self.bins[idx]) & (input < self.bins[idx + 1])] = idx
        height, width, _ = img.shape
        for h in range(height):
            for w in range(width):
                b_idx = bins_idx[tuple(img[h, w])]
                hist[b_idx] += 1
        return hist

    def make_indices(self, samples):
        if self.h_type == 'global':
            pass
        elif self.h_type == 'region':
            pass
        else:
            raise Exception("不支持的特征类型")
