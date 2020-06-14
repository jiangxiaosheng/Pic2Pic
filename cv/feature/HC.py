import os

from feature.utils import *
from feature.extractor import Extractor
from imageio import imread
import numpy as np
import itertools
from six.moves import cPickle

slices = config['HC']['slices']
n_bin = config['HC']['n_bin']

class HC(Extractor):
    def __init__(self, slices=slices, n_bin=n_bin):
        self.slices = slices
        self.n_bin = n_bin
        self.bins = np.linspace(0, 256, self.n_bin + 1, endpoint=True)

    def extract_feature(self, resource):
        image = imread(resource, pilmode='RGB')
        height, width, channel = image.shape
        feature = np.zeros((self.slices, self.slices, self.n_bin ** channel))
        width_slices = np.linspace(0, width, self.slices + 1, endpoint=True).astype(int)
        height_slices = np.linspace(0, height, self.slices + 1, endpoint=True).astype(int)
        for h in range(len(height_slices) - 1):
            for w in range(len(width_slices) - 1):
                region = image[height_slices[h]: height_slices[h + 1], width_slices[w]: width_slices[w + 1]]
                feature[h, w] = self._compute_hist(region, channel)

        feature /= np.sum(feature)
        return feature.flatten()

    def _compute_hist(self, input, channel):
        img = input.copy()
        bins_idx = {key: idx for idx, key in
                    enumerate(itertools.product(np.arange(self.n_bin), repeat=channel))}
        hist = np.zeros(self.n_bin ** channel)

        for idx in range(len(self.bins) - 1):
            img[(input >= self.bins[idx]) & (input < self.bins[idx + 1])] = idx
        height, width, _ = img.shape
        for h in range(height):
            for w in range(width):
                b_idx = bins_idx[tuple(img[h, w])]
                hist[b_idx] += 1
        return hist

    def make_indices(self, samples):
        sample_path = 'HC'
        try:
            images = cPickle.load(open(os.path.join('indices', sample_path), 'rb', True))
        except:
            images = []
            data = samples.get_data()
            i = 1
            for d in data.itertuples():
                print(i)
                i += 1
                d_img, d_cls = getattr(d, "img"), getattr(d, "cls")
                feature = self.extract_feature(d_img)
                images.append({
                    'img': d_img,
                    'cls': d_cls,
                    'feature': feature
                })
            cPickle.dump(images, open(os.path.join('indices', sample_path), "wb", True))
        return images


if __name__ == '__main__':
    samples = Samples('dataset')
    hc = HC()
    hc.make_indices(samples)
    img_path = 'tests/aimer.jpg'
    print(hc.extract_feature(img_path).shape)

