import os

from feature.utils.samples import Samples
from feature.extractor import Extractor
from imageio import imread
import numpy as np
import itertools
from six.moves import cPickle


class HC(Extractor):
    def __init__(self, slices=3, n_bin=12, h_type='global'):
        self.h_type = h_type
        self.slices = slices
        self.n_bin = n_bin
        self.bins = np.linspace(0, 256, self.n_bin + 1, endpoint=True)

    def extract_feature(self, resource):
        image = imread(resource, pilmode='RGB')
        height, width, channel = image.shape
        if self.h_type == 'global':
            feature = self._compute_hist(image, channel)
        elif self.h_type == 'region':
            feature = np.zeros((self.slices, self.slices, self.n_bin ** channel))
            width_slices = np.linspace(0, width, self.slices + 1, endpoint=True).astype(int)
            height_slices = np.linspace(0, height, self.slices + 1, endpoint=True).astype(int)
            for h in range(len(height_slices) - 1):
                for w in range(len(width_slices) - 1):
                    region = image[height_slices[h]: height_slices[h + 1], width_slices[w]: width_slices[w + 1]]
                    feature[h, w] = self._compute_hist(region, channel)
        else:
            raise Exception("不支持的特征类型")

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
        if self.h_type == 'global':
            sample_path = 'HC_{}_bin{}'.format(self.h_type, self.n_bin)
        elif self.h_type == 'region':
            sample_path = 'HC_{}_bin{}_slices{}'.format(self.h_type, self.n_bin, self.slices)
        else:
            raise Exception("不支持的特征类型")

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


if __name__ == '__main__':
    samples = Samples('dataset')
    hc = HC()
    hc.make_indices(samples)

