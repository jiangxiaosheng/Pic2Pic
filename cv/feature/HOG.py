import os

from feature.utils import otsu
from feature.utils.samples import Samples
from feature.extractor import Extractor
from imageio import imread
import numpy as np
from skimage import color
from skimage.feature import hog
from six.moves import cPickle


class HOG(Extractor):
    def __init__(self, slices=6, n_bin=10, orientation=8, ppc=(2, 2), ppb=(1, 1)):
        self.slices = slices
        self.n_bin = n_bin
        self.orientation = orientation
        self.ppc = ppc
        self.ppb = ppb

    def extract_feature(self, resource):
        image = imread(resource, pilmode='RGB')
        height, width, channel = image.shape
        feature = np.zeros((self.slices, self.slices, self.n_bin))
        width_slices = np.linspace(0, width, self.slices + 1, endpoint=True).astype(int)
        height_slices = np.linspace(0, height, self.slices + 1, endpoint=True).astype(int)
        for h in range(len(height_slices) - 1):
            for w in range(len(width_slices) - 1):
                region = image[height_slices[h]: height_slices[h + 1], width_slices[w]: width_slices[w + 1]]
                feature[h, w] = self._compute_hist(region)

        feature /= np.sum(feature)
        return feature.flatten()

    def _compute_hist(self, input):
        image = color.rgb2gray(input)
        image = otsu(image)
        hog_hist = hog(image, orientations=self.orientation, pixels_per_cell=self.ppc, cells_per_block=self.ppb, transform_sqrt=True)
        bins = np.linspace(0, np.max(hog_hist), self.n_bin + 1, endpoint=True)
        hist, _ = np.histogram(hog_hist, bins=bins)
        hist = np.array(hist) / np.sum(hist)
        return hist

    def make_indices(self, samples):
        sample_path = 'HOG'
        try:
            images = cPickle.load(open(os.path.join('indices', sample_path), 'rb', True))
        except:
            images = []
            data = samples.get_images_by_class()
            i = 1
            for d in data:
                print(i)
                i += 1
                d_img, d_cls = d["img"], d["cls"]
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
    myHOG = HOG()
    myHOG.make_indices(samples)
    img_path = 'tests/aimer.jpg'
    print(myHOG.extract_feature(img_path).shape)
