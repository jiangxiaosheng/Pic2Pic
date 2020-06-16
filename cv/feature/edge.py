import os

from feature.extractor import Extractor
from feature.utils import config, otsu, Samples
import numpy as np
from PIL import Image
from six.moves import cPickle

stride = eval(config['edge']['stride'])
slices = config['edge']['slices']


class Edge(Extractor):
    def __init__(self, stride=stride, slices=slices):
        self.kernels = np.array([
            # sobel滤波器
            [
                [-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1]
            ],
            [
                [-1, -2, -1],
                [0, 0, 0],
                [1, 2, 1]
            ],
            # laplacian滤波器
            [
                [1, 1, 1],
                [1, -8, 1],
                [1, 1, 1]
            ]
        ])
        self.stride = stride
        self.slices = slices

    def extract_feature(self, resource):
        image = Image.open(resource)
        # RGB信息对边缘检测帮助不大，所以先转换为灰度图处理
        image = image.convert('L')
        image = np.asarray(image)
        image = otsu(image)
        height, width = image.shape
        feature = np.zeros((self.slices, self.slices, self.kernels.shape[0]))
        width_slices = np.linspace(0, width, self.slices + 1, endpoint=True).astype(int)
        height_slices = np.linspace(0, height, self.slices + 1, endpoint=True).astype(int)
        for h in range(len(height_slices) - 1):
            for w in range(len(width_slices) - 1):
                region = image[height_slices[h]: height_slices[h + 1], width_slices[w]: width_slices[w + 1]]
                feature[h, w] = self._conv(region)
        feature /= np.sum(feature) + 1e-6
        return feature.flatten()

    def _conv(self, input):
        height, width = input.shape
        conv_kernels = np.expand_dims(self.kernels, axis=3)
        stride_h, stride_w = self.stride
        kernel_n, kernel_h, kernel_w = self.kernels.shape

        histogram = np.zeros(kernel_n)
        h_steps = int((height - kernel_h) / stride_h + 1)
        w_steps = int((width - kernel_w) / stride_w + 1)

        for index, k in enumerate(conv_kernels):
            for h in range(h_steps):
                hs = int(h * stride_h)
                he = int(h * stride_h + kernel_h)
                for w in range(w_steps):
                    ws = int(w * stride_w)
                    we = int(w * stride_w + kernel_w)
                    histogram[index] += np.sum(input[hs: he, ws: we] * k)

        histogram /= np.sum(histogram) + 1e-6
        return histogram

    def make_indices(self, samples):
        sample_path = 'edge'
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


if __name__ == '__main__':
    samples = Samples('dataset')
    edge = Edge()
    edge.make_indices(samples)
