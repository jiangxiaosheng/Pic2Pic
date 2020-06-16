import os

from feature.utils import *
from feature.extractor import Extractor
from feature.evaluate import getExtractor
from sklearn.random_projection import johnson_lindenstrauss_min_dim, GaussianRandomProjection, SparseRandomProjection
from six.moves import cPickle
import numpy as np

ratio = config['LSH']['ratio']
project_type = config['LSH']['type']


def _transform_to_dict(feature):
    transformed_feature = {}
    for feat in feature:
        transformed_feature[feat['img']] = {
            'cls': feat['cls'],
            'feature': feat['feature']
        }
    return transformed_feature


class LSH:
    def __init__(self, feature_pooling, samples, classes=None, ratio=ratio, project_type=project_type):
        self.feature_pooling = feature_pooling
        self.ratio = ratio
        self.project_type = project_type
        self.samples = samples
        self.classses = classes

    def extract_feature(self, resource):
        extractors = self._get_extractors()
        sample_features = []
        for extractor in extractors:
            feature = {'feature': extractor.extract_feature(resource), 'img': resource}
            _samples = extractor.make_indices(self.samples)
            samples_for_classes = []
            for s in _samples:
                if self.classses is not None and s['cls'] not in self.classses:
                    continue
                else:
                    samples_for_classes.append(s)
            _samples = np.append(feature, samples_for_classes)
            sample_features.append(_samples)
        compact_features, flag = self._compact_features(sample_features)
        print(compact_features[0]['feature'].shape, compact_features[1]['feature'].shape)
        return compact_features

    def make_indices(self):
        sample_path = 'LSH_('
        for i, feature_name in enumerate(self.feature_pooling):
            sample_path += feature_name
            if i != len(self.feature_pooling) - 1:
                sample_path += ','
        sample_path += ')'
        try:
            images = cPickle.load(open(os.path.join('indices', sample_path), 'rb', True))
        except:
            features = self._get_samples_features(self.samples)
            images, flag = self._compact_features(features)
            if not flag:
                return []
            cPickle.dump(images, open(os.path.join('indices', sample_path), 'wb', True))
        return images

    def _get_extractors(self):
        extractor = []
        for name in self.feature_pooling:
            extractor.append(getExtractor(name))
        return extractor

    def _get_samples_features(self, samples):
        samples_features = []
        extractors = self._get_extractors()
        for extractor in extractors:
            samples_features.append(extractor.make_indices(samples))
        return samples_features

    def _compact_features(self, features):
        first_feature = features[0]
        for another_feature in features[1:]:
            for index in range(len(first_feature)):
                first_feature[index]['feature'] = np.append(first_feature[index]['feature'], another_feature[index]['feature'])
        _features = np.array([f['feature'] for f in first_feature])
        eps = self._compute_eps(_features)
        if eps == -1:
            print('The parameters don\'t fit the requirements of random projection!')
            return first_feature, False
        if self.project_type == 'gaussian':
            transformer = GaussianRandomProjection(eps=eps)
        elif self.project_type == 'sparse':
            transformer = SparseRandomProjection(eps=eps)
        compact_features = transformer.fit_transform(_features)
        for i in range(len(first_feature)):
            first_feature[i]['feature'] = compact_features[i]
        return first_feature, True

    def _compute_eps(self, samples):
        samples_count, dim = samples.shape
        new_dim = dim * self.ratio
        precision = 1e-3
        n_iters = int(1 / precision)
        for i in range(1, n_iters):
            eps = i / n_iters
            evaluate_dim = johnson_lindenstrauss_min_dim(samples_count, eps=eps)
            if evaluate_dim <= new_dim:
                return eps
        return -1


if __name__ == '__main__':
    samples = Samples('dataset')
    lsh = LSH(['HOG', 'VGG'], samples)
    # indices = lsh.make_indices()
    query_img = 'dataset/accordion/image_0001.jpg'
    print(lsh.extract_feature(query_img))
    # n_samples = 855
    # new_dim = 0.2 * n_samples
    # for i in range(1, 1000):
    #     eps = i / 1000
    #     evaluate_dim = johnson_lindenstrauss_min_dim(n_samples, eps=eps)
    #     if evaluate_dim <= new_dim:
    #         print(eps)

