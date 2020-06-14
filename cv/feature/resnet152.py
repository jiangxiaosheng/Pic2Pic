import os

from feature.utils import Samples
from feature.extractor import Extractor
from keras.applications.resnet import ResNet152, preprocess_input
from keras.preprocessing import image
import numpy as np
from six.moves import cPickle


class ResNet(Extractor):
    def __init__(self, pooling='max'):
        self.image_size = (224, 224, 3)
        self.pooling = pooling
        self.weights = 'imagenet'
        self.res_model = ResNet152(
            weights=self.weights,
            input_shape=(self.image_size[0], self.image_size[1], self.image_size[2]),
            pooling=self.pooling,
            include_top=False
        )

    def extract_feature(self, resource):
        img = image.load_img(resource, target_size=(self.image_size[0], self.image_size[1]))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        feature = self.res_model.predict(img)
        return feature[0] / np.linalg.norm(feature[0])

    def make_indices(self, samples):
        sample_path = 'resnet152_pooling_{}'.format(self.pooling)
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
                    'hist': feature
                })
        cPickle.dump(images, open(os.path.join('indices', sample_path), "wb", True))


if __name__ == '__main__':
    samples = Samples('dataset')
    hc = ResNet()
    hc.make_indices(samples)