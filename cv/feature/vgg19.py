import os

from feature.utils.samples import Samples
from feature.extractor import Extractor
from keras.applications.vgg19 import VGG19, preprocess_input
from keras.preprocessing import image
import numpy as np
from six.moves import cPickle


class VGGNet(Extractor):
    def __init__(self, pooling='max'):
        self.image_size = (224, 224, 3)
        self.weight = 'imagenet'
        # 最大池化可以保留纹理特征
        # 平均池化可以保留背景信息
        self.pooling = pooling
        self.vgg_model = VGG19(
            weights=self.weight,
            input_shape=(self.image_size[0], self.image_size[1], self.image_size[2]),
            pooling=self.pooling,
            include_top=False
        )

    def extract_feature(self, input):
        img = image.load_img(input, target_size=(self.image_size[0], self.image_size[1]))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img, mode='caffe')
        feature = self.vgg_model.predict(img)
        return feature[0] / np.linalg.norm(feature[0])

    def make_indices(self, samples):
        sample_path = 'vgg19'
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
    hc = VGGNet()
    hc.make_indices(samples)
