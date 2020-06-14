from feature.utils import *
from feature import HC, HOG, VGGNet, ResNet, Edge


def query(query_img, feature_pool, samples, query_classes=None, distance_method='cosine', top_n=10):
    extractor = getExtractor(feature_pool[0])
    q_feature = extractor.extract_feature(query_img)
    if query_classes is None:
        results = []
        for index, sample in enumerate(samples):
            s_img, s_cls, s_feature = samples['img'], samples['cls'], samples['feature']
            if s_img == query_img:
                continue
            results.append(
                {
                    'distance': distance()
                }
            )
    else:
        pass


def getExtractor(name):
    if name == 'HC':
        return HC()
    elif name == 'HOG':
        return HOG()
    elif name == 'VGG':
        return VGGNet()
    elif name == 'Res':
        return ResNet()
    elif name == 'edge':
        return Edge()