from feature.utils import *
from feature import HC, HOG, VGGNet, ResNet, Edge


def query(query_img, feature_pool, samples, query_classes=None, distance_method='euclidean', top_n=10):
    extractor = getExtractor(feature_pool[0])
    q_feature = extractor.extract_feature(query_img)
    if query_classes is None:
        results = []
        indices = extractor.make_indices(samples)
        for index, sample in enumerate(indices):
            s_img, s_cls, s_feature = sample['img'], sample['cls'], sample['feature']
            if s_img == query_img:
                continue
            results.append(
                {
                    'distance': distance(q_feature, s_feature, distance_method),
                    'cls': s_cls,
                    'img': s_img
                }
            )
        results = sorted(results, key=lambda x: x['distance'])[: top_n]
        return results
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


if __name__ == '__main__':
    query_img = 'dataset/accordion/image_0001.jpg'
    feature_pool = ['Res']
    samples = Samples('dataset')
    results = query(query_img, feature_pool, samples, distance_method='cosine')
    for r in results:
        print(r)
