import time
import timeit
from feature import query, query_lsh
from feature.utils import Samples


def clock(func):
    def clocked(*args):
        t0 = timeit.default_timer()
        result = func(*args)
        elapsed = timeit.default_timer() - t0
        name = func.__name__
        arg_str = ', '.join(repr(arg) for arg in args)
        print('[%0.8fs] %s(%s)' % (elapsed, name, arg_str))
        return result

    return clocked


@clock
def timing(func):
    func()


if __name__ == '__main__':
    # timing(time.sleep, 2)
    samples = Samples('dataset')
    query_img = 'dataset/accordion/image_0001.jpg'
    # query(query_img, 'Res', samples, query_classes=['accordion'], distance_method='cosine')
    timing(
        lambda: print(query(query_img, 'HC', samples, query_classes=['accordion'], distance_method='cosine'))
    )
