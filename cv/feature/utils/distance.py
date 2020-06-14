import numpy as np
from scipy.spatial.distance import mahalanobis, cityblock, euclidean, chebyshev, cosine, correlation


def distance(v1, v2, method):
    if method == 'cityblock':
        return cityblock(v1, v2)
    elif method == 'euclidean':
        return euclidean(v1, v2)
    elif method == 'cosine':
        return cosine(v1, v2)
    elif method == 'chebyshev':
        return chebyshev(v1, v2)
    elif method == 'correlation':
        return correlation(v1, v2)
    else:
        return None


def distance_mahala(v1, v2, samples):
    v = np.vstack(samples)
    S = np.cov(v)
    SI = np.linalg.inv(S)
    try:
        return mahalanobis(v1, v2, SI)
    except:
        return distance(v1, v2, method='cosine')

