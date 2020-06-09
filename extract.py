from cv.feature.samples import Samples
from cv.feature.HC import HC

samples_dir = 'samples'

if __name__ == "__main__":
    hc = HC()
    samples = Samples(samples_dir)
    hc.make_indices(samples)