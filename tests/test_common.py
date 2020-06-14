import unittest
from utils.download import download
import numpy as np
from feature.utils.samples import *


class CommonTest(unittest.TestCase):
    def test_download(self):
        download('.', 'img.jpg', 'https://ss3.bdstatic.com/70cFv8Sh_Q1YnxGkpoWK1HF6hhy/it/u=4042775187,64433586&fm=26&gp=0.jpg')

    def test_numpy(self):
        a = np.array([1, 2, 3, 4])
        b = np.array([2, 3, 4, 5])
        print(np.sum((a - b) ** 2))
        print(np.linspace(0, 4, 3))

    def test_samples(self):
        s = Samples('../dataset')
        print(s.get_data())