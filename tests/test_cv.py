import unittest
from feature.utils import *
from feature import *
from PIL import Image
import numpy as np


class CVTest(unittest.TestCase):
    def test_otsu(self):
        img_path = 'mesoshe.jpg'
        img = Image.open(img_path)
        img = img.convert('L')
        img_array = np.asarray(img)
        otsu_img_array = otsu(img_array)
        img = Image.fromarray(otsu_img_array)
        img.show()
