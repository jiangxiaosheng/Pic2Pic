import unittest
# from feature import *
from PIL import Image
from skimage import color
from imageio import imread
import numpy as np


class CVTest(unittest.TestCase):
    def test_otsu(self):
        img_path = 'mesoshe.jpg'
        img = imread(img_path, pilmode='RGB')
        img = color.rgb2gray(img)
        print(img)
        img = Image.fromarray(img * 255)
        img.show()
        img = Image.open(img_path)
        img = img.convert('L')
        img_array = np.asarray(img)
        print(img_array)
        img.show()

    def test_nearpy(self):
        from nearpy import Engine
        from nearpy.hashes import RandomBinaryProjections

        # Dimension of our vector space
        dimension = 3

        # Create a random binary hash with 10 bits
        rbp = RandomBinaryProjections('rbp', 10)

        # Create engine with pipeline configuration
        engine = Engine(dimension, lshashes=[rbp])

        # Index 1000000 random vectors (set their data to a unique string)
        # for index in range(100000):
        #     v = np.random.randn(dimension)
        #     engine.store_vector(v, 'data_%d' % index)
        v = np.array([1, 2, 3])
        engine.store_vector(v, 'data_%d' % 0)
        v = np.array([1, 2, 4])
        engine.store_vector(v, 'data_%d' % 1)
        v = np.array([4, 2, 3])
        engine.store_vector(v, 'data_%d' % 2)
        v = np.array([1, 2, 3.2])
        engine.store_vector(v, 'data_%d' % 3)



        # Create random query vector
        # query = np.random.randn(dimension)
        query = np.array([1, 2, 3])

        # Get nearest neighbours
        N = engine.neighbours(query)
        # print(N[0][2], N[1][2], N[2][2], N[3][2])
        print(N)