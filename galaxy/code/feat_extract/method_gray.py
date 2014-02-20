""" simple feature extraction using black and white features"""
from PIL import Image
import numpy as np


class GrayExtractor():
    def __init__(self, data):
        (self.train, self.labels), (self.test, self.test_ids) = data
        self.pixelSize = 4
        self.img_size = 36

    def convert_one_img(self, pic):
        pic = pic.convert('L')
        # pixelate the image with pixelSize
        pic = pic.resize(
            (pic.size[0] / self.pixelSize, pic.size[1] / self.pixelSize),
            Image.NEAREST)
        pic = pic.resize((pic.size[0] * self.pixelSize, pic.size[1] * self.pixelSize), Image.NEAREST)
        vectors = []
        pixel = pic.load()
        for i in xrange(0, pic.size[0], self.pixelSize):
            for y in xrange(0, pic.size[1], self.pixelSize):
                vectors.append(round(pixel[y, i] / float(255), 3))
        return np.array(vectors, dtype='float32')

    def produce_train(self):
        train_feat = [self.convert_one_img(pic) for pic in self.train]
        return np.array(train_feat)

    def produce_test(self):
        test_feat = [self.convert_one_img(pic) for pic in self.test]
        return np.array(test_feat)
