""" simple feature extraction using black and white features"""
from PIL import Image
import numpy as np
from code.feat_extract.utils import save_testing, save_training, load_data, prepare_train_valid_test


class GrayExtractor():
    def __init__(self):
        self.pixelSize = 4
        self.img_size = 36
        self.method_name = "method_gray"

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

    def run(self):
        filter = lambda img: self.convert_one_img(img)
        save_training(self.method_name, prepare_train_valid_test(load_data(filter, set='train')))
        save_testing(self.method_name, load_data(filter, set='test'))


