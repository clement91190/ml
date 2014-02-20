""" Step 1 : generate feature vector from images 

For each method, we add a file in the data folder, and store 2 files : train_feat.tkl, test_feat.tkl created 
using a given method. 

train_feat should be 
"""
import os
import cPickle
import csv
from PIL import Image
import numpy as np

#extraction methods:
from method_gray import GrayExtractor

methods_map = {
    'method_gray': GrayExtractor}  # feature_extraction is a function that take

pixelSize = 4
img_size = 36
size = 424
small = (size - img_size * pixelSize) / 2
crop_dimensions = (small, small, small + img_size * pixelSize, small + img_size * pixelSize)
#print crop_dimensions


def load_img(num, path):
    """ return the numpy array of the img """
    pic = Image.open(path + num + '.jpg')
    pic = pic.crop(crop_dimensions)
    #t = np.asarray(pic)
    return pic


def load_data(set='train', max=1000000):
    """ load the training or testing set, and return
    all the images -> memory intensive but ok !
    if training + labels
    if testing + ids """
    if set == 'train':
        label_path = '../../data/original_data/training_solutions_rev1.csv'
    else:
        label_path = '../../data/original_data/all_ones_benchmark.csv'
    pics = []
    labels_or_ids = []
    print "... loading " + set
    with open(label_path) as label_f:
        reader = csv.reader(label_f)
        for i, row in enumerate(reader):
            if max > i > 0:
                if i % 1000 == 0:
                    print "loading image ", i
                #for angle in xrange(0, 360, 360/nb_img):
                #gray = load_img_hog(row[0])
                img = load_img(row[0], '../../data/original_data/' + set + 'ing_set/img/')
                pics.append(img)
                if set == 'train':
                    labels_or_ids.append([float(v) for v in row[1:]])
                else:
                    labels_or_ids.append(row[0])
    if set == 'train':
        total_training_y = np.array(labels_or_ids, dtype='float32')
        return (pics, total_training_y)
    else:
        return (pics, labels_or_ids)


def apply_transformation(method, data):
    (train, labels), (test, ids) = data
    feature_extractor = methods_map[method](data)
    train_feat = feature_extractor.produce_train()
    test_feat = feature_extractor.produce_test()
    return (prepare_train_valid_test((train_feat, labels)), (test_feat, ids))


def prepare_train_valid_test((total_x, total_y), tr_size=60, val_size=20):
    """ cut the training set in train, valid, test """
    print "preparing learning datasets"
    last_train = tr_size * total_x.shape[0] / 100
    last_valid = last_train + val_size * total_x.shape[0] / 100
    train_x, train_y = total_x[:last_train], total_y[:last_train]
    valid_x, valid_y = total_x[last_train:last_valid], total_y[last_train:last_valid]
    test_x, test_y = total_x[last_valid:], total_y[last_valid:]
    rval = [(train_x, train_y), (valid_x, valid_y), (test_x, test_y)]
    return rval


def save_training(method, rval):
    print " ... saving training"
    try:
        os.mkdir('../../data/' + method)
    except:
        pass
    with open('../../data/' + method + '/training_set.pkl', 'w') as f:
        cPickle.dump(rval, f)


def save_testing(method, test_features):
    print "... saving testing"
    try:
        os.mkdir('../../data/' + method)
    except:
        pass
    with open('../../data/' + method + '/testing_set.pkl', 'w') as f:
        cPickle.dump(test_features, f)


def main():
    method = "method_gray"
    data_train = load_data(set='train')
    data_test = load_data(set='test')
    print "... apply the transformation"
    (rval, test_features) = apply_transformation(method, (data_train, data_test))
    save_training(method, rval)
    save_testing(method, test_features)
    print "...done"


if __name__ == "__main__":
    main()
