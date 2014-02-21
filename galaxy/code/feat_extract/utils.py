from PIL import Image
import cPickle
import numpy as np
import csv
import os

pixelSize = 4
img_size = 36
size = 424
small = (size - img_size * pixelSize) / 2
crop_dimensions = (small, small, small + img_size * pixelSize, small + img_size * pixelSize)


def load_img(num, path):
    """ return the numpy array of the img """
    pic = Image.open(path + num + '.jpg')
    #pic = pic.crop(crop_dimensions)
    #t = np.asarray(pic)
    return pic


def load_train_img(num):
    img = load_img(num, '../../data/original_data/' + set + 'ing_set/img/')
    return img


def load_test_img(num):
    img = load_img(num, '../../data/original_data/' + set + 'ing_set/img/')
    return img


def load_img_and_filter(num, path, filter):
    return filter(load_img(num, path))


def load_data(filter, set='train', max=1000000):
    """ load the training or testing set, and return
    all the images -> apply filter directly on images
    if training + labels
    if testing + ids """
    if set == 'train':
        label_path = '../../data/original_data/training_solutions_rev1.csv'
    else:
        label_path = '../../data/original_data/all_ones_benchmark.csv'
    pics = []
    labels = []
    ids = []
    print "... loading " + set
    with open(label_path) as label_f:
        reader = csv.reader(label_f)
        for i, row in enumerate(reader):
            if max > i > 0:
                if i % 1000 == 0:
                    print "loading image ", i
                #for angle in xrange(0, 360, 360/nb_img):
                #gray = load_img_hog(row[0])
                img = load_img_and_filter(row[0], '../../data/original_data/' + set + 'ing_set/img/', filter)
                pics.append(img)
                if set == 'train':
                    labels.append([float(v) for v in row[1:]])
                ids.append(row[0])
    pics = np.array(pics)
    if set == 'train':
        total_training_y = np.array(labels, dtype='float32')
        return (pics, total_training_y, ids)
    else:
        return (pics, ids)


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


def prepare_train_valid_test((total_x, total_y, total_ids), tr_size=60, val_size=20):
    """ cut the training set in train, valid, test """
    print "preparing learning datasets"
    last_train = tr_size * total_x.shape[0] / 100
    last_valid = last_train + val_size * total_x.shape[0] / 100
    train_x, train_y, train_ids = total_x[:last_train], total_y[:last_train], total_ids[:last_train] 
    valid_x, valid_y, valid_ids = total_x[last_train:last_valid], total_y[last_train:last_valid], total_ids[last_train:last_valid]
    test_x, test_y, test_ids = total_x[last_valid:], total_y[last_valid:], total_ids[last_valid:]
    rval = [(train_x, train_y, train_ids), (valid_x, valid_y, valid_ids), (test_x, test_y, test_ids)]
    return rval


