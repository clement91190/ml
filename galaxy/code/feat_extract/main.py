""" Step 1 : generate feature vector from images 

For each method, we add a file in the data folder, and store 2 files : train_feat.tkl, test_feat.tkl created 
using a given method. 

train_feat should be 
"""

#extraction methods:
from method_gray import GrayExtractor
from method_gray_center import GrayCenterExtractor

methods_map = {
    'method_gray': GrayExtractor,
    'method_gray_center': GrayCenterExtractor,
    }  # feature_extraction is a function that take

pixelSize = 4
img_size = 36
size = 424
small = (size - img_size * pixelSize) / 2
crop_dimensions = (small, small, small + img_size * pixelSize, small + img_size * pixelSize)
#print crop_dimensions


def main():

    method = "method_gray_center"
    feature_extractor = methods_map[method]()
    feature_extractor.run()
    print "...done"


if __name__ == "__main__":
    main()
