""" Step 1 : generate feature vector from images 

For each method, we add a file in the data folder, and store 2 files : train_feat.tkl, test_feat.tkl created 
using a given method. 

train_feat should be 
"""

#extraction methods:
from code.feat_extract.method_gray import GrayExtractor
from code.feat_extract.method_gray_center import GrayCenterExtractor
from code.feat_extract.method_gray_ultra_center import GrayUltraCenterExtractor
from code.feat_extract.method_gray_large import GrayLargeExtractor
from code.feat_extract.method_hog import HOGExtractor
from code.feat_extract.method_LLE import LLEExtractor
from code.feat_extract.method_PCA import PCAExtractor
from code.feat_extract.method_PCA_max_var1 import PCAVAR1Extractor

methods_map = {
    'method_gray': GrayExtractor,
    'method_gray_center': GrayCenterExtractor,
    'method_gray_ultra_center': GrayUltraCenterExtractor,
    'method_gray_large': GrayLargeExtractor,
    'method_LLE': LLEExtractor,
    'method_PCA': PCAExtractor,
    'method_PCA_var1': PCAVAR1Extractor,
    'method_hog': HOGExtractor}  # feature_extraction is a function that take

pixelSize = 4
img_size = 36
size = 424
small = (size - img_size * pixelSize) / 2
crop_dimensions = (small, small, small + img_size * pixelSize, small + img_size * pixelSize)
#print crop_dimensions


def main():

    method = "method_PCA_var1"
    feature_extractor = methods_map[method]()
    feature_extractor.run()
    print "...done"


if __name__ == "__main__":
    main()
