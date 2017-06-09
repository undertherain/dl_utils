import begin
import imp
import numpy as np

try:
    imp.find_module('cv2')
except ImportError:
    HAS_OPENCV = False
    import scipy.misc
else:
    HAS_OPENCV = True
    import cv2


class ImageLoader:
    def __init__(self):
        pass

    def resize(self, img, shape, interpolation=3):
        if HAS_OPENCV:
            return cv2.resize(img, shape, interpolation)
        else:
            return scipy.misc.imresize(img, shape)

    def pad(self, img, dim_max=5616):
        pad1 = dim_max - img.shape[0]
        pad1 = max(0, pad1)
        pad2 = dim_max - img.shape[1]
        pad2 = max(0, pad2)
        res = np.lib.pad(img, ((0, pad1), (0, pad2), (0, 0)), 'constant', constant_values=0)
        return res

    def load_image(self, path, grey_scale=False):
        # todo: check if file exists, don't return None, raise proper exception
        if HAS_OPENCV:
            if grey_scale:
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            else:
                img = cv2.imread(path, cv2.IMREAD_COLOR)
            # img = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
        else:
            img = scipy.misc.imread(path, grey_scale)

        return (img / 255).astype(np.float32)

    def make_channels_first(self, img):
        img = np.rollaxis(img, 2, 0)
        return img

@begin.start
def main():
    print("testing data loader")
    il = ImageLoader()
    img = il.load_image("test_data/ireland.jpg")
    img = il.resize(img, (64, 64))
    print(img.shape)
    # todo: proper unit tests for different image formats
