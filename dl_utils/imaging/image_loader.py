import os
import begin
import errno
import imp
import numpy as np
from scipy.misc import imsave
from scipy.ndimage.interpolation import rotate
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import zoom

try:
    imp.find_module('cv2')
except ImportError:
    HAS_OPENCV = False
    import scipy.misc
else:
    HAS_OPENCV = True
    import cv2

try:
    imp.find_module('tifffile')
except ImportError:
    HAS_TIFFFILE = False
    print("No module named 'tifffile'")
    print("Cannot process .tif format.")
else:
    HAS_TIFFFILE = True
    import tifffile as tiff


class ImageLoader:
    def __init__(self, scroll_count=8):
        self._flipper = {0: lambda img, axes, angle: img,
                         1: lambda img, axes, angle: np.rot90(m=img, k=1, axes=axes),
                         2: lambda img, axes, angle: np.rot90(m=img, k=2, axes=axes),
                         3: lambda img, axes, angle: np.rot90(m=img, k=3, axes=axes),
                         4: lambda img, axes, angle: np.flipud(img),
                         5: lambda img, axes, angle: np.fliplr(img),
                         6: lambda img, axes, angle: rotate(np.rot90(m=img, k=np.random.randint(4), axes=axes),
                                                            angle=angle, axes=axes, reshape=False)}

        self._scroller = {0: lambda img, axes: np.roll(img, self._scroll_img(img, axes[0]), axis=axes[0]),
                          1: lambda img, axes: np.roll(img, self._scroll_img(img, axes[1]), axis=axes[1])}

        self._blurrer = {0: lambda img, axes: self._blur_img(img, axes),
                         1: lambda img, axes: img}

        self._scroll_count = scroll_count

    def center_crop(self, img, axes, new_shape=(64, 64)):
        old_shape = [img.shape[axes[0]], img.shape[axes[1]]]
        left1 = np.floor((old_shape[0] - new_shape[0]) / 2.).astype(np.int32)
        right1 = np.floor((old_shape[0] + new_shape[0]) / 2.).astype(np.int32)
        left2 = np.floor((old_shape[1] - new_shape[1]) / 2.).astype(np.int32)
        right2 = np.floor((old_shape[1] + new_shape[1]) / 2.).astype(np.int32)
        slc = [slice(None)] * len(img.shape)
        slc[axes[0]] = slice(left1, right1)
        slc[axes[1]] = slice(left2, right2)
        return img[slc]

    def mosaic_pad(self, img, axes):
        pad_axis_width = (img.shape[axes[0]], img.shape[axes[1]])
        pad_width = [(0, 0), (0, 0), (0, 0)]
        pad_width[axes[0]] = pad_axis_width
        pad_width[axes[1]] = pad_axis_width
        return np.lib.pad(img, pad_width=tuple(pad_width), mode='reflect')

    def _blur_img(self, img, axes):
        pix = np.random.randint(2, 5)
        sigma = np.zeros(3, dtype=np.int32)
        idx = (np.array(list(axes), dtype=np.int32),)
        sigma[idx] = pix
        return gaussian_filter(img, sigma=tuple(sigma))

    def _scroll_img(self, img, axes):
        unit_shift = int(img.shape[axes] / self._scroll_count)
        return unit_shift * np.random.randint(unit_shift)

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

    def random_rotate(self, img, axes=(0, 1), angle=15):
        rot_idx = np.random.randint(7)
        return self._flipper[rot_idx](img, axes, angle), True if rot_idx == 6 else False

    def random_scroll(self, img, axes=(0, 1)):
        return self._scroller[np.random.randint(2)](img, axes)

    def random_blur(self, img, axes=(0, 1), proba=0.2):
        return self._blurrer[1 if np.random.randint(int(1 / proba)) > 0 else 0](img, axes)

    def random_zoom(self, img, axes=(0, 1), rot=False, low=0.5, high=1.0):
        if rot:
            high = 0.8
        cur_low = round(np.random.uniform(low=low, high=high), 1)
        zoom_size = [1, 1, 1]
        zoom_size[axes[0]] = zoom_size[axes[1]] = cur_low
        return zoom(img, zoom=tuple(zoom_size), order=1, mode='reflect')

    def load_image(self, path, grey_scale=False):
        if os.path.isfile(path) is False:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path)

        img_type = path.split('.')[-1]

        if img_type in ['tif', 'tiff']:
            if HAS_TIFFFILE:
                img = tiff.imread(path)
            else:
                raise ImportError("No module named 'tifffile'")
        else:
            if HAS_OPENCV:
                if grey_scale:
                    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                else:
                    img = cv2.imread(path, cv2.IMREAD_COLOR)
            else:
                img = scipy.misc.imread(path, grey_scale)

        return (img / float(np.iinfo(img.dtype).max)).astype(np.float32)

    def save_image(self, img, path):
        if HAS_OPENCV:
            cv2.imwrite(path, img * 255)
        else:
            imsave(path, img)

    def make_channels_first(self, img):
        axes = img.shape
        img = np.rollaxis(img, axes.index(min(axes)), 0)
        return img


@begin.start
def main():
    print("testing data loader")
    il = ImageLoader()
    img = il.load_image("test_data/ireland.jpg")
    # img = il.mosaic_pad(img=img, axes=(1, 2))
    # img = il.random_rotate(img=img, axes=(1, 2))
    # img = il.center_crop(img=img, axes=(1, 2))
    img = il.random_blur(img=img, axes=(0, 1), proba=1)
    # img = il.resize(img, (64, 64))

    print(img.shape)
    if HAS_OPENCV:
        cv2.imwrite('outfile.jpg', img)
    else:
        scipy.misc.imsave('outfile.jpg', img)
