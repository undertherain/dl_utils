import unittest
from dl_utils.imaging import ImageLoader


il = ImageLoader()
path_img = "dl_utils/imaging/test_data/ireland.jpg"


class Tests(unittest.TestCase):

    def test_load(self):
        img = il.load_image(path_img)
        print(img.shape)

    def test_blur(self):
        img = il.load_image(path_img)
        img = il.random_blur(img=img, axes=(0, 1), proba=1)
        il.save_image(img, "/tmp/img_blurred.jpg")
