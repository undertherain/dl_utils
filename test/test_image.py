import unittest
from dl_utils.imaging import ImageLoader


il = ImageLoader()
path_img = "dl_utils/imaging/test_data/ireland.jpg"


class Tests(unittest.TestCase):

    def test_load(self):
        img = il.load_image(path_img)
        print(img.shape)
