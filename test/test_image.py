import unittest
from dl_utils.imaging import ImageLoader


il = ImageLoader()
path_img = "dl_utils/imaging/test_data/ireland.jpg"


class Tests(unittest.TestCase):

    def test_load(self):
        img = il.load_image(path_img)
        print(img.shape)

    def test_resize(self):
        img = il.load_image(path_img)
        img = il.resize(img, (512, 512))
        il.save_image(img, "/tmp/img_resized.jpg")

    def test_blur(self):
        img = il.load_image(path_img)
        img = il.random_blur(img=img, axes=(0, 1), proba=1)
        il.save_image(img, "/tmp/img_blurred.jpg")

    def test_rotate(self):
        img = il.load_image(path_img)
        img, rot = il.random_rotate(img=img, axes=(0, 1))
        il.save_image(img, "/tmp/img_rotated.jpg")

    def test_pad(self):
        img = il.load_image(path_img)
        img = il.resize(img, (512, 512))
        img = il.mosaic_pad(img=img, axes=(0, 1))
        il.save_image(img, "/tmp/img_padded.jpg")

    def test_scroll(self):
        img = il.load_image(path_img)
        img = il.random_scroll(img=img, axes=(0, 1))
        il.save_image(img, "/tmp/img_scrolled.jpg")

    def test_zoom(self):
        img = il.load_image(path_img)
        img = il.resize(img, (512, 512))
        img = il.mosaic_pad(img=img, axes=(0, 1))
        img = il.random_zoom(img=img, axes=(0, 1))
        img = il.center_crop(img=img, axes=(0, 1), new_shape=(512, 512))
        il.save_image(img, "/tmp/img_zoomed.jpg")
