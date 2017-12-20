import numpy as np

# def rotate(img, axes):
    # rotate(np.rot90(m=img, k=np.random.randint(4), axes=axes),


def rotate_90(img, axes=(1, 2)):
    k = np.random.randint(4)
    result = np.rot90(m=img, k=k, axes=axes)
    return result


def flip(img, axes=(1, 2)):
    k = np.random.randint(4)
    result = img
    if k == 1:
        result = np.fliplr(img)
    if k == 2:
        result = np.flipud(img)
    return result

                         #4: lambda img, axes, angle: np.flipud(img),
                         #5: lambda img, axes, angle: np.fliplr(img),

#            self._flipper = {0: lambda img, axes, angle: img,
                         #1: lambda img, axes, angle: np.rot90(m=img, k=1, axes=axes),
                         #2: lambda img, axes, angle: np.rot90(m=img, k=2, axes=axes),
                         #3: lambda img, axes, angle: np.rot90(m=img, k=3, axes=axes),
                         #4: lambda img, axes, angle: ,
                         #6: lambda img, axes, angle: rotate(np.rot90(m=img, k=np.random.randint(4), axes=axes),
                                                            #angle=angle, axes=axes, reshape=False)}
