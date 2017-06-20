import os
from scipy import misc
import matplotlib.image as mpimg
import numpy as np
import cv2
from rawkit.raw import Raw
from PIL import Image

def load_image(infilename):
    data = mpimg.imread(infilename)
    return data


def load_image_noSTD(infilename):
    data = misc.imread(infilename)
    return data


def loading_training_data(image_dir, recto, size):
    '''
    Loading images: 1-split_coef_test for training and split_coef_test for testing.
    Return training and testing images.
    
    Args:
        :param (string): The folder path with images to be loaded
        :param (boolean): If true rectro images will be loaded
        :param (int): Maximum number of images to be loaded

    Returns:
        :return two numpy matrices of the images and masks
    '''
    files = os.listdir(image_dir)
    print("Loading " + str(len(files)) + " images for training")
    files = sorted(files)
    imgs = []
    gt_imgs = []

    for i in range(min(len(files), size)):
        if recto:
            if "recto" in files[i]:
                if "truth" not in files[i]:
                    imgs.append(load_image_noSTD(image_dir + files[i]))
                else:
                    gt_imgs.append(load_image_noSTD(image_dir + files[i]))
            else:
                continue
        else:
            if "verso" in files[i]:
                if "truth" not in files[i]:
                    imgs.append(load_image_noSTD(image_dir + files[i]))
                else:
                    gt_imgs.append(load_image_noSTD(image_dir + files[i]))
            else:
                continue
    return np.stack(imgs), np.stack(gt_imgs)


def get_patches(images, ver, hor):
    '''
        Loading images: 1-split_coef_test for training and split_coef_test for testing.
        Return training and testing images.

        Args:
            :param (string): The folder path with images to be loaded
            :param (boolean): If true rectro images will be loaded
            :param (int): Maximum number of images to be loaded

        Returns:
            :return two numpy matrices of the images and masks
        '''
    arr = []
    for k in range(len(images)):
        img = images[k]
        for i in range(ver):
            for j in range(hor):
                arr.append(img[(i * img.shape[0] // ver):((i + 1) * img.shape[0] // ver),
                           (j * img.shape[1] // hor):((j + 1) * img.shape[1] // hor)])
    return np.stack(arr, axis=0)

def load_images(image_dir):
    '''
        Loading images: 1-split_coef_test for training and split_coef_test for testing.
        Return training and testing images.

        Args:
            :param (string): The folder path with images to be loaded
            :param (boolean): If true rectro images will be loaded
            :param (int): Maximum number of images to be loaded

        Returns:
            :return two numpy matrices of the images and masks
        '''
    imgs = []
    files = os.listdir(image_dir)
    for f in files:
        imgs.append(load_image_noSTD(image_dir + f))
    return imgs

def load_jpg_file_to_image(filename):
    """
    Load a jpg file
    :param filename:
    :return: PIL.Image
    """
    image = Image.open(filename)
    return image

def load_raw_file_to_image(filename):
    """
    Load a Raw file
    :param filename:
    :return: PIL.Image
    """
    with Raw(filename=filename) as raw:
        raw.options.rotation = 0
        w = raw.data.contents.sizes.width
        h = raw.data.contents.sizes.height
        buffered_image = np.array(raw.to_buffer())
        image = Image.frombytes('RGB', (w, h), buffered_image)
        return image

def batch_iter(y, tx, batch_size, epochs):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)
    num_batches = int(np.ceil(data_size / batch_size))
    print(y.shape)
    print(tx.shape)

    for epoch in range(epochs):
        transformed_y = []
        transformed_x = []
        for i in range(data_size):
            tmp_y = y[i]
            tmp_x = tx[i]

            if (np.random.random_sample() < 0.3):
                tmp_y = cv2.flip(tmp_y, 0)
                tmp_x = cv2.flip(tmp_x, 0)

            if (np.random.random_sample() < 0.3):
                tmp_y = cv2.flip(tmp_y, 1)
                tmp_x = cv2.flip(tmp_x, 1)

            if (np.random.random_sample() < -1):
                M = cv2.getRotationMatrix2D((tmp_y.shape[1] / 2, tmp_y.shape[0] / 2), 90, 1)
                tmp_y = cv2.warpAffine(tmp_y, M, (tmp_y.shape[1], tmp_y.shape[0]))
                tmp_x = cv2.warpAffine(tmp_x, M, (tmp_x.shape[1], tmp_x.shape[0]))

            elif (np.random.random_sample() < -1):
                M = cv2.getRotationMatrix2D((tmp_y.shape[1] / 2, tmp_y.shape[0] / 2), 270, 1)
                tmp_y = cv2.warpAffine(tmp_y, M, (tmp_y.shape[1], tmp_y.shape[0]))
                tmp_x = cv2.warpAffine(tmp_x, M, (tmp_x.shape[1], tmp_x.shape[0]))


            transformed_y.append(tmp_y)
            transformed_x.append(tmp_x)

        transformed_y = np.stack(transformed_y, axis=0)
        transformed_x = np.stack(transformed_x, axis=0)

        shuffle_indices = np.random.permutation(np.arange(data_size))
        transformed_y = transformed_y[shuffle_indices]
        transformed_x = transformed_x[shuffle_indices]

        for batch_num in range(num_batches):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            if start_index != end_index:
                yield transformed_y[start_index:end_index], transformed_x[start_index:end_index]