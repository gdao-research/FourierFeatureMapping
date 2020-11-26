import imageio
import numpy as np
import tensorflow as tf


def get_data():
    image_url = 'https://live.staticflickr.com/7492/15677707699_d9d67acf9d_b.jpg'
    img = (imageio.imread(image_url)[..., :3] / 255.).astype('float32')
    c = [img.shape[0]//2, img.shape[1]//2]
    r = 256
    img = img[c[0]-r:c[0]+r, c[1]-r:c[1]+r]
    coords = np.linspace(0, 1, img.shape[0], endpoint=False)
    x_test = np.stack(np.meshgrid(coords, coords), -1).astype('float32')
    test_data = [x_test, img]  # coords, image
    train_data = [x_test[::2,::2], img[::2,::2]]  # down sampling coords, image
    return train_data, test_data

def loss_fn(y, yHat):
    return 0.5*tf.reduce_mean((y - yHat)**2)

def psnr_fn(loss):
    return -10.*tf.math.log(2.*loss)/tf.math.log(10.)