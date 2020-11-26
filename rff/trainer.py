from tqdm import tqdm
import numpy as np
import tensorflow as tf
from rff.model import Generator
from rff.utils import *


class Trainer:
    def __init__(self, num_layers=4, units=128, dim=256, preset=True):
        self.train_data, self.test_data = get_data()
        self.num_layers = num_layers
        self.units = units
        self.dim = dim
        self.model = Generator(self.num_layers, self.units, self.dim, preset)
        self.optim = tf.keras.optimizers.Adam(1e-4)

    @tf.function
    def _train(self, use_B=True):
        with tf.GradientTape() as tape:
            rgbs =  self.model(self.train_data[0], use_B)
            loss = loss_fn(self.train_data[1], rgbs)
        psnr = psnr_fn(loss)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optim.apply_gradients(zip(grads, self.model.trainable_variables))
        return psnr

    @tf.function
    def _test(self, use_B=True):
        rgbs = self.model(self.test_data[0], use_B)
        loss = loss_fn(self.test_data[1], rgbs)
        psnr = psnr_fn(loss)
        return psnr, rgbs

    def train(self, iters=2000, use_B=True):
        psnrs = []
        psnrs_test = []
        for i in tqdm(range(iters+1)):
            psnr = self._train(use_B)
            psnr_test, rgbs = self._test(use_B)
            if i % 25 == 0:
                psnrs.append(psnr.numpy())
                psnrs_test.append(psnr_test)
        return psnrs, psnrs_test, rgbs