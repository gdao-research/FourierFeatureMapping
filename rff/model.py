import numpy as np
import tensorflow as tf
tfkl = tf.keras.layers


class FourierFeature(tfkl.Layer):
    def __init__(self, units=128, preset=True):
        super(FourierFeature, self).__init__()
        self.PI = tf.constant(np.pi, dtype=tf.float32)
        if preset:
            self.B = tf.constant(10.*np.random.randn(2, units), dtype=tf.float32)
        else:
            self.B = self.add_weight(shape=(2, units), initializer=tf.keras.initializers.RandomUniform(-15.0, 15.0), dtype=tf.float32, trainable=True)

    def call(self, x):
        proj = tf.matmul((2*self.PI*x), self.B)
        return tf.concat([tf.sin(proj), tf.cos(proj)], axis=-1)


class Generator(tf.keras.Model):
    def __init__(self, n_layers=4, units=128, dim=256, preset=True):
        super(Generator, self).__init__()
        self.rff = FourierFeature(units, preset)
        self.denses = []
        for i in range(n_layers):
            self.denses.append(tfkl.Dense(dim, activation='relu'))
        self.final = tfkl.Dense(3, activation='sigmoid')

    def call(self, x, use_B=True):
        if use_B:
           x = self.rff(x)
        for dense in self.denses:
            x = dense(x)
        return self.final(x)