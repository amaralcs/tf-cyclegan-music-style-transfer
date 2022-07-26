import os
import logging
import numpy as np
from glob import glob
from random import choice

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Layer, Input, Conv2D, Lambda, ReLU, Conv2DTranspose
from tensorflow.data import Dataset

from reverse_pianoroll import piano_roll_to_pretty_midi

logging.basicConfig(
    format="%(asctime)s : %(name)s [%(levelname)s] : %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("utils_logger")


class InstanceNorm(Layer):
    """Custom implementation of Layer Normalization"""

    def __init__(self, epsilon=1e-5, **kwargs):
        super(InstanceNorm, self).__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        self.scale = tf.Variable(
            initial_value=np.random.normal(1.0, 0.02, input_shape[-1:]),
            trainable=True,
            name="SCALE",
            dtype=tf.float32,
        )
        self.offset = tf.Variable(
            initial_value=np.zeros(input_shape[-1:]),
            trainable=True,
            name="OFFSET",
            dtype=tf.float32,
        )
        super(InstanceNorm, self).build(input_shape)

    def call(self, inputs):
        mean, variance = tf.nn.moments(inputs, axes=[1, 2], keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (inputs - mean) * inv
        return self.scale * normalized + self.offset

    def get_config(self):
        base_config = super(InstanceNorm, self).get_config()
        return {**base_config, "epsilon": self.epsilon}


class ResNetBlock(Layer):
    """Implements a ResNet block with convolutional layers"""

    def __init__(
        self,
        n_units,
        kernel_size=3,
        strides=1,
        activation="relu",
        kernel_initializer=tf.random_normal_initializer(0, 0.2),
    ):
        """A custom implementation of a ResNet block, as used by the original authors.

        Parameters
        ----------
        n_units : int
            The number of units in each convolutional layer.
        kernel_size : int, Optional
        strides : int, Optional
        activation : str, Optional
            Name of activation to use for the block.
        kernel_initializer : tf.Initializer, Optional
            The initialization function to use.
        """
        super(ResNetBlock, self).__init__()
        self.n_units = n_units
        self.kernel_size = kernel_size
        self.kernel_initializer = kernel_initializer
        self.strides = strides
        self.pad_size = (self.kernel_size - 1) // 2
        self.padding = "valid"
        self.activation = keras.activations.get(activation)

    def build(self, input_shape):
        self.padding_1 = Lambda(
            input_padding, arguments={"pad_size": self.pad_size}, name=f"padding_1"
        )
        self.padding_2 = Lambda(
            input_padding, arguments={"pad_size": self.pad_size}, name=f"padding_1"
        )
        self.conv2d_1 = Conv2D(
            self.n_units,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            kernel_initializer=self.kernel_initializer,
            use_bias=False,
            name="conv2D_1",
        )
        self.conv2d_2 = Conv2D(
            self.n_units,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            kernel_initializer=self.kernel_initializer,
            use_bias=False,
            name="conv2D_2",
        )
        self.instance_norm = InstanceNorm()

        super(ResNetBlock, self).build(input_shape)

    def call(self, inputs):
        """
        Parameters
        ----------
        inputs : tf.tensor
            Input tensor.
        """
        X = self.padding_1(inputs)
        X = self.conv2d_1(X)
        X = self.instance_norm(X)
        X = self.padding_2(X)
        X = self.conv2d_2(X)
        X = self.instance_norm(X)
        return self.activation(X + inputs)

    def get_config(self):
        base_config = super(ResNetBlock, self).get_config()
        return {
            **base_config,
            "n_units": self.n_units,
            "kernel_size": self.kernel_size,
            "kernel_initializer": self.kernel_initializer,
            "strides": self.strides,
            "pad_size": self.pad_size,
            "padding": self.padding,
            "activation": tf.keras.activations.serialize(self.activation),
        }


class Conv2DBlock(Layer):
    def __init__(
        self, n_units, kernel_size, strides, padding, kernel_initializer, **kwargs
    ):
        super(Conv2DBlock, self).__init__(**kwargs)
        self.n_units = n_units
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.kernel_initializer = kernel_initializer

        self.conv2D = Conv2D(
            self.n_units,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            kernel_initializer=self.kernel_initializer,
            use_bias=False,
        )
        self.instance_norm = InstanceNorm()
        self.relu = ReLU()

    def call(self, inputs):
        X = inputs
        X = self.conv2D(X)
        X = self.instance_norm(X)
        X = self.relu(X)
        return X

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "n_units": self.n_units,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "padding": self.padding,
            "kernel_initializer": self.kernel_initializer,
        }


class Deconv2DBlock(Layer):
    def __init__(
        self, n_units, kernel_size, strides, padding, kernel_initializer, **kwargs
    ):
        super(Deconv2DBlock, self).__init__(**kwargs)
        self.n_units = n_units
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.kernel_initializer = kernel_initializer

        self.deconv2D = Conv2DTranspose(
            self.n_units,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            kernel_initializer=self.kernel_initializer,
            use_bias=False,
        )
        self.instance_norm = InstanceNorm()
        self.relu = ReLU()

    def call(self, inputs):
        X = inputs
        X = self.deconv2D(X)
        X = self.instance_norm(X)
        X = self.relu(X)
        return X

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "n_units": self.n_units,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "padding": self.padding,
            "kernel_initializer": self.kernel_initializer,
        }


def input_padding(X, pad_size=3):
    """A custom padding implemented by the authors for some of the inputs

    Parameters
    ----------
    X : tf.tensor
        Input tensor to pad
    pad_size : int, Optional
        The size of the padding.

    Returns
    -------
        Input tenser with paddings of shape (pad_size, pad_size) applied to the
        2nd and 3rd dimensions.
    """
    return tf.pad(
        X, [[0, 0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]], mode="REFLECT"
    )


def join_datasets(
    dataset_a, dataset_b, batch_size=16, shuffle=False, shuffle_buffer=15_000
):
    """Joins two given datasets to create inputs of the form ((a1, b1), (a2, b2), ...)

    Parameters
    ----------
    dataset_a : tf.data.Dataset
        Dataset with songs from genre A.
    dataset_b : tf.data.Dataset
        Dataset with songs from genre B.
    shuffle : bool, Optional
        Whether to shuffle the resulting dataset.
    shuffle_buffer : int, Optional
        The size of the shuffle buffer

    Returns
    -------
    tf.data.Dataset

    """
    logger.info("Joining datasets")
    ds = tf.data.Dataset.zip((dataset_a, dataset_b))
    if shuffle:
        ds = ds.shuffle(shuffle_buffer)

    return ds.batch(batch_size).prefetch(1)


def parse_tfr_tensor(tensor):
    """Converts the string representation of a tensor to a float32 tensor.

    Parameters
    ----------
    tensor : tf.Tensor
        The tensor to be parsed. Must have dtype = tf.string.

    Returns
    -------
    tf.data.Dataset
    """
    tensor = tf.io.parse_tensor(tensor, out_type=tf.float32)

    # Prevents dataset.from_tensor_slices from using the first dimension as the
    # dataset dimension
    tensor = tf.expand_dims(tensor, axis=0)
    return Dataset.from_tensor_slices(tensor)


def load_tfrecords(path, set_type, cycle_length):
    """Loads data in the tfrecords format and converts it to a dataset.

    Parameters
    ----------
    path : str
        Path to the prepared phrases.
    set_type: str, Optional
        Whether to load from train/test folder.

    Returns
    -------
    tf.data.Dataset
    """
    logger.info(f"Loading tfrecords from {path}/{set_type}")
    fnames = glob(os.path.join(path, set_type, "*.tfrecord"))

    dataset = tf.data.TFRecordDataset(fnames)
    dataset = dataset.interleave(parse_tfr_tensor, cycle_length=cycle_length)
    return dataset


def load_data(path_a, path_b, set_type, batch_size=16, shuffle=False, cycle_length=50):
    """Helper function for loading tfrecords into a tensorflow dataset

    Parameters
    ----------
    path_a : str
        Path to where phrases of genre A are stored.
    path_b : str
        Path to where phrases of genre B are stored.
    set_type: str, Optional
        Whether to load from train/test folder.
    batch_size : int
        Batch size to use for the dataset.
    shuffle : bool, Optional
        Whether to shuffle the resulting dataset.
    sample_size : int
        If not-null, use only the defined number of samples from each of the datasets.
        This is useful for testing runs.

    Returns
    -------
    tf.data.Dataset
    """
    dataset_a = load_tfrecords(path_a, set_type, cycle_length=cycle_length)
    dataset_b = load_tfrecords(path_b, set_type, cycle_length=cycle_length)
    return join_datasets(dataset_a, dataset_b, batch_size, shuffle=shuffle)


def save_midis(piano_roll, file_name, **kwargs):
    """Take a numpy array and convert it to a midi file.

    The input piano roll is padded so that it fits the number of midi instruments (128)
    and reshaped to shape=(128, frames) where frames is the length of the song.

    Parameters
    ----------
    piano_roll : np.array
        Array of shape (n_timesteps, 84, 1) where n_timesteps is the number of timesteps
        which the model was trained on.
    file_name: str
        Name of the output file.
    """
    piano_roll = piano_roll.squeeze()  # remove unnecessary dimension
    piano_roll = np.concatenate(
        (
            np.zeros((piano_roll.shape[0], 24)),
            piano_roll,
            np.zeros((piano_roll.shape[0], 20)),
        ),
        axis=1,
    )
    # shape = (128, n_timesteps), maximize note velocity
    piano_roll = piano_roll.T * 127 

    pm = piano_roll_to_pretty_midi(piano_roll, **kwargs)
    pm.write(file_name)
