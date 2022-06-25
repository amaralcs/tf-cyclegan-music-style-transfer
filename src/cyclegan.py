import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, Input, Lambda, LeakyReLU

from utils import Conv2DBlock, Deconv2DBlock, ResNetBlock, input_padding


class CycleGAN(Model):
    def __init__(
        self,
        genre_A,
        genre_B,
        pitch_range=84,
        n_timesteps=64,
        n_units_discriminator=64,
        n_units_generator=64,
        l1_lambda=10.0,
        gamma=1.0,
        sigma_d=0.01,
        initializer=tf.random_normal_initializer(0, 0.02),
        **kwargs,
    ):
        super(CycleGAN, self).__init__(**kwargs)
        self.genre_A = genre_A
        self.genre_B = genre_B
        self.pitch_range = pitch_range
        self.n_timesteps = n_timesteps
        self.n_units_discriminator = n_units_discriminator
        self.n_units_generator = n_units_generator
        self.l1_lambda = l1_lambda
        self.gamma = gamma
        self.sigma_d = sigma_d
        self.initializer = initializer

        self.build_model()

    def build_model(
        self,
        d_optimizers=None,
        g_optimizers=None,
        default_init={"learning_rate": 2e-4, "beta_1": 0.5},
    ):
        """Builds the CycleGAN model

        Parameters
        ----------
        d_optimizers : List[tf.keras.optimizer, tf.keras.optimizer]
            List with optimizers to use for d_A and d_B, respectively.
        g_optimizers : List[tf.keras.optimizer, tf.keras.optimizer]
            List with optimizers to use for g_A2B and g_B2A, respectively.
        default_init : dict, Optional
            Predefined parameter values for Adam optimzer to use as the default.

        Returns
        -------
        None
        """
        # Set the default optimizers for the discriminator
        if not d_optimizers:
            d_optimizers = [Adam(**default_init), Adam(**default_init)]
        # and for the generator
        if not g_optimizers:
            g_optimizers = [Adam(**default_init), Adam(**default_init)]

        self.d_A_opt, self.d_B_opt = d_optimizers
        self.g_A2B_opt, self.g_B2A_opt = g_optimizers

        self.discriminator_A = self.build_discriminator("discriminator_A")
        self.discriminator_B = self.build_discriminator("discriminator_B")

        self.generator_A2B = self.build_generator("generator_A2B")
        self.generator_B2A = self.build_generator("generator_B2A")

        super(CycleGAN, self).compile()

    def get_config(self):
        """Loads the model configuration"""
        base_config = super().get_config()
        return {
            **base_config,
            "genre_A": self.genre_A,
            "genre_B": self.genre_B,
            "pitch_range": self.pitch_range,
            "n_timesteps": self.n_timesteps,
            "n_units_discriminator": self.n_units_discriminator,
            "n_units_generator": self.n_units_generator,
            "l1_lambda": self.l1_lambda,
            "gamma": self.gamma,
            "sigma_d": self.sigma_d,
            "initializer": self.initializer,
            "d_loss_generated": self.d_loss_generated,
            "d_loss_real": self.d_loss_real,
            "d_loss_single": self.d_loss_single,
            "cycle_loss": self.cycle_loss,
            "g_loss_single": self.f_loss_single,
        }

    def d_loss_generated(self, y_preds):
        """Calculates the MSE between y_preds and a tensor of zeros of shape y_preds.shape

        Parameters
        ----------
        y_preds : tf.tensor

        Returns
        -------
            Tensor with MSE result.
        """
        y_true = tf.zeros_like(y_preds)
        return tf.reduce_mean(tf.square(y_true - y_preds))

    def d_loss_real(self, y_preds):
        """Calculates the MSE between y_preds and a tensor of ones of shape y_preds.shape

        Parameters
        ----------
        y_preds : tf.tensor

        Returns
        -------
            Tensor with MSE result.
        """
        y_true = tf.ones_like(y_preds)
        return tf.reduce_mean(tf.square(y_true - y_preds))

    def d_loss_single(self, y_true, y_generated):
        """The overall loss for a single discriminator

        Parameters
        ----------
        y_true : tf.tensor
            Logits of the discriminator after receiving a true sample.
        y_generated : tf.tensor
            Logits of the discriminator after receiving a generated sample.

        Returns
        -------
            Tensor with the discriminator loss.
        """
        loss_generated = self.d_loss_generated(y_generated)
        loss_real = self.d_loss_real(y_true)
        return (loss_generated + loss_real) / 2

    def build_discriminator(self, name):
        """Builds CycleGAN discriminator

        Returns
        -------
        tf.keras.models.Model
        """
        # There's only '1' channel in the MIDI data, hence final dim = 1
        inputs = Input(shape=(self.n_timesteps, self.pitch_range, 1))
        X = inputs

        X = Conv2D(
            self.n_units_discriminator,
            kernel_size=7,
            strides=2,
            padding="same",
            kernel_initializer=self.initializer,
            use_bias=False,
            name="conv2D_1",
        )(X)
        X = LeakyReLU(alpha=0.2)(X)
        X = Conv2D(
            self.n_units_discriminator * 4,
            kernel_size=7,
            strides=2,
            padding="same",
            kernel_initializer=self.initializer,
            use_bias=False,
            name="conv2D_2",
        )(X)
        X = LeakyReLU(alpha=0.2)(X)

        outputs = Conv2D(
            1,
            kernel_size=7,
            strides=1,
            padding="same",
            kernel_initializer=self.initializer,
            use_bias=False,
            name="conv2D_3",
        )(X)
        discriminator = Model(inputs=inputs, outputs=outputs, name=name)
        return discriminator

    def cycle_loss(self, y_true, y_preds):
        """Calculates the cycle consistency loss given by

            cycle_loss = lambda * (MAE(X_a, X'_a) + MAE(X_b, X'b))

        Where:
            lambda is a hyperparameter set during model initialization
            X_a is the input song from style A
            X'_a is the result of converting X_a to genre B and then back to genre A
            X_b is the input song from style B
            X'_b is the result of converting X_b to genre A and then back to genre B

        Parameters
        ----------
        y_true : tuple(tf.tensor, tf.tensor)
            Tuple containing the original songs X_a and X_b
        y_preds : tuple(tf.tensor, tf.tensor)
            Tuple containing the cycled songs X'_a and X'_b

        Returns
        -------
            Tensor with the cycle loss.
        """
        X_a, X_b = y_true
        X_a_cycle, X_b_cycle = y_preds

        return self.l1_lambda * (
            tf.reduce_mean(tf.abs(X_a - X_a_cycle))
            + tf.reduce_mean(tf.abs(X_b - X_b_cycle))
        )

    def g_loss_single(self, y_true, cycle_loss):
        """Computes loss for a single generator.

        The loss for one generator is the MSE between the outputs of the discriminator of the
        target genre and a tensor of ones of the same shape, added to the cycle loss.

        Parameters
        ----------
        y_true : tf.tensor
            Tensor with predictions for the discriminator of target genre
        y_preds : tuple(tf.tensor, tf.tensor)
            Tensor with predictions for the discriminator of target genre
        Returns
        -------
            Tensor with the generator loss.
        """
        return cycle_loss + tf.reduce_mean(tf.square(y_true - tf.ones_like(y_true)))

    def build_generator(self, name):
        """Builds CycleGAN generator

        Returns
        -------
        tf.keras.models.Model
        """
        # There's only '1' channel in the MIDI data, hence final dim = 1
        inputs = Input(shape=(self.n_timesteps, self.pitch_range, 1))

        X = inputs
        X = Lambda(input_padding, name="padding_1")(X)

        X = Conv2DBlock(
            self.n_units_generator,
            kernel_size=7,
            strides=1,
            padding="valid",
            kernel_initializer=self.initializer,
            name="conv2D_1",
        )(X)
        X = Conv2DBlock(
            self.n_units_generator * 2,
            kernel_size=3,
            strides=2,
            padding="same",
            kernel_initializer=self.initializer,
            name="conv2D_2",
        )(X)
        X = Conv2DBlock(
            self.n_units_generator * 4,
            kernel_size=3,
            strides=2,
            padding="same",
            kernel_initializer=self.initializer,
            name="conv2D_3",
        )(X)

        for _ in range(10):
            X = ResNetBlock(
                self.n_units_generator * 4, kernel_initializer=self.initializer
            )(X)

        X = Deconv2DBlock(
            self.n_units_generator * 2,
            kernel_size=3,
            strides=2,
            padding="same",
            kernel_initializer=self.initializer,
            name=f"deconv2D_1",
        )(X)
        X = Deconv2DBlock(
            self.n_units_generator,
            kernel_size=3,
            strides=2,
            padding="same",
            kernel_initializer=self.initializer,
            name=f"deconv2D_2",
        )(X)

        X = Lambda(input_padding, name="padding_2")(X)

        outputs = Conv2D(
            1,
            kernel_size=7,
            strides=1,
            padding="valid",
            kernel_initializer=self.initializer,
            activation="sigmoid",
            use_bias=False,
            name="conv2D_4",
        )(X)
        generator = Model(inputs=inputs, outputs=outputs, name=name)
        return generator

    def gaussian_noise(self):
        """Generates Gaussian noise sampled form N(0, sigma_d) with shape:
            [n_timesteps, pitch_range, 1]

        Note that the absolute value of the sampled values are returned.

        Returns
        -------
        A tensor of the specified shape filled with random normal values.
        """
        sample = tf.random.normal(
            shape=[self.n_timesteps, self.pitch_range, 1],
            mean=0,
            stddev=self.sigma_d,
            dtype=tf.float32,
        )
        return tf.abs(sample)

    def train_step(self, inputs):
        """The training step.

        Parameters
        ----------
        inputs : List[np.array, np.array]
            Input samples from genre_a and genre_b respectively

        Returns
        -------
        dict
            Dictionary of losses
        """
        X_a, X_b = inputs  # unpack the inputs

        noise = self.gaussian_noise()

        with tf.GradientTape(persistent=True) as tape:
            # X_a in the style of X_b
            X_a_transfer = self.generator_A2B(X_a, training=True)
            X_a_cycle = self.generator_B2A(X_a_transfer, training=True)

            X_b_transfer = self.generator_B2A(
                X_b, training=True
            )  # X_b in the style of X_a
            X_b_cycle = self.generator_A2B(X_b_transfer, training=True)

            # discriminator evaluation
            d_a_real_logits = self.discriminator_A(X_a + noise, training=True)
            d_a_generated_logits = self.discriminator_A(
                X_b_transfer + noise, training=True
            )

            d_b_real_logits = self.discriminator_B(X_b + noise, training=True)
            d_b_generated_logits = self.discriminator_B(
                X_a_transfer + noise, training=True
            )

            # discriminator losses
            d_A_loss = self.d_loss_single(d_a_real_logits, d_a_generated_logits)
            d_B_loss = self.d_loss_single(d_b_real_logits, d_b_generated_logits)

            # generator losses
            cycle_loss = self.cycle_loss((X_a, X_b), (X_a_cycle, X_b_cycle))
            g_A2B_loss = self.g_loss_single(d_a_real_logits, cycle_loss)
            g_B2A_loss = self.g_loss_single(d_b_real_logits, cycle_loss)
            g_loss = g_A2B_loss + g_B2A_loss - cycle_loss

        # generator gradients
        g_A2B_gradients = tape.gradient(
            g_A2B_loss, self.generator_A2B.trainable_variables
        )
        g_B2A_gradients = tape.gradient(
            g_B2A_loss, self.generator_B2A.trainable_variables
        )
        self.g_A2B_opt.apply_gradients(
            zip(g_A2B_gradients, self.generator_A2B.trainable_variables)
        )
        self.g_B2A_opt.apply_gradients(
            zip(g_B2A_gradients, self.generator_B2A.trainable_variables)
        )

        # discriminator gradients
        d_A_gradients = tape.gradient(
            d_A_loss, self.discriminator_A.trainable_variables
        )
        d_B_gradients = tape.gradient(
            d_B_loss, self.discriminator_B.trainable_variables
        )
        self.d_A_opt.apply_gradients(
            zip(d_A_gradients, self.discriminator_A.trainable_variables)
        )
        self.d_B_opt.apply_gradients(
            zip(d_B_gradients, self.discriminator_B.trainable_variables)
        )

        return {
            "d_A_loss": d_A_loss,
            "d_B_loss": d_B_loss,
            "g_A2B_loss": g_A2B_loss,
            "g_B2A_loss": g_B2A_loss,
            "cycle_loss": cycle_loss,
        }

    def call(self, inputs, direction="A2B"):
        """Converts the inputs to the trained style and generates the cyle back to the original style

        Parameters
        ----------
        inputs : tf.data.Dataset
            Datset with songs from the given genres.
        direction : str
            The direction of the transfer. One of "A2B" or "B2A".

        Returns
        -------
        original : np.ndarray
            A numpy array of shape (batch_size, n_timesteps, note_range, 1) with the original song.
        transfer : np.ndarray
            A numpy array of shape (batch_size, n_timesteps, note_range, 1) with the transferred song.
        cycled : np.ndarray
            A numpy array of shape (batch_size, n_timesteps, note_range, 1) with the cycled song.
        """
        X_a, X_b = inputs

        if direction == "A2B":
            original = X_a
            transfer = self.generator_A2B(X_a)
            cycle = self.generator_B2A(transfer)
        elif direction == "A2B":
            original = X_b
            transfer = self.generator_B2A(X_b)
            cycle = self.generator_B2A(transfer)
        else:
            raise Exception(f"Transfer direction '{direction}' not understood")

        return original.numpy(), transfer.numpy(), cycle.numpy()
