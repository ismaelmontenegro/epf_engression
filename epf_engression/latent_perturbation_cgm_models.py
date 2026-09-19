import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.losses import Loss
from tensorflow.keras.layers import Layer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay


class ShapeLayer(Layer):
    def call(self, inputs):
        return tf.shape(inputs)[0]


class SampleLayer(Layer):
    def __init__(self, latent_dist="normal", **kwargs):
        super(SampleLayer, self).__init__(*kwargs)
        self.latent_dist = latent_dist

    def call(self, inputs):
        bs, dim1, dim2 = inputs
        if self.latent_dist == "uniform":
            epsilon = tf.random.uniform(shape=(bs, dim1, dim2), minval=-1.0, maxval=1.0)
        elif self.latent_dist == "normal":
            epsilon = tf.random.normal(shape=(bs, dim1, dim2), mean=0.0, stddev=1.0)
        return epsilon


# define energy score
def energy_score(y_true, y_pred):
    """
    Computes energy score efficiently.
    Parameters
    ----------
    y_true : tf tensor of shape (BATCH_SIZE, D, 1)
        True values.
    y_pred : tf tensor of shape (BATCH_SIZE, D, N_SAMPLES)
        Predictive samples.
    Returns
    -------
    tf tensor of shape (BATCH_SIZE,)
        Scores.
    """
    n_samples_model = tf.cast(tf.shape(y_pred)[2], dtype=tf.float32)

    es_12 = tf.reduce_sum(tf.sqrt(tf.clip_by_value(tf.matmul(y_true, y_true,
                                                             transpose_a=True,
                                                             transpose_b=False) +
                                                   tf.square(tf.linalg.norm(y_pred,
                                                                            axis=1,
                                                                            keepdims=True)) -
                                                   2 * tf.matmul(y_true, y_pred,
                                                                 transpose_a=True,
                                                                 transpose_b=False),
                                                   K.epsilon(), 1e10)),
                          axis=(1, 2))
    G = tf.linalg.matmul(y_pred, y_pred, transpose_a=True, transpose_b=False)
    d = tf.reduce_sum(tf.square(y_pred), axis=1, keepdims=True)
    es_22 = tf.reduce_sum(tf.sqrt(tf.clip_by_value(d +
                                                   tf.transpose(d, perm=(0, 2, 1)) -
                                                   2 * G,
                                                   K.epsilon(), 1e10)),
                          axis=(1, 2))

    loss = es_12 / (n_samples_model) - es_22 / (2 * n_samples_model * (n_samples_model - 1))

    return tf.reduce_mean(loss)


def custom_loss(y_true, y_pred):
    """
    Computes energy score efficiently.
    Parameters
    ----------
    y_true : tf tensor of shape (BATCH_SIZE, 1)
        True values.
    y_pred : tf tensor of shape (BATCH_SIZE, D, N_SAMPLES)
        Predictive samples.
    Returns
    -------
    tf tensor of shape (BATCH_SIZE,)
        Scores.
    """
    # Find the index of the maximum value along axis=1
    max_indices = tf.argmax(y_pred, axis=1)
    # (BATCH_SIZE, N_SAMPLES)

    # Ensure max_indices and y_true are of the same data type
    max_indices = tf.cast(max_indices, dtype=tf.float32)
    y_true = tf.cast(y_true, dtype=tf.float32)

    # Example loss calculation (mean squared error between max_indices and y_true)
    loss_square = tf.square(max_indices - y_true) / 100
    loss = tf.reduce_mean(loss_square, axis=1)
    mean_loss = tf.reduce_mean(loss)

    return mean_loss


# subclass Keras loss
class EnergyScore(Loss):
    def __init__(self, weight=1.0, name="EnergyScore", **kwargs):
        super().__init__(name=name, **kwargs)
        self.weight = weight

    def call(self, y_true, y_pred):
        y_true_price = y_true[:, 1:, :]
        y_true_index = y_true[:, 0, :]
        ES = energy_score(y_true_price, y_pred)
        index_diff = custom_loss(y_true_index, y_pred)

        # Use tf.print to print the values
        # tf.print("ES:", ES, "Index:", index_diff)

        return (ES / 2) * (1 - self.weight) + index_diff * self.weight


class cgm_engression_hybrid(object):

    def __init__(self, dim_out, dim_in_features, dim_in_past, dim_latent, n_samples_train, loss_weight, emb_size=2):

        super(cgm_engression_hybrid, self).__init__()

        self.shape_layer = ShapeLayer()
        self.sample_layer = SampleLayer(latent_dist='normal')
        self.loss_weight = loss_weight
        self.emb_size = emb_size

        self.n_samples_train = n_samples_train  # 100
        self.dim_out = dim_out  # 10
        self.dim_latent = dim_latent  # 50
        self.dim_in_features = dim_in_features  # 43
        self.dim_in_past = dim_in_past  # 165

        # self.model, self.model_delta = self._build_model()
        self.model = self._build_model()

    def _build_model(self):

        ### Inputs ###
        input_past = keras.Input(shape=(20, self.dim_in_past), name="input_past")
        input_std = keras.Input(shape=(self.dim_in_past,), name="input_std")
        input_all = keras.Input(shape=(self.dim_in_features,), name="input_all")
        input_weekday = keras.Input(shape=(1,), name="input_weekday")
        bs = self.shape_layer(input_all)

        ### Embeddings of week day information ###
        emb = layers.Embedding(8, self.emb_size)(input_weekday)
        emb = layers.Flatten()(emb)

        ##### Time series forecast part ####
        tsp = layers.Dense(512, activation='elu')(input_past)
        # tsp = layers.Dense(64, activation = 'elu')(tsp)
        tsp = layers.Dense(128, activation='elu')(tsp)  # added
        tsp = layers.Dense(32, activation='elu')(tsp)  # added
        tsp = layers.Dense(1, activation='linear')(tsp)
        # (, 20, 1)
        tsp = layers.Flatten()(tsp)
        # (, 20)

        ##### Conditional noise part ####
        delta = layers.Dense(512, activation='elu')(input_std)
        delta = layers.Dense(256, activation='elu')(delta)
        delta = layers.Dense(self.dim_latent, activation='exponential')(delta)
        # (, dim_latent)
        delta_z = layers.Reshape((1, self.dim_latent))(delta)
        # (, 1, dim_latent)

        # Convert dimensions to tensors
        n_samples = tf.constant(self.n_samples_train)
        d_latent = tf.constant(self.dim_latent)
        # Use the custom sampling layer
        epsilon = self.sample_layer([bs, n_samples, d_latent])
        # (, n_samples, dim_latent)
        z = layers.Multiply(name="scaled_noise")([delta_z, epsilon])
        # (, n_samples, dim_latent)

        ##### Weights part ####
        features_in = layers.Concatenate(axis=1, name="features_with_emb")([input_all, emb])
        context = layers.Concatenate(axis=1, name="context")([features_in, tsp])
        # shape: (batch, dim_in_features + emb_size + 20)

        ##### Engression-style latent predictor state z0 #####
        z0 = layers.Dense(512, activation='elu', name="z0_dense_1")(context)
        z0 = layers.Dense(256, activation='elu', name="z0_dense_2")(z0)
        z0 = layers.Dense(128, activation='elu', name="z0_dense_3")(z0)
        z0 = layers.Dense(self.dim_latent, activation='linear', name="z0_latent")(z0)
        # shape: (batch, dim_latent)

        z0_rep = layers.RepeatVector(self.n_samples_train, name="z0_repeat")(z0)
        # shape: (batch, n_samples, dim_latent)

        ##### Optional stochastic latent transform #####
        eta = layers.Dense(self.dim_latent, activation='linear', name="eta_latent")(z)
        # shape: (batch, n_samples, dim_latent)

        ##### Pre-additive latent perturbation #####
        latent = layers.Add(name="latent_perturbed")([z0_rep, eta])
        # shape: (batch, n_samples, dim_latent)

        ##### Nonlinear decoder g(z) #####
        h = layers.Dense(512, activation='elu', name="decoder_dense_1")(latent)
        h = layers.Dense(256, activation='elu', name="decoder_dense_2")(h)
        h = layers.Dense(128, activation='elu', name="decoder_dense_3")(h)
        y_base = layers.Dense(self.dim_out, activation='linear', name="decoder_out")(h)
        # shape: (batch, n_samples, dim_out)

        ##### Deterministic skip from context #####
        #skip = layers.Dense(128, activation='elu', name="skip_dense_1")(context)
        #skip = layers.Dense(self.dim_out, activation='linear', name="skip_out")(skip)
        #skip = layers.RepeatVector(self.n_samples_train, name="skip_repeat")(skip)
        # shape: (batch, n_samples, dim_out)

        ##### Final output #####
        y_noise = y_base
        # shape: (batch, n_samples, dim_out)

        y = layers.Permute((2, 1), name="forecast_samples")(y_noise)
        # shape: (batch, dim_out, n_samples)

        model = Model(
            inputs=[input_past, input_std, input_all, input_weekday],
            outputs=y,
            name="cgm_engression_hybrid"
        )

        return model

    def fit(self, x, y, batch_size=64, epochs=300, verbose=0, callbacks=None,
            validation_split=0.0, validation_data=None, sample_weight=None, learningrate=0.01):

        if learningrate == 'decay':
            # Define an exponential learning rate decay schedule
            initial_learning_rate = 1e-3
            lr_schedule = ExponentialDecay(initial_learning_rate, decay_steps=3, decay_rate=0.8)
            # Compile the model with the learning rate schedule
            opt = Adam(learning_rate=lr_schedule)
        else:
            opt = Adam(learning_rate=learningrate)

        self.model.compile(loss=EnergyScore(weight=self.loss_weight), optimizer=opt)
        self.history = self.model.fit(x=x,
                                      y=y,
                                      batch_size=batch_size,
                                      epochs=epochs,
                                      verbose=verbose,
                                      callbacks=callbacks,
                                      validation_split=validation_split,
                                      validation_data=validation_data,
                                      shuffle=True,
                                      sample_weight=sample_weight)

        return self

    def predict(self, x_test, n_samples=1, verbose=0):
        repetitions = np.int32(np.ceil(n_samples / self.n_samples_train))
        S = np.tile(self.model.predict(x_test, verbose=verbose), (1, 1, repetitions))
        return S[:, :, 0:n_samples]

    def get_model(self):
        return self.model