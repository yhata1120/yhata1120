#!/usr/bin/env python
import os
import pickle
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model, clone_model, model_from_json
from tensorflow.keras.layers import Dense, Input, Average, Lambda, Flatten, Reshape, BatchNormalization, Layer, Masking, LSTM
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from sklearn.model_selection import KFold

physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
        print('{} memory growth: {}'.format(device, tf.config.experimental.get_memory_growth(device)))
else:
    print("Not enough GPU hardware devices available")


def make_simple_nn(n_input, n_output, n_layer, n_node, activation_hidden="relu", activation_output="linear", weights=None, get_config=False):
    inputs = Input(n_input, name="input")
    kernel_initializer_dict = {"relu": "he_normal"}
    hidden_layers_list = [Dense(n_node, activation=activation_hidden, kernel_initializer=kernel_initializer_dict.get(activation_hidden, "glorot_normal"), name=f"hidden_{i + 1}") for i in range(n_layer)]
    x = inputs
    for layer in hidden_layers_list:
        x = layer(x)
    outputs = Dense(n_output, activation=activation_output, name="output")(x)
    model = Model(inputs, outputs)
    if weights is not None:
        model.load_weights(weights)
    if get_config:
        return model.get_config()
    else:
        return model


def get_padded_relative_coords(v, n_max, constant_values=0.0):
    v_sorted = v[np.argsort(np.sum(np.square(v), axis=1))][1::]
    return np.pad(v_sorted, [[0, n_max - len(v_sorted)], [0, 0]], mode="constant", constant_values=constant_values)


def make_simple_lstm(n_input, n_output, n_lstm_units, n_layer, n_node, mask_value=0.0, random_rotate=False, activation_hidden="relu", activation_output="linear", weights=None, get_config=False):
    inputs = Input(n_input, name="input")
    kernel_initializer_dict = {"relu": "he_normal"}
    hidden_layers_list = [Dense(n_node, activation=activation_hidden, kernel_initializer=kernel_initializer_dict.get(activation_hidden, "glorot_normal"), name=f"hidden_{i + 1}") for i in range(n_layer)]
    if random_rotate:
        x = Random3DPointRotate(name="random_rotate")(inputs)
    else:
        x = inputs
    x = Masking(mask_value=mask_value, name="masking")(x)
    x = LSTM(n_lstm_units, return_sequences=False, name="lstm")(x)
    for layer in hidden_layers_list:
        x = layer(x)
    outputs = Dense(n_output, activation=activation_output, name="output")(x)
    model = Model(inputs, outputs)
    if weights is not None:
        model.load_weights(weights)
    if get_config:
        return model.get_config()
    else:
        return model


def make_simple_lstm_dos(n_input, n_output, n_lstm_units, n_layer, n_node, mask_value=0.0, random_rotate=False, activation_hidden="relu", activation_output="linear", weights=None, get_config=False):
    inputs = Input(n_input, name="input")
    kernel_initializer_dict = {"relu": "he_normal"}
    hidden_layers_list = [Dense(n_node, activation=activation_hidden, kernel_initializer=kernel_initializer_dict.get(activation_hidden, "glorot_normal"), name=f"hidden_{i + 1}") for i in range(n_layer)]
    if random_rotate:
        x = Random3DPointRotate(name="random_rotate")(inputs)
    else:
        x = inputs
    x = Masking(mask_value=mask_value, name="masking")(x)
    x = LSTM(n_lstm_units, return_sequences=False, name="lstm")(x)
    for layer in hidden_layers_list:
        x = layer(x)
    x = Dense(np.prod(n_output), activation=activation_output, name="output")(x)
    outputs = Reshape(n_output, name="reshape")(x)
    model = Model(inputs, outputs)
    if weights is not None:
        model.load_weights(weights)
    if get_config:
        return model.get_config()
    else:
        return model


def kfold_cv(model_config, X_trainval, y_trainval, optimizer="adam", loss="mse", n_splits=5, batch_size=32, epochs=1000, es_patience=100, save_dir=".", prefix="kfold_cv", callbacks=list(), eval_func=None, random_state=0):
    val_scores = []
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    for fold, (train_indices, val_indices) in enumerate(kf.split(X_trainval)):
        # Prepare dataset
        X_train, X_val = X_trainval[train_indices], X_trainval[val_indices]
        y_train, y_val = y_trainval[train_indices], y_trainval[val_indices]
        # Create model from model_config
        if isinstance(model_config, dict):
            model = Model.from_config(model_config)
            model.compile(optimizer=optimizer, loss=loss)
        elif isinstance(model_config, str):
            if os.path.isfile(model_config):
                with open(model_config, "rt") as f:
                    json_string = f.read()
            else:
                json_string = model_config
            model = model_from_json(json_string)
            model.compile(optimizer=optimizer, loss=loss)
        elif callable(model_config):
            model = model_config()
        else:
            raise RuntimeError(f"unknown type of model_config: {type(model_config)}")
        model._name = f"{prefix}_{model.name}"
        # Save model architecture
        if fold == 0:
            with open(os.path.join(save_dir, f"{prefix}_architecture.json"), 'wt') as f:
                f.write(model.to_json())
        # Prepare callbacks
        rlr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.1,
            patience=10,
            verbose=0,
            min_delta=1e-4
        )
        ckp = ModelCheckpoint(
            os.path.join(save_dir, f'{prefix}_cv{fold}_weight.hdf5'),
            monitor='val_loss',
            verbose=0,
            save_best_only=True,
            save_weights_only=True
        )
        es = EarlyStopping(
            monitor='val_loss',
            patience=es_patience,
            restore_best_weights=True,
            verbose=0
        )
        callbacks_default = [rlr, ckp, es]

        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks + callbacks_default,
            verbose=0
        )
        # Save learning history
        with open(os.path.join(save_dir, f'{prefix}_cv{fold}_history.pickle'), 'wb') as f:
            pickle.dump(history.history, f)
        # Evaluate model by validation data
        if eval_func is None:
            score = model.evaluate(X_val, y_val)
        else:
            y_val_pred = model.predict(X_val)
            score = eval_func(y_val, y_val_pred)
        print(f'fold {fold} score: {score}')
        val_scores.append(score)
        # Delete model and clear session
        del model
        K.clear_session()
    # Get average of validation score
    cv_score = np.mean(val_scores)
    print(f'CV score: {cv_score}')
    return cv_score


def make_ensemble_model(model_config, weights_list):
    if isinstance(model_config, dict):
        model_base = Model.from_config(model_config)
    elif isinstance(model_config, str):
        if os.path.isfile(model_config):
            with open(model_config, "r") as f:
                json_string = f.read()
        else:
            json_string = model_config
        model_base = model_from_json(json_string)
    model_list = list()
    for i, weights in enumerate(weights_list):
        model = clone_model(model_base)
        model.load_weights(weights)
        model._name = f"ensemble_{i+1}_{model.name}"
        for layer in model.layers:
            layer.trainable = False
            layer._name = f"ensemble_{i+1}_{layer.name}"
        model_list.append(model)
    emsemble_inputs_list = [Input(shape=inputs.get_shape().as_list()[1:], name=f"ensemble_input_{i + 1}") for i, inputs in enumerate(model.inputs)]
    outputs_list = [model(emsemble_inputs_list) for model in model_list]
    emsemble_outputs = Average(name="ensemble_average")(outputs_list)
    emsemble_model = Model(emsemble_inputs_list, emsemble_outputs, name="emsemble")
    return emsemble_model


class VAE():
    def __init__(self, input_dim=(28, 28), intermediate_dim=64, activation_hidden="relu", latent_dim=2, n_layer_encoder=2, n_layer_decoder=2, reconstruction_loss_func=binary_crossentropy, kl_lambda=1, optimizer='adam', trainable=True, vae_weights=None, **kwargs):
        self.input_dim = input_dim
        self.intermediate_dim = intermediate_dim
        self.n_layer_encoder = n_layer_encoder
        self.n_layer_decoder = n_layer_decoder
        self.activation_hidden = activation_hidden
        self.latent_dim = latent_dim
        self.kl_lambda = kl_lambda
        self.optimizer = optimizer
        self.trainable = trainable
        self.vae_weights = vae_weights
        self.kernel_initializer_dict = {"relu": "he_normal"}
        self.reconstruction_loss_func = reconstruction_loss_func
        self.build_vae()
        # binary_crossentropy,mse(*args)

    def build_vae(self):
        encoder_inputs = Input(shape=self.input_dim, name="encoder_input")
        x = Flatten(name="flatten_input")(encoder_inputs)
        x = BatchNormalization(name="batchnormalization")(x)
        encoder_hidden_layers_list = [
            Dense(
                self.intermediate_dim,
                activation=self.activation_hidden,
                kernel_initializer=self.kernel_initializer_dict.get(self.activation_hidden, "glorot_normal"),
                name=f"encoder_hidden_{i + 1}"
            )
            for i in range(self.n_layer_encoder)
        ]
        for layer in encoder_hidden_layers_list:
            x = layer(x)
        z_mean = Dense(self.latent_dim, activation='linear', name='z_mean')(x)
        z_log_var = Dense(self.latent_dim, activation='linear', name='z_log_var')(x)

        z = Lambda(self.sampling, output_shape=(self.latent_dim,), name='z')([z_mean, z_log_var])

        # instantiate encoder model
        self.encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name='encoder')

        # build decoder model
        latent_inputs = Input(shape=(self.latent_dim,), name='z_sampling')
        y = latent_inputs
        decoder_hidden_layers_list = [
            Dense(
                self.intermediate_dim,
                activation=self.activation_hidden,
                kernel_initializer=self.kernel_initializer_dict.get(self.activation_hidden, "glorot_normal"),
                name=f"decoder_hidden_{i + 1}"
            )
            for i in range(self.n_layer_decoder)
        ]
        for layer in decoder_hidden_layers_list:
            y = layer(y)
        y = Dense(np.prod(self.input_dim), activation='sigmoid', name="decoder_output")(y)
        decoder_outputs = Reshape(self.input_dim, name="decoder_output_reshape")(y)

        # instantiate decoder model
        self.decoder = Model(latent_inputs, decoder_outputs, name='decoder')

        # instantiate VAE model
        decoder_outputs = self.decoder(self.encoder(encoder_inputs)[2])
        self.vae = Model(encoder_inputs, decoder_outputs, name='vae_mlp')
        if self.trainable:
            vae_loss = Lambda(self.get_vae_loss, name="vae_loss")(
                [
                    Lambda(self.get_reconstruction_loss, name="reconstruction_loss")([encoder_inputs, decoder_outputs]),
                    Lambda(self.get_kl_loss, name="kl_loss")([z_mean, z_log_var])
                ]
            )
            self.vae.add_loss(vae_loss)
            self.vae.compile(optimizer=self.optimizer)
        if self.vae_weights is not None:
            self.vae.load_weights(self.vae_weights)

    def get_vae_loss(self, args):
        reconstruction_loss, kl_loss = args
        return K.mean(reconstruction_loss + self.kl_lambda * kl_loss)

    def get_reconstruction_loss(self, args):
        reconstruction_loss = self.reconstruction_loss_func(*args)
        reconstruction_loss *= np.prod(self.input_dim)
        return reconstruction_loss

    def get_kl_loss(self, args):
        z_mean, z_log_var = args
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss)
        kl_loss *= -0.5
        return kl_loss

    def sampling(self, args):
        """Reparameterization trick by sampling from an isotropic unit Gaussian.
        # Arguments
            args (tensor): mean and log of variance of Q(z|X)
        # Returns
            z (tensor): sampled latent vector
        """
        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        # by default, random_normal has mean = 0 and std = 1.0
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon


def uniform_random_rotate():
    x0 = np.random.random()
    y1 = 2*np.pi*np.random.random()
    y2 = 2*np.pi*np.random.random()
    r1 = np.sqrt(1.0-x0)
    r2 = np.sqrt(x0)
    u0 = np.cos(y2)*r2
    u1 = np.sin(y1)*r1
    u2 = np.cos(y1)*r1
    u3 = np.sin(y2)*r2
    coefi = 2.0*u0*u0-1.0
    coefuu = 2.0
    coefe = 2.0*u0
    r = np.zeros(shape=(3, 3))
    r[0, 0] = coefi+coefuu*u1*u1
    r[1, 1] = coefi+coefuu*u2*u2
    r[2, 2] = coefi+coefuu*u3*u3

    r[1, 2] = coefuu*u2*u3-coefe*u1
    r[2, 0] = coefuu*u3*u1-coefe*u2
    r[0, 1] = coefuu*u1*u2-coefe*u3

    r[2, 1] = coefuu*u3*u2+coefe*u1
    r[0, 2] = coefuu*u1*u3+coefe*u2
    r[1, 0] = coefuu*u2*u1+coefe*u3
    return r


class Random3DPointRotate(Layer):
    def __init__(self, **kwargs):
        super(Random3DPointRotate, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Random3DPointRotate, self).build(input_shape)

    def call(self, x, training=None):
        if training:
            return tf.matmul(x, tf.convert_to_tensor(uniform_random_rotate().astype(np.float32)))
        else:
            return x

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)


class PoissonSampling(Layer):
    def __init__(self, mean_count, pre_normalize=True, post_normalize=True, only_training=True, **kwargs):
        super(PoissonSampling, self).__init__(**kwargs)
        self.mean_count = mean_count
        self.pre_normalize = pre_normalize
        self.post_normalize = post_normalize
        self.only_training = only_training

    def build(self, input_shape):
        super(PoissonSampling, self).build(input_shape)

    def call(self, x, training=None):
        if training or not self.only_training:
            if self.pre_normalize:
                x = x / tf.reduce_sum(x, [i for i in range(len(x.shape))][1:])
            x = tf.random.poisson(shape=[], lam=self.mean_count * x * np.prod(list(x.shape[1:])))
        if self.post_normalize:
            x = x / tf.reduce_max(x, [i for i in range(len(x.shape))][1:])
        return x

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)