import copy
import numpy as np
import tensorflow as tf


def lenet(input_shape):
    input_tensor = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Activation(activation="relu")(input_tensor)
    x = tf.keras.layers.MaxPool2D(pool_size=8, strides=3, padding='same')(x)
    x = tf.keras.layers.Activation(activation='sigmoid')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=5, strides=1, padding='same')(x)
    x = tf.keras.layers.Activation(activation='softplus')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=7, strides=8, padding='same', data_format='channels_last')(x)
    x = tf.keras.layers.Flatten(data_format='channels_first')(x)
    x = tf.keras.layers.Dense(units=2, activation='exponential', kernel_constraint=None, bias_regularizer=None, use_bias=False, bias_initializer='he_uniform', activity_regularizer=None, kernel_initializer='ones', bias_constraint=None, kernel_regularizer=None)(x)
    x = tf.keras.layers.Dense(units=7, activation='exponential', use_bias=True, kernel_initializer='glorot_uniform', kernel_constraint=None, bias_regularizer=None)(x)
    output_tensor = tf.keras.layers.Flatten(data_format='channels_first')(x)
    tail_flatten = tf.keras.layers.Flatten()(output_tensor)
    tail_fc = tf.keras.layers.Dense(units=10)(tail_flatten)
    model = tf.keras.models.Model(inputs=input_tensor, outputs=tail_fc)
    return model


def chebyshev_distance(A: np.ndarray, B: np.ndarray):
    if A is None or B is None:
        return 0.0
    if A.shape != B.shape:
        return 9999999
    else:
        return float(np.max(np.abs(A - B)))


def train(inp, label):
    flag = True
    label = tf.convert_to_tensor(label)
    model_g = lenet(inp.shape[1:])
    model_g.load_weights("./output_dict/grad_diff_initial_weights.h5")
    with tf.device('GPU'):
        with tf.GradientTape() as tape:
            output_g = model_g(inp)
            loss_g = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(label, output_g)
        gradients_g = tape.gradient(loss_g, model_g.trainable_variables)
        gradients_dic_g = {}
        for var, gradient in zip(model_g.trainable_variables, gradients_g):
            if gradient != None:
                gradients_dic_g.setdefault(var.name.replace('/', '.')[:-2], gradient)

    model_c = copy.deepcopy(model_g)
    model_c.load_weights("./output_dict/grad_diff_initial_weights.h5")
    with tf.device('CPU'):
        with tf.GradientTape() as tape:
            output_c = model_c(inp)
            loss_c = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(label, output_c)
        gradients_c = tape.gradient(loss_c, model_c.trainable_variables)
        gradients_dic_c = {}
        for var, gradient in zip(model_c.trainable_variables, gradients_c):
            if gradient != None:
                gradients_dic_c.setdefault(var.name.replace('/', '.')[:-2], gradient)
    if chebyshev_distance(output_c.numpy(), output_g.numpy()) > 1.0:
        flag = False
        return flag, 'Output diff too big'
    if abs(loss_c - loss_g) > 0.1:
        flag = False
        return flag, 'Loss diff too big'
    for name in gradients_dic_c.keys():
        if name in gradients_dic_g.keys():
            if chebyshev_distance(gradients_dic_c[name], gradients_dic_g[name]) > 0.1:
                flag = False
                return flag, 'Grad diff too big'
    for name in gradients_dic_g.keys():
        if name in gradients_dic_c.keys():
            if chebyshev_distance(gradients_dic_g[name], gradients_dic_c[name]) > 0.1:
                flag = False
                return flag, 'Grad diff too big'
    return flag, ''


data = np.load("./output_dict/grad_diff_input.npz")
inp = data['inp']
label = data['label']
print(train(inp, label))