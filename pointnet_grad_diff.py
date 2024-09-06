import copy
import numpy as np
import tensorflow as tf

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
tf.config.experimental.enable_op_determinism()

def pointnet(input_shape):
    input_tensor = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv1D(filters=64, kernel_size=7, padding="valid")(input_tensor)
    x = tf.keras.layers.Softmax()(x)
    x = tf.keras.layers.Activation(activation='softplus')(x)
    x = tf.keras.layers.MaxPool1D(padding='valid', strides=4)(x)
    x = tf.keras.layers.BatchNormalization(center=False, momentum=0.07244736627895476, epsilon=0.3847853359642447)(x)
    x = tf.keras.layers.Activation(activation='exponential')(x)
    x = tf.keras.layers.MaxPooling1D(padding='same', strides=6, data_format='channels_last')(x)
    x = tf.keras.layers.Softmax(axis=2)(x)
    x = tf.keras.layers.Activation(activation='hard_sigmoid')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(units=1, use_bias=False, activation='softplus', bias_constraint=None, bias_regularizer=None, kernel_regularizer=None, kernel_constraint=None, activity_regularizer=None, bias_initializer='identity', kernel_initializer='truncated_normal')(x)
    x = tf.keras.layers.LayerNormalization(center=True, axis=1, scale=True, beta_constraint=None, epsilon=0.7136230859666187, gamma_initializer='glorot_normal', beta_regularizer=None, gamma_constraint=None, beta_initializer='zeros', gamma_regularizer=None)(x)
    x = tf.keras.layers.Activation(activation='selu')(x)
    x = tf.keras.layers.Dense(units=8, use_bias=False, activation='selu', kernel_regularizer=None, bias_constraint=None, bias_regularizer=None, bias_initializer='ones', kernel_initializer='glorot_uniform', kernel_constraint=None, activity_regularizer=None)(x)
    tail_flatten = tf.keras.layers.Flatten()(x)
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
    model_g = pointnet(inp.shape[1:])
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
                print(chebyshev_distance(gradients_dic_c[name], gradients_dic_g[name]))
                flag = False
                return flag, 'Grad diff too big'
    for name in gradients_dic_g.keys():
        if name in gradients_dic_c.keys():
            if chebyshev_distance(gradients_dic_g[name], gradients_dic_c[name]) > 0.1:
                print(gradients_dic_c[name], gradients_dic_g[name])
                flag = False
                return flag, 'Grad diff too big'
    return flag, ''


data = np.load("./output_dict/grad_diff_input.npz")
inp = data['inp']
label = data['label']
print(train(inp, label))
