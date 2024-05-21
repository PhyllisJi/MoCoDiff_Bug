import tensorflow as tf
import numpy as np
import os


os.environ['CUDA_VISIBLE_DEVICES'] = ''
output_dir = "./tensorflow-LeNet/LeNet-12-654/case/tensorflow_cpu/layer_outputs/"
os.makedirs(output_dir, exist_ok=True)
def construct_sub_model(tf_model, tf_input, layer_index):
    layer_output = tf_model.layers[layer_index].output
    layer_name = tf_model.layers[layer_index].name
    sub_model = tf.keras.Model(inputs = tf_model.input, outputs = layer_output)
    last_layer_output = sub_model(tf_input)
    output_filename = "./tensorflow-LeNet/LeNet-12-654/case/tensorflow_cpu/layer_outputs/" + layer_name + ".npz"
    np.savez(output_filename, arr_0 = last_layer_output)

def Model_VlysjQxB81qtaIXsA_VkCXmPGmE7aDNP(input):
    input = tf.keras.Input(shape=input)
    _zeropadding_input = tf.keras.layers.ZeroPadding2D(padding=((0, 0), (0, 0)))(input)
    conv1_output = tf.keras.layers.Conv2DTranspose(filters=6, kernel_size=(5, 5), strides=(1, 1), padding="valid", output_padding=(0, 0), data_format="channels_last", dilation_rate=(1, 1), use_bias=True, name="conv1_mutated")(input)
    relu1_output = tf.nn.relu(conv1_output)
    _zeropadding_relu1_output = tf.keras.layers.ZeroPadding2D(padding=((0, 0), (0, 0)))(relu1_output)
    maxpool1_output = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid", data_format="channels_last", name="pool1")(_zeropadding_relu1_output)
    _zeropadding_maxpool1_output = tf.keras.layers.ZeroPadding2D(padding=((0, 0), (0, 0)))(maxpool1_output)
    conv2_output = tf.keras.layers.Conv2D(filters=16, kernel_size=(6, 8), strides=(1, 1), padding="valid", data_format="channels_last", dilation_rate=(1, 1), groups=1, use_bias=True, name="conv2_mutated")(_zeropadding_maxpool1_output)
    relu2_output = tf.math.softsign(conv2_output)
    _zeropadding_relu2_output = tf.keras.layers.ZeroPadding2D(padding=((0, 0), (0, 0)))(relu2_output)
    maxpool2_output = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid", data_format="channels_last", name="pool2")(_zeropadding_relu2_output)
    output_transpose = [(0), (0, 1), (0, 2, 1), (0, 3, 1, 2), (0, 4, 1, 2, 3)]
    maxpool2_output = tf.transpose(maxpool2_output, list(output_transpose[(len(maxpool2_output.shape) - 1)]))
    flatten_output = tf.keras.layers.Flatten(data_format="channels_last", name="flatten")(maxpool2_output)
    fc1_output = tf.keras.layers.Dense(units=120, use_bias=True, name="linear1")(flatten_output)
    relu3_output = tf.keras.layers.ThresholdedReLU(theta=0.1, name="relu3_mutated")(fc1_output)
    fc2_output = tf.keras.layers.Dense(units=84, use_bias=True, name="linear2_mutated")(relu3_output)
    relu4_output = tf.math.erf(fc2_output)
    fc3_output = tf.keras.layers.Dense(units=10, use_bias=True, name="linear3_mutated")(relu4_output)
    output_transpose = [(0), (0, 1), (0, 2, 1), (0, 3, 1, 2), (0, 4, 1, 2, 3)]
    fc3_output = tf.transpose(fc3_output, list(output_transpose[(len(fc3_output.shape) - 1)]))
    tail_flatten_output = tf.keras.layers.Flatten(data_format="channels_last", name="tail_flatten")(fc3_output)
    tail_fc_output = tf.keras.layers.Dense(units=10, use_bias=True, name="tail_fc")(tail_flatten_output)

    tail_fc_output = tail_fc_output
    model = tf.keras.models.Model(inputs=input, outputs=tail_fc_output)
    return model


def go():
    with tf.device('/CPU:0'):
        try:
            shape = [1, 1, 28, 28]
            _numpy = np.random.random(shape).astype(np.float32)
            tf_input = tf.convert_to_tensor(_numpy.transpose(0, 2, 3, 1), dtype=tf.float32)
            tf_model = Model_VlysjQxB81qtaIXsA_VkCXmPGmE7aDNP(tf_input.shape[1:])
            tf_output = tf_model(tf_input)
            flag = True
        except Exception:
            flag = False
        return flag


def initialize(model):
    module_dir = os.path.dirname(__file__)
    gradient_transpose = [(0,), (1, 0), (2, 1, 0), (2, 3, 1, 0), (2, 3, 4, 1, 0)]
    for layer in model.layers:
        matrix_path = module_dir + '/../initializer/' + layer.name
        if hasattr(layer, 'kernel_initializer'):
            weight_init_path = matrix_path + '/weight.npz'
            weight_init = np.load(weight_init_path)
            weight_init = weight_init['matrix']
            tf_weight = tf.convert_to_tensor(weight_init, dtype=tf.float32)
            tf_weight = tf.transpose(tf_weight, gradient_transpose[len(tf_weight.shape) - 1])
            layer.kernel.assign(tf.keras.initializers.Constant(tf_weight)(layer.kernel.shape))
        if hasattr(layer, 'bias_initializer') and layer.use_bias:
            bias_init_path = matrix_path + '/bias.npz'
            bias_init = np.load(bias_init_path)
            bias_init = bias_init['matrix']
            tf_bias = tf.convert_to_tensor(bias_init, dtype=tf.float32)
            tf_bias = tf.transpose(tf_bias, gradient_transpose[len(tf_bias.shape) - 1])
            layer.bias.assign(tf.keras.initializers.Constant(tf_bias)(layer.bias.shape))

def train(inp, label):
    with tf.device('/CPU:0'):
        shape = inp.shape
        tf_input = tf.convert_to_tensor(inp.transpose(0, 2, 3, 1), dtype=tf.float32)
        tf_model = Model_VlysjQxB81qtaIXsA_VkCXmPGmE7aDNP(tf_input.shape[1:])

        initialize(tf_model)
        layer_index = len(tf_model.layers) - 4
        construct_sub_model(tf_model, tf_input, layer_index)
        construct_sub_model(tf_model, tf_input, layer_index-1)
        tf_output = tf_model(tf_input)
        output_transpose = [(0), (0, 1), (0, 2, 1), (0, 3, 1, 2), (0, 4, 1, 2, 3)]
        tf_output_trans = tf.transpose(tf_output, list(output_transpose[(len(tf_output.shape) - 1)])).numpy()
        
        tf_targets = tf.convert_to_tensor(label)
        with tf.GradientTape() as tape:
            tf_predictions = tf_model(tf_input)
            tf_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(tf_targets, tf_predictions)
        tf_gradients = tape.gradient(tf_loss, tf_model.trainable_variables)
        tf_gradients_dic = {}
        for var, gradient in zip(tf_model.trainable_variables, tf_gradients):
            gradient_transpose = [(0, ), (1, 0), (2, 0, 1), (3, 2, 0, 1), (4, 3, 0, 1, 2)]
            tf_gradient = tf.transpose(gradient, list(gradient_transpose[len(gradient.shape) - 1])).numpy()
            tf_gradients_dic.setdefault(var.name.replace('/', '.')[:-2], tf_gradient)
        
        return tf_gradients_dic, float(tf_loss.numpy()), tf_output_trans
inp =np.load("./tensorflow-LeNet/LeNet-12-654/case/input.npz")
gradients, loss, output = train(inp['inp'], inp['label'])
print(output)
