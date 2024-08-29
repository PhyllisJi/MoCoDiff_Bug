import h5py
import tensorflow as tf
import numpy as np

tf.random.set_seed(42)

def chebyshev_distance(A: np.ndarray, B: np.ndarray):
    if A is None or B is None:
        return 0.0
    if A.shape != B.shape:
        return 9999999
    else:
        return float(np.max(np.abs(A - B)))

h5_file_path = "./output_diff_initial_weights.h5"
npz_path = "./output_diff_input.npz"
conv1d_layer = tf.keras.layers.Conv1D(filters=64, kernel_size=7, padding="valid")
layer_name = 'conv1d_2351'

data = np.load(npz_path)
inp = data['inp']
input_shape = inp.shape[1:]

with h5py.File(h5_file_path, 'r') as h5_file:
    weights = h5_file[f'{layer_name}/{layer_name}/kernel:0'][:]
    biases = h5_file[f'{layer_name}/{layer_name}/bias:0'][:]
    

conv1d_layer.build(input_shape)
conv1d_layer.set_weights([weights, biases])


with tf.device('/CPU:0'):
    x_cpu = tf.constant(inp, dtype=tf.float32)
    output_cpu = conv1d_layer(x_cpu)


if tf.config.list_physical_devices('GPU'):
    with tf.device('/GPU:0'):
        x_gpu = tf.constant(inp, dtype=tf.float32)
        output_gpu = conv1d_layer(x_gpu)

else:
    print("GPU not available.")
    
output_diff = chebyshev_distance(output_cpu.numpy(), output_gpu.numpy())
print(output_diff)
