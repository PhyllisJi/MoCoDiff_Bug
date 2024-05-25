# MoCoDiff_Bug

#LeNet-11-821

### The diff between layers

```
# W0525 05:58:23.551180  1942 gpu_resources.cc:119] Please NOTE: device: 0, GPU Compute Capability: 8.6, Driver API Version: 12.2, Runtime API Version: 11.8
# W0525 05:58:23.552209  1942 gpu_resources.cc:164] device: 0, cuDNN Version: 8.9.

relu2_output.npz 1.1175870895385742e-08
maxpool2_output.npz 1.1175870895385742e-08
conv1_output.npz 0.0
relu4_output.npz 1.0
fc2_output.npz 0.0001531541347503662
conv2_output.npz 0.0
relu3_output.npz 2.9802322387695312e-08
flatten_output.npz 1.1175870895385742e-08
output.npz 0.248759925365448
relu1_output.npz 0.0
fc1_output.npz 5.960464477539063e-08
maxpool1_output.npz 0.0
```

```
# W0525 05:54:59.668944   515 gpu_resources.cc:119] Please NOTE: device: 0, GPU Compute Capability: 7.5, Driver API Version: 12.2, Runtime API Version: 11.8
# W0525 05:54:59.670363   515 gpu_resources.cc:164] device: 0, cuDNN Version: 8.9.

relu2_output.npz 1.1175870895385742e-08
maxpool2_output.npz 1.1175870895385742e-08
conv1_output.npz 0.0
relu4_output.npz 0.0
fc2_output.npz 2.086162567138672e-07
conv2_output.npz 0.0
relu3_output.npz 2.9802322387695312e-08
flatten_output.npz 1.1175870895385742e-08
output.npz 0.0
relu1_output.npz 0.0
fc1_output.npz 5.960464477539063e-08
maxpool1_output.npz 0.0
```

### Steps to Reproduction

```bash
git clone -b paddle-issue#64591 https://github.com/PhyllisJi/MoCoDiff_Bug.git
cd MoCoDiff_Bug/
cd paddle_bug/
python ./layer_diff.py
```

```
The outputs are the Chebyshev distance for the last few layers
The output of each layer is stored in the corresponding folder layer_outputs
input.npz is the input used and we also provide the initialisation parameters.
```