# MoCoDiff_Bug

#LeNet-10-300

### The diff between layers

```
# cuda 11.6
# W0522 14:31:24.082360  3337 gpu_resources.cc:119] Please NOTE: device: 0, GPU Compute Capability: 7.5, Driver API Version: 11.6, Runtime API Version: 11.8
# W0522 14:31:24.083253  3337 gpu_resources.cc:164] device: 0, cuDNN Version: 8.9.
# W0522 14:31:24.083271  3337 gpu_resources.cc:196] WARNING: device: 0. The installed 	Paddle is compiled with CUDA 11.8, but CUDA runtime version in your machine is 11.6, which may cause serious incompatible bug. Please recompile or reinstall Paddle with compatible CUDA version.

relu2_output.npz 0.0
maxpool2_output.npz 0.0
fc2_output.npz 1.6689300537109375e-06
conv2_output.npz 0.0
relu3_output.npz 0.0
flatten_output.npz 0.0
output.npz 1.430511474609375e-06
fc1_output.npz 1.1920928955078125e-06
```

```
# cuda 11.8
# W0522 14:19:26.351598   496 gpu_resources.cc:119] Please NOTE: device: 0, GPU Compute Capability: 8.6, Driver API Version: 12.2, Runtime API Version: 11.8
# W0522 14:19:26.352345   496 gpu_resources.cc:164] device: 0, cuDNN Version: 8.9.

conv2_output.npz 0.0
relu2_output.npz 0.0
maxpool2_output.npz 0.0
flatten_output.npz 0.0
fc1_output.npz 0.0008640289306640625
relu3_output.npz 1.0
fc2_output.npz 0.17127180099487305
output.npz 0.3886955976486206
```

### Steps to Reproduction

```bash
git clone -b paddle-issue#64537 https://github.com/PhyllisJi/MoCoDiff_Bug.git
cd MoCoDiff_Bug/
cd paddle_bug/
python ./layer_diff.py
```

```
The outputs are the Chebyshev distance for the last few layers
The output of each layer is stored in the corresponding folder layer_outputs
input.npz is the input used and we also provide the initialisation parameters.
```