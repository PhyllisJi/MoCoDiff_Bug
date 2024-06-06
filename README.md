# MoCoDiff_Bug

#MobileNet-38-384

### The diff between layers

```
Standard Output of pytorch:
 tail_flatten_output:  tensor([[ 7.9025e+10, -3.6124e+10, -1.1924e+11,  1.9557e+11,  3.4007e+10,
         -5.6296e+09,  7.8256e+10,  7.9971e+10]], device='cuda:0',
       grad_fn=<ViewBackward0>)

Standard Output of paddle:
 tail_flatten_output:  Tensor(shape=[1, 8], dtype=float32, place=Place(gpu:0), stop_gradient=False,
       [[ 79025455104. , -36124397568. , -119238516736.,  195569451008.,
          34007255040. , -5629640704.  ,  78255857664. ,  79970967552. ]])

8
conv5_output.npz 0.0
relu4_output.npz 0.0
conv19_output.npz 0.0
relu18_output.npz 0.0
conv20_output.npz 0.0
tail_flatten_output.npz 0.0
tail_fc_output.npz 4096.0
output.npz 4096.0
```

### Steps to Reproduction

```bash
git clone -b paddle-issue#64976 https://github.com/PhyllisJi/MoCoDiff_Bug.git
cd MoCoDiff_Bug/
cd paddle_bug/
python ./layer_diff.py
python ./grad_diff.py
```

```
The outputs are the Chebyshev distance for the last few layers
The output of each layer is stored in the corresponding folder layer_outputs
input.npz is the input used and we also provide the initialisation parameters.
```