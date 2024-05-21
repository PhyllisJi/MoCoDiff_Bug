# MoCoDiff_Bug


#LeNet-12-654

### The diff between layers
```
linear3_mutated.npz 0.048886626958847046
linear2_mutated.npz 0.01712418906390667
linear1.npz 0.0003943890333175659
tf.compat.v1.transpose.npz 0.0004679672420024872
relu3_mutated.npz 0.10010143369436264
pool2.npz 0.0004679672420024872
flatten.npz 0.0004679672420024872
tf.math.erf.npz 0.01932205818593502
```

### Steps to Reproduction
```
git clone -b tf-issue#67829 https://github.com/PhyllisJi/MoCoDiff_Bug.git
cd MoCoDiff_Bug/
cd tensorflow_bug/
python ./layer_diff.py
```
```
The outputs are the Chebyshev distance for the last few layers
The output of each layer is stored in the corresponding folder layer_outputs
input.npz is the input used and we also provide the initialisation parameters.
```
