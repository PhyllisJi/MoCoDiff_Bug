# MoCoDiff_Bug

#GoogleNet-8-93

### Grad diff

```
tail_fc.bias: 梯度数据一致
conv3_mutated.weight: 梯度数据一致
conv1_mutated.bias: 梯度数据一致
conv2_mutated.weight: 梯度数据一致
conv3_mutated.bias: 梯度数据一致
conv2_mutated.bias: 梯度数据一致
conv1_mutated.weight: 梯度数据不一致, 差值:768.0
tail_fc.weight: 梯度数据一致
```

### Steps to Reproduction

```bash
git clone -b paddle-issue#64722 https://github.com/PhyllisJi/MoCoDiff_Bug.git
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