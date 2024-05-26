# MoCoDiff_Bug

#LeNet-11-302

### The diff between layers

```
fc2_output.npz 0.00019089877605438232
relu4_output.npz 2942823.25
output.npz 717287.0
```

### Grad diff

```
tail_fc.bias: 梯度数据不一致, 差值:0.004240369889885187
conv1_mutated.bias: 梯度数据不一致, 差值:38971648.0
linear1_mutated.bias: 梯度数据不一致, 差值:154666512.0
linear2.weight: 梯度数据不一致, 差值:271438272.0
linear1_mutated.weight: 梯度数据不一致, 差值:41292152.0
conv1_mutated.weight: 梯度数据不一致, 差值:109647968.0
tail_fc.weight: 梯度数据不一致, 差值:2942.853759765625
conv2_mutated.bias: 梯度数据不一致, 差值:348457472.0
conv2_mutated.weight: 梯度数据不一致, 差值:223159072.0
linear2.bias: 梯度数据不一致, 差值:682456960.0
```

### Steps to Reproduction

```bash
git clone -b paddle-issue#64606 https://github.com/PhyllisJi/MoCoDiff_Bug.git
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