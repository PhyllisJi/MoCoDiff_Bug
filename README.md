# MoCoDiff_Bug

#PointNet-20-238

### grad_diff between torch cpu and gpu

```
conv1_mutated.weight
0.054493383
conv1_mutated.bias
5.5010105e-08
# bn1-torch.nn.BatchNorm1d
bn1.weight
0.11135915
bn1.bias
0.09993172
conv2_mutated.weight
0.04085891
conv2_mutated.bias
0.00016388169
bn2.weight
0.052734636
bn2.bias
9.678304e-06
conv3_mutated.weight
0.04560852
conv3_mutated.bias
1.1920929e-06
bn3.weight
0.030974608
bn3.bias
0.029964829
conv4_mutated.weight
0.021974621
conv4_mutated.bias
1.7881393e-07
bn4.weight
0.026008025
bn4.bias
0.001900834
conv5_mutated.weight
0.016805578
conv5_mutated.bias
2.561137e-08
bn5.weight
0.017251626
bn5.bias
0.0034310077
linear1.weight
0.019508496
linear1.bias
1.5832484e-07
bn6.weight
0.00020761788
bn6.bias
0.013012487
tail_fc.weight
0.00012109056
tail_fc.bias
4.3287873e-06
```

### Steps to Reproduction

```bash
git clone -b torch-issue-batchnorm1d https://github.com/PhyllisJi/MoCoDiff_Bug.git
cd MoCoDiff_Bug/
cd pytorch_bug/
python ./cmp_output_torch.py
```
