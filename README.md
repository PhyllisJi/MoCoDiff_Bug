# MoCoDiff_Bug

#PointNet-1-1

### grad_diff between torch cpu and gpu

```
conv1_mutated.weight
0.17957306
conv1_mutated.bias
1.7881393e-07
tail_fc.weight
6.1035156e-05
tail_fc.bias
1.0430813e-07
```

### Steps to Reproduction

```bash
git clone -b torch-issue-conv1d https://github.com/PhyllisJi/MoCoDiff_Bug.git
cd MoCoDiff_Bug/
cd pytorch_bug/
python ./cmp_output_torch.py
```
