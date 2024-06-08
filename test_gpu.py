import mindspore
import numpy as np
# import torch

mindspore.context.set_context(device_target='GPU')

inp = np.load('/mnt/test_inp.npz', allow_pickle=True)['inp']
print(inp)

ms_inp = mindspore.Tensor(inp.astype(np.float32))

mindspore_layer = mindspore.ops.reciprocal(ms_inp)

print(mindspore_layer)