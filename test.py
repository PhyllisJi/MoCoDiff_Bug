# import torch
# import numpy as np
# input = np.load("./cpu_input.npz")['arr_0']
# print(input.shape)
# output = torch.reciprocal(torch.from_numpy(input))
# np.savez("./cpu_output.npz",output.cpu().detach().numpy())
import numpy as np
import torch
import torch.nn.functional as F

# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 读取 NumPy 数组
cpu_input = np.load("./cpu_input.npz")['arr_0']
gpu_input = np.load("./gpu_input.npz")['arr_0']

cpu_input_tensor = torch.from_numpy(cpu_input)
gpu_input_tensor = torch.from_numpy(gpu_input).to(device)

gpu_output = torch.reciprocal(gpu_input_tensor)
np.savez("./gpu_output.npz",gpu_output.cpu().detach().numpy())
gpu_output = gpu_output.to('cpu')

cpu_output = torch.reciprocal(cpu_input_tensor)
np.savez("./cpu_output.npz",cpu_output.cpu().detach().numpy())

# 使用切比雪夫距离比较
output_chebyshev_distance = torch.max(torch.abs(cpu_output - gpu_output))
input_chebyshev_distance = torch.max(torch.abs(cpu_input_tensor - torch.from_numpy(gpu_input)))
print(f"Chebyshev distance between inputs: {input_chebyshev_distance.item()}")
print(f"Chebyshev distance between outputs: {output_chebyshev_distance.item()}")
