import numpy as np
import torch
import torch.nn.functional as F
"""
The commented out code is used to load our inputs and outputs, 
and a direct comparison shows that only torch-gpu differs significantly from torch-cpu, mindspore-gpu, and mindspor-cpu.
"""
# m_gpu_o = np.load("./mindspore_gpu_output.npz")
# t_gpu_o = np.load("./pytorch_gpu_output.npz")

# m_cpu_o = np.load("./mindspore_cpu_output.npz")
# t_cpu_o = np.load("./pytorch_cpu_output.npz")


# lst = [m_gpu_o, t_gpu_o, m_cpu_o, t_cpu_o]
# name = ['m_gpu', 't_gpu', 'm_cpu', 't_cpu']
# for i in range(0, 4):
#     for j in range(i+1, 4):
#         print(name[i].upper() + "-" + name[j].upper())
#         different_positions = np.where(lst[i]['relu3_output'] != lst[j]['relu3_output'])
#         print("position:", different_positions)
        
#         for pos in zip(*different_positions):
#             v1 = lst[i]['fc1_output'][pos]
#             v2 = lst[j]['fc1_output'][pos]
#             print(f"Specific values for locations {pos} where discrepancies exist in the input of round: {name[i].upper()}={v1}, {name[j].upper()}={v2}")

#             v3 = lst[i]['relu3_output'][pos]
#             v4 = lst[j]['relu3_output'][pos]
#             print(f"Specific values for locations {pos} where discrepancies exist in the output of round: {name[i].upper()}={v3}, {name[j].upper()}={v4}")


# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 读取 NumPy 数组
cpu_input = np.load("./pytorch_cpu_output.npz")['fc1_output']
gpu_input = np.load("./pytorch_gpu_output.npz")['fc1_output']

cpu_input_tensor = torch.from_numpy(cpu_input)
gpu_input_tensor = torch.from_numpy(gpu_input).to(device)

gpu_output = torch.reciprocal(gpu_input_tensor)
#np.savez("./gpu_output.npz",gpu_output.cpu().detach().numpy())
gpu_output = gpu_output.to('cpu')

cpu_output = torch.reciprocal(cpu_input_tensor)
#np.savez("./cpu_output.npz",cpu_output.cpu().detach().numpy())

# 使用切比雪夫距离比较
output_chebyshev_distance = torch.max(torch.abs(cpu_output - gpu_output))
input_chebyshev_distance = torch.max(torch.abs(cpu_input_tensor - torch.from_numpy(gpu_input)))
print(f"Chebyshev distance between inputs: {input_chebyshev_distance.item()}")
print(f"Chebyshev distance between outputs: {output_chebyshev_distance.item()}")
