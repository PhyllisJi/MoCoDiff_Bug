import os
import numpy as np
import subprocess

def get_all_npz_files(directory):
    npz_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.npz'):
                npz_files.append(os.path.join(file))
    return npz_files

def chebyshev_distance(A: np.ndarray, B: np.ndarray):
    if A is None or B is None:
        return 0.0
    # if A.shape != B.shape:
    #     return 100
    else:
        return float(np.max(np.abs(A - B)))

def is_all_zero_in_npz(file_path):
    with np.load(file_path) as data:
        for key in data:
            if not np.all(data[key] == 0):
                return False
        return True

def find_difference_positions(array1, array2, distance=1):
    diff = np.abs(array1 - array2)
    positions = np.argwhere(diff == distance)
    values = [(array1[tuple(pos)], array2[tuple(pos)]) for pos in positions]
    return positions, values,

# res_path = "pytorch_gpu-paddle_gpu-LeNet"
model = "LeNet"
level = "12"
order = "654"
library_1 = "tensorflow"
library_2 = "paddle"
comp_type = "same"

if comp_type == "same":
    gpu_code_path = f'./{model}-{level}-{order}/case/{library_1}_gpu/{model}-{level}-{order}_{library_1}_gpu_debug.py'
    cpu_code_path = f'./{model}-{level}-{order}/case/{library_1}_cpu/{model}-{level}-{order}_{library_1}_cpu_debug.py'
    
    gpu_result = subprocess.run(['python', gpu_code_path])
    cpu_result = subprocess.run(['python', cpu_code_path])
    
    cpu_path =f'./{model}-{level}-{order}/case/{library_1}_cpu/layer_outputs/'
    gpu_path =f'./{model}-{level}-{order}/case/{library_1}_gpu/layer_outputs/'
    npz_files = get_all_npz_files(cpu_path)
    print(len(npz_files))
    for i in range(len(npz_files)):
        cpu_output = np.load(cpu_path + npz_files[i])['arr_0']
        gpu_output = np.load(gpu_path + npz_files[i])['arr_0']
        output_diff = chebyshev_distance(cpu_output, gpu_output)
        print(npz_files[i], output_diff)
        positions, values = find_difference_positions(cpu_output, gpu_output, distance=output_diff)
        # print(positions, values)



    


