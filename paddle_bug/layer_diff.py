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
    if A.shape != B.shape:
        return 100
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
    return positions, values


# res_path = "pytorch_gpu-paddle_gpu-LeNet"
res_path = "paddle-GoogleNet"
model = "GoogleNet"
level = "8"
order = "93"
library_1 = "paddle"
library_2 = "paddle"
comp_type = "same"

if comp_type == "same":
    gpu_code_path = f'./{res_path}/{model}-{level}-{order}/case/{library_1}_gpu/{model}-{level}-{order}_{library_1}_gpu_debug.py'
    cpu_code_path = f'./{res_path}/{model}-{level}-{order}/case/{library_1}_cpu/{model}-{level}-{order}_{library_1}_cpu_debug.py'

    gpu_result = subprocess.run(['python', gpu_code_path])
    cpu_result = subprocess.run(['python', cpu_code_path])

    cpu_path = f'./{res_path}/{model}-{level}-{order}/case/{library_1}_cpu/layer_outputs/'
    gpu_path = f'./{res_path}/{model}-{level}-{order}/case/{library_1}_gpu/layer_outputs/'
    npz_files = get_all_npz_files(cpu_path)
    print(len(npz_files))
    for i in range(len(npz_files)):
        cpu_output = np.load(cpu_path + npz_files[i])['arr_0']
        gpu_output = np.load(gpu_path + npz_files[i])['arr_0']
        output_diff = chebyshev_distance(cpu_output, gpu_output)
        print(npz_files[i], output_diff)
        positions, values = find_difference_positions(cpu_output, gpu_output, distance=output_diff)
        # print(positions, values)

else:
    code_path_1 = f'./{res_path}/{model}-{level}-{order}/case/{library_1}_gpu/{model}-{level}-{order}_{library_1}_gpu_debug.py'
    code_path_2 = f'./{res_path}/{model}-{level}-{order}/case/{library_2}_gpu/{model}-{level}-{order}_{library_2}_gpu_debug.py'

    try:
        result_1 = subprocess.run(['python', code_path_1], capture_output=True, text=True)
        print(f"Standard Output of {library_1}:\n", result_1.stdout)
        if result_1.returncode != 0:
            print(f"Standard Error of {library_1}:\n", result_1.stderr)
            print(f"Return Code of {library_1}:", result_1.returncode)
    except Exception as e:
        print(f"An error occurred with {library_1}:", str(e))
    # except subprocess.CalledProcessError as e:
    #     print(f"Standard Error of {library_1}:\n", e.stderr)
    #     print(f"Return Code of {library_1}:", e.returncode)

    try:
        result_2 = subprocess.run(['python', code_path_2], capture_output=True, text=True)
        print(f"Standard Output of {library_2}:\n", result_2.stdout)
        if result_2.returncode != 0:
            print(f"Standard Error of {library_2}:\n", result_2.stderr)
            print(f"Return Code of {library_2}:", result_2.returncode)
    except Exception as e:
        print(f"An error occurred with {library_2}:", str(e))
    # except subprocess.CalledProcessError as e:
    #     print(f"Standard Error of {library_2}:\n", e.stderr)
    #     print(f"Return Code of {library_2}:", e.returncode)

    output_path_1 = f'./{res_path}/{model}-{level}-{order}/case/{library_1}_gpu/layer_outputs/'
    output_path_2 = f'./{res_path}/{model}-{level}-{order}/case/{library_2}_gpu/layer_outputs/'
    npz_files = get_all_npz_files(output_path_1)
    print(len(npz_files))
    for i in range(len(npz_files)):
        output_1 = np.load(output_path_1 + npz_files[i])['arr_0']
        output_2 = np.load(output_path_2 + npz_files[i])['arr_0']
        output_diff = chebyshev_distance(output_1, output_2)
        print(npz_files[i], output_diff)
        positions, values = find_difference_positions(output_1, output_2, distance=output_diff)
        # print(positions)
        # if "fc2_output" in npz_files[i] or "relu4" in npz_files[i]:
        #     print(positions, values)
