import os
import numpy as np


def chebyshev_distance(A: np.ndarray, B: np.ndarray):
    if A is None or B is None:
        return 0.0
    if A.shape != B.shape:
        return 100
    else:
        return float(np.max(np.abs(A - B)))


def write_file(file_path, txt):
    with open(file_path, "a", encoding="utf-8") as writing_file:
        writing_file.write(txt)


diff_file = "./grad_diff.txt"
grad_gpu = np.load('./grad-gpu.npz')
grad_cpu = np.load('./grad-cpu.npz')

keys_cpu = set(grad_cpu.keys())
keys_gpu = set(grad_gpu.keys())
if keys_cpu != keys_gpu:
    write_file(diff_file, "keys are different\n")
else:
    for key in keys_cpu:
        compare_result = chebyshev_distance(grad_cpu[key], grad_gpu[key])
        if compare_result == 100:
            print(f"{key}: shape inconsistency")
        elif compare_result == 0.0:
            if grad_cpu[key] is None:
                print(f"grad_pytorch[{key}] is none")
                write_file(diff_file, f"grad_pytorch[{key}] is none\n")
            elif grad_gpu[key] is None:
                print(f"grad_paddle[{key}] is none")
                write_file(diff_file, f"grad_paddle[{key}] is none\n")
            else:
                print(f"{key}: gradient consistency")
                no_use = 1
        else:
            print(f"{key}: gradient inconsistency, distance:{compare_result}")
            write_file(diff_file, f"{key}: gradient inconsistency, distance:{compare_result}\n"               
            write_file(diff_file, f"grad_cpu[{key}]: {grad_cpu[key]}\n")
            write_file(diff_file, f"grad_gpu[{key}]: {grad_gpu[key]}\n")
