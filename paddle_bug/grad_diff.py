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


def read_grad(model, path):
    diff_file = f"{model}/{path}/grad_diff.txt"
    # with open(diff_file, "w", encoding="utf-8") as f:
    #     f.write(f"{path}\n")
    # with open(f"{model}/{path}/case/case_info.txt", "r") as f:
    #     for line in f:
    #         if "mutate info:" in line:
    #             print(line.strip())
    #             write_file(diff_file, line + "\n")

    grad_gpu = np.load(f'{model}/{path}/case/paddle_gpu/grad-new.npz')
    grad_cpu = np.load(f'{model}/{path}/case/paddle_cpu/grad-new.npz')

    # output_pytorch = np.load(f'{model}/{path}/case/pytorch_gpu/output.npz')
    # output_pytorch = output_pytorch['output']
    # output_paddle = np.load(f'{model}/{path}/case/pytorch_cpu/output.npz')
    # output_paddle = output_paddle['output']

    # output_diff = chebyshev_distance(output_pytorch, output_paddle)
    # print(f"output_diff: {output_diff}")
    # write_file(diff_file, f"output_diff: {output_diff}\n\n")

    keys_cpu = set(grad_cpu.keys())
    keys_gpu = set(grad_gpu.keys())
    if keys_cpu != keys_gpu:
        print("警告：CPU和GPU梯度数据的键不匹配。")
        write_file(diff_file, "警告：CPU和GPU梯度数据的键不匹配。\n")
    else:
        print("键匹配，开始比较值...")
        for key in keys_cpu:
            compare_result = chebyshev_distance(grad_cpu[key], grad_gpu[key])
            if compare_result == 100:
                print(f"{key}: 形状不一致")
                # write_file(diff_file, f"{key}: 形状不一致\n")
            elif compare_result == 0.0:
                if grad_cpu[key] is None:
                    print(f"grad_paddle[{key}] is none")
                    write_file(diff_file, f"grad_pytorch[{key}] is none\n")
                elif grad_gpu[key] is None:
                    print(f"grad_paddle[{key}] is none")
                    write_file(diff_file, f"grad_paddle[{key}] is none\n")
                else:
                    print(f"{key}: 梯度数据一致")
                    no_use = 1
            else:
                print(f"{key}: 梯度数据不一致, 差值:{compare_result}")
                write_file(diff_file, f"{key}: 梯度数据不一致, 差值:{compare_result}\n")
                # print(f"grad_cpu[{key}]: {grad_cpu[key]}")
                # print(f"grad_gpu[{key}]: {grad_gpu[key]}")
                write_file(diff_file, f"grad_cpu[{key}]: {grad_cpu[key]}\n")
                write_file(diff_file, f"grad_gpu[{key}]: {grad_gpu[key]}\n")


if __name__ == "__main__":
    # root_dir = "paddle-ResNet18"
    # directories = []
    # for root, dirs, files in os.walk(root_dir):
    #     directories = dirs
    #     break
    # for directory in directories:
    #     read_grad(root_dir, directory)
    #     print("_______________________________________________________________________")

    read_grad("paddle-GoogleNet", "GoogleNet-8-93")