from importlib import import_module
import inspect
import torch
import numpy as np
import sys
import json
import os
import re
import io
import contextlib
from collections import OrderedDict

variable_values = OrderedDict()


def process_valid_num(x):
    return x if not (np.isinf(x) or np.isnan(x)) else ('INF' if np.isinf(x) else 'NaN')


def add_save_statements(match):
    indent = match.group(1)
    statement = match.group(2)
    var_name = statement.split('=')[0].strip()
    if var_name != "x":
        return f'{indent}{statement}\n{indent}variable_values[\"{var_name}\"]={var_name}'
    else:
        output_name = statement.split('=')[1].replace("self.", "").strip().rstrip()
        output_name = output_name.split("(")[0] if '(' in output_name else output_name
        output_name = output_name + "_output"
        if "_mutated" in output_name:
            output_name = output_name.replace("_mutated", "")
        return f'{indent}{statement}\n{indent}variable_values[\"{output_name}\"]={var_name}'


def gen_new_forward(forward_code):
    pattern = r'(\s*)(\w+\s*=\s*self\.[^\n]+)'
    matches = re.findall(pattern, forward_code)
    output_layer = {}
    for match in matches:
        statement = match[1]
        var_name = statement.split('=')[0].strip()
        layer_name = statement.split('=')[1].replace("self.", "").strip().rstrip()
        layer_name = layer_name.split("(")[0] if '(' in layer_name else layer_name
        if var_name != "x":
            output_layer[var_name] = layer_name
        else:
            output_name = layer_name + "_output"
            if "_mutated" in output_name:
                output_name = output_name.replace("_mutated", "")
            output_layer[output_name] = layer_name

    for key in output_layer.keys():
        print(key, "=======>", output_layer[key])

    # 插入打印语句
    modified_source_code = re.sub(pattern, add_save_statements, forward_code)

    # 移除函数定义行前的多余缩进
    modified_source_code = re.sub(r'^\s*def', 'def', modified_source_code, flags=re.MULTILINE)
    return modified_source_code, output_layer


def find_difference_positions(array1, array2, distance=1):
    diff = np.abs(array1 - array2)
    positions = np.argwhere(diff == distance)
    values = [(array1[tuple(pos)], array2[tuple(pos)]) for pos in positions]
    return positions, values


def obtain_torch_layer_outputs(case_path, py_name, input_tensor, label, is_gpu):
    if case_path not in sys.path:
        sys.path.append(case_path)
    model_module = import_module(py_name.replace(".py", ""))

    for name, cls in model_module.__dict__.items():
        if inspect.isclass(cls) and "Model_" in name:
            model_class = cls
            break

    # 获取 forward 方法
    forward_method = getattr(model_class, 'forward')
    forward_code = inspect.getsource(forward_method)
    # 定义一个正则表达式模式来匹配每次赋值操作并插入打印语句

    modified_forward_code, output_layer = gen_new_forward(forward_code)
    # 执行修改后的源代码
    exec_dict = {}
    exec(modified_forward_code, globals(), exec_dict)
    # 将修改后的 forward 方法替换到类中
    setattr(model_class, 'forward', exec_dict['forward'])
    init_code = inspect.getsource(getattr(model_class, "__init__"))
    pattern = r'self\.(\w+)\s*=\s*([\w\.]+(?:\([^)]*\))?)'
    matches = re.findall(pattern, init_code)
    layer_type = {}
    for var_name, assignment in matches:
        # 提取类名或函数名，去掉括号及其内容
        class_name = assignment.split('(')[0] if '(' in assignment else assignment
        layer_type[var_name] = class_name.strip().rstrip()

    for key in layer_type.keys():
        print(key, "==>", layer_type[key])

    if is_gpu:
        model = model_class().to('cuda')
    else:
        model = model_class()

    initialize_model = getattr(model_module, 'initialize', None)
    initialize_model(model)
    # 前向传播
    output = model(input_tensor)
    layer_outputs = OrderedDict()
    for var_name, value in variable_values.items():
        layer_name = output_layer[var_name]
        layer_outputs[layer_name + "-" + layer_type[layer_name]] = value
    variable_values.clear()
    # 获取梯度
    grads, loss = obtain_grads(model, output, label, is_gpu)
    return layer_outputs, output, grads, loss


def obtain_grads(model, output, label, is_gpu):
    if is_gpu:
        target = torch.from_numpy(label).to('cuda')
    else:
        target = torch.from_numpy(label).to('cpu')
    gradients = {}
    try:
        loss = torch.nn.CrossEntropyLoss()(output, target)
        loss.backward()
        for name, param in model.named_parameters():
            if param.grad is not None:
                gradients[name] = param.grad.to('cpu').numpy()
            else:
                gradients[name] = None
        return gradients, loss.item()
    except Exception as e:
        print(f"Unexpected error: {e}")
        for name, param in model.named_parameters():
            gradients[name] = None
        return gradients, None


def process_valid_num_dict(diff_dict):
    for key in diff_dict.keys():
        if key != "type":
            diff_dict[key] = process_valid_num(diff_dict[key])
    return diff_dict


def cmp_outputs(library, case_path, node, input_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cpu_path = f"{case_path}/{library}_cpu/"
    cpu_py = f"{node}_{library}_cpu.py"
    gpu_path = f"{case_path}/{library}_gpu/"
    gpu_py = f"{node}_{library}_gpu.py"

    if library == "pytorch":
        cpu_input_tensor = torch.from_numpy(np.load(input_path)['inp'].astype(np.float32))
        cpu_outputs, cpu_output, cpu_grads, cpu_loss = obtain_torch_layer_outputs(cpu_path, cpu_py, cpu_input_tensor,
                                                                                  np.load(input_path)['label'], False)
        gpu_input_tensor = torch.from_numpy(np.load(input_path)['inp'].astype(np.float32)).to(device)
        gpu_outputs, gpu_output, gpu_grads, gpu_loss = obtain_torch_layer_outputs(gpu_path, gpu_py, gpu_input_tensor,
                                                                                  np.load(input_path)['label'], True)
        output_diff_dict = OrderedDict()
        for key in cpu_outputs.keys():
            print(key)
            c_o = cpu_outputs[key].detach().to('cpu').numpy()
            g_o = gpu_outputs[key].detach().to('cpu').numpy()
            output_mean_diff = np.mean(np.abs(c_o - g_o))
            output_max_diff = np.max(np.abs(c_o - g_o))
            output_min_diff = np.min(np.abs(c_o - g_o))
            output_diff_dict[key] = {"mean": process_valid_num(output_mean_diff),
                                     "max": process_valid_num(output_max_diff),
                                     "min": process_valid_num(output_min_diff)}
        ns_res = judge_numerical_stability(cpu_outputs, gpu_outputs, output_diff_dict)
        grad_diff_dict = OrderedDict()
        if (gpu_loss is None) | (cpu_loss is None):
            print("Train Error!!")
            return ns_res, output_diff_dict, None, None, None
        if len(set(cpu_grads.keys())) != len(set(gpu_grads.keys())):
            grad_diff_dict["type"] = "key inconsistent"
        else:
            grad_diff_dict["type"] = "same key"
            max_grad_diff = 0.0
            for i in range(len(cpu_grads.keys())):
                if (cpu_grads[list(cpu_grads.keys())[i]] is not None) & (
                        gpu_grads[list(gpu_grads.keys())[i]] is not None):
                    current_grad_diff = np.max(
                        np.abs(cpu_grads[list(cpu_grads.keys())[i]] - gpu_grads[list(gpu_grads.keys())[i]]))
                    grad_diff_dict[list(cpu_grads.keys())[i]] = current_grad_diff
                    if current_grad_diff > max_grad_diff:
                        max_grad_diff = current_grad_diff
                else:
                    grad_diff_dict[list(cpu_grads.keys())[i]] = "NaN"

        loss_diff = process_valid_num(np.abs(cpu_loss - gpu_loss))
        grad_diff_dict = process_valid_num_dict(grad_diff_dict)
        print("loss diff: ", loss_diff)
        return ns_res, output_diff_dict, max_grad_diff, grad_diff_dict, loss_diff


def judge_numerical_stability(cpu_outputs, gpu_outputs, output_diff_dict):
    diff_dict_items = list(output_diff_dict.items())
    for i in range(len(diff_dict_items)):
        current_key, current_value = diff_dict_items[i]
        # print("current_key: ", current_key)
        if current_key.split("-")[1] == "torch.reciprocal":
            last_key, last_value = diff_dict_items[i - 1]
            last_cpu_output = cpu_outputs[last_key].detach().to('cpu').numpy()
            if float(abs(np.mean(last_cpu_output))) < 10 ** -4:
                # if (float(last_value["mean"]) < 10**-6) & (cpu_outputs[last_key]< 10**-4):
                uncheck_reason = "The input is too small " + str(float(abs(np.mean(last_cpu_output))))
                return uncheck_reason, False
            else:
                print("torch.reciprocal", float(abs(np.mean(last_cpu_output))) < 10 ** -4)
                continue
                # check_reason = "The input is not too small " + str(last_value["mean"])
                # return check_reason, True
        elif current_key.split("-")[1] in ["torch.floor", "torch.ceil", "torch.round", "F.threshold"]:
            last_key, last_value = diff_dict_items[i - 1]
            last_cpu_output = cpu_outputs[last_key].detach().to('cpu').numpy()
            last_gpu_output = gpu_outputs[last_key].detach().to('cpu').numpy()
            last_positions = np.where(last_cpu_output != last_gpu_output)
            current_cpu_output = cpu_outputs[current_key].detach().to('cpu').numpy()
            current_gpu_output = gpu_outputs[current_key].detach().to('cpu').numpy()
            cur_positions = np.where(current_cpu_output != current_gpu_output)
            if np.all(np.isin(cur_positions, last_positions)):
                print("torch.floor", "torch.ceil", "torch.round", "F.threshold do not check")
                return "floor ceil round et.al position match", False
            else:
                cur_as_lists = [x.tolist() for x in cur_positions]
                last_as_lists = [x.tolist() for x in last_positions]
                # print(cur_as_lists, last_as_lists)
                check_reason = {"cur_pos": cur_as_lists, "last_pos": last_as_lists}
                return check_reason, True
        elif i == len(diff_dict_items) - 3:
            last_key, last_value = diff_dict_items[i - 1]
            if float(last_value["max"]) > 0.01:
                print("input has difference, do not check")
                uncheck_reason = "input has difference, do not check " + str(last_value["max"])
                return uncheck_reason, False
            else:
                continue

    return "No match to existing situation", True


def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


library = "pytorch"
case_path = "./pytorch-PointNet/PointNet-20-238/case"
input_path = case_path + "/input.npz"
node = "PointNet-20-238"
ns_res, output_diff_dict, max_grad_diff, grad_diff_dict, loss_diff = cmp_outputs(library, case_path, node, input_path)
for key in output_diff_dict.keys():
    print(key)
    print(output_diff_dict[key])

for key in grad_diff_dict.keys():
    print(key)
    print(grad_diff_dict[key])

