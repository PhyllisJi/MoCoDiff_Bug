import torch
import torch.nn as nn
import numpy as np
from torch import optim
import os
import torch.nn.functional as F


class Model_GPfAjTK5qHBhJhySk153X61UxWVTmZvI(nn.Module):
    def __init__(self):
        super(Model_GPfAjTK5qHBhJhySk153X61UxWVTmZvI, self).__init__()
        self.conv1_mutated = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=[7, 7], stride=[2, 2], padding=[0, 0], dilation=[1, 1], groups=1, bias=True)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=[3, 3], stride=[2, 2], padding=[0, 0], dilation=[1, 1], ceil_mode=False)
        self.conv2_mutated = torch.nn.Conv2d(in_channels=3, out_channels=2, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[7, 6], groups=1, bias=True)
        self.relu1 = torch.nn.ReLU()
        self.tail_flatten = torch.nn.Flatten()
        self.tail_fc = torch.nn.Linear(in_features=5832, out_features=1000)

    def forward(self, x):
        x = self.conv1_mutated(x)
        x = self.pool1(x)
        x = self.conv2_mutated(x)
        x = self.relu1(x)
        tail_flatten_output = self.tail_flatten(x)
        tail_fc_output = self.tail_fc(tail_flatten_output)

        tail_fc_output = tail_fc_output
        return tail_fc_output


def go():
    try:
        model = Model_GPfAjTK5qHBhJhySk153X61UxWVTmZvI()
        x = torch.randn([1, 3, 224, 224])
        y = model(x)
        flag = True
    except Exception:
        flag = False
    return flag


def initialize(model):
    module_dir = os.path.dirname(__file__)
    for name, param in model.named_parameters():
        layer_name, matrix_name = name.rsplit('.', 1)
        matrix_path = module_dir + './initializer/' + layer_name + '/' + matrix_name + '.npz'
        data = np.load(matrix_path)
        tensor = torch.from_numpy(data['matrix']).float()
        tensor = tensor.to(param.device)
        param.data = tensor


def train(inp, label):
    model = Model_GPfAjTK5qHBhJhySk153X61UxWVTmZvI().to('cpu')
    initialize(model)
    my_input = torch.from_numpy(inp).to('cpu')
    output = model(my_input)
    target = torch.from_numpy(label).to('cpu')
    loss = nn.CrossEntropyLoss()(output, target)
    loss.backward()
    gradients = {name: param.grad.to('cpu').numpy() for name, param in model.named_parameters()}
    return gradients, loss.item(), output.detach().to('cpu').numpy()
inp = np.load("./input.npz")
gradients, loss, output = train(inp['inp'], inp['label'])
print(output)
np.savez("./grad-cpu.npz", **gradients)
