import torch
import torch.nn as nn
import numpy as np
from torch import optim
import os
import torch.nn.functional as F


class Model_gGjaaC_PbmJv5McK74QSRSo6yZEqY0Nb(nn.Module):
    def __init__(self):
        super(Model_gGjaaC_PbmJv5McK74QSRSo6yZEqY0Nb, self).__init__()
        self.conv1_mutated = torch.nn.ConvTranspose2d(in_channels=1, out_channels=6, kernel_size=[5, 5], stride=[1, 1], padding=[0, 0], output_padding=[0, 0], dilation=[1, 1], groups=1, bias=True)
        self.relu1 = torch.nn.ReLU()
        self.pool1_mutated = torch.nn.MaxPool2d(kernel_size=[2, 2], stride=[1, 2], padding=[0, 0], dilation=1, ceil_mode=False)
        self.conv2_mutated = torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=[5, 5], stride=[8, 8], padding=[0, 0], dilation=[1, 1], groups=1, bias=True)
        self.relu2 = torch.nn.ReLU()
        self.pool2_mutated = torch.nn.MaxPool2d(kernel_size=[2, 2], stride=[8, 8], padding=[0, 0], dilation=1, ceil_mode=False)
        self.flatten = torch.nn.Flatten()
        self.linear1 = torch.nn.Linear(in_features=16, out_features=120)
        self.relu3_mutated = torch.round
        self.linear2_mutated = torch.nn.Linear(in_features=120, out_features=84)
        self.tail_flatten = torch.nn.Flatten()
        self.tail_fc = torch.nn.Linear(in_features=84, out_features=10)

    def forward(self, input):
        conv1_output = self.conv1_mutated(input)
        relu1_output = self.relu1(conv1_output)
        maxpool1_output = self.pool1_mutated(relu1_output)
        conv2_output = self.conv2_mutated(maxpool1_output)
        relu2_output = self.relu2(conv2_output)
        maxpool2_output = self.pool2_mutated(relu2_output)
        flatten_output = self.flatten(maxpool2_output)
        fc1_output = self.linear1(flatten_output)
        relu3_output = self.relu3_mutated(fc1_output)
        fc2_output = self.linear2_mutated(relu3_output)
        tail_flatten_output = self.tail_flatten(fc2_output)
        tail_fc_output = self.tail_fc(tail_flatten_output)

        tail_fc_output = tail_fc_output
        
        parent_dir = os.path.dirname(os.path.abspath(script_path_os))
        np.savez(f"{parent_dir}/layers_output.npz", 
            conv1_output=conv1_output.cpu().detach().numpy(),
            relu1_output=relu1_output.cpu().detach().numpy(),
            maxpool1_output=maxpool1_output.cpu().detach().numpy(),
            conv2_output=conv2_output.cpu().detach().numpy(),
            relu2_output=relu2_output.cpu().detach().numpy(),
            maxpool2_output=maxpool2_output.cpu().detach().numpy(),
            flatten_output=flatten_output.cpu().detach().numpy(),
            fc1_output=fc1_output.cpu().detach().numpy(),
            relu3_output=relu3_output.cpu().detach().numpy(),
            fc2_output=fc2_output.cpu().detach().numpy(),
            tail_flatten_output=tail_flatten_output.cpu().detach().numpy(),
            tail_fc_output=tail_fc_output.cpu().detach().numpy()
            )
        
        return tail_fc_output


def go():
    try:
        model = Model_gGjaaC_PbmJv5McK74QSRSo6yZEqY0Nb().to('cuda')
        x = torch.randn([1, 1, 28, 28]).to('cuda')
        y = model(x)
        flag = True
    except Exception:
        flag = False
    return flag


def initialize(model):
    module_dir = os.path.dirname(__file__)
    for name, param in model.named_parameters():
        layer_name, matrix_name = name.rsplit('.', 1)
        matrix_path = module_dir + '/initializer/' + layer_name + '/' + matrix_name + '.npz'
        data = np.load(matrix_path)
        tensor = torch.from_numpy(data['matrix']).float()
        tensor = tensor.to(param.device)
        param.data = tensor


def train(inp, label):
    model = Model_gGjaaC_PbmJv5McK74QSRSo6yZEqY0Nb().to('cuda')
    initialize(model)
    my_input = torch.from_numpy(inp).to('cuda')
    output = model(my_input)
    target = torch.from_numpy(label).to('cuda')
    loss = nn.CrossEntropyLoss()(output, target)
    loss.backward()
    gradients = {name: param.grad.to('cpu').numpy() for name, param in model.named_parameters()}
    return gradients, loss.item(), output.detach().to('cpu').numpy()

input = np.load("./input.npz")
train(input['inp'], input['label'])
