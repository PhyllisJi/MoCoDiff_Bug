import torch
import torch.nn as nn
import numpy as np
from torch import optim
import os
import torch.nn.functional as F


class Model_7iqqNJoc8LgdfHpFybB049DQYfURiOHg(nn.Module):
    def __init__(self):
        super(Model_7iqqNJoc8LgdfHpFybB049DQYfURiOHg, self).__init__()
        self.conv1_mutated = torch.nn.ConvTranspose2d(in_channels=1, out_channels=6, kernel_size=[5, 5], stride=[1, 1], padding=[0, 0], output_padding=[0, 0], dilation=[1, 1], groups=1, bias=True)
        self.relu1 = torch.nn.ReLU()
        self.pool1 = torch.nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=[0, 0], dilation=1, ceil_mode=False)
        self.conv2_mutated = torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=[6, 8], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True)
        self.relu2_mutated = torch.nn.Softsign()
        self.pool2 = torch.nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=[0, 0], dilation=1, ceil_mode=False)
        self.flatten = torch.nn.Flatten()
        self.linear1_mutated = torch.nn.Linear(in_features=320, out_features=120)
        self.relu3 = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(in_features=120, out_features=84)
        self.relu4_mutated = torch.reciprocal
        self.tail_flatten = torch.nn.Flatten()
        self.tail_fc = torch.nn.Linear(in_features=84, out_features=10)

    def forward(self, input):
        conv1_output = self.conv1_mutated(input)
        relu1_output = self.relu1(conv1_output)
        maxpool1_output = self.pool1(relu1_output)
        conv2_output = self.conv2_mutated(maxpool1_output)
        relu2_output = self.relu2_mutated(conv2_output)
        maxpool2_output = self.pool2(relu2_output)
        flatten_output = self.flatten(maxpool2_output)
        fc1_output = self.linear1_mutated(flatten_output)
        relu3_output = self.relu3(fc1_output)
        fc2_output = self.linear2(relu3_output)
        relu4_output = self.relu4_mutated(fc2_output)
        tail_flatten_output = self.tail_flatten(relu4_output)
        tail_fc_output = self.tail_fc(tail_flatten_output)

        tail_fc_output = tail_fc_output
        return tail_fc_output


def go():
    try:
        model = Model_7iqqNJoc8LgdfHpFybB049DQYfURiOHg().to('cuda')
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
        matrix_path = module_dir + '/../initializer/' + layer_name + '/' + matrix_name + '.npz'
        data = np.load(matrix_path)
        tensor = torch.from_numpy(data['matrix']).float()
        tensor = tensor.to(param.device)
        param.data = tensor


def train(inp, label):
    model = Model_7iqqNJoc8LgdfHpFybB049DQYfURiOHg().to('cuda')
    initialize(model)
    my_input = torch.from_numpy(inp).to('cuda')
    output = model(my_input)
    target = torch.from_numpy(label).to('cuda')
    loss = nn.CrossEntropyLoss()(output, target)
    loss.backward()
    gradients = {name: param.grad.to('cpu').numpy() for name, param in model.named_parameters()}
    return gradients, loss.item(), output.detach().to('cpu').numpy()
