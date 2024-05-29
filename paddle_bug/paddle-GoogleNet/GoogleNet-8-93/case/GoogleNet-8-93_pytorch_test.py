import torch
import torch.nn as nn
import numpy as np
from torch import optim
import os
import torch.nn.functional as F


class Model_8MALSkW5Wc363zuitXTnAvl4NzTqrvbx(nn.Module):
    def __init__(self):
        super(Model_8MALSkW5Wc363zuitXTnAvl4NzTqrvbx, self).__init__()
        self.conv1_mutated = torch.nn.Conv2d(in_channels=3, out_channels=4, kernel_size=[7, 7], stride=[2, 2], padding=[3, 3], dilation=[8, 2], groups=1, bias=True)
        self.relu1 = torch.nn.ReLU()
        self.pool1_mutated = torch.nn.MaxPool2d(kernel_size=[3, 6], stride=[2, 2], padding=[0, 0], dilation=[1, 1], ceil_mode=True)
        self.conv2_mutated = torch.nn.ConvTranspose2d(in_channels=4, out_channels=4, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], output_padding=[0, 0], dilation=[1, 1], groups=1, bias=True)
        self.relu2_mutated = torch.nn.Tanh()
        self.conv3_mutated = torch.nn.ConvTranspose2d(in_channels=4, out_channels=12, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], output_padding=[0, 0], dilation=[1, 1], groups=1, bias=True)
        self.relu3_mutated = torch.reciprocal
        self.pool2 = torch.nn.MaxPool2d(kernel_size=[3, 3], stride=[2, 2], padding=[0, 0], dilation=[1, 1], ceil_mode=True)
        self.tail_flatten = torch.nn.Flatten()
        self.tail_fc = torch.nn.Linear(in_features=6864, out_features=1000)

    def forward(self, input):
        conv1_output = self.conv1_mutated(input)
        relu1_output = self.relu1(conv1_output)
        pool1_output = self.pool1_mutated(relu1_output)
        conv2_output = self.conv2_mutated(pool1_output)
        relu2_output = self.relu2_mutated(conv2_output)
        conv3_output = self.conv3_mutated(relu2_output)
        relu3_output = self.relu3_mutated(conv3_output)
        inception1_input = self.pool2(relu3_output)
        tail_flatten_output = self.tail_flatten(inception1_input)
        tail_fc_output = self.tail_fc(tail_flatten_output)

        tail_fc_output = tail_fc_output
        return tail_fc_output


def go():
    try:
        model = Model_8MALSkW5Wc363zuitXTnAvl4NzTqrvbx().to('cuda')
        x = torch.randn([1, 3, 224, 224]).to('cuda')
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
    model = Model_8MALSkW5Wc363zuitXTnAvl4NzTqrvbx().to('cuda')
    initialize(model)
    my_input = torch.from_numpy(inp).to('cuda')
    output = model(my_input)
    target = torch.from_numpy(label).to('cuda')
    loss = nn.CrossEntropyLoss()(output, target)
    loss.backward()
    gradients = {name: param.grad.to('cpu').numpy() for name, param in model.named_parameters()}
    return gradients, loss.item(), output.detach().to('cpu').numpy()
