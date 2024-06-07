import torch
import torch.nn as nn
import numpy as np
from torch import optim
import os
import torch.nn.functional as F


class Model_Gh8lMCGxHLli_0NjVOnFISjc2dkuC9Mz(nn.Module):
    def __init__(self):
        super(Model_Gh8lMCGxHLli_0NjVOnFISjc2dkuC9Mz, self).__init__()
        self.conv1_mutated = torch.nn.ConvTranspose1d(in_channels=3, out_channels=64, kernel_size=[1], stride=[1], padding=[0], output_padding=[0], dilation=[1], groups=1, bias=True)
        self.bn1 = torch.nn.BatchNorm1d(num_features=64, eps=1e-05, momentum=0.9)
        self.relu1 = torch.nn.ReLU()
        self.conv2_mutated = torch.nn.Conv1d(in_channels=64, out_channels=64, kernel_size=[1], stride=[1], padding=[0], dilation=[1], groups=1, bias=True)
        self.bn2 = torch.nn.BatchNorm1d(num_features=64, eps=1e-05, momentum=0.9)
        self.relu2_mutated = torch.nn.AdaptiveAvgPool1d(output_size=1)
        self.conv3_mutated = torch.nn.Conv1d(in_channels=64, out_channels=64, kernel_size=[1], stride=[1], padding=[0], dilation=[1], groups=1, bias=True)
        self.bn3 = torch.nn.BatchNorm1d(num_features=64, eps=1e-05, momentum=0.9)
        self.relu3 = torch.nn.ReLU()
        self.conv4_mutated = torch.nn.Conv1d(in_channels=64, out_channels=128, kernel_size=[1], stride=[1], padding=[1], dilation=[1], groups=1, bias=True)
        self.bn4 = torch.nn.BatchNorm1d(num_features=128, eps=1e-05, momentum=0.9)
        self.relu4_mutated = torch.erf
        self.conv5_mutated = torch.nn.ConvTranspose1d(in_channels=128, out_channels=1024, kernel_size=[1], stride=[1], padding=[0], output_padding=[0], dilation=[1], groups=1, bias=True)
        self.bn5 = torch.nn.BatchNorm1d(num_features=1024, eps=1e-05, momentum=0.9)
        self.relu5 = torch.nn.ReLU()
        self.globalpool = torch.nn.AdaptiveMaxPool1d(output_size=1)
        self.flatten = torch.nn.Flatten()
        self.linear1 = torch.nn.Linear(in_features=1024, out_features=512)
        self.bn6 = torch.nn.BatchNorm1d(num_features=512, eps=1e-05, momentum=0.9)
        self.relu6 = torch.nn.ReLU()
        self.tail_flatten = torch.nn.Flatten()
        self.tail_fc = torch.nn.Linear(in_features=512, out_features=10)

    def forward(self, x):
        x = self.conv1_mutated(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2_mutated(x)
        x = self.bn2(x)
        x = self.relu2_mutated(x)
        x = self.conv3_mutated(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.conv4_mutated(x)
        x = self.bn4(x)
        x = self.relu4_mutated(x)
        x = self.conv5_mutated(x)
        x = self.bn5(x)
        x = self.relu5(x)
        x = self.globalpool(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.bn6(x)
        x = self.relu6(x)
        tail_flatten_output = self.tail_flatten(x)
        tail_fc_output = self.tail_fc(tail_flatten_output)

        tail_fc_output = tail_fc_output
        return tail_fc_output


def go():
    try:
        model = Model_Gh8lMCGxHLli_0NjVOnFISjc2dkuC9Mz()
        x = torch.randn([2, 3, 2048])
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
    model = Model_Gh8lMCGxHLli_0NjVOnFISjc2dkuC9Mz().to('cpu')
    initialize(model)
    my_input = torch.from_numpy(inp).to('cpu')
    output = model(my_input)
    target = torch.from_numpy(label).to('cpu')
    loss = nn.CrossEntropyLoss()(output, target)
    loss.backward()
    gradients = {name: param.grad.to('cpu').numpy() for name, param in model.named_parameters()}
    return gradients, loss.item(), output.detach().to('cpu').numpy()
