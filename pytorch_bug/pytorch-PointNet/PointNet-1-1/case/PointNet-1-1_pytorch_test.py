import torch
import torch.nn as nn
import numpy as np
from torch import optim
import os
import torch.nn.functional as F


class Model_yycZ9fPDLJratiKMIYEkMCSp2MNnhZ7p(nn.Module):
    def __init__(self):
        super(Model_yycZ9fPDLJratiKMIYEkMCSp2MNnhZ7p, self).__init__()
        self.conv1_mutated = torch.nn.Conv1d(in_channels=3, out_channels=64, kernel_size=[1], stride=[1], padding=[1], dilation=[1], groups=1, bias=True)
        self.tail_flatten = torch.nn.Flatten()
        self.tail_fc = torch.nn.Linear(in_features=131200, out_features=10)

    def forward(self, x):
        x = self.conv1_mutated(x)
        tail_flatten_output = self.tail_flatten(x)
        tail_fc_output = self.tail_fc(tail_flatten_output)

        tail_fc_output = tail_fc_output
        return tail_fc_output


def go():
    try:
        model = Model_yycZ9fPDLJratiKMIYEkMCSp2MNnhZ7p().to('cuda')
        x = torch.randn([2, 3, 2048]).to('cuda')
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
    model = Model_yycZ9fPDLJratiKMIYEkMCSp2MNnhZ7p().to('cuda')
    initialize(model)
    my_input = torch.from_numpy(inp).to('cuda')
    output = model(my_input)
    target = torch.from_numpy(label).to('cuda')
    loss = nn.CrossEntropyLoss()(output, target)
    loss.backward()
    gradients = {name: param.grad.to('cpu').numpy() for name, param in model.named_parameters()}
    return gradients, loss.item(), output.detach().to('cpu').numpy()
