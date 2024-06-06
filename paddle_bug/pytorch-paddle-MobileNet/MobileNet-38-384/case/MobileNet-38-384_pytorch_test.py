import torch
import torch.nn as nn
import numpy as np
from torch import optim
import os
import torch.nn.functional as F


class Model_4djfs4q1dzA0papKTAk16yoN_b_SC511(nn.Module):
    def __init__(self):
        super(Model_4djfs4q1dzA0papKTAk16yoN_b_SC511, self).__init__()
        self.conv1_mutated = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=True)
        self.conv2_mutated = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=[3, 3], stride=[8, 8], padding=[1, 1], dilation=[1, 1], groups=3, bias=True)
        self.relu1_mutated = torch.nn.AdaptiveMaxPool2d(output_size=1)
        self.conv3_mutated = torch.nn.Conv2d(in_channels=3, out_channels=4, kernel_size=[1, 1], stride=[5, 7], padding=[0, 0], dilation=[1, 1], groups=1, bias=True)
        self.relu2 = torch.nn.ReLU()
        self.conv4_mutated = torch.nn.ConvTranspose2d(in_channels=4, out_channels=4, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], output_padding=[0, 0], dilation=[1, 1], groups=4, bias=True)
        self.relu3 = torch.nn.ReLU()
        self.conv5_mutated = torch.nn.Conv2d(in_channels=4, out_channels=5, kernel_size=[1, 1], stride=[8, 8], padding=[0, 0], dilation=[1, 1], groups=1, bias=True)
        self.relu4_mutated = torch.floor
        self.conv6_mutated = torch.nn.Conv2d(in_channels=5, out_channels=5, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=5, bias=True)
        self.relu5 = torch.nn.ReLU()
        self.conv7_mutated = torch.nn.ConvTranspose2d(in_channels=5, out_channels=6, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], output_padding=[0, 0], dilation=[1, 1], groups=1, bias=True)
        self.relu6 = torch.nn.ReLU()
        self.conv8_mutated = torch.nn.ConvTranspose2d(in_channels=6, out_channels=6, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], output_padding=[0, 0], dilation=[1, 1], groups=6, bias=True)
        self.relu7 = torch.nn.ReLU()
        self.conv9_mutated = torch.nn.ConvTranspose2d(in_channels=6, out_channels=7, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], output_padding=[0, 0], dilation=[1, 1], groups=1, bias=True)
        self.relu8 = torch.nn.ReLU()
        self.conv10_mutated = torch.nn.ConvTranspose2d(in_channels=7, out_channels=7, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], output_padding=[0, 0], dilation=[1, 1], groups=7, bias=True)
        self.relu9_mutated = torch.nn.Tanh()
        self.conv11_mutated = torch.nn.ConvTranspose2d(in_channels=7, out_channels=7, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], output_padding=[0, 0], dilation=[1, 1], groups=1, bias=True)
        self.relu10 = torch.nn.ReLU()
        self.conv12_mutated = torch.nn.ConvTranspose2d(in_channels=7, out_channels=7, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], output_padding=[0, 0], dilation=[1, 1], groups=7, bias=True)
        self.relu11 = torch.nn.ReLU()
        self.conv13_mutated = torch.nn.ConvTranspose2d(in_channels=7, out_channels=8, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], output_padding=[0, 0], dilation=[1, 1], groups=1, bias=True)
        self.relu12 = torch.nn.ReLU()
        self.conv14_mutated = torch.nn.ConvTranspose2d(in_channels=8, out_channels=8, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], output_padding=[0, 0], dilation=[1, 1], groups=8, bias=True)
        self.relu13 = torch.nn.ReLU()
        self.conv15_mutated = torch.nn.ConvTranspose2d(in_channels=8, out_channels=8, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], output_padding=[0, 0], dilation=[1, 1], groups=1, bias=True)
        self.relu14 = torch.nn.ReLU()
        self.conv16_mutated = torch.nn.ConvTranspose2d(in_channels=8, out_channels=8, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], output_padding=[0, 0], dilation=[1, 1], groups=8, bias=True)
        self.relu15 = torch.nn.ReLU()
        self.conv17_mutated = torch.nn.ConvTranspose2d(in_channels=8, out_channels=8, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], output_padding=[0, 0], dilation=[1, 1], groups=1, bias=True)
        self.relu16 = torch.nn.ReLU()
        self.conv18_mutated = torch.nn.ConvTranspose2d(in_channels=8, out_channels=8, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], output_padding=[0, 0], dilation=[1, 1], groups=8, bias=True)
        self.relu17 = torch.nn.ReLU()
        self.conv19_mutated = torch.nn.ConvTranspose2d(in_channels=8, out_channels=8, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], output_padding=[0, 0], dilation=[1, 1], groups=1, bias=True)
        self.relu18_mutated = torch.reciprocal
        self.conv20_mutated = torch.nn.ConvTranspose2d(in_channels=8, out_channels=8, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], output_padding=[0, 0], dilation=[1, 1], groups=8, bias=True)
        self.tail_flatten = torch.nn.Flatten()
        self.tail_fc = torch.nn.Linear(in_features=8, out_features=1000)

    def forward(self, input):
        conv1_output = self.conv1_mutated(input)
        conv2_output = self.conv2_mutated(conv1_output)
        relu1_output = self.relu1_mutated(conv2_output)
        conv3_output = self.conv3_mutated(relu1_output)
        relu2_output = self.relu2(conv3_output)
        conv4_output = self.conv4_mutated(relu2_output)
        relu3_output = self.relu3(conv4_output)
        conv5_output = self.conv5_mutated(relu3_output)
        relu4_output = self.relu4_mutated(conv5_output)
        conv6_output = self.conv6_mutated(relu4_output)
        relu5_output = self.relu5(conv6_output)
        conv7_output = self.conv7_mutated(relu5_output)
        relu6_output = self.relu6(conv7_output)
        conv8_output = self.conv8_mutated(relu6_output)
        relu7_output = self.relu7(conv8_output)
        conv9_output = self.conv9_mutated(relu7_output)
        relu8_output = self.relu8(conv9_output)
        conv10_output = self.conv10_mutated(relu8_output)
        relu9_output = self.relu9_mutated(conv10_output)
        conv11_output = self.conv11_mutated(relu9_output)
        relu10_output = self.relu10(conv11_output)
        conv12_output = self.conv12_mutated(relu10_output)
        relu11_output = self.relu11(conv12_output)
        conv13_output = self.conv13_mutated(relu11_output)
        relu12_output = self.relu12(conv13_output)
        conv14_output = self.conv14_mutated(relu12_output)
        relu13_output = self.relu13(conv14_output)
        conv15_output = self.conv15_mutated(relu13_output)
        relu14_output = self.relu14(conv15_output)
        conv16_output = self.conv16_mutated(relu14_output)
        relu15_output = self.relu15(conv16_output)
        conv17_output = self.conv17_mutated(relu15_output)
        relu16_output = self.relu16(conv17_output)
        conv18_output = self.conv18_mutated(relu16_output)
        relu17_output = self.relu17(conv18_output)
        conv19_output = self.conv19_mutated(relu17_output)
        relu18_output = self.relu18_mutated(conv19_output)
        conv20_output = self.conv20_mutated(relu18_output)
        tail_flatten_output = self.tail_flatten(conv20_output)
        tail_fc_output = self.tail_fc(tail_flatten_output)

        tail_fc_output = tail_fc_output
        return tail_fc_output


def go():
    try:
        model = Model_4djfs4q1dzA0papKTAk16yoN_b_SC511().to('cuda')
        x = torch.randn([1, 3, 224, 224]).to('cuda')
        y = model(x)
        flag = True
    except Exception:
        flag = False
    return flag


def initialize(model):
    module_dir = os.path.dirname(__file__)
    for name, param in model.named_parameters():
        layer_name, matrix_name = name.split('.')
        matrix_path = module_dir + '/../initializer/' + layer_name + '/' + matrix_name + '.npz'
        data = np.load(matrix_path)
        tensor = torch.from_numpy(data['matrix']).float()
        tensor = tensor.to(param.device)
        param.data = tensor


def train(inp, label):
    model = Model_4djfs4q1dzA0papKTAk16yoN_b_SC511().to('cuda')
    initialize(model)
    my_input = torch.from_numpy(inp).to('cuda')
    output = model(my_input)
    target = torch.from_numpy(label).to('cuda')
    loss = nn.CrossEntropyLoss()(output, target)
    loss.backward()
    gradients = {name: param.grad.to('cpu').numpy() for name, param in model.named_parameters()}
    return gradients, loss.item(), output.detach().to('cpu').numpy()
