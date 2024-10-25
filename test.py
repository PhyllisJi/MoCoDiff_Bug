import importlib.util
import json
import os
import sys
import time

import torch
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np
import random


with np.load("./input.npz") as input_data:
    inp, label = input_data["inp"], input_data["label"]

seed = 42
random.seed(seed)
np.random.seed(seed)
paddle.seed(seed)
paddle.framework.set_flags({"FLAGS_cudnn_deterministic": True})
paddle.framework.set_flags({"FLAGS_cudnn_exhaustive_search": False})


class Model_1729248955(nn.Layer):
    def __init__(self):
        super(Model_1729248955, self).__init__()
        self.conv1_mutated = paddle.nn.Conv2D(in_channels=3, out_channels=4, kernel_size=[8, 8], stride=[4, 4], padding=[2, 2], dilation=[1, 1], groups=1, bias_attr=None)
        self.relu1 = paddle.nn.ReLU()
        self.pool1 = paddle.nn.MaxPool2D(kernel_size=[3, 3], stride=[2, 2], padding=[0, 0], ceil_mode=False)
        self.conv2_mutated = paddle.nn.Conv2DTranspose(in_channels=4, out_channels=6, kernel_size=[5, 5], stride=[1, 1], padding=[2, 2], output_padding=[0, 0], dilation=[1, 1], groups=1, bias_attr=None)
        self.relu2_mutated = paddle.round
        self.pool2_mutated = paddle.nn.MaxPool2D(kernel_size=[8, 8], stride=[2, 2], padding=[0, 0], ceil_mode=False)
        self.conv3_mutated = paddle.nn.Conv2DTranspose(in_channels=6, out_channels=8, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], output_padding=[0, 0], dilation=[1, 1], groups=1, bias_attr=None)
        self.relu3_mutated = paddle.round
        self.conv4_mutated = paddle.nn.Conv2D(in_channels=8, out_channels=10, kernel_size=[3, 3], stride=1, padding=[4, 5], dilation=[1, 1], groups=1, bias_attr=None)
        self.relu4_mutated = paddle.reciprocal
        self.conv5_mutated = paddle.nn.Conv2DTranspose(in_channels=10, out_channels=12, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], output_padding=[0, 0], dilation=[1, 1], groups=1, bias_attr=None)
        self.relu5_mutated = paddle.reciprocal
        self.pool3_mutated = paddle.nn.MaxPool2D(kernel_size=[8, 8], stride=[2, 2], padding=[0, 0], ceil_mode=False)
        self.avgpool_mutated = paddle.nn.AdaptiveMaxPool2D(output_size=1)
        self.tail_flatten = paddle.nn.Flatten()
        self.tail_fc = paddle.nn.Linear(in_features=12, out_features=1000)

    def forward(self, input):
        conv1_output = self.conv1_mutated(input)
        relu1_output = self.relu1(conv1_output)
        maxpool1_output = self.pool1(relu1_output)
        conv2_output = self.conv2_mutated(relu1_output)
        relu2_output = self.relu2_mutated(conv2_output)
        maxpool2_output = self.pool2_mutated(relu2_output)
        conv3_output = self.conv3_mutated(maxpool2_output)
        relu3_output = self.relu3_mutated(conv3_output)
        conv4_output = self.conv4_mutated(relu3_output)
        relu4_output = self.relu4_mutated(conv4_output)
        conv5_output = self.conv5_mutated(relu4_output)
        relu5_output = self.relu5_mutated(conv5_output)
        maxpool3_output = self.pool3_mutated(relu5_output)
        avgpool_output = self.avgpool_mutated(maxpool3_output)
        tail_flatten_output = self.tail_flatten(avgpool_output)
        tail_fc_output = self.tail_fc(tail_flatten_output)

        tail_fc_output = tail_fc_output
        return tail_fc_output

    
paddle.set_device("cpu")
paddle_model_cpu = Model_1729248955()
paddle_model_cpu.eval()
input = paddle.to_tensor(inp).astype('float32')
paddle_output_cpu = paddle_model_cpu(input)
target = paddle.to_tensor(label, dtype='int64')
loss = paddle.nn.CrossEntropyLoss()(paddle_output_cpu, target)
loss.backward()
gradients = {name: param.grad.to('cpu').numpy() for name, param in model.named_parameters() if param.grad is not None }
for key in gradients.keys():
    if len(gradients[key].shape) == 2:
        gradients[key] = gradients[key].T
print(gradients)
print(loss.item())
print(paddle_output_cpu.detach().to('cpu').numpy())

