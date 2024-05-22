import paddle
import paddle.nn as nn
import numpy as np
import os
import paddle.nn.functional as F


class Model_1715506754(nn.Layer):
    def __init__(self):
        super(Model_1715506754, self).__init__()
        self.conv1_mutated = paddle.nn.Conv2DTranspose(in_channels=1, out_channels=6, kernel_size=[5, 5], stride=[1, 1], padding=[0, 0], output_padding=[0, 0], dilation=[1, 1], groups=1, bias_attr=None)
        self.relu1 = paddle.nn.ReLU()
        self.pool1_mutated = paddle.nn.MaxPool2D(kernel_size=[3, 1], stride=[2, 2], padding=[0, 0], ceil_mode=False)
        self.conv2_mutated = paddle.nn.Conv2D(in_channels=6, out_channels=16, kernel_size=[5, 5], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias_attr=None)
        self.relu2_mutated = paddle.ceil
        self.pool2 = paddle.nn.MaxPool2D(kernel_size=[2, 2], stride=[2, 2], padding=[0, 0], ceil_mode=False)
        self.flatten = paddle.nn.Flatten()
        self.linear1_mutated = paddle.nn.Linear(in_features=672, out_features=120)
        self.relu3_mutated = paddle.round
        self.linear2_mutated = paddle.nn.Linear(in_features=120, out_features=84)
        self.tail_flatten = paddle.nn.Flatten()
        self.tail_fc = paddle.nn.Linear(in_features=84, out_features=10)

    def forward(self, input):
        conv1_output = self.conv1_mutated(input)
        relu1_output = self.relu1(conv1_output)
        maxpool1_output = self.pool1_mutated(relu1_output)
        conv2_output = self.conv2_mutated(maxpool1_output)
        relu2_output = self.relu2_mutated(conv2_output)
        maxpool2_output = self.pool2(relu2_output)
        flatten_output = self.flatten(maxpool2_output)
        fc1_output = self.linear1_mutated(flatten_output)
        relu3_output = self.relu3_mutated(fc1_output)
        fc2_output = self.linear2_mutated(relu3_output)
        tail_flatten_output = self.tail_flatten(fc2_output)
        tail_fc_output = self.tail_fc(tail_flatten_output)

        tail_fc_output = tail_fc_output
        return tail_fc_output


def go():
    try:
        model = Model_1715506754().to('gpu')
        x = paddle.randn([1, 1, 28, 28]).to('gpu')
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
        tensor = paddle.to_tensor(data['matrix'], dtype='float32', place=param.place)
        if "weight" in matrix_name:
           if data['matrix'].shape == (param.shape[1], param.shape[0]):
               tensor = paddle.to_tensor(data['matrix'].T, dtype='float32', place=param.place)
        param.set_value(tensor)


def train(inp, label):
    model = Model_1715506754().to('gpu')
    initialize(model)
    my_input = paddle.to_tensor(inp).to('gpu')
    output = model(my_input)
    target = paddle.to_tensor(label, dtype='int64').to('gpu')
    loss = nn.CrossEntropyLoss()(output, target)
    loss.backward()
    gradients = {name: param.grad.to('cpu').numpy() for name, param in model.named_parameters()}
    for key in gradients.keys():
        if len(gradients[key].shape) == 2:
            gradients[key] = gradients[key].T
    return gradients, loss.item(), output.detach().to('cpu').numpy()
