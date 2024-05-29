import paddle
import paddle.nn as nn
import numpy as np
import os
import paddle.nn.functional as F


class Model_1716690305(nn.Layer):
    def __init__(self):
        super(Model_1716690305, self).__init__()
        self.conv1_mutated = paddle.nn.Conv2D(in_channels=3, out_channels=4, kernel_size=[7, 7], stride=[2, 2], padding=[3, 3], dilation=[8, 2], groups=1, bias_attr=None)
        self.relu1 = paddle.nn.ReLU()
        self.pool1_mutated = paddle.nn.MaxPool2D(kernel_size=[3, 6], stride=[2, 2], padding=[0, 0], ceil_mode=True)
        self.conv2_mutated = paddle.nn.Conv2DTranspose(in_channels=4, out_channels=4, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], output_padding=[0, 0], dilation=[1, 1], groups=1, bias_attr=None)
        self.relu2_mutated = paddle.nn.Tanh()
        self.conv3_mutated = paddle.nn.Conv2DTranspose(in_channels=4, out_channels=12, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], output_padding=[0, 0], dilation=[1, 1], groups=1, bias_attr=None)
        self.relu3_mutated = paddle.reciprocal
        self.pool2 = paddle.nn.MaxPool2D(kernel_size=[3, 3], stride=[2, 2], padding=[0, 0], ceil_mode=True)
        self.tail_flatten = paddle.nn.Flatten()
        self.tail_fc = paddle.nn.Linear(in_features=6864, out_features=1000)

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
        model = Model_1716690305()
        x = paddle.randn([1, 3, 224, 224]).astype('float32')
        y = model(x)
        flag = True
    except Exception:
        flag = False
    return flag


def initialize(model):
    module_dir = os.path.dirname(__file__)
    for name, param in model.named_parameters():
        if '_mean' in name or '_variance' in name:
            continue
        layer_name, matrix_name = name.rsplit('.', 1)
        matrix_path = module_dir + '/../initializer/' + layer_name + '/' + matrix_name + '.npz'
        data = np.load(matrix_path)
        tensor = paddle.to_tensor(data['matrix'], dtype='float32', place=param.place)
        if "weight" in matrix_name and 'batchnorm' not in layer_name and 'bn' not in layer_name:
           if data['matrix'].shape == (param.shape[1], param.shape[0]):
               tensor = paddle.to_tensor(data['matrix'].T, dtype='float32', place=param.place)
        param.set_value(tensor)


def train(inp, label):
    model = Model_1716690305().to('cpu')
    initialize(model)
    my_input = paddle.to_tensor(inp).astype('float32').to('cpu')
    output = model(my_input)
    target = paddle.to_tensor(label, dtype='int64').to('cpu')
    loss = nn.CrossEntropyLoss()(output, target)
    loss.backward()
    gradients = {}
    for name, param in model.named_parameters():
        if '_mean' in name or '_variance' in name:
            continue
        if param.grad is not None:
            gradients[name] = param.grad.to('cpu').numpy()
    for key in gradients.keys():
        if len(gradients[key].shape) == 2:
            gradients[key] = gradients[key].T
    return gradients, loss.item(), output.detach().to('cpu').numpy()
