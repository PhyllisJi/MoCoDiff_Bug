import paddle
import paddle.nn as nn
import numpy as np
import os
import paddle.nn.functional as F


output_dir = "./pytorch-paddle-MobileNet/MobileNet-38-384/case/paddle_gpu/layer_outputs/"
os.makedirs(output_dir, exist_ok=True)
class Model_1716301515(nn.Layer):
    def __init__(self):
        super(Model_1716301515, self).__init__()
        self.conv1_mutated = paddle.nn.Conv2D(in_channels=3, out_channels=3, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias_attr=None)
        self.conv2_mutated = paddle.nn.Conv2D(in_channels=3, out_channels=3, kernel_size=[3, 3], stride=[8, 8], padding=[1, 1], dilation=[1, 1], groups=3, bias_attr=None)
        self.relu1_mutated = paddle.nn.AdaptiveMaxPool2D(output_size=1)
        self.conv3_mutated = paddle.nn.Conv2D(in_channels=3, out_channels=4, kernel_size=[1, 1], stride=[5, 7], padding=[0, 0], dilation=[1, 1], groups=1, bias_attr=None)
        self.relu2 = paddle.nn.ReLU()
        self.conv4_mutated = paddle.nn.Conv2DTranspose(in_channels=4, out_channels=4, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], output_padding=[0, 0], dilation=[1, 1], groups=4, bias_attr=None)
        self.relu3 = paddle.nn.ReLU()
        self.conv5_mutated = paddle.nn.Conv2D(in_channels=4, out_channels=5, kernel_size=[1, 1], stride=[8, 8], padding=[0, 0], dilation=[1, 1], groups=1, bias_attr=None)
        self.relu4_mutated = paddle.floor
        self.conv6_mutated = paddle.nn.Conv2D(in_channels=5, out_channels=5, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=5, bias_attr=None)
        self.relu5 = paddle.nn.ReLU()
        self.conv7_mutated = paddle.nn.Conv2DTranspose(in_channels=5, out_channels=6, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], output_padding=[0, 0], dilation=[1, 1], groups=1, bias_attr=None)
        self.relu6 = paddle.nn.ReLU()
        self.conv8_mutated = paddle.nn.Conv2DTranspose(in_channels=6, out_channels=6, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], output_padding=[0, 0], dilation=[1, 1], groups=6, bias_attr=None)
        self.relu7 = paddle.nn.ReLU()
        self.conv9_mutated = paddle.nn.Conv2DTranspose(in_channels=6, out_channels=7, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], output_padding=[0, 0], dilation=[1, 1], groups=1, bias_attr=None)
        self.relu8 = paddle.nn.ReLU()
        self.conv10_mutated = paddle.nn.Conv2DTranspose(in_channels=7, out_channels=7, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], output_padding=[0, 0], dilation=[1, 1], groups=7, bias_attr=None)
        self.relu9_mutated = paddle.nn.Tanh()
        self.conv11_mutated = paddle.nn.Conv2DTranspose(in_channels=7, out_channels=7, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], output_padding=[0, 0], dilation=[1, 1], groups=1, bias_attr=None)
        self.relu10 = paddle.nn.ReLU()
        self.conv12_mutated = paddle.nn.Conv2DTranspose(in_channels=7, out_channels=7, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], output_padding=[0, 0], dilation=[1, 1], groups=7, bias_attr=None)
        self.relu11 = paddle.nn.ReLU()
        self.conv13_mutated = paddle.nn.Conv2DTranspose(in_channels=7, out_channels=8, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], output_padding=[0, 0], dilation=[1, 1], groups=1, bias_attr=None)
        self.relu12 = paddle.nn.ReLU()
        self.conv14_mutated = paddle.nn.Conv2DTranspose(in_channels=8, out_channels=8, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], output_padding=[0, 0], dilation=[1, 1], groups=8, bias_attr=None)
        self.relu13 = paddle.nn.ReLU()
        self.conv15_mutated = paddle.nn.Conv2DTranspose(in_channels=8, out_channels=8, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], output_padding=[0, 0], dilation=[1, 1], groups=1, bias_attr=None)
        self.relu14 = paddle.nn.ReLU()
        self.conv16_mutated = paddle.nn.Conv2DTranspose(in_channels=8, out_channels=8, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], output_padding=[0, 0], dilation=[1, 1], groups=8, bias_attr=None)
        self.relu15 = paddle.nn.ReLU()
        self.conv17_mutated = paddle.nn.Conv2DTranspose(in_channels=8, out_channels=8, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], output_padding=[0, 0], dilation=[1, 1], groups=1, bias_attr=None)
        self.relu16 = paddle.nn.ReLU()
        self.conv18_mutated = paddle.nn.Conv2DTranspose(in_channels=8, out_channels=8, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], output_padding=[0, 0], dilation=[1, 1], groups=8, bias_attr=None)
        self.relu17 = paddle.nn.ReLU()
        self.conv19_mutated = paddle.nn.Conv2DTranspose(in_channels=8, out_channels=8, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], output_padding=[0, 0], dilation=[1, 1], groups=1, bias_attr=None)
        self.relu18_mutated = paddle.reciprocal
        self.conv20_mutated = paddle.nn.Conv2DTranspose(in_channels=8, out_channels=8, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], output_padding=[0, 0], dilation=[1, 1], groups=8, bias_attr=None)
        self.tail_flatten = paddle.nn.Flatten()
        self.tail_fc = paddle.nn.Linear(in_features=8, out_features=1000)

    def forward(self, input):
        conv1_output = self.conv1_mutated(input)
        conv2_output = self.conv2_mutated(conv1_output)
        relu1_output = self.relu1_mutated(conv2_output)
        conv3_output = self.conv3_mutated(relu1_output)
        relu2_output = self.relu2(conv3_output)
        conv4_output = self.conv4_mutated(relu2_output)
        relu3_output = self.relu3(conv4_output)
        conv5_output = self.conv5_mutated(relu3_output)
        np.savez("./pytorch-paddle-MobileNet/MobileNet-38-384/case/paddle_gpu/layer_outputs/conv5_output.npz",conv5_output.cpu().detach().numpy())
        relu4_output = self.relu4_mutated(conv5_output)
        np.savez("./pytorch-paddle-MobileNet/MobileNet-38-384/case/paddle_gpu/layer_outputs/relu4_output.npz",relu4_output.cpu().detach().numpy())
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
        np.savez("./pytorch-paddle-MobileNet/MobileNet-38-384/case/paddle_gpu/layer_outputs/conv19_output.npz",conv19_output.cpu().detach().numpy())
        relu18_output = self.relu18_mutated(conv19_output)
        np.savez("./pytorch-paddle-MobileNet/MobileNet-38-384/case/paddle_gpu/layer_outputs/relu18_output.npz",relu18_output.cpu().detach().numpy())
        conv20_output = self.conv20_mutated(relu18_output)
        np.savez("./pytorch-paddle-MobileNet/MobileNet-38-384/case/paddle_gpu/layer_outputs/conv20_output.npz",conv20_output.cpu().detach().numpy())
        tail_flatten_output = self.tail_flatten(conv20_output)
        print("tail_flatten_output: ", tail_flatten_output)
        np.savez("./pytorch-paddle-MobileNet/MobileNet-38-384/case/paddle_gpu/layer_outputs/tail_flatten_output.npz",tail_flatten_output.cpu().detach().numpy())
        tail_fc_output = self.tail_fc(tail_flatten_output)
        # print(tail_fc_output.shape)
        np.savez("./pytorch-paddle-MobileNet/MobileNet-38-384/case/paddle_gpu/layer_outputs/tail_fc_output.npz",tail_fc_output.cpu().detach().numpy())
        tail_fc_output = tail_fc_output
        return tail_fc_output


def go():
    try:
        model = Model_1716301515().to('gpu')
        x = paddle.randn([1, 3, 224, 224]).to('gpu')
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
    model = Model_1716301515().to('gpu')
    initialize(model)
    my_input = paddle.to_tensor(inp).astype('float32').to('gpu')
    output = model(my_input)
    target = paddle.to_tensor(label, dtype='int64').to('gpu')
    loss = nn.CrossEntropyLoss()(output, target)
    loss.backward()
    gradients = {}
    for name, param in model.named_parameters():
        if '_mean' in name or '_variance' in name:
            continue
        gradients[name] = param.grad.to('cpu').numpy()
    for key in gradients.keys():
        if len(gradients[key].shape) == 2:
            gradients[key] = gradients[key].T
    return gradients, loss.item(), output.detach().to('cpu').numpy()
inp = np.load("./pytorch-paddle-MobileNet/MobileNet-38-384/case/input.npz")
gradients, loss, output = train(inp['inp'], inp['label'])
# print(output)
np.savez("./pytorch-paddle-MobileNet/MobileNet-38-384/case/paddle_gpu/layer_outputs/output.npz",output)
np.savez("./pytorch-paddle-MobileNet/MobileNet-38-384/case/paddle_gpu/grad-new.npz", **gradients)