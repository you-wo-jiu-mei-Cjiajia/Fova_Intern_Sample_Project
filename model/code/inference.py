import json
import logging
import os
import requests

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride,
                      padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim,
                       stride=stride, groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=10, width_mult=1.0):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280

        # CIFAR10
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 1],  # Stride 2 -> 1 for CIFAR-10
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        # END

        # building first layer
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * max(1.0, width_mult))

        # CIFAR10: stride 2 -> 1
        features = [ConvBNReLU(3, input_channel, stride=1)]
        # END

        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(
                    block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNReLU(
            input_channel, self.last_channel, kernel_size=1))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = x.mean([2, 3])
        x = self.classifier(x)
        return x


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def model_fn(model_dir):
    """
    Loads the pretrained mobilenetv2 model, works as the entry point for the sagemaker API

    Args:
        model_dir (string): the absolute path of the directory holding the saved .pt
                            file of the loading model

    Rets:
        (DataParallel): the pretrained and loaded mobilenetv2 model
    """
    logger.info('Loading model...')

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MobileNetV2()

    # if torch.cuda.device_count() > 1:
    #     logger.info("Gpu count: {}".format(torch.cuda.device_count()))
    #     model = nn.DataParallel(model)

    # logger.info('Current device: {}'.format(device))

    with open(os.path.join(model_dir, 'model.pt'), 'rb') as f:
        model.load_state_dict(torch.load(f))
    model.to(device).eval()

    logger.info('Loading complete.')

    return model.to(device)


def input_fn(request_body, request_content_type):
    """
    Deserializes JSON encoded data into a torch.Tensor

    Args:
        request_body (buffer): a single json list compatible with the loaded model
                               or a PIL image object
        requested_content_type (str): specifies input data type

    Rets:
        (Compose): A transformed tensor ready to be passed to predict_fn
    """
    logger.info('Deserializing the input data...')
    if request_content_type == 'application/json':
        input_data = json.loads(request_body)
        url = input_data['url']
        logger.info('Loading image: %s', url)
        image_data = Image.open(requests.get(url, stream=True).raw)

    elif request_content_type == 'image/*':
        image_data = request_body

    else:
        raise Exception('Unsupported input type')

    # normalize the image data
    image_transform = transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465],
                             [0.2023, 0.1994, 0.2010])
    ])

    return image_transform(image_data)


def predict_fn(input_data, model):
    """
    Takes the input object and performs an inference agianst the loaded model.

    Args:
        input_data (Compose): a transformed tensor object from input_fn
        model (DataParallel): a pretrained and loaded mobilenetv2 model from model_fn

    Rets:
        (Tensor): a torch.Tensor object containing the predition
    """
    logger.info('Performing inference based on the input parameter...')
    if torch.cuda.is_available():
        input_data = input_data.view(1, 3, 224, 224).cuda()
    else:
        input_data = input_data.view(1, 3, 224, 224)

    with torch.no_grad():
        model.eval()
        output = model(input_data)
        prediction = torch.exp(output)

    return prediction


def output_fn(prediction, response_content_type='application/json'):
    """
    Serialize the prediction result into the desired response content type

    Args:
        prediction (Tensor): a torch.Tensor object representing the prediction from predict_fn
        response_content_type (str): specifies desired output data

    Rets:
        (str): a json formatted string from the prediction tensor object
    """
    logger.info('Serializing the output...')
    classes = {0: 'plane', 1: 'car', 2: 'bird', 3: 'cat', 4: 'deer',
               5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}

    topk, topclass = prediction.topk(3, dim=1)
    result = []

    for i in range(3):
        pred = {'prediction': classes[topclass.cpu().numpy(
        )[0][i]], 'score': f'{topk.cpu().numpy()[0][i] * 100}%'}
        logger.info('Adding prediction: %s', pred)
        result.append(pred)

    if response_content_type == 'application/json':
        return json.dumps(result), response_content_type
    raise Exception('Unsupported output type.')
