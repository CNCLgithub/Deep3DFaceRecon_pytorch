import torch
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

import torchvision.models.resnet as torch_resnet
from torchvision.models.resnet import BasicBlock, Bottleneck
from torchvision.models._utils import _ovewrite_named_param


class ResNet(torch_resnet.ResNet):
    def __init__(self, block, layers, use_last_fc, **kwargs):
        super(ResNet, self).__init__(block, layers, **kwargs)
        self.use_last_fc = use_last_fc

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        if self.use_last_fc:
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x

def _resnet(block, layers, weights, progress, use_last_fc, **kwargs):
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = ResNet(block, layers, use_last_fc, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model

def resnet18(*, weights, progress = True, use_last_fc=True, **kwargs):
    weights = torch_resnet.ResNet18_Weights.verify(weights)
    return _resnet(BasicBlock, [2, 2, 2, 2], weights, progress, use_last_fc, **kwargs)

def resnet50(*, weights, progress = True, use_last_fc=True, **kwargs):
    weights = torch_resnet.ResNet50_Weights.verify(weights)
    return _resnet(Bottleneck, [3, 4, 6, 3], weights, progress, use_last_fc, **kwargs)
