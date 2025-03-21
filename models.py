import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange, Reduce

import pretrainedmodels
import pretrainedmodels.utils as utils

from fabulous.color import fg256

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#alexnet = pretrainedmodels.__dict__['alexnet'](num_classes=1000, pretrained=None).cuda()
#resnet  = pretrainedmodels.__dict__['resnet18'](num_classes=1000, pretrained=None).cuda()
alexnet = pretrainedmodels.__dict__['alexnet'](num_classes=1000, pretrained='imagenet').cuda()
resnet  = pretrainedmodels.__dict__['resnet18'](num_classes=1000, pretrained='imagenet').cuda()
print(fg256("green", 'Successfully loaded INet weights.'))

def get_encoder():
    return Encoder_AL()

def get_regressor():
    return Regressor_AL(64)

def get_task_header():
    return ERM_FC(64, 2)

def nn_output():
    encoder2 = get_encoder().to(device)
    regressor2 = get_regressor().to(device)
    task_header2 = get_task_header().to(device)
    return encoder2, regressor2, task_header2

class Encoder_AL(nn.Module):
    def __init__(self):
        super(Encoder_AL, self).__init__()
        self.features = alexnet._features
        self.avgpool = alexnet.avgpool

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(self.avgpool(x), 1)
        return x


class Encoder_R18(nn.Module):

    def __init__(self):
        super(Encoder_R18, self).__init__()

        self.conv1 = resnet.conv1
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        self.last_linear = resnet.last_linear

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)  # shape: [1, 512, 8, 8]

        x = torch.flatten(self.avgpool(x), 1)
        x = self.last_linear(x)  # [1000]
        return x


class Regressor_AL(nn.Module):

    def __init__(self, no_domain):
        super(Regressor_AL, self).__init__()
        self.avgpool = alexnet.avgpool
        self.lin0 = nn.Linear(9216, 256)
        self.lin1 = nn.Linear(256, no_domain)
        self.relu0 = alexnet.relu0
        self.relu1 = alexnet.relu1
        self.drop0 = alexnet.dropout0
        self.drop1 = alexnet.dropout0
        self.bn = nn.BatchNorm1d(no_domain, affine=True)

    def forward(self, x):
        x = self.relu0(self.lin0(self.drop0(x)))
        latent_variable = self.relu1(self.bn(self.lin1(self.drop1(x))))
        return latent_variable


class Regressor_R18(nn.Module):

    def __init__(self, no_domain):
        super(Regressor_R18, self).__init__()
        self.lin0 = nn.Linear(1000, 512)
        self.lin1 = nn.Linear(512, 256)
        self.lin2 = nn.Linear(256, no_domain)
        self.bn1  = nn.BatchNorm1d(256, affine=True)
        self.bn2  = nn.BatchNorm1d(no_domain, affine=True)

    def forward(self, x):
        x = F.relu(self.lin0(F.dropout2d(x)))
        x = F.relu(self.bn1(self.lin1(F.dropout2d(x))))
        latent_variable = F.relu(self.bn2(self.lin2(F.dropout2d(x))))
        return latent_variable


class ERM_FC(nn.Module):

    def __init__(self, erm_input_dim, erm_output_dim):
        super(ERM_FC, self).__init__()

        self.erm_input_dim = erm_input_dim
        self.erm_hidden_dim = self.erm_input_dim * 20  #10
        self.erm_output_dim = erm_output_dim
        self.linear1 = nn.Linear(self.erm_input_dim, self.erm_hidden_dim)
        self.linear2 = nn.Linear(self.erm_hidden_dim, self.erm_output_dim)
        self.bn1 = nn.BatchNorm1d(self.erm_hidden_dim, affine=True)

        self.layer_blocks = nn.Sequential(
            self.linear1,
            self.bn1,
            nn.ReLU(inplace=True),
            self.linear2,
        )

    def forward(self, inputs, train=True):
        if train:
            return self.layer_blocks(inputs)
        else:
            x = self.layer_blocks(inputs)
            x = torch.clamp(x, min=-1., max=1.)
            return F.tanh(x)


def encoder_AL():
    encoder = Encoder_AL()
    return encoder
def encoder_R18():
    encoder = Encoder_R18()
    return encoder

def regressor_AL(latent_dim):
    domainregressor = Regressor_AL(latent_dim)
    return domainregressor
def regressor_R18(latent_dim):
    domainregressor = Regressor_R18(latent_dim)
    return domainregressor

def load_ERM_FC(erm_input_dim, erm_output_dim):
    erm_fc = ERM_FC(erm_input_dim, erm_output_dim)
    return erm_fc


if __name__ == "__main__":
    from pytorch_model_summary import summary

    # Encoder_AL test
    print(fg256("cyan", summary(
        Encoder_AL().to(device),
        torch.ones_like(torch.empty(1, 3, 255, 255)).to(device),
        show_input=True
    )))

    # Regressor_AL test
    print(fg256("orange", summary(
        Regressor_AL(1).to(device),
        torch.ones_like(torch.empty(1, 256 * 6 * 6)).to(device), 
        show_input=True
    )))

    # ERM_FC test
    print(fg256("yellow", "ERM_FC", summary(
        ERM_FC(64, 2).to(device),
        torch.ones_like(torch.empty(10, 64)).to(device), 
        show_input=True
    )))

    # Encoder_R18 test
    print(fg256("magenta", summary(
        Encoder_R18().to(device),
        torch.ones_like(torch.empty(1, 3, 255, 255)).to(device), 
        show_input=True
    )))

    # Regressor_R18 test
    print(fg256("blue", summary(
        Regressor_R18(1).to(device),
        torch.ones_like(torch.empty(1, 1000)).to(device), 
        show_input=True
    )))

