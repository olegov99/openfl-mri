import torch
from torch import nn
import torchvision.models as models
from torch.nn import functional as F

import constants

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.map = nn.Conv2d(in_channels=4, out_channels=1, kernel_size=1)
        self.map1 = nn.Conv2d(in_channels=4, out_channels=1, kernel_size=1)
        self.map2 = nn.Conv2d(in_channels=4, out_channels=1, kernel_size=1)
        self.map3 = nn.Conv2d(in_channels=4, out_channels=1, kernel_size=1)
        self.FT = nn.Conv2d(in_channels=4, out_channels=3, kernel_size=1)
        # self.net = efficientnet_pytorch.EfficientNet.from_name("efficientnet-b0")
        # checkpoint = torch.load("./input/efficientnet-pytorch/efficientnet-b0-08094119.pth")
        # self.net.load_state_dict(checkpoint)
        self.net = models.resnet34(pretrained=False)
        n_features = self.net.fc.in_features
        self.net.fc = nn.Linear(in_features=n_features, out_features=constants.CNN_FEATURES, bias=True)

    def forward(self, x):
        x1 = F.relu(self.map(x))
        x2 = F.relu(self.map1(x))
        x3 = F.relu(self.map2(x))
        x4 = F.relu(self.map3(x))
        x_cat = torch.cat((x1, x2, x3, x4), dim=1)
        net_in = F.relu(self.FT(x_cat))
        out = self.net(net_in)
        return out


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.cnn = CNN()
        self.rnn = nn.LSTM(constants.CNN_FEATURES, constants.LSTM_HIDDEN, 2, batch_first=True)
        self.fc = nn.Linear(constants.LSTM_HIDDEN, 1, bias=True)

    def forward(self, x):
        # x shape: BxTxCxHxW
        batch_size, timesteps, C, H, W = x.size()
        c_in = x.view(batch_size * timesteps, C, H, W)
        c_out = self.cnn(c_in)
        r_in = c_out.view(batch_size, timesteps, -1)
        output, (hn, cn) = self.rnn(r_in)

        out = self.fc(hn[-1])
        return out
