import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, input_layer=3):
        super(ResNet, self).__init__()
        self.inplanes = 8
        self.conv1 = nn.Conv2d(input_layer, 8, kernel_size=3, stride=1, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 8, layers[0])
        self.layer2 = self._make_layer(block, 16, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 32, layers[2], stride=2)
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        return x

def resnet18(input_layers):
    model = ResNet(block=BasicBlock, layers=[3, 3, 3], input_layer=input_layers)
    return model


class Model(nn.Module):
    def __init__(self, input_layer, board_size):
        super(Model, self).__init__()
        self.model = resnet18(input_layers=input_layer)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        # value head
        self.value_conv1 = nn.Conv2d(kernel_size=1, in_channels=32, out_channels=16)
        self.value_bn1 = nn.BatchNorm2d(16)
        self.value_fc1 = nn.Linear(in_features=16 * 4 * 4, out_features=256)
        self.value_fc2 = nn.Linear(in_features=256, out_features=1)
        # policy head
        self.policy_conv1 = nn.Conv2d(kernel_size=1, in_channels=32, out_channels=16)
        self.policy_bn1 = nn.BatchNorm2d(16)
        self.policy_fc1 = nn.Linear(in_features=16 * 4 * 4, out_features=board_size * board_size)

    def forward(self, state):
        s = self.model(state)

        # value head part
        v = self.value_conv1(s)
        v = self.relu(self.value_bn1(v)).view(-1, 16*4*4)
        v = self.relu(self.value_fc1(v))
        value = self.tanh(self.value_fc2(v))

        # policy head part
        p = self.policy_conv1(s)
        p = self.relu(self.policy_bn1(p)).view(-1, 16*4*4)
        prob = self.policy_fc1(p)
        return prob, value

class neuralnetwork:
    def __init__(self, input_layers, board_size, use_cuda=True, learning_rate=0.1):
        self.use_cuda = use_cuda
        if use_cuda:
            self.model = Model(input_layer=input_layers, board_size=board_size).cuda().double()
        else:
            self.model = Model(input_layer=input_layers, board_size=board_size)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)

        self.mse = nn.MSELoss()
        self.crossloss = nn.CrossEntropyLoss()


    def train(self, data_loader, game_time):
        self.model.train()
        for batch_idx, (state, distrib, winner) in enumerate(data_loader):
            state, distrib, winner = Variable(state).unsqueeze(1).double(), Variable(distrib).double(), Variable(winner).unsqueeze(1).double()
            if self.use_cuda:
                state, distrib, winner = state.cuda(), distrib.cuda(), winner.cuda()
            self.opt.zero_grad()
            prob, value = self.model(state)
            output = F.log_softmax(prob, dim=1)

            # loss1 = F.kl_div(output, distrib)
            # loss2 = F.mse_loss(value, winner)
            # loss1.backward(retain_graph=True)
            # loss2.backward()
            cross_entropy = - torch.mean(torch.sum(distrib*output, 1))
            mse = F.mse_loss(value, winner)
            # loss = F.mse_loss(value, winner) - torch.mean(torch.sum(distrib*output, 1))
            loss = cross_entropy + mse
            loss.backward()

            self.opt.step()
            if batch_idx % 20 == 0:
                print("We have played {} games, and batch {}, the cross entropy loss is {}, the mse loss is {}".format(game_time, batch_idx, cross_entropy.data, mse.data))


    def eval(self, state):
        self.model.eval()
        if self.use_cuda:
            state = torch.from_numpy(state).unsqueeze(0).unsqueeze(0).double().cuda()
        else:
            state = torch.from_numpy(state).unsqueeze(0).unsqueeze(0).double()
        with torch.no_grad():
            prob, value = self.model(state)
        return F.softmax(prob, dim=1), value