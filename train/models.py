import torch
import torch.nn as nn


class ID(nn.Module):
    def __init__(self, num_classes=10, dataset="cifar10", target=None):
        super(ID, self).__init__()
        self.dataset = dataset
        self.layer1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.layer3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()
        self.layer4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU()
        self.layer5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.relu5 = nn.ReLU()
        self.up3 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.up2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.layer6 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(192)
        self.relu6 = nn.ReLU()
        self.layer7 = nn.Conv2d(in_channels=192, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm2d(64)
        self.relu7 = nn.ReLU()
        self.layer8 = nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1)
        self.bn8 = nn.BatchNorm2d(96)
        self.relu8 = nn.ReLU()
        self.layer9 = nn.Conv2d(in_channels=96, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.bn9 = nn.BatchNorm2d(3)
        self.relu9 = nn.ReLU()
        self.target = target

    def forward(self, inputs):
        x1 = self.layer1(inputs)
        x1 = self.bn1(x1)
        x1 = self.relu1(x1)
        #         (32,32,32)
        x2 = self.layer2(x1)
        x2 = self.bn2(x2)
        x2 = self.relu2(x2)
        x2 = self.layer3(x2)
        x2 = self.bn3(x2)
        x2 = self.relu3(x2)
        #         (64,16,16)
        x3 = self.layer4(x2)
        x3 = self.bn4(x3)
        x3 = self.relu4(x3)
        x3 = self.layer5(x3)
        x3 = self.bn5(x3)
        x3 = self.relu5(x3)
        #         (128,8,8)
        x3 = self.up3(x3)
        x2 = torch.cat((x2, x3), 1)
        #         (192,16,16)
        x2 = self.layer6(x2)
        x2 = self.bn6(x2)
        x2 = self.relu6(x2)
        x2 = self.layer7(x2)
        x2 = self.bn7(x2)
        x2 = self.relu7(x2)
        #         (64,16,16)
        x2 = self.up2(x2)
        x1 = torch.cat((x1, x2), 1)
        #         (96,32,32)
        x1 = self.layer8(x1)
        x1 = self.bn8(x1)
        x1 = self.relu8(x1)
        x1 = self.layer9(x1)
        x1 = self.bn9(x1)
        noise = self.relu9(x1)
        #         (3,32,32)
        y = torch.add(noise, inputs)
        outputs = self.target(y)
        return outputs

    def freezeTarget(self):
        # 锁定target模型为不可训练
        for name, value in self.target.named_parameters():
            value.requires_grad = False

class HR(nn.Module):

    def __init__(self):
        super(HR, self).__init__()
        self.name = "HiddenDRestorer"
        self.model = self._make_layers()

    def _make_layers(self):
        model = []
        model.append(nn.Linear(512, 256))
        model.append(nn.ReLU())
        model.append(nn.Linear(256, 128))
        model.append(nn.ReLU())
        model.append(nn.Linear(128, 32))
        model.append(nn.ReLU())
        model.append(nn.Linear(32, 128))
        model.append(nn.ReLU())
        model.append(nn.Linear(128, 256))
        model.append(nn.ReLU())
        
        model.append(nn.Linear(256, 512))
        model.append(nn.ReLU())

        return nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

