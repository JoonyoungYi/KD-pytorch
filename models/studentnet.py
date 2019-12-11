import torch
import torch.nn as nn


class StudentNet(nn.Module):
    def __init__(self):
        super(StudentNet, self).__init__()
        self.features = self._make_layers([64, 'M', 64, 'M', 64, 'M', 64, 'M'])
        self.classifier = nn.Linear(256, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [
                    nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                    nn.BatchNorm2d(x),
                    nn.ReLU(inplace=True)
                ]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def test():
    net = StudentNet()
    x = torch.randn(2, 3, 32, 32)
    y = net(x)
    print(y.size())


if __name__ == '__main__':
    test()
