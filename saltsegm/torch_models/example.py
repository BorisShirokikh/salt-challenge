import torch.nn as nn


def get_Example():
    model = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False),
        nn.ReLU(),
        nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False),
        nn.ReLU(),
        nn.Conv2d(in_channels=16, out_channels=128, kernel_size=1, stride=1, padding=0, bias=False),
        nn.ReLU(),
        nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False),
    )
    model.cuda()
    return model
