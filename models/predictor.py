import torch
import torch.nn as nn
import torchvision
class Predict(nn.Module):
    def __init__(self):
        super(Predict, self).__init__()
        self.net = nn.Sequential(
            *list(torchvision.models.resnet50(pretrained=True).children())[:-1]
        )
        self.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 17),
            nn.ReLU()
        )

    def forward(self, img):
        res = self.net(img)
        res = res.view(img.size(0), -1)
        return self.fc(res)