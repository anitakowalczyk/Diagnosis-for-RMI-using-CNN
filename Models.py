from torch import squeeze, nn
import torch
import torchvision.models as models


class CnnBasic(nn.Module):
    def __init__(self):
        super().__init__()
        self.sequential = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=(4, 4), padding=(2, 2)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(64, 128, kernel_size=5, padding=(2, 2)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(256, 256, kernel_size=3, padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(3 * 256, 1)

    def forward(self, axial, sagittal, coronal):
        axial = squeeze(axial, dim=0)
        sagittal = squeeze(sagittal, dim=0)
        coronal = squeeze(coronal, dim=0)

        axial = self.sequential(axial)
        sagittal = self.sequential(sagittal)
        coronal = self.sequential(coronal)

        axial = self.avg_pooling(axial).view(axial.size(0), -1)
        sagittal = self.avg_pooling(sagittal).view(sagittal.size(0), -1)
        coronal = self.avg_pooling(coronal).view(coronal.size(0), -1)

        axial = torch.max(axial, 0, keepdim=True)[0]
        sagittal = torch.max(sagittal, 0, keepdim=True)[0]
        coronal = torch.max(coronal, 0, keepdim=True)[0]

        sample = torch.cat((axial, sagittal, coronal), 1)
        output = self.classifier(sample)
        return output


class CnnReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.sequential = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=(4, 4), padding=(2, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(64, 128, kernel_size=5, padding=(2, 2)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(256, 256, kernel_size=3, padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(3 * 256, 1)

    def forward(self, axial, sagittal, coronal):
        axial = squeeze(axial, dim=0)
        sagittal = squeeze(sagittal, dim=0)
        coronal = squeeze(coronal, dim=0)

        axial = self.sequential(axial)
        sagittal = self.sequential(sagittal)
        coronal = self.sequential(coronal)

        axial = self.avg_pooling(axial).view(axial.size(0), -1)
        sagittal = self.avg_pooling(sagittal).view(sagittal.size(0), -1)
        coronal = self.avg_pooling(coronal).view(coronal.size(0), -1)

        axial = torch.max(axial, 0, keepdim=True)[0]
        sagittal = torch.max(sagittal, 0, keepdim=True)[0]
        coronal = torch.max(coronal, 0, keepdim=True)[0]

        sample = torch.cat((axial, sagittal, coronal), 1)
        output = self.classifier(sample)
        return output


class CnnLeaky(nn.Module):
    def __init__(self):
        super().__init__()
        self.sequential = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=(4, 4), padding=(2, 2)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Dropout(0.2),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(64, 128, kernel_size=5, padding=(2, 2)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Dropout(0.5),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Dropout(0.5),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(256, 256, kernel_size=3, padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Dropout(0.5),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(3 * 256, 1)

    def forward(self, axial, sagittal, coronal):
        axial = squeeze(axial, dim=0)
        sagittal = squeeze(sagittal, dim=0)
        coronal = squeeze(coronal, dim=0)

        axial = self.sequential(axial)
        sagittal = self.sequential(sagittal)
        coronal = self.sequential(coronal)

        axial = self.avg_pooling(axial).view(axial.size(0), -1)
        sagittal = self.avg_pooling(sagittal).view(sagittal.size(0), -1)
        coronal = self.avg_pooling(coronal).view(coronal.size(0), -1)

        axial = torch.max(axial, 0, keepdim=True)[0]
        sagittal = torch.max(sagittal, 0, keepdim=True)[0]
        coronal = torch.max(coronal, 0, keepdim=True)[0]

        sample = torch.cat((axial, sagittal, coronal), 1)
        output = self.classifier(sample)
        return output


class CnnPretrained(nn.Module):
    def __init__(self):
        super().__init__()
        self.alexnet = models.alexnet(pretrained=True)
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(3 * 256, 1)

    def forward(self, axial, sagittal, coronal):
        axial = squeeze(axial, dim=0)
        sagittal = squeeze(sagittal, dim=0)
        coronal = squeeze(coronal, dim=0)

        axial = self.alexnet.features(axial)
        sagittal = self.alexnet.features(sagittal)
        coronal = self.alexnet.features(coronal)

        axial = self.avg_pooling(axial).view(axial.size(0), -1)
        sagittal = self.avg_pooling(sagittal).view(sagittal.size(0), -1)
        coronal = self.avg_pooling(coronal).view(coronal.size(0), -1)

        axial = torch.max(axial, 0, keepdim=True)[0]
        sagittal = torch.max(sagittal, 0, keepdim=True)[0]
        coronal = torch.max(coronal, 0, keepdim=True)[0]

        sample = torch.cat((axial, sagittal, coronal), 1)
        output = self.classifier(sample)
        return output
