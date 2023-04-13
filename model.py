from torch import nn
import torchvision
import torch
from torch import Tensor

__all__ = [
    "AlexNet",
    "alexnet",
]


class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 4) -> None:
        super(AlexNet, self).__init__()

        self.features_hh = nn.Sequential(
            nn.Conv2d(1, 64, (11, 11), (4, 4), (2, 2)),
            nn.ReLU(True),
            nn.MaxPool2d((3, 3), (2, 2)),

            nn.Conv2d(64, 192, (5, 5), (1, 1), (2, 2)),
            nn.ReLU(True),
            nn.MaxPool2d((3, 3), (2, 2)),

            nn.Conv2d(192, 384, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
            nn.Conv2d(384, 256, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
            nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
            nn.MaxPool2d((3, 3), (2, 2)),
            ########################
            nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
            nn.MaxPool2d((3, 3), (2, 2)),
        )

        self.features_hv = nn.Sequential(
            nn.Conv2d(1, 64, (11, 11), (4, 4), (2, 2)),
            nn.ReLU(True),
            nn.MaxPool2d((3, 3), (2, 2)),

            nn.Conv2d(64, 192, (5, 5), (1, 1), (2, 2)),
            nn.ReLU(True),
            nn.MaxPool2d((3, 3), (2, 2)),

            nn.Conv2d(192, 384, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
            nn.Conv2d(384, 256, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
            nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
            nn.MaxPool2d((3, 3), (2, 2)),
            ###############################
            nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
            nn.MaxPool2d((3, 3), (2, 2)),
        )

        self.features_a_hh = nn.Sequential(
            nn.Conv2d(1, 64, (11, 11), (4, 4), (2, 2)),
            nn.ReLU(True),
            nn.MaxPool2d((3, 3), (2, 2)),

            nn.Conv2d(64, 192, (5, 5), (1, 1), (2, 2)),
            nn.ReLU(True),
            nn.MaxPool2d((3, 3), (2, 2)),

            nn.Conv2d(192, 384, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
            nn.Conv2d(384, 256, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
            nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
            nn.MaxPool2d((3, 3), (2, 2)),
        )

        self.features_a_hv = nn.Sequential(
            nn.Conv2d(1, 64, (11, 11), (4, 4), (2, 2)),
            nn.ReLU(True),
            nn.MaxPool2d((3, 3), (2, 2)),

            nn.Conv2d(64, 192, (5, 5), (1, 1), (2, 2)),
            nn.ReLU(True),
            nn.MaxPool2d((3, 3), (2, 2)),

            nn.Conv2d(192, 384, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
            nn.Conv2d(384, 256, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
            nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
            nn.MaxPool2d((3, 3), (2, 2)),
        )

        self.features_a_hh_hv = nn.Sequential(
            nn.Conv2d(1, 64, (11, 11), (4, 4), (2, 2)),
            nn.ReLU(True),
            nn.MaxPool2d((3, 3), (2, 2)),

            nn.Conv2d(64, 192, (5, 5), (1, 1), (2, 2)),
            nn.ReLU(True),
            nn.MaxPool2d((3, 3), (2, 2)),

            nn.Conv2d(192, 384, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
            nn.Conv2d(384, 256, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
            nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
            nn.MaxPool2d((3, 3), (2, 2)),
        )

        self.features_a_hv_hh = nn.Sequential(
            nn.Conv2d(1, 64, (11, 11), (4, 4), (2, 2)),
            nn.ReLU(True),
            nn.MaxPool2d((3, 3), (2, 2)),

            nn.Conv2d(64, 192, (5, 5), (1, 1), (2, 2)),
            nn.ReLU(True),
            nn.MaxPool2d((3, 3), (2, 2)),

            nn.Conv2d(192, 384, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
            nn.Conv2d(384, 256, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
            nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
            nn.MaxPool2d((3, 3), (2, 2)),
        )


        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1536 * 6 * 6, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, num_classes),
        )
        self.Sigmoid = nn.Sigmoid()
        self.SoftMax = nn.Softmax()

    def forward(self, hh, hv, a_hh, a_hv, a_hh_hv, a_hv_hh):
        out_hh = self.features_hh(hh)
        out_hv = self.features_hv(hv)
        out_a_hh = self.features_a_hh(a_hh)
        out_a_hv = self.features_a_hv(a_hv)
        out_a_hh_hv = self.features_a_hh_hv(a_hh_hv)
        out_a_hv_hh = self.features_a_hv_hh(a_hv_hh)
        #print(hh.shape, out_hh.shape, out_hv.shape, out_a_hh.shape, out_a_hv.shape, out_a_hh_hv.shape, out_a_hv_hh.shape)
        combined = torch.cat((out_hh, out_hv, out_a_hh, out_a_hv, out_a_hh_hv, out_a_hv_hh), dim=1)
        out = self.avgpool(combined)
        #print(out.shape)
        #out = self.features(x)
        #out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.classifier(out)

        return self.Sigmoid(out)

class AlexNet64(nn.Module):
    def __init__(self, num_classes: int = 4) -> None:
        super(AlexNet64, self).__init__()

        self.features_hh = nn.Sequential(
            nn.Conv2d(1, 32, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
            nn.MaxPool2d((2, 2), (2, 2), (1,1)),

            nn.Conv2d(32, 96, (5, 5), (1, 1), (2, 2)),
            nn.ReLU(True),
            nn.MaxPool2d((2, 2), (2, 2)),

            nn.Conv2d(96, 192, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
            nn.Conv2d(192, 128, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
            nn.Conv2d(128, 32, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
            nn.MaxPool2d((2, 2), (2, 2)),            
        )

        self.features_hv = nn.Sequential(
            nn.Conv2d(1, 32, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
            nn.MaxPool2d((2, 2), (2, 2), (1,1)),

            nn.Conv2d(32, 96, (5, 5), (1, 1), (2, 2)),
            nn.ReLU(True),
            nn.MaxPool2d((2, 2), (2, 2)),

            nn.Conv2d(96, 192, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
            nn.Conv2d(192, 128, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
            nn.Conv2d(128, 32, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
            nn.MaxPool2d((2, 2), (2, 2)), 
        )

        self.features_a_hh = nn.Sequential(
            nn.Conv2d(1, 96, (5, 5), (1, 1), (2, 2)),
            nn.ReLU(True),
            nn.MaxPool2d((2, 2), (2, 2)),

            nn.Conv2d(96, 192, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
            nn.Conv2d(192, 128, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
            nn.Conv2d(128, 32, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
            nn.MaxPool2d((2, 2), (2, 2)), 
        )

        self.features_a_hv = nn.Sequential(
            nn.Conv2d(1, 96, (5, 5), (1, 1), (2, 2)),
            nn.ReLU(True),
            nn.MaxPool2d((2, 2), (2, 2)),

            nn.Conv2d(96, 192, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
            nn.Conv2d(192, 128, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
            nn.Conv2d(128, 32, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
            nn.MaxPool2d((2, 2), (2, 2)), 
        )

        self.features_a_hh_hv = nn.Sequential(
            nn.Conv2d(1, 96, (5, 5), (1, 1), (2, 2)),
            nn.ReLU(True),
            nn.MaxPool2d((2, 2), (2, 2)),

            nn.Conv2d(96, 192, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
            nn.Conv2d(192, 128, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
            nn.Conv2d(128, 32, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
            nn.MaxPool2d((2, 2), (2, 2)), 
        )

        self.features_a_hv_hh = nn.Sequential(
            nn.Conv2d(1, 96, (5, 5), (1, 1), (2, 2)),
            nn.ReLU(True),
            nn.MaxPool2d((2, 2), (2, 2)),

            nn.Conv2d(96, 192, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
            nn.Conv2d(192, 128, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
            nn.Conv2d(128, 32, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
            nn.MaxPool2d((2, 2), (2, 2)), 
        )


        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(6912, 512),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(True),
            nn.Linear(128, num_classes),
        )
        self.Sigmoid = nn.Sigmoid()
        self.SoftMax = nn.Softmax()

    def forward(self, hh, hv, a_hh, a_hv, a_hh_hv, a_hv_hh):
        out_hh = self.features_hh(hh)
        out_hv = self.features_hv(hv)
        out_a_hh = self.features_a_hh(a_hh)
        out_a_hv = self.features_a_hv(a_hv)
        out_a_hh_hv = self.features_a_hh_hv(a_hh_hv)
        out_a_hv_hh = self.features_a_hv_hh(a_hv_hh)

        #print(out_hh.shape, out_hv.shape, out_a_hh.shape, out_a_hv.shape, out_a_hh_hv.shape, out_a_hv_hh.shape)
        combined = torch.cat((out_hh, out_hv, out_a_hh, out_a_hv, out_a_hh_hv, out_a_hv_hh), dim=1)
        out = self.avgpool(combined)

        #out = self.features(x)
        #out = self.avgpool(out)
        out = torch.flatten(out, 1)
        #print(out.shape)
        out = self.classifier(out)

        return self.Sigmoid(out)


class AlexNetNOT(nn.Module):
    def __init__(self, num_classes: int = 4) -> None:
        super(AlexNetNOT, self).__init__()

        self.features_hh = nn.Sequential(
            nn.Conv2d(1, 64, (11, 11), (4, 4), (2, 2)),
            nn.ReLU(True),
            nn.MaxPool2d((3, 3), (2, 2)),

            nn.Conv2d(64, 192, (5, 5), (1, 1), (2, 2)),
            nn.ReLU(True),
            nn.MaxPool2d((3, 3), (2, 2)),

            nn.Conv2d(192, 384, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
            nn.Conv2d(384, 256, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
            nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
            nn.MaxPool2d((3, 3), (2, 2)),
            ########################
            nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
            nn.MaxPool2d((3, 3), (2, 2)),
        )

        self.features_hv = nn.Sequential(
            nn.Conv2d(1, 64, (11, 11), (4, 4), (2, 2)),
            nn.ReLU(True),
            nn.MaxPool2d((3, 3), (2, 2)),

            nn.Conv2d(64, 192, (5, 5), (1, 1), (2, 2)),
            nn.ReLU(True),
            nn.MaxPool2d((3, 3), (2, 2)),

            nn.Conv2d(192, 384, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
            nn.Conv2d(384, 256, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
            nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
            nn.MaxPool2d((3, 3), (2, 2)),
            ###############################
            nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
            nn.MaxPool2d((3, 3), (2, 2)),
        )

        self.features_a_hh = nn.Sequential(
            nn.Conv2d(1, 64, (11, 11), (4, 4), (2, 2)),
            nn.ReLU(True),
            nn.MaxPool2d((3, 3), (2, 2)),

            nn.Conv2d(64, 192, (5, 5), (1, 1), (2, 2)),
            nn.ReLU(True),
            nn.MaxPool2d((3, 3), (2, 2)),

            nn.Conv2d(192, 384, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
            nn.Conv2d(384, 256, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
            nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
            nn.MaxPool2d((3, 3), (2, 2)),
        )

        self.features_a_hv = nn.Sequential(
            nn.Conv2d(1, 64, (11, 11), (4, 4), (2, 2)),
            nn.ReLU(True),
            nn.MaxPool2d((3, 3), (2, 2)),

            nn.Conv2d(64, 192, (5, 5), (1, 1), (2, 2)),
            nn.ReLU(True),
            nn.MaxPool2d((3, 3), (2, 2)),

            nn.Conv2d(192, 384, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
            nn.Conv2d(384, 256, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
            nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
            nn.MaxPool2d((3, 3), (2, 2)),
        )

        self.features_a_hh_hv = nn.Sequential(
            nn.Conv2d(1, 64, (11, 11), (4, 4), (2, 2)),
            nn.ReLU(True),
            nn.MaxPool2d((3, 3), (2, 2)),

            nn.Conv2d(64, 192, (5, 5), (1, 1), (2, 2)),
            nn.ReLU(True),
            nn.MaxPool2d((3, 3), (2, 2)),

            nn.Conv2d(192, 384, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
            nn.Conv2d(384, 256, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
            nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
            nn.MaxPool2d((3, 3), (2, 2)),
        )

        self.features_a_hv_hh = nn.Sequential(
            nn.Conv2d(1, 64, (11, 11), (4, 4), (2, 2)),
            nn.ReLU(True),
            nn.MaxPool2d((3, 3), (2, 2)),

            nn.Conv2d(64, 192, (5, 5), (1, 1), (2, 2)),
            nn.ReLU(True),
            nn.MaxPool2d((3, 3), (2, 2)),

            nn.Conv2d(192, 384, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
            nn.Conv2d(384, 256, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
            nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
            nn.MaxPool2d((3, 3), (2, 2)),
        )


        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512* 6 * 6, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, num_classes),
        )
        self.Sigmoid = nn.Sigmoid()
        self.SoftMax = nn.Softmax()

    def forward(self, hh, hv, a_hh, a_hv, a_hh_hv, a_hv_hh):
        out_hh = self.features_hh(hh)
        out_hv = self.features_hv(hv)
        #out_a_hh = self.features_a_hh(a_hh)
        #out_a_hv = self.features_a_hv(a_hv)
        #out_a_hh_hv = self.features_a_hh_hv(a_hh_hv)
        #out_a_hv_hh = self.features_a_hv_hh(a_hv_hh)
        #print(hh.shape, out_hh.shape, out_hv.shape, out_a_hh.shape, out_a_hv.shape, out_a_hh_hv.shape, out_a_hv_hh.shape)
        combined = torch.cat((out_hh, out_hv), dim=1)
        out = self.avgpool(combined)
        #print(out.shape)
        #out = self.features(x)
        #out = self.avgpool(out)
        #print(out.shape)
        out = torch.flatten(out, 1)
        out = self.classifier(out)

        return self.Sigmoid(out)



def get_model():
    model = AlexNetNOT()

    return model
