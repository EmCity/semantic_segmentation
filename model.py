import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class SegmentationNN(nn.Module):
    def __init__(self, pre_model="vgg11", num_classes=23):
        super(SegmentationNN, self).__init__()
        self.num_classes = num_classes

        if pre_model == "vgg11":
            self.vgg_feat = models.vgg11(pretrained=True).features
        elif pre_model == "vgg16":
            self.vgg_feat = models.vgg16(pretrained=True).features
        elif pre_model == "vgg19":
            self.vgg_feat = models.vgg19(pretrained=True).features
        else:
            self.vgg_feat = models.vgg11(pretrained=True).features

        self.fcn = nn.Sequential(
                                nn.Conv2d(512, 1024, 7),
                                nn.ReLU(inplace=True),
                                nn.Dropout(),
                                nn.Conv2d(1024, 2048, 1),
                                nn.ReLU(inplace=True),
                                nn.Dropout(),
                                nn.Conv2d(2048, num_classes, 1)
                                )

    def forward(self, x):
        x_input = x
        x = self.vgg_feat(x)
        x = self.fcn(x)
        x = F.upsample(x, x_input.size()[2:], mode='bilinear').contiguous()

        return x

    @property
    def is_cuda(self):
        return next(self.parameters()).is_cuda

    def save(self, path):
        print('Saving model... %s' % path)
        torch.save(self, path)
