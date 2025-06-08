import math
import torch
import torch.nn as nn
import torchvision.models as models

means = [100, 100, 100] 

def mobilenet(pretrained=True):
    """
    Constructs a MobileNetV2 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = models.mobilenet_v2(pretrained=pretrained)
    return model

def mean_image_subtraction(images, means=means):
    '''
    image normalization
    :param images: bs * w * h * channel
    :param means:
    :return:
    '''
    num_channels = images.data.shape[1]
    if len(means) != num_channels:
        raise ValueError('len(means) must match the number of channels')
    for i in range(num_channels):
        images.data[:, i, :, :] -= means[i]

    return images

class East(nn.Module):
    def __init__(self):
        super(East, self).__init__()
        self.mobilenet = mobilenet(True)
        # self.si for stage i
        self.s1 = nn.Sequential(*list(self.mobilenet.children())[0][0:4])
        self.s2 = nn.Sequential(*list(self.mobilenet.children())[0][4:7])
        self.s3 = nn.Sequential(*list(self.mobilenet.children())[0][7:14])
        self.s4 = nn.Sequential(*list(self.mobilenet.children())[0][14:17])

        self.conv1 = nn.Conv2d(160+96, 128, 1)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(128+32, 64, 1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU()

        self.conv5 = nn.Conv2d(64+24, 64, 1)
        self.bn5 = nn.BatchNorm2d(64)
        self.relu5 = nn.ReLU()

        self.conv6 = nn.Conv2d(64, 32, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(32)
        self.relu6 = nn.ReLU()

        self.conv7 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn7 = nn.BatchNorm2d(32)
        self.relu7 = nn.ReLU()

        self.conv8 = nn.Conv2d(32, 1, 1)
        self.sigmoid1 = nn.Sigmoid()
        self.conv9 = nn.Conv2d(32, 4, 1)
        self.sigmoid2 = nn.Sigmoid()
        self.conv10 = nn.Conv2d(32, 1, 1)
        self.sigmoid3 = nn.Sigmoid()
        self.unpool1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.unpool2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.unpool3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, images):
        images = mean_image_subtraction(images)

        f0 = self.s1(images)
        f1 = self.s2(f0)
        f2 = self.s3(f1)
        f3 = self.s4(f2)

        h = f3  
        g = (self.unpool1(h))
        c = self.conv1(torch.cat((g, f2), 1))
        c = self.bn1(c)
        c = self.relu1(c)

        h = self.conv2(c) 
        h = self.bn2(h)
        h = self.relu2(h)
        g = self.unpool2(h) 
        c = self.conv3(torch.cat((g, f1), 1))
        c = self.bn3(c)
        c = self.relu3(c)

        h = self.conv4(c) 
        h = self.bn4(h)
        h = self.relu4(h)
        g = self.unpool3(h) 
        c = self.conv5(torch.cat((g, f0), 1))
        c = self.bn5(c)
        c = self.relu5(c)

        h = self.conv6(c) 
        h = self.bn6(h)
        h = self.relu6(h)
        g = self.conv7(h) 
        g = self.bn7(g)
        g = self.relu7(g)

        F_score = self.conv8(g) 
        F_score = self.sigmoid1(F_score)
        geo_map = self.conv9(g)
        geo_map = self.sigmoid2(geo_map) * 512
        angle_map = self.conv10(g)
        angle_map = self.sigmoid3(angle_map)
        angle_map = (angle_map - 0.5) * math.pi / 2

        F_geometry = torch.cat((geo_map, angle_map), 1) 

        return F_score, F_geometry