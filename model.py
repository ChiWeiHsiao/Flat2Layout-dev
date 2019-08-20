import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import functools


ENCODER_RESNET = [
    'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
    'resnext50_32x4d', 'resnext101_32x8d'
]

'''
Encoders
'''
class Resnet(nn.Module):
    def __init__(self, backbone='resnext50_32x4d', pretrained=True, dilate_scale=8):
        super(Resnet, self).__init__()
        assert backbone in ENCODER_RESNET
        if dilate_scale == 8:
            replace_stride_with_dilation = [False, True, True]
        elif dilate_scale == 16:
            replace_stride_with_dilation = [False, False, True]
        elif dilate_scale == 32:
            replace_stride_with_dilation = None
        else:
            raise NotImplementedError()
        self.encoder = getattr(models, backbone)(
            pretrained=pretrained,
            replace_stride_with_dilation=replace_stride_with_dilation
        )
        del self.encoder.fc, self.encoder.avgpool

    def forward(self, x):
        features = []
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = self.encoder.maxpool(x)

        x = self.encoder.layer1(x);  features.append(x)  # 1/4
        x = self.encoder.layer2(x);  features.append(x)  # 1/8
        x = self.encoder.layer3(x);  features.append(x)  # 1/16
        x = self.encoder.layer4(x);  features.append(x)  # 1/32
        return features

    def list_blocks(self):
        lst = [m for m in self.encoder.children()]
        block0 = lst[:4]
        block1 = lst[4:5]
        block2 = lst[5:6]
        block3 = lst[6:7]
        block4 = lst[7:8]
        return block0, block1, block2, block3, block4


'''
Final models
'''
class SimpleFlattenHead(nn.Module):
    def __init__(self, in_low, in_high):
        super(SimpleFlattenHead, self).__init__()
        self.row_attention = nn.Sequential(
            nn.Conv2d(in_high, in_high//8, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_high//8),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_high//8, 1, kernel_size=1)
        )
        self.row_attention[-1].bias.data.fill_(0)

        self.fc = nn.Sequential(
            nn.Conv2d(in_low+in_high, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, kernel_size=1)
        )

    def forward(self, x_low, x_high, w):
        att_high = self.row_attention(x_high)
        att_low = F.interpolate(att_high, size=x_low.shape[2:], mode='bilinear', align_corners=True)
        att_high = F.softmax(att_high, dim=2)
        att_low = F.softmax(att_low, dim=2)

        x_high = (x_high * att_high).sum(2)
        x_low = (x_low * att_low).sum(2)

        x_final = torch.cat([
            F.interpolate(x_high, size=w, mode='linear', align_corners=True),
            F.interpolate(x_low, size=w, mode='linear', align_corners=True)
        ], 1).unsqueeze(2)

        out = self.fc(x_final)

        return out, att_low

class SimpleModel(nn.Module):
    def __init__(self, init_bias=[-0.5, 0.5], backbone='resnext50_32x4d', dilate_scale=8):
        super(SimpleModel, self).__init__()
        # Encoder
        self.encoder = Resnet(backbone, pretrained=True, dilate_scale=dilate_scale)
        with torch.no_grad():
            # Inference the channels of skip connection
            dummy = torch.rand(1, 3, 512, 512)
            in_c = [int(v.shape[1]) for v in self.encoder(dummy)]

        # Simple head
        self.y_c = SimpleFlattenHead(in_c[0], in_c[-1])
        self.y_f = SimpleFlattenHead(in_c[0], in_c[-1])

        self.y_c.fc[-1].bias.data.fill_(init_bias[0])
        self.y_f.fc[-1].bias.data.fill_(init_bias[1])

    def forward(self, x):
        x1, x2, x3, x4 = self.encoder(x)
        y_c, y_c_att = self.y_c(x1, x4, x.shape[3])
        y_f, y_f_att = self.y_f(x1, x4, x.shape[3])
        y_reg = torch.cat([y_c, y_f], 1).squeeze(2)
        y_att = torch.cat([y_c_att, y_f_att], 1)
        return y_reg, y_att


if __name__ == '__main__':
    encoder = Resnet(backbone='resnet50', dilate_scale=8)
    x = torch.rand(4, 3, 512, 512)
    with torch.no_grad():
        for f in encoder(x):
            print(f.shape)
    print()

    net = SimpleModel(backbone='resnext50_32x4d', dilate_scale=8)
    with torch.no_grad():
        print(net(x).shape)
