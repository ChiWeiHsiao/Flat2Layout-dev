import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import functools

import model_ADE20k_encoder


ENCODER_RESNET = [
    'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
    'resnext50_32x4d', 'resnext101_32x8d',
    'ade20k_resnet50'
]
ENCODER_DENSENET = [
    'densenet121', 'densenet169', 'densenet161', 'densenet201'
]


'''
Encoder
'''
class Resnet(nn.Module):
    def __init__(self, backbone='resnet50', pretrained=True):
        super(Resnet, self).__init__()
        assert backbone in ENCODER_RESNET
        self.encoder = getattr(models, backbone)(pretrained=pretrained)
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


class Densenet(nn.Module):
    def __init__(self, backbone='densenet169', pretrained=True):
        super(Densenet, self).__init__()
        assert backbone in ENCODER_DENSENET
        self.encoder = getattr(models, backbone)(pretrained=pretrained)
        self.final_relu = nn.ReLU(inplace=True)
        del self.encoder.classifier

    def forward(self, x):
        lst = []
        for m in self.encoder.features.children():
            x = m(x)
            lst.append(x)
        features = [lst[4], lst[6], lst[8], self.final_relu(lst[11])]
        return features

    def list_blocks(self):
        lst = [m for m in self.encoder.features.children()]
        block0 = lst[:4]
        block1 = lst[4:6]
        block2 = lst[6:8]
        block3 = lst[8:10]
        block4 = lst[10:]
        return block0, block1, block2, block3, block4


'''
Decoder
'''
class ConvCompressH(nn.Module):
    ''' Reduce feature height by factor of two '''
    def __init__(self, in_c, out_c, ks=3):
        super(ConvCompressH, self).__init__()
        assert ks % 2 == 1
        self.layers = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=ks, stride=(2, 1), padding=ks//2),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )
        n = self.layers[0].kernel_size[0] * self.layers[0].kernel_size[1] * self.layers[0].out_channels
        self.layers[0].weight.data.normal_(0, np.sqrt(2. / n))
        self.layers[1].weight.data.fill_(1)
        self.layers[1].bias.data.zero_()

    def forward(self, x):
        return self.layers(x)


class GlobalHeightConv(nn.Module):
    def __init__(self, in_c, out_c):
        super(GlobalHeightConv, self).__init__()
        self.layer = nn.Sequential(
            ConvCompressH(in_c, in_c//2),
            ConvCompressH(in_c//2, in_c//2),
            ConvCompressH(in_c//2, in_c//4),
            ConvCompressH(in_c//4, out_c),
        )

    def forward(self, x, out_w):
        x = self.layer(x)

        assert out_w % x.shape[3] == 0
        factor = out_w // x.shape[3]
        x = torch.cat([x[..., -1:], x, x[..., :1]], 3)
        x = F.interpolate(x, size=(x.shape[2], out_w + 2 * factor), mode='bilinear', align_corners=False)
        x = x[..., factor:-factor]
        return x


class GlobalHeightStage(nn.Module):
    def __init__(self, c1, c2, c3, c4, out_scale=8):
        ''' Process 4 blocks from encoder to single multiscale features '''
        super(GlobalHeightStage, self).__init__()
        self.out_scale = out_scale
        self.ghc_lst = nn.ModuleList([
            GlobalHeightConv(c1, c1//out_scale),
            GlobalHeightConv(c2, c2//out_scale),
            GlobalHeightConv(c3, c3//out_scale),
            GlobalHeightConv(c4, c4//out_scale),
        ])

    def forward(self, conv_list, out_w):
        assert len(conv_list) == 4
        bs = conv_list[0].shape[0]
        feature = torch.cat([
            f(x, out_w).reshape(bs, -1, out_w)
            for f, x in zip(self.ghc_lst, conv_list)
        ], dim=1)
        return feature


'''
HorizonNet
'''
class HorizonNet(nn.Module):
    def __init__(self, backbone, use_rnn=True, init_bias=[-0.5, 0.5], drop_p=0.5):
        super(HorizonNet, self).__init__()
        self.out_c = 2  # 2=y1,y2 3=y1,y2,c
        self.backbone = backbone
        self.use_rnn = use_rnn
        self.out_scale = 8
        self.step_cols = 4
        self.rnn_hidden_size = 512
        self.drop_p = drop_p

        # Encoder
        if backbone.startswith('res'):
            self.feature_extractor = Resnet(backbone, pretrained=True)
        elif backbone.startswith('dense'):
            self.feature_extractor = Densenet(backbone, pretrained=True)
        else:
            raise NotImplementedError()

        # Inference channels number from each block of the encoder
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 512, 1024)
            c1, c2, c3, c4 = [b.shape[1] for b in self.feature_extractor(dummy)]
            c_last = (c1*8 + c2*4 + c3*2 + c4*1) // self.out_scale

        # Convert features from 4 blocks of the encoder into B x C x 1 x W'
        self.reduce_height_module = GlobalHeightStage(c1, c2, c3, c4, self.out_scale)

        # 1D prediction
        if self.use_rnn:
            self.bi_rnn = nn.LSTM(input_size=c_last,
                                  hidden_size=self.rnn_hidden_size,
                                  num_layers=2,
                                  dropout=self.drop_p,
                                  batch_first=False,
                                  bidirectional=True)
            self.drop_out = nn.Dropout(self.drop_p)
            self.linear = nn.Linear(in_features=2 * self.rnn_hidden_size,
                                    out_features=self.out_c * self.step_cols)
            self.linear.bias.data[0*self.step_cols:1*self.step_cols].fill_(init_bias[0])
            self.linear.bias.data[1*self.step_cols:2*self.step_cols].fill_(init_bias[1])
            if self.out_c == 3:
                self.linear.bias.data[2*self.step_cols:3*self.step_cols].fill_(init_bias[2])  # c:-1
        else:
            self.linear = nn.Sequential(
                nn.Linear(c_last, self.rnn_hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(self.rnn_hidden_size, self.out_c * self.step_cols),
            )
            self.linear[-1].bias.data[0*self.step_cols:1*self.step_cols].fill_(init_bias[0])
            self.linear[-1].bias.data[1*self.step_cols:2*self.step_cols].fill_(init_bias[1])
            if self.out_c == 3:
                self.linear[-1].bias.data[2*self.step_cols:3*self.step_cols].fill_(init_bias[2])  # c:-1

    def forward(self, x):
        conv_list = self.feature_extractor(x)
        feature = self.reduce_height_module(conv_list, x.shape[3]//self.step_cols)

        # rnn
        if self.use_rnn:
            feature = feature.permute(2, 0, 1)  # [w, b, c*h]
            output, hidden = self.bi_rnn(feature)  # [seq_len, b, num_directions * hidden_size]
            output = self.drop_out(output)
            output = self.linear(output)  # [seq_len, b, 3 * step_cols]
            output = output.view(output.shape[0], output.shape[1], self.out_c, self.step_cols)  # [seq_len, b, 3, step_cols]
            output = output.permute(1, 2, 0, 3)  # [b, 3, seq_len, step_cols]
            output = output.contiguous().view(output.shape[0], self.out_c, -1)  # [b, 3, seq_len*step_cols]
        else:
            feature = feature.permute(0, 2, 1)  # [b, w, c*h]
            output = self.linear(feature)  # [b, w, 3 * step_cols]
            output = output.view(output.shape[0], output.shape[1], self.out_c, self.step_cols)  # [b, w, 3, step_cols]
            output = output.permute(0, 2, 1, 3)  # [b, 3, w, step_cols]
            output = output.contiguous().view(output.shape[0], self.out_c, -1)  # [b, 3, w*step_cols]

        return output
        #  # output.shape => B x 3 x W
        #  cor = output[:, :1]  # B x 1 x W
        #  bon = output[:, 1:]  # B x 2 x W

        #  return bon, cor


class LowResHorizonNet(nn.Module):
    def __init__(self, backbone, use_rnn=True, pred_cor=False, pred_key=False,
                 init_bias=[-0.5, 0.5, -3, -3, -3, -3],
                 bn_momentum=None,
                 branches=1,
                 finetune_cor=0,
                 gray_mode=0,
                 drop_p=0.5):
        super(LowResHorizonNet, self).__init__()
        assert not use_rnn or branches == 1
        assert finetune_cor == 0 or branches == 1
        if pred_cor and pred_key and not finetune_cor:
            self.out_c = 6  # y1,y2,c1,c2,k1,k2
        elif pred_cor and not finetune_cor:
            self.out_c = 4  # y1,y2,c1,c2
        else:
            self.out_c = 2  # y1,y2
        print('self.out_c:', self.out_c)

        self.backbone = backbone
        self.use_rnn = use_rnn
        self.rnn_hidden_size = 256
        self.finetune_cor = finetune_cor
        self.gray_mode = gray_mode
        self.drop_p = drop_p

        # Encoder
        if backbone.startswith('res'):
            self.feature_extractor = Resnet(backbone, pretrained=True)
        elif backbone.startswith('dense'):
            self.feature_extractor = Densenet(backbone, pretrained=True)
        elif backbone == 'ade20k_resnet50':
            self.feature_extractor = model_ADE20k_encoder.resnet50()
        else:
            raise NotImplementedError()

        # Inference channels number from each block of the encoder
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 512, 1024)
            c1, c2, c3, c4 = [b.shape[1] for b in self.feature_extractor(dummy)]

        # Convert features from block 4 of the encoder into B x C x 1 x W'
        self.branches = branches
        if self.branches == 1:
            self.reduce_height_module = nn.Sequential(
                ConvCompressH(c4, c4//2),
                ConvCompressH(c4//2, c4//2),
                ConvCompressH(c4//2, c4//4),
                ConvCompressH(c4//4, c4//8),
            )
        else:
            self.reduce_height_module = nn.ModuleList([
                nn.Sequential(
                    ConvCompressH(c4, c4//2),
                    ConvCompressH(c4//2, c4//2),
                    ConvCompressH(c4//2, c4//4),
                    ConvCompressH(c4//4, c4//8),
                )
                for _ in range(self.branches)
            ])
        c_last = 2*c4//8  # h*c

        # 1D prediction
        if self.use_rnn:
            self.bi_rnn = nn.LSTM(input_size=c_last,
                                  hidden_size=self.rnn_hidden_size,
                                  num_layers=2,
                                  dropout=self.drop_p,
                                  batch_first=False,
                                  bidirectional=True)
            self.drop_out = nn.Dropout(self.drop_p)
            self.linear = nn.Linear(in_features=2 * self.rnn_hidden_size,
                                    out_features=self.out_c)
            for i in range(len(init_bias)):
                self.linear.bias.data[i].fill_(init_bias[i])
        elif branches == 1:
            self.linear = nn.Sequential(
                nn.Linear(c_last, self.rnn_hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(self.drop_p),
                nn.Linear(self.rnn_hidden_size, self.out_c),
            )
            for i in range(len(init_bias)):
                self.linear[-1].bias.data[i].fill_(init_bias[i])
        else:
            self.linear = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(c_last, self.rnn_hidden_size),
                    nn.ReLU(inplace=True),
                    nn.Dropout(self.drop_p),
                    nn.Linear(self.rnn_hidden_size, self.out_c//self.branches),
                )
                for _ in range(self.branches)
            ])
            i, j = 0, 0
            for v in init_bias:
                self.linear[i][-1].bias.data[j].fill_(v)
                j += 1
                if j >= len(self.linear[i][-1].bias.data):
                    i += 1
                    j = 0

        if self.finetune_cor:
            self.cor_reduce_height_module = nn.Sequential(
                ConvCompressH(c4, c4//2),
                ConvCompressH(c4//2, c4//2),
                ConvCompressH(c4//2, c4//4),
                ConvCompressH(c4//4, c4//8),
            )
            self.cor_bi_rnn = nn.LSTM(input_size=c_last,
                                  hidden_size=self.rnn_hidden_size,
                                  num_layers=2,
                                  dropout=self.drop_p,
                                  batch_first=False,
                                  bidirectional=True)
            self.cor_drop_out = nn.Dropout(self.drop_p)
            self.cor_linear = nn.Linear(in_features=2 * self.rnn_hidden_size,
                                        out_features=self.out_c)
            self.cor_linear.bias.data[0].fill_(init_bias[2])
            self.cor_linear.bias.data[1].fill_(init_bias[3])

        if bn_momentum is not None:
            for m in self.modules():
                if isinstance(m, nn.modules.batchnorm._BatchNorm):
                    m.momentum = bn_momentum

    def forward(self, x):
        if self.gray_mode:
            x = (0.2989*x[:,[0]] + 0.5870*x[:,[1]] + 0.1140*x[:,[2]]).repeat(1, 3, 1, 1)
        conv_list = self.feature_extractor(x)
        last_block = conv_list[-1]

        if self.finetune_cor:
            feature = self.reduce_height_module(last_block)  # [b, c=256, h=2, w=20(=640/32)]
            feature = feature.reshape(feature.shape[0], -1, feature.shape[3]) # [b, c*h, w]
            feature = feature.permute(2, 0, 1)  # [w, b, c*h]
            output, hidden = self.bi_rnn(feature)  # [seq_len, b, num_directions * hidden_size]
            output = self.drop_out(output)
            output = self.linear(output)  # [seq_len, b, 3]
            output = output.view(output.shape[0], output.shape[1], self.out_c)  # [seq_len, b, 3]
            output = output.permute(1, 2, 0)  # [b, 3, seq_len]
            feature = self.cor_reduce_height_module(last_block)  # [b, c=256, h=2, w=20(=640/32)]
            feature = feature.reshape(feature.shape[0], -1, feature.shape[3]) # [b, c*h, w]
            feature = feature.permute(2, 0, 1)  # [w, b, c*h]
            output_cor, hidden = self.cor_bi_rnn(feature)  # [seq_len, b, num_directions * hidden_size]
            output_cor = self.cor_drop_out(output_cor)
            output_cor = self.cor_linear(output_cor)  # [seq_len, b, 3]
            output_cor = output_cor.view(output_cor.shape[0], output_cor.shape[1], self.out_c)  # [seq_len, b, 3]
            output_cor = output_cor.permute(1, 2, 0)  # [b, 3, seq_len]
        elif self.use_rnn:
            feature = self.reduce_height_module(last_block)  # [b, c=256, h=2, w=20(=640/32)]
            feature = feature.reshape(feature.shape[0], -1, feature.shape[3]) # [b, c*h, w]
            feature = feature.permute(2, 0, 1)  # [w, b, c*h]
            output, hidden = self.bi_rnn(feature)  # [seq_len, b, num_directions * hidden_size]
            output = self.drop_out(output)
            output = self.linear(output)  # [seq_len, b, 3]
            output = output.view(output.shape[0], output.shape[1], self.out_c)  # [seq_len, b, 3]
            output = output.permute(1, 2, 0)  # [b, 3, seq_len]
        elif self.branches == 1:
            feature = self.reduce_height_module(last_block)  # [b, c=256, h=2, w=20(=640/32)]
            feature = feature.reshape(feature.shape[0], -1, feature.shape[3]) # [b, c*h, w]
            feature = feature.permute(0, 2, 1)  # [b, w, c*h]
            output = self.linear(feature)  # [b, w, 3]
            output = output.view(output.shape[0], output.shape[1], self.out_c)  # [b, w, 3]
            output = output.permute(0, 2, 1)  # [b, 3, w]
        else:
            branches = [f(last_block) for f in self.reduce_height_module]
            branches = [v.reshape(v.shape[0], -1, v.shape[3]) for v in branches]
            branches = [v.permute(0, 2, 1) for v in branches]
            output = [f(v) for f, v in zip(self.linear, branches)]
            output = [v.view(v.shape[0], v.shape[1], self.out_c//self.branches) for v in output]
            output = [v.permute(0, 2, 1) for v in output]
            output = torch.cat(output, 1)

        if self.finetune_cor:
            return output, output_cor
        elif self.out_c == 2:
            return output
        elif self.out_c == 4:
            bon = output[:, :2, :]
            cor = output[:, 2:, :]
            return bon, cor
        else:
            bon = output[:, :2, :]
            cor = output[:, 2:4, :]
            key = output[:, 4:, :]
            return bon, cor, key



if __name__ == '__main__':
    net = LowResHorizonNet(backbone='resnet50', pred_cor=True)
    dummy = torch.zeros(1, 3, 640, 640)
    bon, cor = net(dummy)
    print('output shape w/ rnn: ', bon.shape, cor.shape)  # (1, 2, 20)

    net = LowResHorizonNet(backbone='resnet50', use_rnn=False, pred_cor=False)
    pred = net(dummy)
    print('output shape w/o rnn: ', pred.shape)
