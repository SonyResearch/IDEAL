import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel

nz = 128
nc = 3


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.layer1 = nn.Sequential(  # 输入1*28*28
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # 32*28*28
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32*14*14
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 64*14*14
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64*7*7
            nn.ReLU(inplace=True),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 128*7*7
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # 256*7*7
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # 256*7*7
            nn.MaxPool2d(kernel_size=3, stride=2),  # 256*3*3
            nn.ReLU(inplace=True),
        )
        self.fc1 = nn.Linear(256 * 3 * 3, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(-1, 256 * 3 * 3)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class VGG(nn.Module):

    def __init__(self, cfg, batch_norm=False, num_classes=1000):
        super(VGG, self).__init__()
        self.block0 = self._make_layers(cfg[0], batch_norm, 3)
        self.block1 = self._make_layers(cfg[1], batch_norm, cfg[0][-1])
        self.block2 = self._make_layers(cfg[2], batch_norm, cfg[1][-1])
        self.block3 = self._make_layers(cfg[3], batch_norm, cfg[2][-1])
        self.block4 = self._make_layers(cfg[4], batch_norm, cfg[3][-1])

        self.pool0 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool4 = nn.AdaptiveAvgPool2d((1, 1))
        # self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.classifier = nn.Linear(512, num_classes)
        self._initialize_weights()

    def get_feat_modules(self):
        feat_m = nn.ModuleList([])
        feat_m.append(self.block0)
        feat_m.append(self.pool0)
        feat_m.append(self.block1)
        feat_m.append(self.pool1)
        feat_m.append(self.block2)
        feat_m.append(self.pool2)
        feat_m.append(self.block3)
        feat_m.append(self.pool3)
        feat_m.append(self.block4)
        feat_m.append(self.pool4)
        return feat_m

    def get_bn_before_relu(self):
        bn1 = self.block1[-1]
        bn2 = self.block2[-1]
        bn3 = self.block3[-1]
        bn4 = self.block4[-1]
        return [bn1, bn2, bn3, bn4]

    def forward(self, x, return_features=False):
        h = x.shape[2]
        x = F.relu(self.block0(x))
        x = self.pool0(x)
        f1=x = self.block1(x)
        x = F.relu(x)
        x = self.pool1(x)
        f2=x = self.block2(x)
        x = F.relu(x)
        x = self.pool2(x)
        f3=x = self.block3(x)
        x = F.relu(x)
        if h == 64:
            x = self.pool3(x)
        f4=x = self.block4(x)
        x = F.relu(x)
        x = self.pool4(x)
        features = x.view(x.size(0), -1)
        x = self.classifier(features)
        if return_features:
            return x, [f1,f2,f3,f4]
        else:
            return x

    @staticmethod
    def _make_layers(cfg, batch_norm=False, in_channels=3):
        layers = []
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        layers = layers[:-1]
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


cfg = {
    'A': [[64], [128], [256, 256], [512, 512], [512, 512]],
    'B': [[64, 64], [128, 128], [256, 256], [512, 512], [512, 512]],
    'D': [[64, 64], [128, 128], [256, 256, 256], [512, 512, 512], [512, 512, 512]],
    'E': [[64, 64], [128, 128], [256, 256, 256, 256], [512, 512, 512, 512], [512, 512, 512, 512]],
    'S': [[64], [128], [256], [512], [512]],
}



def vgg8_bn(**kwargs):
    """VGG 8-layer model (configuration "S")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(cfg['S'], batch_norm=True, **kwargs)
    return model

# class AlexNet(nn.Module):
#
#     def __init__(self, num_classes=12):
#         super(AlexNet, self).__init__()
#         self.features = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2),
#             nn.Conv2d(64, 192, kernel_size=5, padding=2),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2),
#             nn.Conv2d(192, 384, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(384, 256, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(256, 256, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2),
#         )
#         self.classifier = nn.Sequential(
#             nn.Dropout(),
#             nn.Linear(256 * 6 * 6, 1024),
#             nn.ReLU(inplace=True),
#             nn.Dropout(),
#             nn.Linear(1024, 1024),
#             nn.ReLU(inplace=True),
#             nn.Linear(1024, num_classes),
#         )
#
#     def forward(self, x):
#         x = self.features(x)
#         x = x.view(x.size(0), 256 * 1 * 1)
#         x = self.classifier(x)
#         return x




# 建立一个四层感知机网络
class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(784, 512)
        self.fc2 = torch.nn.Linear(512, 128)
        self.fc3 = torch.nn.Linear(128, 10)

    def forward(self, din):
        din = din.view(-1, 28 * 28)
        out = F.relu(self.fc1(din))
        out = F.relu(self.fc2(out))
        out = F.softmax(self.fc3(out), dim=1)
        return out


class GeneratorA(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=1, img_size=32, final_bn=True):
        super(GeneratorA, self).__init__()

        # if activation is None:
        #     raise ValueError("Provide a valid activation function")
        # self.activation = activation

        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(nz, ngf * 2 * self.init_size ** 2))

        self.conv_blocks0 = nn.Sequential(
            nn.BatchNorm2d(ngf * 2),
        )
        self.conv_blocks1 = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf * 2, 3, stride=1, padding=1),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )

        if final_bn:
            self.conv_blocks2 = nn.Sequential(
                nn.Conv2d(ngf * 2, ngf, 3, stride=1, padding=1),
                nn.BatchNorm2d(ngf),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ngf, nc, 3, stride=1, padding=1),
                # nn.Tanh(),
                nn.BatchNorm2d(nc, affine=False)
            )
        else:
            self.conv_blocks2 = nn.Sequential(
                nn.Conv2d(ngf * 2, ngf, 3, stride=1, padding=1),
                nn.BatchNorm2d(ngf),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ngf, nc, 3, stride=1, padding=1),
                # nn.Tanh(),
                # nn.BatchNorm2d(nc, affine=False)
            )

    def forward(self, z):
        out = self.l1(z.view(z.shape[0], -1))
        out = out.view(out.shape[0], -1, self.init_size, self.init_size)
        img = self.conv_blocks0(out)
        img = nn.functional.interpolate(img, scale_factor=2)
        img = self.conv_blocks1(img)
        img = nn.functional.interpolate(img, scale_factor=2)
        img = self.conv_blocks2(img)

        return img
        # if pre_x:
        #     return img
        # else:
        #     # img = nn.functional.interpolate(img, scale_factor=2)
        #     return self.activation(img)


class pre_conv(nn.Module):
    def __init__(self):
        super(pre_conv, self).__init__()
        self.nf = 64
        G_type = 1

        if G_type == 1:
            # ipdb.set_trace()
            self.pre_conv = nn.Sequential(
                nn.Conv2d(nz, self.nf * 2, 3, 1, 1, bias=False),  # torch.Size([50, 128, 1, 1])
                nn.BatchNorm2d(self.nf * 2),
                nn.LeakyReLU(0.2, inplace=True),

                nn.ConvTranspose2d(self.nf * 2, self.nf * 2, 4, 2, 1, bias=False),  # torch.Size([50, 128, 2, 2])
                nn.BatchNorm2d(self.nf * 2),
                nn.LeakyReLU(0.2, inplace=True),

                nn.ConvTranspose2d(self.nf * 2, self.nf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.nf * 2),
                nn.LeakyReLU(0.2, inplace=True),

                nn.ConvTranspose2d(self.nf * 2, self.nf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.nf * 2),
                nn.LeakyReLU(0.2, inplace=True),

                nn.ConvTranspose2d(self.nf * 2, self.nf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.nf * 2),
                nn.LeakyReLU(0.2, inplace=True),

                nn.ConvTranspose2d(self.nf * 2, self.nf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.nf * 2),
                nn.LeakyReLU(0.2, inplace=True)
            )
        elif G_type == 2:
            self.pre_conv = nn.Sequential(
                nn.Conv2d(self.nf * 8, self.nf * 8, 3, 1, round((self.shape[0] - 1) / 2), bias=False),
                nn.BatchNorm2d(self.nf * 8),
                nn.ReLU(True),  # added
                nn.Conv2d(self.nf * 8, self.nf * 8, 3, 1, round((self.shape[0] - 1) / 2), bias=False),
                nn.BatchNorm2d(self.nf * 8),
                nn.ReLU(True),

                nn.Conv2d(self.nf * 8, self.nf * 4, 3, 1, 1, bias=False),
                nn.BatchNorm2d(self.nf * 4),
                nn.ReLU(True),

                nn.Conv2d(self.nf * 4, self.nf * 2, 3, 1, 1, bias=False),
                nn.BatchNorm2d(self.nf * 2),
                nn.ReLU(True),

                nn.Conv2d(self.nf * 2, self.nf, 3, 1, 1, bias=False),
                nn.BatchNorm2d(self.nf),
                nn.ReLU(True),

                nn.Conv2d(self.nf, self.shape[0], 3, 1, 1, bias=False),
                nn.BatchNorm2d(self.shape[0]),
                nn.ReLU(True),

                nn.Conv2d(self.shape[0], self.shape[0], 3, 1, 1, bias=False),

                nn.Sigmoid()
            )

    def forward(self, input):
        output = self.pre_conv(input)
        return output


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.nf = 64
        self.num_class = 10
        G_type = 1
        if G_type == 1:
            self.main = nn.Sequential(
                nn.Conv2d(self.nf * 2, self.nf * 4, 3, 1, 0, bias=False),
                nn.BatchNorm2d(self.nf * 4),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(self.nf * 4, self.nf * 8, 3, 1, 0, bias=False),
                nn.BatchNorm2d(self.nf * 8),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(self.nf * 8, self.nf * 4, 3, 1, 1, bias=False),
                nn.BatchNorm2d(self.nf * 4),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(self.nf * 4, self.nf * 2, 3, 1, 1, bias=False),
                nn.BatchNorm2d(self.nf * 2),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(self.nf * 2, self.nf, 3, 1, 1, bias=False),
                nn.BatchNorm2d(self.nf),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(self.nf, nc, 3, 1, 1, bias=False),
                nn.BatchNorm2d(nc),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(nc, nc, 3, 1, 1, bias=False),
                nn.Sigmoid()
            )
        elif G_type == 2:
            self.main = nn.Sequential(
                nn.Conv2d(nz, self.nf * 2, 3, 1, 1, bias=False),
                nn.BatchNorm2d(self.nf * 2),
                nn.ReLU(True),

                nn.ConvTranspose2d(self.nf * 2, self.nf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.nf * 2),
                nn.ReLU(True),

                nn.ConvTranspose2d(self.nf * 2, self.nf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.nf * 4),
                nn.ReLU(True),

                nn.ConvTranspose2d(self.nf * 4, self.nf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.nf * 4),
                nn.ReLU(True),

                nn.ConvTranspose2d(self.nf * 4, self.nf * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.nf * 8),
                nn.ReLU(True),

                nn.ConvTranspose2d(self.nf * 8, self.nf * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.nf * 8),
                nn.ReLU(True),

                nn.Conv2d(self.nf * 8, self.nf * 8, 3, 1, 1, bias=False),
                nn.BatchNorm2d(self.nf * 8),
                nn.ReLU(True)
            )

    def forward(self, input):
        output = self.main(input)
        return output


class Generator_2(nn.Module):
    def __init__(self, nz=100, ngf=64, img_size=32, nc=3):
        super(Generator_2, self).__init__()

        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(nz, ngf * 2 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(ngf * 2),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(ngf * 2, ngf * 2, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(ngf * 2, ngf, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf, nc, 3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], -1, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


def UpSampling(x, H, W):
    B, N, C = x.size()
    assert N == H * W
    x = x.permute(0, 2, 1)
    x = x.view(-1, C, H, W)
    x = nn.PixelShuffle(2)(x)
    B, C, H, W = x.size()
    x = x.view(-1, C, H * W)
    x = x.permute(0, 2, 1)
    return x, H, W


# class MLP(nn.Module):
#     def __init__(self, in_feat, hid_feat=None, out_feat=None,
#                  dropout=0.):
#         super().__init__()
#         if not hid_feat:
#             hid_feat = in_feat
#         if not out_feat:
#             out_feat = in_feat
#         self.fc1 = nn.Linear(in_feat, hid_feat)
#         self.act = nn.GELU()
#         self.fc2 = nn.Linear(hid_feat, out_feat)
#         self.droprateout = nn.Dropout(dropout)
#
#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.act(x)
#         x = self.fc2(x)
#         return self.droprateout(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, attention_dropout=0., proj_dropout=0.):
        super().__init__()
        self.heads = heads
        self.scale = 1. / dim ** 0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(proj_dropout)
        )

    def forward(self, x):
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.heads, c // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)

        dot = (q @ k.transpose(-2, -1)) * self.scale
        attn = dot.softmax(dim=-1)
        attn = self.attention_dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = self.out(x)
        return x


class Encoder_Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4, drop_rate=0.):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, drop_rate, drop_rate)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, dim * mlp_ratio, dropout=drop_rate)

    def forward(self, x):
        x1 = self.ln1(x)
        x = x + self.attn(x1)
        x2 = self.ln2(x)
        x = x + self.mlp(x2)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, depth, dim, heads, mlp_ratio=4, drop_rate=0.):
        super().__init__()
        self.Encoder_Blocks = nn.ModuleList([
            Encoder_Block(dim, heads, mlp_ratio, drop_rate)
            for i in range(depth)])

    def forward(self, x):
        for Encoder_Block in self.Encoder_Blocks:
            x = Encoder_Block(x)
        return x


class Generator_3(nn.Module):
    """docstring for Generator"""

    def __init__(self, depth1=5, depth2=4, depth3=2, initial_size=8, dim=384, heads=4, mlp_ratio=4,
                 drop_rate=0.):
        super(Generator_3, self).__init__()

        self.initial_size = initial_size
        self.dim = dim
        self.depth1 = depth1
        self.depth2 = depth2
        self.depth3 = depth3
        self.heads = heads
        self.mlp_ratio = mlp_ratio
        self.droprate_rate = drop_rate

        self.mlp = nn.Linear(1024, (self.initial_size ** 2) * self.dim)

        self.positional_embedding_1 = nn.Parameter(torch.zeros(1, (8 ** 2), 384))
        self.positional_embedding_2 = nn.Parameter(torch.zeros(1, (8 * 2) ** 2, 384 // 4))
        self.positional_embedding_3 = nn.Parameter(torch.zeros(1, (8 * 4) ** 2, 384 // 16))

        self.TransformerEncoder_encoder1 = TransformerEncoder(depth=self.depth1, dim=self.dim, heads=self.heads,
                                                              mlp_ratio=self.mlp_ratio, drop_rate=self.droprate_rate)
        self.TransformerEncoder_encoder2 = TransformerEncoder(depth=self.depth2, dim=self.dim // 4, heads=self.heads,
                                                              mlp_ratio=self.mlp_ratio, drop_rate=self.droprate_rate)
        self.TransformerEncoder_encoder3 = TransformerEncoder(depth=self.depth3, dim=self.dim // 16, heads=self.heads,
                                                              mlp_ratio=self.mlp_ratio, drop_rate=self.droprate_rate)

        self.linear = nn.Sequential(nn.Conv2d(self.dim // 16, 3, 1, 1, 0))

    def forward(self, noise):
        x = self.mlp(noise).view(-1, self.initial_size ** 2, self.dim)

        x = x + self.positional_embedding_1
        H, W = self.initial_size, self.initial_size
        x = self.TransformerEncoder_encoder1(x)

        x, H, W = UpSampling(x, H, W)
        x = x + self.positional_embedding_2
        x = self.TransformerEncoder_encoder2(x)

        x, H, W = UpSampling(x, H, W)
        x = x + self.positional_embedding_3

        x = self.TransformerEncoder_encoder3(x)
        x = self.linear(x.permute(0, 2, 1).view(-1, self.dim // 16, H, W))

        return x


class CNNCifar10(nn.Module):
    def __init__(self):
        super(CNNCifar10, self).__init__()
        self.conv1 = nn.Conv2d(3, 256, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(256, 256, 3)
        self.conv3 = nn.Conv2d(256, 128, 3)
        self.fc1 = nn.Linear(128 * 4 * 4, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 128 * 4 * 4)
        x = self.fc1(x)
        return x


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(7 * 7 * 32, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, return_features=False):
        x = self.conv1(x)
        x = self.bn1(x)
        out = F.relu(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        feature = out.view(out.size(0), -1)
        out = self.linear(feature)

        if return_features:
            return out, feature
        else:
            return out


def resnet18(num_classes=10):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)


def resnet34(num_classes=10):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)


def resnet50(num_classes=10):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes)


def resnet101(num_classes=10):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes)


def resnet152(num_classes=10):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes)
