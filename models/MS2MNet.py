import torch
import torch.nn as nn
from einops import rearrange
import torchvision
from timm.models.layers import DropPath
import torch.nn.functional as F
import cv2
import numpy as np
import math

class MAF(nn.Module):
    def __init__(self, dim, fc_ratio, dilation=[3, 5, 7], dropout=0., num_classes=6):
        super(MAF, self).__init__()

        self.conv0 = nn.Conv2d(dim, dim//fc_ratio, 1)
        self.bn0 = nn.BatchNorm2d(dim//fc_ratio)

        self.conv1_1 = nn.Conv2d(dim//fc_ratio, dim//fc_ratio, 3, padding=dilation[-3], dilation=dilation[-3], groups=dim//fc_ratio)
        self.bn1_1 = nn.BatchNorm2d(dim//fc_ratio)
        self.conv1_2 = nn.Conv2d(dim//fc_ratio, dim, 1)
        self.bn1_2 = nn.BatchNorm2d(dim)

        self.conv2_1 = nn.Conv2d(dim//fc_ratio, dim//fc_ratio, 3, padding=dilation[-2], dilation=dilation[-2], groups=dim//fc_ratio)
        self.bn2_1 = nn.BatchNorm2d(dim//fc_ratio)
        self.conv2_2 = nn.Conv2d(dim//fc_ratio, dim, 1)
        self.bn2_2 = nn.BatchNorm2d(dim)

        self.conv3_1 = nn.Conv2d(dim//fc_ratio, dim//fc_ratio, 3, padding=dilation[-1], dilation=dilation[-1], groups=dim//fc_ratio)
        self.bn3_1 = nn.BatchNorm2d(dim//fc_ratio)
        self.conv3_2 = nn.Conv2d(dim//fc_ratio, dim, 1)
        self.bn3_2 = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU6()

        self.conv4 = nn.Conv2d(dim, dim, 1)
        self.bn4 = nn.BatchNorm2d(dim)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(dim, dim//fc_ratio, 1, 1),
            nn.ReLU6(),
            nn.Conv2d(dim//fc_ratio, dim, 1, 1),
            nn.Sigmoid()
        )

        self.s_conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=5, padding=2)
        self.sigmoid = nn.Sigmoid()

        self.head = nn.Sequential(SeparableConvBNReLU(dim, dim, kernel_size=3),
                                  nn.Dropout2d(p=dropout, inplace=True),
                                  Conv(dim, num_classes, kernel_size=1))

    def forward(self, x):
        u = x.clone()

        attn1_0 = self.relu(self.bn0(self.conv0(x)))
        attn1_1 = self.relu(self.bn1_1(self.conv1_1(attn1_0)))
        attn1_1 = self.relu(self.bn1_2(self.conv1_2(attn1_1)))
        attn1_2 = self.relu(self.bn2_1(self.conv2_1(attn1_0)))
        attn1_2 = self.relu(self.bn2_2(self.conv2_2(attn1_2)))
        attn1_3 = self.relu(self.bn3_1(self.conv3_1(attn1_0)))
        attn1_3 = self.relu(self.bn3_2(self.conv3_2(attn1_3)))

        c_attn = self.avg_pool(x)
        c_attn = self.fc(c_attn)
        c_attn = u * c_attn

        s_max_out, _ = torch.max(x, dim=1, keepdim=True)
        s_avg_out = torch.mean(x, dim=1, keepdim=True)
        s_attn = torch.cat((s_avg_out, s_max_out), dim=1)
        s_attn = self.s_conv(s_attn)
        s_attn = self.sigmoid(s_attn)
        s_attn = u * s_attn

        attn = attn1_1 + attn1_2 + attn1_3
        attn = self.relu(self.bn4(self.conv4(attn)))
        attn = u * attn

        out = self.head(attn + c_attn + s_attn)

        return out

class LayerNorm1D(nn.Module):
    """LayerNorm for channels of 1D tensor(B C L)"""

    def __init__(self, num_channels, eps=1e-5, affine=True):
        super(LayerNorm1D, self).__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.weight = nn.Parameter(torch.ones(1, num_channels, 1))
            self.bias = nn.Parameter(torch.zeros(1, num_channels, 1))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)  # (B, 1, H, W)
        var = x.var(dim=1, keepdim=True, unbiased=False)  # (B, 1, H, W)

        x_normalized = (x - mean) / torch.sqrt(var + self.eps)  # (B, C, H, W)

        if self.affine:
            x_normalized = x_normalized * self.weight + self.bias

        return x_normalized


class ConvLayer2D(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, norm=nn.BatchNorm2d,
                 act_layer=nn.ReLU, bn_weight_init=1):
        super(ConvLayer2D, self).__init__()
        self.conv = nn.Conv2d(
            in_dim,
            out_dim,
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
            padding=(padding, padding),
            dilation=(dilation, dilation),
            groups=groups,
            bias=False
        )
        self.norm = norm(num_features=out_dim) if norm else None
        self.act = act_layer() if act_layer else None

        if self.norm:
            torch.nn.init.constant_(self.norm.weight, bn_weight_init)
            torch.nn.init.constant_(self.norm.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x


class ConvLayer1D(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, norm=nn.BatchNorm1d,
                 act_layer=nn.ReLU, bn_weight_init=1):
        super(ConvLayer1D, self).__init__()
        self.conv = nn.Conv1d(
            in_dim,
            out_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False
        )
        self.norm = norm(num_features=out_dim) if norm else None
        self.act = act_layer() if act_layer else None

        if self.norm:
            torch.nn.init.constant_(self.norm.weight, bn_weight_init)
            torch.nn.init.constant_(self.norm.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x

class SqueezeExcite(nn.Module):
    def __init__(self, dim, reduction_ratio=4):
        super().__init__()
        hidden_dim = max(dim // reduction_ratio, 8)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, hidden_dim, 1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.gate(x)

class WASEFusion(nn.Module):
    def __init__(self, channel,
                 reduction=16, activation=nn.ReLU(inplace=True)):
        super(WASEFusion, self).__init__()
        self.fc_x = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1),
            activation,
            nn.Conv2d(channel // reduction, channel, kernel_size=1),
            nn.Sigmoid()
        )
        self.fc_y = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1),
            activation,
            nn.Conv2d(channel // reduction, channel, kernel_size=1),
            nn.Sigmoid()
        )
        # 添加一个简单的门控参数来调整两个分支的相对重要性
        self.weight_x = nn.Parameter(torch.ones(1))
        self.weight_y = nn.Parameter(torch.ones(1))

    def forward(self, x, y):
        weighting_x = F.adaptive_avg_pool2d(x, 1)
        weighting_x = self.fc_x(weighting_x)

        weighting_y = F.adaptive_avg_pool2d(y, 1)
        weighting_y = self.fc_y(weighting_y)

        # 添加可学习权重，但保持结构简单
        weight_sum = torch.abs(self.weight_x) + torch.abs(self.weight_y) + 1e-8
        normalized_weight_x = torch.abs(self.weight_x) / weight_sum
        normalized_weight_y = torch.abs(self.weight_y) / weight_sum

        # 使用归一化权重重新加权
        output = normalized_weight_x * weighting_x * x + normalized_weight_y * weighting_y * y

        return output

class ResNet50(nn.Module):
    def __init__(self, pretrained=True, in_channels=3):
        super(ResNet50, self).__init__()
        pretrained = torchvision.models.resnet50(pretrained=pretrained)

        for module_name in [
            "conv1",
            "bn1",
            "relu",
            "maxpool",
            "layer1",
            "layer2",
            "layer3",
            "layer4",
        ]:
            self.add_module(module_name, getattr(pretrained, module_name))

        if in_channels != 3:
            self.conv1 = torch.nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            self.conv1.weight.data = torch.unsqueeze(torch.mean(pretrained.conv1.weight.data, dim=1),
                                                     dim=1)

    def forward(self, x):
        b0 = self.relu(self.bn1(self.conv1(x)))
        b = self.maxpool(b0)
        b1 = self.layer1(b)
        b2 = self.layer2(b1)
        b3 = self.layer3(b2)
        return b0, b1, b2, b3

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d,
                 groups=1, bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=groups),
            norm_layer(out_channels),
            nn.ReLU6()
        )

class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d,
                 bias=False):
        super(ConvBN, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels)
        )

class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )

class SeparableConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            norm_layer(out_channels),
            nn.ReLU6()
        )

class E_FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, ksize=5, act_layer=nn.ReLU6, drop=0.):
        super(E_FFN, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = ConvBNReLU(in_channels=in_features, out_channels=hidden_features, kernel_size=1)
        self.conv1 = ConvBNReLU(in_channels=hidden_features, out_channels=hidden_features, kernel_size=ksize,
                                groups=hidden_features)
        self.conv2 = ConvBNReLU(in_channels=hidden_features, out_channels=hidden_features, kernel_size=3,
                                groups=hidden_features)
        self.fc2 = ConvBN(in_channels=hidden_features, out_channels=out_features, kernel_size=1)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x = self.fc2(x1 + x2)
        x = self.act(x)
        return x

class MMPADBlock(nn.Module):
    def __init__(self, dim=512, fc_ratio=4, dilation=[3, 5, 7], pool_ratio=16):
        super(MMPADBlock, self).__init__()
        self.conv0_1 = nn.Conv2d(dim, dim // fc_ratio, 1)
        self.bn0_1 = nn.BatchNorm2d(dim // fc_ratio)
        self.conv0_2 = nn.Conv2d(dim // fc_ratio, dim // fc_ratio, 3, padding=dilation[-3], dilation=dilation[-3],
                                 groups=dim // fc_ratio)
        self.bn0_2 = nn.BatchNorm2d(dim // fc_ratio)
        self.conv0_3 = nn.Conv2d(dim // fc_ratio, dim, 1)
        self.bn0_3 = nn.BatchNorm2d(dim)

        self.conv1_2 = nn.Conv2d(dim // fc_ratio, dim // fc_ratio, 3, padding=dilation[-2], dilation=dilation[-2],
                                 groups=dim // fc_ratio)
        self.bn1_2 = nn.BatchNorm2d(dim // fc_ratio)
        self.conv1_3 = nn.Conv2d(dim // fc_ratio, dim, 1)
        self.bn1_3 = nn.BatchNorm2d(dim)

        self.conv2_2 = nn.Conv2d(dim // fc_ratio, dim // fc_ratio, 3, padding=dilation[-1], dilation=dilation[-1],
                                 groups=dim // fc_ratio)
        self.bn2_2 = nn.BatchNorm2d(dim // fc_ratio)
        self.conv2_3 = nn.Conv2d(dim // fc_ratio, dim, 1)
        self.bn2_3 = nn.BatchNorm2d(dim)

        # 增加一个小尺寸卷积分支，捕获更精细的局部特征
        self.conv3_2 = nn.Conv2d(dim // fc_ratio, dim // fc_ratio, 1)
        self.bn3_2 = nn.BatchNorm2d(dim // fc_ratio)
        self.conv3_3 = nn.Conv2d(dim // fc_ratio, dim, 1)
        self.bn3_3 = nn.BatchNorm2d(dim)

        self.conv4 = nn.Conv2d(dim, dim, 1)
        self.bn4 = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU6()

        self.Avg = nn.AdaptiveAvgPool2d(pool_ratio)

    def forward(self, x):
        u = x.clone()

        attn0_1 = self.relu(self.bn0_1(self.conv0_1(x)))
        attn0_2 = self.relu(self.bn0_2(self.conv0_2(attn0_1)))
        attn0_3 = self.relu(self.bn0_3(self.conv0_3(attn0_2)))

        attn1_2 = self.relu(self.bn1_2(self.conv1_2(attn0_1)))
        attn1_3 = self.relu(self.bn1_3(self.conv1_3(attn1_2)))

        attn2_2 = self.relu(self.bn2_2(self.conv2_2(attn0_1)))
        attn2_3 = self.relu(self.bn2_3(self.conv2_3(attn2_2)))

        # 新增的小尺寸分支
        attn3_2 = self.relu(self.bn3_2(self.conv3_2(attn0_1)))
        attn3_3 = self.relu(self.bn3_3(self.conv3_3(attn3_2)))

        attn = attn0_3 + attn1_3 + attn2_3 + attn3_3
        attn = self.relu(self.bn4(self.conv4(attn)))
        attn = attn * u

        pool = self.Avg(attn)

        return pool

class Mutilscal_MHSA(nn.Module):
    def __init__(self, dim, num_heads, atten_drop=0., proj_drop=0., dilation=[3, 5, 7], fc_ratio=4, pool_ratio=16):
        super(Mutilscal_MHSA, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.atten_drop = nn.Dropout(atten_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        # 使用改进的EnhancedMutilScal
        self.MSC = MMPADBlock(dim=dim, fc_ratio=fc_ratio, dilation=dilation, pool_ratio=pool_ratio)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim // fc_ratio, kernel_size=1),
            nn.ReLU6(),
            nn.Conv2d(in_channels=dim // fc_ratio, out_channels=dim, kernel_size=1),
            nn.Sigmoid()
        )
        self.kv = Conv(dim, 2 * dim, 1)

    def forward(self, x):
        u = x.clone()
        B, C, H, W = x.shape
        kv = self.MSC(x)
        kv = self.kv(kv)

        B1, C1, H1, W1 = kv.shape

        q = rearrange(x, 'b (h d) (hh) (ww) -> (b) h (hh ww) d', h=self.num_heads,
                      d=C // self.num_heads, hh=H, ww=W)
        k, v = rearrange(kv, 'b (kv h d) (hh) (ww) -> kv (b) h (hh ww) d', h=self.num_heads,
                         d=C // self.num_heads, hh=H1, ww=W1, kv=2)

        dots = (q @ k.transpose(-2, -1)) * self.scale
        attn = dots.softmax(dim=-1)
        attn = self.atten_drop(attn)
        attn = attn @ v

        attn = rearrange(attn, '(b) h (hh ww) d -> b (h d) (hh) (ww)', h=self.num_heads,
                         d=C // self.num_heads, hh=H, ww=W)
        c_attn = self.avgpool(x)
        c_attn = self.fc(c_attn)
        c_attn = c_attn * u
        return attn + c_attn

class Block(nn.Module):
    def __init__(self, dim=512, num_heads=16, mlp_ratio=4, pool_ratio=16, drop=0., dilation=[3, 5, 7],
                 drop_path=0., act_layer=nn.ReLU6, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Mutilscal_MHSA(dim, num_heads=num_heads, atten_drop=drop, proj_drop=drop, dilation=dilation,
                                   pool_ratio=pool_ratio, fc_ratio=mlp_ratio)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim // mlp_ratio)

        self.mlp = E_FFN(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer,
                         drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.norm1(self.attn(x)))
        x = x + self.drop_path(self.mlp(x))
        return x

class HFPDecoder(nn.Module):
    def __init__(self,
                 encode_channels=[256, 512, 1024, 2048],
                 decode_channels=512,
                 dilation=[[1, 3, 5], [3, 5, 7], [5, 7, 9], [7, 9, 11]],
                 fc_ratio=4,
                 dropout=0.1,
                 num_classes=6):
        super(HFPDecoder, self).__init__()

        # 特征转换模块
        self.Conv1 = ConvBNReLU(encode_channels[-1], decode_channels, 1)
        self.Conv2 = ConvBNReLU(encode_channels[-2], decode_channels, 1)

        # 高级特征处理与增强
        self.b4 = Block(dim=decode_channels, num_heads=16, mlp_ratio=4, pool_ratio=16, dilation=dilation[0])

        # 使用增强的特征融合模块
        self.p3 = ASAFFusion(decode_channels)
        self.b3 = Block(dim=decode_channels, num_heads=16, mlp_ratio=4, pool_ratio=16, dilation=dilation[1])

        self.p2 = ASAFFusion(decode_channels)
        self.b2 = Block(dim=decode_channels, num_heads=16, mlp_ratio=4, pool_ratio=16, dilation=dilation[2])

        self.Conv3 = ConvBN(encode_channels[-3], encode_channels[-4], 1)

        self.p1 = ASAFFusion(encode_channels[-4])

        # 使用增强的分割头
        self.seg_head = MAF(encode_channels[-4], fc_ratio=fc_ratio, dilation=dilation[3],
                                  dropout=dropout, num_classes=num_classes)
                                  
        # 添加深度监督辅助分类器，处理不同尺度的特征
        self.aux_head3 = nn.Sequential(
            ConvBNReLU(decode_channels, 256, 3, 1, 1),
            nn.Dropout2d(dropout),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )
        
        self.aux_head2 = nn.Sequential(
            ConvBNReLU(decode_channels, 256, 3, 1, 1),
            nn.Dropout2d(dropout),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )

        self.init_weight()

    def forward(self, res1, res2, res3, res4, h, w):
        # 特征转换
        res4 = self.Conv1(res4)
        res3 = self.Conv2(res3)

        # 高级特征处理
        x = self.b4(res4)

        # 中级特征融合
        x = self.p3(x, res3)
        aux_out3 = self.aux_head3(x)  # 辅助输出1
        x = self.b3(x)

        # 低级特征融合
        x = self.p2(x, res2)
        aux_out2 = self.aux_head2(x)  # 辅助输出2
        x = self.b2(x)

        # 最低级特征融合
        x = self.Conv3(x)
        x = self.p1(x, res1)

        # 分割头处理
        main_out = self.seg_head(x)

        # 上采样到原始尺寸
        main_out = F.interpolate(main_out, size=(h, w), mode='bilinear', align_corners=False)
        
        # 上采样辅助输出
        aux_out3 = F.interpolate(aux_out3, size=(h, w), mode='bilinear', align_corners=False)
        aux_out2 = F.interpolate(aux_out2, size=(h, w), mode='bilinear', align_corners=False)

        if self.training:
            return main_out, aux_out3, aux_out2
        else:
            return main_out

    def init_weight(self):
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class MS2MNet(nn.Module):
    def __init__(self,
                 encode_channels=[64, 256, 512, 1024],
                 decode_channels=256,
                 dropout=0.1,
                 num_classes=6,
                 residualRate=0.5):
        super().__init__()

        self.rgb_backbone = ResNet50(in_channels=3)
        self.dsm_backbone = ResNet50(in_channels=1)

        # 使用优化的解码器
        self.decoder = HFPDecoder(encode_channels, decode_channels, dropout=dropout, num_classes=num_classes)

        # 融合模块 - 保持原有的融合模块
        self.waseFusion1 = WASEFusion(64)
        self.waseFusion2 = WASEFusion(256)
        self.waseFusion3 = WASEFusion(512)
        self.waseFusion4 = WASEFusion(1024)

        # 使用增强版EfficientViMBlock - 根据不同层级特征的特点，调整state_dim
        # 较低层（高分辨率）使用较小state_dim，较高层（低分辨率）使用较大state_dim
        self.vfeBlock1 = VFE(dim=64, state_dim=16)
        self.vfeBlock2 = VFE(dim=256, state_dim=24)
        self.vfeBlock3 = VFE(dim=512, state_dim=28)
        self.vfeBlock4 = VFE(dim=1024, state_dim=32)

        self.residualRate = residualRate
        
        # 训练模式标志
        self.training = True

    def forward(self, rgb, dsm):
        # 边缘增强处理
        dsm = self.dfeBlock(dsm)

        # 提取RGB和DSM特征
        rgb_h, rgb_w = rgb.size()[-2:]
        rgb_res1, rgb_res2, rgb_res3, rgb_res4 = self.rgb_backbone(rgb)
        dsm_res1, dsm_res2, dsm_res3, dsm_res4 = self.dsm_backbone(dsm)

        # 特征融合
        f1 = self.waseFusion1(rgb_res1, dsm_res1)
        f2 = self.waseFusion2(rgb_res2, dsm_res2)
        f3 = self.waseFusion3(rgb_res3, dsm_res3)
        f4 = self.waseFusion4(rgb_res4, dsm_res4)

        f1_enhanced, _ = self.vfeBlock1(f1)
        f1 = f1 + f1_enhanced * self.residualRate  # 添加缩放因子控制特征融合比例
        f2_enhanced, _ = self.vfeBlock2(f2)
        f2 = f2 + f2_enhanced * self.residualRate
        f3_enhanced, _ = self.vfeBlock3(f3)
        f3 = f3 + f3_enhanced * self.residualRate
        f4_enhanced, _ = self.vfeBlock4(f4)
        f4 = f4 + f4_enhanced * self.residualRate

        # 解码并生成分割结果
        if self.training:
            main_out, aux_out3, aux_out2 = self.decoder(f1, f2, f3, f4, rgb_h, rgb_w)
            return main_out, aux_out3, aux_out2
        else:
            x = self.decoder(f1, f2, f3, f4, rgb_h, rgb_w)
            return x
            
    def train(self, mode=True):
        """
        重写train方法以正确设置模型状态
        """
        super().train(mode)
        self.training = mode
        return self
    
    def eval(self):
        """
        重写eval方法以正确设置模型状态
        """
        super().eval()
        self.training = False
        return self


def calculate_fps_two_tensors(model, input_tensor1, input_tensor2, num_runs=100, warmup_runs=10):
    if not torch.cuda.is_available():
        print("GPU不可用，使用CPU版本")
        return calculate_fps_two_tensors_cpu(model, input_tensor1, input_tensor2, num_runs, warmup_runs)

    model.eval()
    model = model.cuda()
    input_tensor1 = input_tensor1.cuda()
    input_tensor2 = input_tensor2.cuda()

    # 预热
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(input_tensor1, input_tensor2)

    # 创建CUDA事件
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # 计时测试
    start_event.record()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(input_tensor1, input_tensor2)
    end_event.record()

    # 等待GPU完成
    torch.cuda.synchronize()

    # 计算时间
    total_time = start_event.elapsed_time(end_event) / 1000.0  # 转换为秒
    fps = num_runs / total_time

    return fps, total_time


def calculate_fps_two_tensors_cpu(model, input_tensor1, input_tensor2, num_runs=100, warmup_runs=10):
    model.eval()

    # 预热
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(input_tensor1, input_tensor2)

    # 计时测试
    import time
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(input_tensor1, input_tensor2)
    end_time = time.time()

    total_time = end_time - start_time
    fps = num_runs / total_time

    return fps, total_time

if __name__ == "__main__":
    from thop import profile, clever_format

    device = torch.device("cuda")

    rgb = torch.randn(1, 3, 256, 256)
    dsm = torch.randn(1, 1, 256, 256)
    net = MS2MNet()

    rgb = rgb.to(device)
    dsm = dsm.to(device)
    net = net.to(device)

    flops, params = profile(net, inputs=(rgb, dsm))
    macs, params = clever_format([flops, params], "%.3f")
    print("FLOPs:", macs)
    print("params:", params)

    # net_fps = MS2MNet()
    # rgb_fps = torch.randn(1, 3, 256, 256)
    # dsm_fps = torch.randn(1, 1, 256, 256)
    # fps, total_time = calculate_fps_two_tensors(net_fps, rgb_fps, dsm_fps)
    # print(f"FPS: {fps:.2f}")
    # print(f"总时间: {total_time:.4f}秒")
    # print(f"平均每帧时间: {total_time / 100 * 1000:.2f}毫秒")