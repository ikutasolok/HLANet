"""HLANet: three-branch ResNet-50 U-Net with height-category guidance."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet50


class ChannelLayerNorm(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


def projection(channels, hidden):
    return nn.Sequential(nn.Conv2d(channels, hidden, 1, bias=False), ChannelLayerNorm(hidden), nn.ReLU(inplace=True))


class NonLocal(nn.Module):
    def __init__(self, channels):
        super().__init__()
        hidden = max(channels // 4, 16)
        self.theta = projection(channels, hidden)
        self.phi = projection(channels, hidden)
        self.value = projection(channels, hidden)
        self.output = nn.Conv2d(hidden, channels, 1)

    def forward(self, x):
        b, _, h, w = x.shape
        q = self.theta(x).flatten(2).transpose(1, 2)
        k = self.phi(x).flatten(2)
        v = self.value(x).flatten(2).transpose(1, 2)
        attention = torch.softmax(torch.bmm(q, k) / math.sqrt(k.shape[1]), dim=-1)
        context = torch.bmm(attention, v).transpose(1, 2).reshape(b, -1, h, w)
        return x + self.output(context)


class CrossGuidance(nn.Module):
    def __init__(self, channels):
        super().__init__()
        hidden = max(channels // 4, 16)
        self.theta_class = projection(channels, hidden)
        self.phi_height = projection(channels, hidden)
        self.value_height = projection(channels, hidden)
        self.output = nn.Conv2d(hidden, channels, 1)
        self.gate = nn.Conv2d(channels, channels, 1)

    def forward(self, class_feature, height_feature):
        b, _, h, w = height_feature.shape
        q = self.theta_class(class_feature).flatten(2).transpose(1, 2)
        k = self.phi_height(height_feature).flatten(2)
        v = self.value_height(height_feature).flatten(2).transpose(1, 2)
        attention = torch.softmax(torch.bmm(q, k) / math.sqrt(k.shape[1]), dim=-1)
        context = torch.bmm(attention, v).transpose(1, 2).reshape(b, -1, h, w)
        context = self.output(context)
        return height_feature + torch.sigmoid(self.gate(context)) * context


class HCGM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.high_nonlocal = NonLocal(channels)
        self.low_nonlocal = NonLocal(channels)
        self.class_nonlocal = NonLocal(channels)
        self.high_bridge = CrossGuidance(channels)
        self.low_bridge = CrossGuidance(channels)

    def forward(self, feature):
        class_feature = self.class_nonlocal(feature)
        high_feature = self.high_bridge(class_feature, self.high_nonlocal(feature))
        low_feature = self.low_bridge(class_feature, self.low_nonlocal(feature))
        return high_feature, low_feature, class_feature


class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class Decoder(nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.up1 = UpBlock(2048, 1024, 512)
        self.up2 = UpBlock(512, 512, 256)
        self.up3 = UpBlock(256, 256, 128)
        self.up4 = UpBlock(128, 64, 64)
        self.head = nn.Conv2d(64, out_channels, 1)

    def forward(self, bottleneck, skips, target_size):
        x = bottleneck
        for block, skip in zip((self.up1, self.up2, self.up3, self.up4), skips):
            x = block(x, skip)
        return F.interpolate(self.head(x), size=target_size, mode='bilinear', align_corners=False)


class HLANet(nn.Module):
    """Input: S2 B2/B3/B4/B8 then S1 VV/VH; class 0/1/2 = background/low/high."""
    def __init__(self):
        super().__init__()
        encoder = resnet50(weights=None)
        encoder.conv1 = nn.Conv2d(6, 64, 7, stride=2, padding=3, bias=False)
        self.stem = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)
        self.pool = encoder.maxpool
        self.layer1, self.layer2 = encoder.layer1, encoder.layer2
        self.layer3, self.layer4 = encoder.layer3, encoder.layer4
        self.guidance = HCGM(2048)
        self.high_decoder = Decoder(1)
        self.low_decoder = Decoder(1)
        self.class_decoder = Decoder(3)
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, image):
        if image.ndim != 4 or image.shape[1] != 6:
            raise ValueError('Expected [batch, 6, height, width]')
        size = image.shape[-2:]
        stem = self.stem(image)
        x1 = self.layer1(self.pool(stem))
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        high_feature, low_feature, class_feature = self.guidance(x4)
        skips = (x3, x2, x1, stem)
        high = self.high_decoder(high_feature, skips, size)
        low = self.low_decoder(low_feature, skips, size)
        logits = self.class_decoder(class_feature, skips, size)
        probs = torch.softmax(logits, dim=1)
        fused = probs[:, 1:2] * low + probs[:, 2:3] * high
        return {'high': high, 'low': low, 'class_logits': logits, 'fused': fused}

    def set_stage(self, stage):
        if stage not in {'warmup', 'low', 'high'}:
            raise ValueError(stage)
        for parameter in self.parameters():
            parameter.requires_grad_(True)
        if stage == 'low':
            frozen = (self.high_decoder, self.guidance.high_nonlocal, self.guidance.high_bridge)
        elif stage == 'high':
            frozen = (self.low_decoder, self.guidance.low_nonlocal, self.guidance.low_bridge)
        else:
            frozen = ()
        for module in frozen:
            for parameter in module.parameters():
                parameter.requires_grad_(False)
