"""
CNN模型定义：可配置BatchNorm位置的ResNet风格网络
研究框架: A(BN位置) → B(梯度稳定性) → C(收敛速度)

基于CIFAR-10设计，支持三种BN位置策略:
- NoBN: 不使用批量归一化
- BN_conv_post: BN置于卷积层后、激活函数前 (Conv→BN→ReLU)
- BN_activation_post: BN置于激活函数后 (Conv→ReLU→BN)
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple


class ConvBnBlock(nn.Module):
    """基本卷积块：Conv + 可配置BN + ReLU"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1, use_bn: bool = True,
                 bn_position: str = 'pre_activation'):
        """
        Args:
            bn_position: 'pre_activation' → Conv→BN→ReLU
                         'post_activation' → Conv→ReLU→BN
        """
        super().__init__()
        self.use_bn = use_bn
        self.bn_position = bn_position

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=not use_bn)
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        if self.use_bn and self.bn_position == 'pre_activation':
            x = self.conv(x)
            x = self.bn(x)
            x = self.relu(x)
        elif self.use_bn and self.bn_position == 'post_activation':
            x = self.conv(x)
            x = self.relu(x)
            x = self.bn(x)
        else:
            x = self.conv(x)
            x = self.relu(x)
        return x


class ResidualBlock(nn.Module):
    """残差块：两层Conv + 跳跃连接"""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
                 use_bn: bool = True, bn_position: str = 'pre_activation'):
        super().__init__()
        self.use_bn = use_bn
        self.bn_position = bn_position

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=not use_bn)
        self.bn1 = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.relu1 = nn.ReLU(inplace=False)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=not use_bn)
        self.bn2 = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.relu2 = nn.ReLU(inplace=False)

        # 跳跃连接的投影层（维度或分辨率不匹配时使用）
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
            )

    def forward(self, x):
        identity = self.shortcut(x)

        if self.use_bn and self.bn_position == 'pre_activation':
            out = self.bn1(self.conv1(x))
            out = self.relu1(out)
            out = self.bn2(self.conv2(out))
        elif self.use_bn and self.bn_position == 'post_activation':
            out = self.relu1(self.conv1(x))
            out = self.bn1(out)
            out = self.relu2(self.conv2(out))
            out = self.bn2(out)
        else:
            out = self.relu1(self.conv1(x))
            out = self.relu2(self.conv2(out))

        out = out + identity  # 使用非inplace加法避免梯度计算冲突
        out = self.relu2(out)
        return out


class ResNet18Light(nn.Module):
    """
    轻量化ResNet-18风格网络

    架构: Conv→[ResBlock×2]→[ResBlock×2]→[ResBlock×2]→AvgPool→FC
    每层均支持可配置BN位置

    参数约0.3M，适合在CPU上进行充分的消融实验
    """

    def __init__(self, num_classes: int = 10, use_bn: bool = True,
                 bn_position: str = 'pre_activation'):
        super().__init__()
        self.use_bn = use_bn
        self.bn_position = bn_position

        self.in_channels = 32

        # 初始卷积层
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1, bias=not use_bn)
        self.bn1 = nn.BatchNorm2d(32) if use_bn else nn.Identity()
        self.relu = nn.ReLU(inplace=False)

        # 3个stage，通道数分别为32, 64, 128
        self.layer1 = self._make_layer(32, 2, stride=1, use_bn=use_bn, bn_position=bn_position)
        self.layer2 = self._make_layer(64, 2, stride=2, use_bn=use_bn, bn_position=bn_position)
        self.layer3 = self._make_layer(128, 2, stride=2, use_bn=use_bn, bn_position=bn_position)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

        # 注册所有需要监控梯度的层名称
        self._register_layer_names()

    def _make_layer(self, out_channels: int, num_blocks: int, stride: int,
                    use_bn: bool, bn_position: str) -> nn.Sequential:
        layers = []
        layers.append(ResidualBlock(self.in_channels, out_channels, stride, use_bn, bn_position))
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, 1, use_bn, bn_position))
        return nn.Sequential(*layers)

    def _register_layer_names(self):
        """构建层名称列表（用于梯度监控标识）"""
        names = ['conv1']
        for li, layer_name in enumerate(['layer1', 'layer2', 'layer3']):
            for bi in range(2):  # 每层2个残差块
                for ci in range(1, 3):  # 每块2个卷积
                    names.append(f'{layer_name}_b{bi}_conv{ci}')
        names.append('fc')
        self._layer_names = names

    def forward(self, x):
        if self.use_bn and self.bn_position == 'pre_activation':
            x = self.relu(self.bn1(self.conv1(x)))
        elif self.use_bn and self.bn_position == 'post_activation':
            x = self.bn1(self.relu(self.conv1(x)))
        else:
            x = self.relu(self.conv1(x))

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def get_linear_conv_layers(self) -> List[nn.Module]:
        """获取所有有梯度的层（用于hook注册）"""
        layers = []
        layers.append(self.conv1)
        for layer_group in [self.layer1, self.layer2, self.layer3]:
            for res_block in layer_group:
                layers.append(res_block.conv1)
                layers.append(res_block.conv2)
        layers.append(self.fc)
        return layers

    def get_layer_names(self) -> List[str]:
        return self._layer_names


class SimpleCNN(nn.Module):
    """
    简单CNN（无残差连接），用于对比残差结构的影响

    4个卷积块 + 全连接分类器
    与ResNet18Light相同的参数化BN配置
    """

    def __init__(self, num_classes: int = 10, use_bn: bool = True,
                 bn_position: str = 'pre_activation'):
        super().__init__()
        self.use_bn = use_bn
        self.bn_position = bn_position

        channels = [3, 32, 64, 128, 256]

        self.blocks = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.blocks.append(ConvBnBlock(
                channels[i], channels[i + 1], 3,
                stride=2 if i >= 2 else 1,
                use_bn=use_bn, bn_position=bn_position
            ))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

        self._layer_names = [f'conv{i+1}' for i in range(len(channels) - 1)] + ['fc']

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def get_linear_conv_layers(self) -> List[nn.Module]:
        layers = [b.conv for b in self.blocks]
        layers.append(self.fc)
        return layers

    def get_layer_names(self) -> List[str]:
        return self._layer_names


def create_model_configs(model_type: str = 'resnet18',
                         num_classes: int = 10) -> Dict[str, nn.Module]:
    """
    创建实验所需的所有模型配置

    Args:
        model_type: 'resnet18' 或 'simple_cnn'
        num_classes: 输出类别数

    Returns:
        配置名称到模型实例的字典
    """
    model_cls = ResNet18Light if model_type == 'resnet18' else SimpleCNN

    configs = {
        'NoBN': model_cls(num_classes=num_classes, use_bn=False),
        'BN_pre': model_cls(num_classes=num_classes, use_bn=True, bn_position='pre_activation'),
        'BN_post': model_cls(num_classes=num_classes, use_bn=True, bn_position='post_activation'),
    }

    return configs


def count_parameters(model: nn.Module) -> int:
    """计算模型可训练参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
