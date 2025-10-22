import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """Basic residual block for ResNet"""
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, dropout=0.0):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        if self.dropout is not None:
            out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu(out)

        return out


class ResNet34(nn.Module):
    """ResNet-34 model for CIFAR datasets (Enhanced with deeper architecture)"""

    def __init__(self, num_classes=100, dropout=0.0):
        super(ResNet34, self).__init__()
        self.in_channels = 64

        # Initial convolution layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # REMOVED: Destroys spatial resolution for CIFAR-100

        # IMPROVED: Deeper architecture [5,5,5,5] instead of [3,4,6,3] for better CIFAR-100 performance
        # More balanced depth across layers, total 20 blocks vs 16 blocks
        self.layer1 = self._make_layer(64, 5, stride=1, dropout=dropout)
        self.layer2 = self._make_layer(128, 5, stride=2, dropout=dropout)
        self.layer3 = self._make_layer(256, 5, stride=2, dropout=dropout)
        self.layer4 = self._make_layer(512, 5, stride=2, dropout=dropout)

        # Final layers - IMPROVED: Increased dropout from 0.25 to 0.3
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

        # Initialize weights
        self._initialize_weights()

    def _make_layer(self, out_channels, blocks, stride, dropout=0.0):
        """Create a residual layer with specified number of blocks"""
        downsample = None
        if stride != 1 or self.in_channels != out_channels * BasicBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * BasicBlock.expansion,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion),
            )

        layers = []
        layers.append(BasicBlock(self.in_channels, out_channels, stride, downsample, dropout))
        self.in_channels = out_channels * BasicBlock.expansion

        for _ in range(1, blocks):
            layers.append(BasicBlock(self.in_channels, out_channels, dropout=dropout))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """Initialize weights using He initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Initial layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        # x = self.maxpool(x)  # REMOVED: Preserves 32x32 spatial resolution for layer1

        # Residual layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Final layers
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        return x


def get_resnet34(num_classes=100, pretrained=False):
    """
    Get ResNet34 model

    Args:
        num_classes (int): Number of output classes (default: 100 for CIFAR-100)
        pretrained (bool): If True, load pretrained weights (not available for CIFAR training)

    Returns:
        ResNet34 model
    """
    model = ResNet34(num_classes=num_classes)

    if pretrained:
        print("Warning: Pretrained weights not available for CIFAR ResNet34")
        print("Training from scratch")

    return model


if __name__ == "__main__":
    # Test the model
    model = get_resnet34(num_classes=100)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Model trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Test forward pass
    x = torch.randn(1, 3, 32, 32)  # CIFAR size (32x32)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
