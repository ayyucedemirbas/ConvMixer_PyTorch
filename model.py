import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torchvision.transforms as transforms


class ActivationBlock(nn.Module):
    def __init__(self, num_features):
        super(ActivationBlock, self).__init__()
        self.activation = nn.GELU()
        self.batch_norm = nn.BatchNorm2d(num_features)

    def forward(self, x):
        x = self.activation(x)
        x = self.batch_norm(x)
        return x


class ConvStem(nn.Module):
    def __init__(self, filters, patch_size):
        super(ConvStem, self).__init__()
        self.conv = nn.Conv2d(3, filters, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.conv(x)
        return x


class ConvMixerBlock(nn.Module):
    def __init__(self, filters, kernel_size):
        super(ConvMixerBlock, self).__init__()
        self.depthwise_conv = nn.Conv2d(
            filters,
            filters,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=filters,
        )
        self.pointwise_conv = nn.Conv2d(filters, filters, kernel_size=1)
        self.activation_block = ActivationBlock(filters)

    def forward(self, x):
        x0 = x
        x = self.depthwise_conv(x)
        x = self.activation_block(x + x0)
        x = self.pointwise_conv(x)
        x = self.activation_block(x)
        return x


class ConvMixer(nn.Module):
    def __init__(
        self, image_size, filters, depth, kernel_size, patch_size, num_classes
    ):
        super(ConvMixer, self).__init__()
        self.image_size = image_size
        self.filters = filters
        self.depth = depth
        self.kernel_size = kernel_size
        self.patch_size = patch_size

        self.data_augmentation = nn.Sequential(
            transforms.ColorJitter(brightness=0.2),
        )
        self.stem = ConvStem(filters, patch_size)

        self.mixer_blocks = nn.Sequential()
        for i in range(depth):
            self.mixer_blocks.add_module(
                f"block_{i}", ConvMixerBlock(filters, kernel_size)
            )

        self.classification_block = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(filters, num_classes),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        x = self.data_augmentation(x)
        x = x / 255.0
        x = self.stem(x)
        x = self.mixer_blocks(x)
        x = self.classification_block(x)
        return x


def create_model(img_height):
    model = ConvMixer(
        image_size=img_height,
        filters=256,
        depth=8,
        kernel_size=5,
        patch_size=2,
        num_classes=6,
    )
    return model
