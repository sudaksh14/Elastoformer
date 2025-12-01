import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = nn.Sequential()
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.downsample(x)
        out = F.relu(out)
        return out
    
# class BottleneckBlock(nn.Module):
#     expansion = 4  # Output channels are 4x the base channels

#     def __init__(self, in_channels, out_channels, stride=1, downsample=None):
#         super(BottleneckBlock, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(out_channels)

#         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride,
#                                padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(out_channels)

#         self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = downsample
#         self.stride = stride

#     def forward(self, x):
#         identity = x

#         out = self.relu(self.bn1(self.conv1(x)))
#         out = self.relu(self.bn2(self.conv2(out)))
#         out = self.bn3(self.conv3(out))

#         if self.downsample is not None:
#             identity = self.downsample(x)

#         out += identity
#         return self.relu(out)


class ResNet(nn.Module):
    def __init__(self, block, num_blocks):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0])
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.fc = nn.Linear(64, 10)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        layers = []
        layers.append(block(self.in_channels, out_channels))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.adaptive_avg_pool2d(out,(1, 1))
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

class ResNet_general(nn.Module):
    def __init__(self, block, num_blocks, channel_dict):
        super(ResNet_general, self).__init__()
        
        self.in_channels = channel_dict['conv1'][0]
        self.conv1 = nn.Conv2d(channel_dict['conv1'][1], channel_dict['conv1'][0], kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channel_dict['conv1'][0])
        self.layer1 = self._make_layer(block, channel_dict['layer1.0.conv1'][0], num_blocks[0])
        self.layer2 = self._make_layer(block, channel_dict['layer2.0.conv1'][0], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, channel_dict['layer3.0.conv1'][0], num_blocks[2], stride=2)
        self.fc = nn.Linear(channel_dict['layer3.2.conv2'][0], 10)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        layers = []
        layers.append(block(self.in_channels, out_channels))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.adaptive_avg_pool2d(out,(1, 1))
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
    
# class ResNet_AnyDepth(nn.Module):
#     def __init__(self, block, layers, channel_dict, num_classes=1000):
#         super(ResNet_AnyDepth, self).__init__()
#         self.in_channels = channel_dict['conv1'][0]

#         self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
#         self.bn1 = nn.BatchNorm2d(self.in_channels)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

#         self.layer1 = self._make_layer(block, channel_dict, 'layer1', layers[0], stride=1)
#         self.layer2 = self._make_layer(block, channel_dict, 'layer2', layers[1], stride=2)
#         self.layer3 = self._make_layer(block, channel_dict, 'layer3', layers[2], stride=2)
#         self.layer4 = self._make_layer(block, channel_dict, 'layer4', layers[3], stride=2)

#         last_block = f'layer4.{layers[3]-1}.conv3'
#         final_out_channels = channel_dict[last_block][0]
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Linear(final_out_channels, num_classes)

#     def _make_layer(self, block, channel_dict, base_name, blocks, stride):
#         layers = []

#         first_block_in = self.in_channels
#         first_block_out = channel_dict[f'{base_name}.0.conv1'][0]
#         downsample = None
#         if stride != 1 or first_block_in != channel_dict[f'{base_name}.0.conv3'][0]:
#             downsample = nn.Sequential(
#                 nn.Conv2d(first_block_in, channel_dict[f'{base_name}.0.conv3'][0],
#                           kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(channel_dict[f'{base_name}.0.conv3'][0])
#             )

#         layers.append(block(first_block_in, first_block_out, stride, downsample))
#         self.in_channels = channel_dict[f'{base_name}.0.conv3'][0]

#         for i in range(1, blocks):
#             block_in = self.in_channels
#             block_out = channel_dict[f'{base_name}.{i}.conv1'][0]
#             layers.append(block(block_in, block_out))

#             self.in_channels = channel_dict[f'{base_name}.{i}.conv3'][0]

#         return nn.Sequential(*layers)

#     def forward(self, x):
#         x = self.relu(self.bn1(self.conv1(x)))
#         x = self.maxpool(x)

#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)

#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         x = self.fc(x)
#         return x


class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, conv1_out, conv2_out, conv3_out, stride=1, downsample=None):
        super(BottleneckBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, conv1_out, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(conv1_out)

        self.conv2 = nn.Conv2d(conv1_out, conv2_out, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(conv2_out)

        self.conv3 = nn.Conv2d(conv2_out, conv3_out, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(conv3_out)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return self.relu(out)

class ResNet_AnyDepth(nn.Module):
    def __init__(self, block, layers, channel_dict, num_classes=1000):
        super(ResNet_AnyDepth, self).__init__()
        self.in_channels = channel_dict['conv1'][0]

        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, channel_dict, 'layer1', layers[0], stride=1)
        self.layer2 = self._make_layer(block, channel_dict, 'layer2', layers[1], stride=2)
        self.layer3 = self._make_layer(block, channel_dict, 'layer3', layers[2], stride=2)
        self.layer4 = self._make_layer(block, channel_dict, 'layer4', layers[3], stride=2)

        last_block = f'layer4.{layers[3]-1}.conv3'
        final_out_channels = channel_dict[last_block][0]
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(final_out_channels, num_classes)

    def _make_layer(self, block, channel_dict, base_name, blocks, stride):
        layers = []

        for i in range(blocks):
            prefix = f'{base_name}.{i}'
            conv1_out = channel_dict[f'{prefix}.conv1'][0]
            conv2_out = channel_dict[f'{prefix}.conv2'][0]
            conv3_out = channel_dict[f'{prefix}.conv3'][0]

            block_stride = stride if i == 0 else 1

            downsample = None
            if block_stride != 1 or self.in_channels != conv3_out:
                downsample = nn.Sequential(
                    nn.Conv2d(self.in_channels, conv3_out,
                              kernel_size=1, stride=block_stride, bias=False),
                    nn.BatchNorm2d(conv3_out)
                )

            layers.append(block(self.in_channels, conv1_out, conv2_out, conv3_out,
                                stride=block_stride, downsample=downsample))
            self.in_channels = conv3_out

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def resnet_generator(arch='resnet18', channel_dict=None, num_classes=1000):
    if channel_dict is None:
        raise ValueError("channel_dict must be provided")
    if arch == 'resnet18':
        return ResNet_AnyDepth(BasicBlock, [2, 2, 2, 2], channel_dict, num_classes)
    elif arch == 'resnet34':
        return ResNet_AnyDepth(BasicBlock, [3, 4, 6, 3], channel_dict, num_classes)
    elif arch == 'resnet50':
        return ResNet_AnyDepth(BottleneckBlock, [3, 4, 6, 3], channel_dict, num_classes)
    elif arch == 'resnet101':
        return ResNet_AnyDepth(BottleneckBlock, [3, 4, 23, 3], channel_dict, num_classes)
    elif arch == 'resnet152':
        return ResNet_AnyDepth(BottleneckBlock, [3, 8, 36, 3], channel_dict, num_classes)