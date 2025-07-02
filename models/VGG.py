import torch
import torch.nn as nn
import torch.nn.functional as F


class VGG(nn.Module):

    """
        Generalized VGG model.
        
        Args:
        - conv_config: A dictionary where each key is the block number (1-indexed) and the value
                       is a list of integers representing the number of filters in each conv layer.
        - num_classes: Number of output classes for the classifier.
        - input_channels: Number of input channels (e.g., 3 for RGB images).
        """

    def __init__(
        self,
        conv_config,
        num_classes: int = 10,
        init_weights: bool = False,
        input_channels=3
    ) -> None:
        super(VGG, self).__init__()
        self.features = self._make_layers(conv_config, input_channels)
        self.classifier = nn.Sequential(
            nn.Linear(conv_config[list(conv_config.keys())[-1]][-1], 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _make_layers(self, conv_config, input_channels):
        layers = []
        for block_idx, filters in conv_config.items():
            for out_channels in filters:
                layers.append(nn.Conv2d(input_channels, out_channels, kernel_size=3, padding=1))
                layers.append(nn.BatchNorm2d(out_channels))
                layers.append(nn.ReLU(inplace=True))
                input_channels = out_channels  # Update input channels for next layer
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))  # Add max pooling after each block
        return nn.Sequential(*layers)

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def VGG_General(channel_dict, variant="VGG16"):

    conv_config = {}
    current_block = 1
    block_layers = []

    # Sort keys numerically for proper order
    sorted_keys = sorted(channel_dict.keys(), key=lambda x: int(x.split('.')[1]))

    for key in sorted_keys:
        out_channels, in_channels = channel_dict[key]
        block_layers.append(out_channels)

        # Determine block boundaries based on VGG pooling structure
        if key.endswith(('3', '10', '20', '30', '40') ):  # End of a block
            conv_config[current_block] = block_layers
            current_block += 1
            block_layers = []

    return VGG(conv_config)


class VGG_AnyDepth(nn.Module):
    def __init__(self, channel_dict, num_classes=1000):
        super(VGG_AnyDepth, self).__init__()
        self.features, last_channels = self._make_features(channel_dict)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        # FOR IMAGENET
        self.classifier = nn.Sequential(
            nn.Linear(last_channels * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )

        # FOR CIFAR
        # self.classifier = nn.Sequential(
        #     nn.Linear(last_channels * 7 * 7, 512),
        #     nn.ReLU(True),
        #     nn.Dropout(),
        #     nn.Linear(512, 512),
        #     nn.ReLU(True),
        #     nn.Dropout(),
        #     nn.Linear(512, num_classes))

    def _make_features(self, channel_dict):
        layers = []
        last_channels = None

        sorted_keys = sorted(channel_dict.keys(), key=lambda x: int(x.split('.')[1]))

        for key in sorted_keys:
            out_ch, in_ch = channel_dict[key]
            idx = int(key.split('.')[1])

             # Special handling for the first layer: always 3 input channels
            if key == 'features.0':
                in_ch = 3

            if in_ch is not None and out_ch is not None:
                layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False))
                layers.append(nn.BatchNorm2d(out_ch))
                layers.append(nn.ReLU(inplace=True))
                last_channels = out_ch

            # Optional: Add pooling at typical VGG locations
            if str(idx) in {'4', '11', '21', '31', '41'}:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        return nn.Sequential(*layers), last_channels

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x