from torch import nn
import ai8x

class Fruits360(nn.Module):
    """
    Simplified CNN model for fruit classification on AI85/AI86
    """
    def __init__(self, num_classes, dimensions=(100, 100), num_channels=3, bias=True, **kwargs):
        super().__init__()

        # Layer 1: Convolution + ReLU + Max Pooling
        self.conv1 = ai8x.FusedConv2dReLU(num_channels, 16, 3, padding=1, bias=False)
        self.pool1 = ai8x.FusedMaxPoolConv2dReLU(16, 16, 3, pool_size=2, pool_stride=2, padding=1, bias=False)

        # Layer 2: Convolution + ReLU + Max Pooling
        self.conv2 = ai8x.FusedConv2dReLU(16, 32, 3, padding=1, bias=bias)
        self.pool2 = ai8x.FusedMaxPoolConv2dReLU(32, 32, 3, pool_size=2, pool_stride=2, padding=1, bias=bias)

        # Layer 3: Convolution + ReLU + Max Pooling
        self.conv3 = ai8x.FusedConv2dReLU(32, 64, 3, padding=1, bias=bias)
        self.pool3 = ai8x.FusedMaxPoolConv2dReLU(64, 64, 3, pool_size=2, pool_stride=2, padding=1, bias=False)

        # # Placeholder for the global average pooling
        # self.avg_pool_size = None  # To be set based on the actual output size of pool3

        # Adaptive Avg Pooling
         # Custom Global Average Pooling
        self.global_avg_pool = ai8x.AvgPool2d(kernel_size=(12, 12), stride=(12, 12))

        # Fully connected layer
        self.fcx = ai8x.Linear(64, num_classes, wide=True, bias=True, **kwargs)  # Changed to 64 to match the output of 64 feature maps

    def forward(self, x):
        """Forward propagation"""
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.global_avg_pool(x)

        # # Dynamically set the pool size based on the output dimensions of pool3
        # if self.avg_pool_size is None:
        #     self.avg_pool_size = (x.size(2), x.size(3))  # Assuming HxW are equal and known here
        #     print(f"Global average pooling size set to: {self.avg_pool_size}")

        # Flatten the output for the fully connected layer
        x = x.view(x.size(0), -1)
        x = self.fcx(x)
        return x


def fruits360(pretrained=False, num_classes=131, **kwargs):
    """
    Constructs a simple fruit classifier model for AI85/AI86.
    """
    assert not pretrained, "Pretrained models are not available for this architecture."
    return Fruits360(num_classes=num_classes, **kwargs)


models = [
    {
        'name': 'fruits360',
        'min_input': 1,
        'dim': 3,
        'num_classes': 131,  # Adjust based on the number of fruit classes in the dataset
    },
]