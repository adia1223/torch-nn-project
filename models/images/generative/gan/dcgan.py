import math

from torch import nn

NGF = 128


class DCGANDownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=4, stride=2, padding=1, bias=False, leaky_relu=0.2):
        super(DCGANDownsampleBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
            bias=bias
        )
        self.bn = nn.BatchNorm2d(
            num_features=out_channels
        )
        self.lr = nn.LeakyReLU(
            negative_slope=leaky_relu
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.lr(x)

        return x


class DCGANDiscriminator(nn.Module):
    def __init__(self, image_size=128, ndf=128, in_channels=3):
        super(DCGANDiscriminator, self).__init__()

        self.ndf = ndf
        self.downsample_num = int(math.log2(image_size)) - 3
        self.in_channels = in_channels

        self.block1 = nn.Sequential(
            nn.Conv2d(self.in_channels, self.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.block2 = nn.Sequential(
            *[
                DCGANDownsampleBlock(in_channels=self.ndf * (2 ** i), out_channels=self.ndf * (2 ** (i + 1)), kernel=4)
                for i in range(self.downsample_num)
            ]
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(self.ndf * (2 ** self.downsample_num), 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        return x


class DCGANUpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=4, stride=2, padding=1, bias=False):
        super(DCGANUpsampleBlock, self).__init__()
        self.convtr = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
            bias=bias
        )
        self.bn = nn.BatchNorm2d(
            num_features=out_channels
        )
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x = self.convtr(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class DCGANGenerator(nn.Module):
    def __init__(self, image_size=128, input_size=128, ngf=128, out_channels=3):
        super(DCGANGenerator, self).__init__()

        self.ngf = ngf
        self.input_size = input_size
        self.upsample_num = int(math.log2(image_size)) - 3
        self.out_channels = out_channels

        self.block1 = DCGANUpsampleBlock(
            in_channels=self.input_size,
            out_channels=self.ngf * (2 ** self.upsample_num),
            kernel=4,
            stride=1,
            padding=0
        )

        self.block2 = nn.Sequential(
            *[
                DCGANUpsampleBlock(in_channels=self.ngf * (2 ** (i + 1)), out_channels=self.ngf * (2 ** i), kernel=4)
                for i in range(self.upsample_num - 1, -1, -1)
            ]
        )

        self.block3 = nn.Sequential(
            nn.ConvTranspose2d(self.ngf, self.out_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        return x
