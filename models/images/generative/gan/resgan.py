import torch
from torch import nn


class ConvolutionalBlock(nn.Module):
    def __init__(self, input_channels, output_channels, stride, kernel_size=3, leaky_relu_param=0.2):
        super(ConvolutionalBlock, self).__init__()

        self.leaky_relu_param = leaky_relu_param
        self.kernel_size = kernel_size

        self.conv = nn.Conv2d(in_channels=input_channels,
                              out_channels=output_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=1)
        self.bn = nn.BatchNorm2d(output_channels)
        self.lrelu = nn.LeakyReLU(leaky_relu_param)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.lrelu(x)

        return x


class ResDiscriminator(nn.Module):
    def __init__(self, in_channels=3, res_ndf=64, leaky_relu_param=0.2):
        super(ResDiscriminator, self).__init__()

        # number of features in the first conv of discriminator, default = 64
        self.res_ndf = res_ndf

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=self.res_ndf,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.LeakyReLU(leaky_relu_param)
        )

        self.block2 = nn.Sequential(
            ConvolutionalBlock(self.res_ndf, self.res_ndf, 2),
            ConvolutionalBlock(self.res_ndf, self.res_ndf * 2, 1),

            ConvolutionalBlock(self.res_ndf * 2, self.res_ndf * 2, 2),
            ConvolutionalBlock(self.res_ndf * 2, self.res_ndf * 4, 1),

            ConvolutionalBlock(self.res_ndf * 4, self.res_ndf * 4, 2),
            ConvolutionalBlock(self.res_ndf * 4, self.res_ndf * 8, 1),

            ConvolutionalBlock(self.res_ndf * 8, self.res_ndf * 8, 2),
        )

        self.block3 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=self.res_ndf * 8,
                      out_channels=self.res_ndf * 8,
                      kernel_size=1,
                      stride=1,
                      padding=0),
            nn.LeakyReLU(leaky_relu_param),
            nn.Conv2d(in_channels=self.res_ndf * 8,
                      out_channels=1,
                      kernel_size=1,
                      padding=0)
        )

    def forward(self, x):
        batch_size = x.size(0)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = torch.sigmoid(x.view(batch_size))

        return x


class ResidualBlock(nn.Module):
    def __init__(self, k=3, n=64, s=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=n,
                               out_channels=n,
                               kernel_size=k,
                               stride=s,
                               padding=1)
        self.bn1 = nn.BatchNorm2d(n)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(in_channels=n,
                               out_channels=n,
                               kernel_size=k,
                               stride=s,
                               padding=1)
        self.bn2 = nn.BatchNorm2d(n)

    def forward(self, x):
        y = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.conv2(x)
        x = self.bn2(x)

        return x + y


class UpsampleBlock(nn.Module):
    def __init__(self, input_channels, scale=2):
        super(UpsampleBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels=input_channels,
                              out_channels=(scale ** 2) * input_channels,
                              kernel_size=3,
                              stride=1,
                              padding=1)
        self.pixshuff = nn.PixelShuffle(scale)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixshuff(x)
        x = self.prelu(x)

        return x


class ResGenerator(nn.Module):
    def __init__(self, input_size=128, out_channels=3, upsample_blocks_num=2, residual_blocks_num=8, res_ngf=64,
                 start_img_size=16):
        super(ResGenerator, self).__init__()

        # number of features in generator, default = 64
        self.res_ngf = res_ngf

        # generator input size (generative noise size), default = 128
        self.input_size = input_size

        # size of initial image transformed from a linear, default = 16
        self.start_img_size = start_img_size

        # number of residual blocks, default = 8
        self.residual_blocks_num = residual_blocks_num

        # number of upsample blocks (each blocks = x2 scaling), default = 2 (x4 scale factor)
        self.upsample_blocks_num = upsample_blocks_num

        self.dense1 = nn.Linear(in_features=self.input_size,
                                out_features=self.res_ngf * self.start_img_size * self.start_img_size)
        self.bn1 = nn.BatchNorm2d(self.res_ngf)
        self.relu1 = nn.ReLU()

        self.res_blocks = nn.Sequential(
            *[ResidualBlock(n=self.res_ngf) for i in range(residual_blocks_num)]
        )

        self.bn2 = nn.BatchNorm2d(res_ngf)
        self.relu2 = nn.ReLU()

        self.upsample_blocks = nn.Sequential(
            *[UpsampleBlock(input_channels=self.res_ngf) for i in range(self.upsample_blocks_num)]
        )

        self.conv1 = nn.Conv2d(in_channels=self.res_ngf,
                               out_channels=out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = self.dense1(x)
        x = x.view(-1, self.res_ngf, self.start_img_size, self.start_img_size)
        x = self.bn1(x)
        state1 = self.relu1(x)
        x = self.res_blocks(state1)
        x = self.bn2(x)
        x = self.relu2(x)
        x += state1
        x = self.upsample_blocks(x)
        x = self.conv1(x)
        x = self.tanh(x)

        return x
