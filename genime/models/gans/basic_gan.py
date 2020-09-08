import torch
from torch.nn import Module, Sequential
from torch.nn import (
    BatchNorm2d
    Conv2d,
    ConvTransposed2d,
    LeakyReLU,
    Sigmoid,
)

from pytorch_lightning import LightningModule


class BasicGenerator(Module):
    def __init__(
            self,
            ngf: int,  # Size of feature maps in generator
        ):
        super().__init__()
        self.ngf = ngf
        self.main = Sequential()

    def forward(self, z):
        output = self.main(z)

        return output


class BasicDiscriminatorBlock(Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
        ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.main = Sequential(
            Conv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            BatchNorm2d(
                num_features=out_channels,
                momentum=0.2,
            ),
            LeakyReLU(
                negative_slope=0.2,
                inplace=True,
            ),
        )

    def forward(self, x):
        output = self.main(x)

        return output


class BasicDiscriminator(Module):  #TODO calc shapes
    def __init__(
            self,
            ndf: int,  # Size of feature maps in discriminator
        ):
        super().__init__()
        self.ndf = ndf
        self.main = Sequential(
            Conv2d(
                in_channels=3,
                out_channels=self.ndf,
                kernel_size=3,
                padding=1,
            ),
            LeakyReLU(
                negative_slope=0.2,
                inplace=True,
            ),
            BasicDiscriminatorBlock(
                in_channels=self.ndf,
                out_channels=self.ndf * 2,
            ),
            BasicDiscriminatorBlock(
                in_channels=self.ndf * 2,
                out_channels=self.ndf * 4,
            ),
            BasicDiscriminatorBlock(
                in_channels=self.ndf * 4,
                out_channels=self.ndf * 8,
            ),
        )

    def forward(self, x):
        pass


class BasicGAN(LightningModule):
    def __init__(
            self,
        ):
        super().__init__()
        self.generator = BasicGenerator()
        self.discriminator = BasicDiscriminator()

    def forward(self, z):
        return self.generator(z)

    def generator_loss(self):
        pass

    def discriminator_loss(self):
        pass

    def generator_step(self):
        pass

    def discriminator_step(self):
        pass

    def training_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        pass

    @static_method
    def add_model_specific_args(parent_parser, args_info):
        parser = ArgumentParser(
            parents=[parent_parser],
            add_help=False,
        )

        for arg_name in args_info:
            parser.add_argument(
                f'--{arg_name}',
                type=args_info[arg_name]['type'],
                default=args_info[arg_name]['default'],
            )

        return parser

