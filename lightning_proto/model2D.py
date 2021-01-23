import pytorch_lightning as pl
from argparse import ArgumentParser
import torch
import torch.nn as nn
import torch.nn.functional as F

def convolution_sequence(in_channels, out_channels, kernel_size, padding=1):
    sequence = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
    )
    return sequence


def encoder_sequence(in_channels, out_channels):
    sequence = nn.Sequential(
        convolution_sequence(in_channels, out_channels, kernel_size=3),
        nn.Dropout2d(p=0.2),
        convolution_sequence(out_channels, out_channels, kernel_size=3),
        convolution_sequence(out_channels,
                             out_channels,
                             kernel_size=1,
                             padding=0),
    )
    return sequence


def decoder_sequence(in_channels, out_channels):
    sequence = nn.Sequential(
        convolution_sequence(in_channels, out_channels, kernel_size=3),
        nn.Dropout2d(p=0.4),
        convolution_sequence(out_channels, out_channels, kernel_size=3),
    )
    return sequence


def transposer_sequence(in_channels, out_channels):
    sequence = nn.Sequential(
        nn.ConvTranspose2d(in_channels,
                           out_channels,
                           kernel_size=2,
                           stride=2,
                           padding=0))
    return sequence


def output_sequence(in_channels, out_channels):
    sequence = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=1))
    return sequence


class UNet(pl.LightningModule):
    @staticmethod
    def add_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--loss_function', type=str, default='F.binary_cross_entropy_with_logits')
        parser.add_argument('--optimizer', type=str, default='torch.optim.Adam')
        parser.add_argument('--encoder_args', type=tuple, default=(
                (1, 32),  # x
                (32, 64),  # x/2
                (64, 128),  # x/4
                (128, 256),  # x/8
                (256, 512),  # x/16
                (512, 1024)  # /32
            )
        )
        parser.add_argument('--output_channels', type=int, default=1)
        parser.add_argument('--learning_rate', type=float, default=1e-3)
        return parser

    def __init__(self, loss_function, optimizer, encoder_args, output_channels, learning_rate):
        super().__init__()
        self.loss_function = eval(loss_function)
        self.optimizer = eval(optimizer)
        self.encoder_args = encoder_args
        self.output_channels = output_channels
        self.learning_rate = learning_rate
        self.init_layers()

    def init_layers(self):
        # Reverse each tuple in a reversed list. Exclude last element
        self.decoder_args = [arg[::-1] for arg in self.encoder_args[::-1]][:-1]
        self.out_args = (self.decoder_args[-1][-1], self.output_channels)

        # Build encoder layers
        self.encoders = nn.ModuleList()
        for args in self.encoder_args:
            self.encoders.append(encoder_sequence(*args))

        # Build transposers and decoders layers
        self.transposers = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for args in self.decoder_args:
            self.transposers.append(transposer_sequence(*args))
            self.decoders.append(decoder_sequence(*args))

        # Build output layer
        self.output = output_sequence(*self.out_args)

    def forward(self, x):
        skips = []

        # Encoding x
        for encoder in self.encoders:
            x = encoder(x)
            if len(skips) < len(self.decoder_args):
                skips.append(x)
                x = nn.MaxPool2d(kernel_size=2)(x)

        skips.reverse()

        # Decoding x
        for decoder, transposer, skip in zip(self.decoders, self.transposers,
                                             skips):
            x = transposer(x)
            x = torch.cat([x, skip], dim=1)
            x = decoder(x)

        return self.output(x)

    # def loss_function(self, output, y):
    #     loss = F.binary_cross_entropy_with_logits(output, y)
    #     return loss

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), self.learning_rate)
        # scheduler = {
        #     'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer),
        #     'monitor': 'val_loss',
        #     'interval': 'epoch',
        # }
        # return [optimizer], [scheduler]
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        loss = self.loss_function(output, y)
        self.log('loss', loss)
        return {'loss':loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        loss = self.loss_function(output, y)
        self.log('val_loss', loss, on_epoch=True)
        # return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        loss = self.loss_function(output, y)
        self.log('test_loss', loss, on_epoch=True)
        # return loss

    # def train_epoch_end(self, outputs):
    #     average_train_loss = torch.tensor([x['loss'] for x in outputs]).mean()
    #     self.log('average_train_loss', average_train_loss)

    # def validation_epoch_end(self, outputs):
    #     average_val_loss = torch.tensor([x['loss'] for x in outputs]).mean()
    #     self.log('average_val_loss', average_val_loss)

    # def test_epoch_end(self, outputs):
    #     average_test_loss = torch.tensor([x['loss'] for x in outputs]).mean()
    #     self.log('average_test_loss', average_test_loss)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = UNet.add_specific_args(parser)
    args = parser.parse_args()

    # NOTE UNet.from_argparse_args(args) not defined in Lightning Module
    model = UNet(args.loss_function,
                 args.optimizer,
                 args.encoder_args,
                 args.output_channels,
                 args.learning_rate)

    batch_size = 2
    input = torch.rand((batch_size, 1, 512, 512))
    output = model(input)

    print(output.shape)
