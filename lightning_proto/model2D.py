import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

# from PIL import Image


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
    def __init__(self, encoder_args=None, output_channels=1):
        super().__init__()
        if encoder_args:
            self.encoder_args = encoder_args
        else:
            # Default args
            self.encoder_args = [
                (1, 32),  # x
                (32, 64),  # x/2
                (64, 128),  # x/4
                (128, 256),  # x/8
                (256, 512),  # x/16
                (512, 1024)  # /32
            ]
        # Reverse each tuple in a reversed list. Exclude last element
        self.decoder_args = [arg[::-1] for arg in self.encoder_args[::-1]][:-1]
        self.out_args = (self.decoder_args[-1][-1], output_channels)

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

    def loss_function(self, output, y):
        loss = F.binary_cross_entropy_with_logits(output, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer),
            'monitor': 'val_loss',
            'interval': 'epoch',
        }
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_index):
        x, y = batch
        output = self(x)
        loss = self.loss_function(output, y)
        self.log('loss', loss)
        return {'loss': loss}

    def validation_step(self, batch, batch_index):
        x, y = batch
        output = self(x)
        val_loss = self.loss_function(output, y)
        self.log('val_loss', val_loss)

    def test_step(self, batch, batch_index):
        x, y = batch
        output = self(x)
        test_loss = self.loss_function(output, y)
        self.log('test_loss', test_loss)


# if __name__ == "__main__":
#     model = UNet()
#     ouput = model(torch.rand(1, 1, 512, 512))
#     assert output.shape == (1, 512, 512)
