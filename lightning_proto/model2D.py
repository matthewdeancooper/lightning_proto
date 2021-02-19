# Copyright (C) 2020 Matthew Cooper

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(pl.LightningModule):
    @staticmethod
    def add_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--loss_function",
                            type=str,
                            default=F.binary_cross_entropy_with_logits)
        parser.add_argument("--optimizer", type=str, default=torch.optim.Adam)
        parser.add_argument("--encoder_args",
                            type=tuple,
                            default=(32, 64, 128, 256, 512, 1024)),
        parser.add_argument("--output_channels", type=int, default=1),
        parser.add_argument("--learning_rate", type=float, default=1e-3)
        return parser

    def __init__(self, loss_function, optimizer, encoder_args, output_channels,
                 learning_rate):
        super().__init__()
        print("\n-------------------------------------")
        print("\nLightningModule: __init__() - Running")
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.encoder_args = encoder_args
        self.output_channels = output_channels
        self.learning_rate = learning_rate
        print("\nBuilding layers...")
        self.build_layers()
        self.save_hyperparameters()
        print("\nLightningModule: __init__() - Completed")

    def build_layers(self):
        def _build_encoder_channels(encoder_args):
            input_arguments = 1, *encoder_args[:-1]
            return tuple(zip(input_arguments, encoder_args))

        def _build_decoder_channels(encoder_args):
            decoder_args = tuple(reversed(encoder_args))
            return tuple(zip(decoder_args[:-1], decoder_args[1:]))

        def _convolution_sequence(in_channels,
                                  out_channels,
                                  kernel_size,
                                  padding=1):
            sequence = nn.Sequential(
                nn.Conv2d(in_channels,
                          out_channels,
                          kernel_size,
                          padding=padding),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            )
            return sequence

        def _encoder_sequence(in_channels, out_channels):
            sequence = nn.Sequential(
                _convolution_sequence(in_channels, out_channels,
                                      kernel_size=3),
                nn.Dropout2d(p=0.2),
                _convolution_sequence(out_channels,
                                      out_channels,
                                      kernel_size=3),
                _convolution_sequence(out_channels,
                                      out_channels,
                                      kernel_size=1,
                                      padding=0),
            )
            return sequence

        def _decoder_sequence(in_channels, out_channels):
            sequence = nn.Sequential(
                _convolution_sequence(in_channels, out_channels,
                                      kernel_size=3),
                nn.Dropout2d(p=0.4),
                _convolution_sequence(out_channels,
                                      out_channels,
                                      kernel_size=3),
            )
            return sequence

        def _transposer_sequence(in_channels, out_channels):
            sequence = nn.Sequential(
                nn.ConvTranspose2d(in_channels,
                                   out_channels,
                                   kernel_size=2,
                                   stride=2,
                                   padding=0))
            return sequence

        def _output_sequence(in_channels, out_channels):
            sequence = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1))
            return sequence

        # Build encoder layers
        encoder_channels = _build_encoder_channels(self.encoder_args)
        print("\nEncoder channels")
        print(encoder_channels)
        self.encoders = nn.ModuleList()
        for args in encoder_channels:
            self.encoders.append(_encoder_sequence(*args))

        # Build transposers and decoders layers
        decoder_channels = _build_decoder_channels(self.encoder_args)
        print("\nDecoder channels")
        print(decoder_channels)
        self.transposers = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for args in decoder_channels:
            self.transposers.append(_transposer_sequence(*args))
            self.decoders.append(_decoder_sequence(*args))

        # Build output layer
        output_channels = (decoder_channels[-1][-1], self.output_channels)
        print("\nOutput channels")
        print(output_channels)
        self.output = _output_sequence(*output_channels)

    def forward(self, x):
        skips = []

        # Encoding x
        for encoder in self.encoders:
            x = encoder(x)
            if len(skips) < len(self.decoders):
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
        self.log("loss", loss)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        loss = self.loss_function(output, y)
        self.log("val_loss", loss, on_epoch=True)
        # return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        loss = self.loss_function(output, y)
        self.log("test_loss", loss, on_epoch=True)
        # return loss


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = UNet.add_specific_args(parser)
    args = parser.parse_args()

    model = UNet(
        args.loss_function,
        args.optimizer,
        args.encoder_args,
        args.output_channels,
        args.learning_rate,
    )

    print("\nLightningModule: Tests - Running")
    batch_size = 5
    test_input = torch.rand((batch_size, 1, 512, 512))
    test_output = model(test_input)
    assert test_output.shape == test_input.shape
    print("\nLightningModule: Tests - Completed")
