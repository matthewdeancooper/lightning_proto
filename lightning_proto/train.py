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
# import torch
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint)

from generator2D import DataModule
from model2D import UNet


def main(args):
    # DataModule
    dm = DataModule.from_argparse_args(args)
    dm.prepare_data()
    dm.setup()

    # LightningModule
    model = UNet(
        args.loss_function,
        args.optimizer,
        args.encoder_args,
        args.output_channels,
        args.learning_rate,
    )

    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    early_stopping = EarlyStopping('val_loss')
    checkpoint = ModelCheckpoint(monitor='val_loss')
    callbacks = [early_stopping, lr_monitor, checkpoint]

    # Trainer
    trainer = pl.Trainer(callbacks=callbacks)
    trainer = trainer.from_argparse_args(args)
    # trainer = pl.Trainer.from_argparse_args(args)

    print("\n-------------------------------------")
    print("TRAINING::")
    trainer.fit(model, dm)
    print("\nTRAINING COMPLETED\n")

    print("\n-------------------------------------")
    print("TESTING:")
    trainer.test(datamodule=dm)
    print("\nTESTING COMPLETED\n")


if __name__ == "__main__":
    # ArgumentParser
    parser = ArgumentParser()
    parser = DataModule.add_specific_args(parser)
    parser = UNet.add_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    main(args)
