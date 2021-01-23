import gc

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint)
from pytorch_lightning.loggers import TensorBoardLogger

from generator2D import DataModule
from model2D import UNet
from argparse import ArgumentParser


def main(args):
    # DataModule
    dm = DataModule.from_argparse_args(args)
    dm.prepare_data()
    dm.setup()

    # LightningModule
    # NOTE UNet.from_argparse_args(args) not defined in Lightning Module
    model = UNet(args.loss_function,
                 args.optimizer,
                 args.encoder_args,
                 args.output_channels,
                 args.learning_rate)

    # Callbacks
    # TODO pytorch_lightning handles some automatically - investigate!
    # lr_monitor = LearningRateMonitor(logging_interval='epoch')
    # early_stopping = EarlyStopping('val_loss')
    # checkpoint = ModelCheckpoint(monitor='val_loss')
    # callbacks = [early_stopping, lr_monitor, checkpoint]

    # Trainer
    trainer = pl.Trainer.from_argparse_args(args)

    print("\n-------------------------------------")
    print("TRAINING::")
    trainer.fit(model, dm)
    print("\nTRAINING COMPLETED\n")

    print("\n-------------------------------------")
    print("TESTING:")
    trainer.test(datamodule=dm)
    print("\nTESTING COMPLETED\n")

if __name__ == '__main__':
    # ArgumentParser
    parser = ArgumentParser()
    parser = DataModule.add_specific_args(parser)
    parser = UNet.add_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    main(args)
