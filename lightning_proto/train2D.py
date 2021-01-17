import gc

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint)
from pytorch_lightning.loggers import TensorBoardLogger

from generator2D import DataModule
from model2D import UNet
from standard_utils import Combinations

# LightningDataModule
batch_size = 10
data_dir = "/home/matthew/github/vacunet/tests/canine_imaging_dataset/"
k_folds = 5

# Trainer
max_epochs = 10

early_stopping = EarlyStopping('val_loss')
lr_monitor = LearningRateMonitor(logging_interval='epoch')
# checkpoint_callback = ModelCheckpoint(monitor='val_loss')
# callbacks = [early_stopping, lr_monitor, checkpoint_callback]
callbacks = [early_stopping, lr_monitor]

#-------------------------------------------------------
run_params = {"k_fold_index": range(k_folds)}
runs = Combinations.get_combinations(run_params)

for run in runs:
    print("New run:", run)

    dm = DataModule(data_dir, batch_size, k_folds, run.k_fold_index)
    dm.prepare_data()
    dm.setup()

    model = UNet()

    logger = TensorBoardLogger('tb_logs', name=str(run))

    trainer = pl.Trainer(callbacks=callbacks,
                         gpus=1,
                         precision=16,
                         max_epochs=max_epochs,
                         limit_train_batches=0.01,
                         logger=logger)

    trainer.fit(model, dm)

    trainer.test(datamodule=dm)

    del trainer
    del model
    del dm
    # del logger
    # del early_stopping
    # del lr_monitor
    # del checkpoint_callback
    # del callbacks
    torch.cuda.empty_cache()
    gc.collect()

# if __name__ == '__main__':
#     cli_main()
