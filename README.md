# lightning_prototype

* Training pipeline assumes data is organised as in the test_dataset folder. The DataModule handles paths, which are read to arrays per batch for training.   
  The image file `test_dataset/<patient>/img/<array>.npy` matches with a mask `test_dataset/<patient>/mask/<array>.npy`.
* Inference pipeline assumes data is a directory path to a DICOM imaging series. Each image in a series will be converted to a array for inference.

## Setup

Activate a python virtual environment
```bash
python3 -m venv env
source env/bin/activate
```
Install requirements via pip
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Training

Run the train.py file specifying any argument required. For example:
```bash
python3 train.py --batch_size=10 --gpus=1 --precision=16
```

Accepted arguments are outlined in `model2D.UNet`, `generator2D.DataModule`, and the pytorch_lightning `Trainer` module, [hyperlinked here](https://pytorch-lightning.readthedocs.io/en/stable/_modules/pytorch_lightning/trainer/trainer.html). The default dataset used is the `test_dataset` included in this repository. 

File:`generator2D.DataModule`  
```python
class DataModule(pl.LightningDataModule):
    @staticmethod
    def add_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--data_dir", type=str, default="../test_dataset")
        parser.add_argument("--batch_size", type=int, default=5)
        parser.add_argument("--k_folds", type=int, default=5)
        parser.add_argument("--k_fold_index", type=int, default=0)
        parser.add_argument("--input_shape", type=tuple, default=(1, 512, 512))
        parser.add_argument("--num_workers", type=int, default=12)
        return parser
```

File:`model2D.UNet` 
```python
class UNet(pl.LightningModule):
    @staticmethod
    def add_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--loss_function",type=str, default=F.binary_cross_entropy_with_logits)
        parser.add_argument("--optimizer", type=str, default=torch.optim.Adam)
        parser.add_argument("--encoder_args",type=tuple, default=(32, 64, 128, 256, 512, 1024)),
        parser.add_argument("--output_channels", type=int, default=1),
        parser.add_argument("--learning_rate", type=float, default=1e-3)
        return parser

```
These arguments can all be passed directly via the command line interface to `train.py` shown above.

## Inference
To be implemented...
