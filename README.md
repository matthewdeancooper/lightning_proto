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

Accepted arguments are outlined in `model2D.UNET`, `generator2D.DataModule`, and the pytorch_lightning `Trainer` module, [hyperlinked here](https://pytorch-lightning.readthedocs.io/en/stable/_modules/pytorch_lightning/trainer/trainer.html). The default dataset used is the `test_dataset` included in this repository.

## Inference
To be implemented...
