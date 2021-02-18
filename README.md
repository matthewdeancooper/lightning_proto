# lightning_prototype

* Assumes data pipeline assumes data is organised as in the test_dataset folder.  
  `test_dataset/<patient>/img/<index>.npy` matches with `test_dataset/<patient>/mask/<index>.npy`

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
