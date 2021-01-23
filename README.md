# lightning_prototype

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
Run the train.py file specifying any argument required. For example:
```bash
python3 train.py --batch_size=10 --gpus=1 --precision=16
```

Accepted arguments are outlined in `model2d.UNET`, `generator2d.DataModule`, and the pytorch_lightning `Trainer` module, [hyperlinked here](https://pytorch-lightning.readthedocs.io/en/stable/_modules/pytorch_lightning/trainer/trainer.html). The default dataset used is the `test_dataset` included in this repository.

## TODO
- Implement data augmentation in `augmentation2D`.
- Implement downloading and preparing data script from PyMedPhys in `generator2D.DataModule.prepare_data`.
- Examine which callbacks pytorch_lightning handles automatically.
- Translate existing TensorFlow code for inference.
- Translate existing TensorFlow metrics code for custom loss functions.
- Merge existing DICOM networking code for deployment.
- Merge with PyMedPhys
