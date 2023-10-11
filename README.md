## TrueDeep: A Systematic of Crack Detection with Less Data
### Installation
- Python Version: 3.8.10
- Install the requirements mentioned in the `requirements.txt`.

### Datasets

- **DeepCrack**
  - This is a crack segmentation dataset that can be downloaded from the [DeepCrack](https://github.com/yhlleo/DeepCrack/blob/master/dataset/DeepCrack.zip) repository.
- **FIVES**
  - This is a vessel segmentation dataset that can be downloaded from [here](https://figshare.com/ndownloader/files/34969398).
- **BCCD**
  - This is a blood cell segmetation dataset that can be downloaded from [here](https://www.kaggle.com/datasets/jeetblahiri/bccd-dataset-with-mask).

### Details

- We assume that a folder containing the dataset must be structured like `{train-test-split}/{images-or-masks}/{filename.ext}`, for example, `train/images/10.jpg`.

- `subset` folder contains the logic to obtain a coreset of images from an overall dataset. Choose any image dataset to subset your overall dataset and
save the subset. Masks can also be saved if a binary segmentation dataset is used.

- `segmentation` folder contains the training methods to train the models for a binary segmentation task on the above mentioned datasets.
  - `training` folder contains the files required to perform training for binary segmentation, run `train.sh` with the required parameters to perform the training.
  - `inference.py` contains the code to perform inference using a trained model and inference can be performed using the script `inference.sh`.
  - Evaluation can be performed using the code provided in the [DeepSegmentor](https://github.com/yhlleo/DeepSegmentor/tree/master/eval) repository.

### Workarounds
-  If you're getting the error **ImportError: cannot import name 'MultiHeadAttention' from 'tensorflow.keras.layers'**, comment out the corresponding lines of code from the `keras_unet_collection` source library.

### Reference
If you find the code useful for your research, please cite our paper:

```
@article{pandey2023truedeep,
  title={TrueDeep: A systematic approach of crack detection with less data},
  author={Pandey, Ram Krishna and Achara, Akshit},
  journal={arXiv preprint arXiv:2305.19088},
  year={2023}
}
```
