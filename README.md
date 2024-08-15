## [TrueDeep: A Systematic Approach of Crack Detection with Less Data](https://www.sciencedirect.com/science/article/abs/pii/S0957417423032876)
### Installation
- Python >= 3.6 Recommended
- Install the requirements mentioned in the `requirements.txt`.

### Datasets

- **DeepCrack**
  - This is a crack segmentation dataset that can be downloaded from the [DeepCrack](https://github.com/yhlleo/DeepCrack/blob/master/dataset/DeepCrack.zip) repository.
- **kaggle-crack-segmentation**
  - This is a crack segmetation dataset containing images from diverse backgrounds and is available [here](https://www.kaggle.com/datasets/lakshaymiddha/crack-segmentation-dataset).
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
  - `augmentation` folder contains the implementation of augmentation techniques performed on crack masks, run `augment_data.sh` to perform the augmentation on the input images.
  - `evaluation` folder contains the implementation of inference and computation of metrics.
    - `inference.py` contains the code to perform inference using a trained model and inference can be performed using the script `inference.sh`.
  - F-score, Precision and Recall can be computed by running `evaluate.sh`. To compute more metrics like global accuracy and class average accuracy, please refer to the [DeepSegmentor](https://github.com/yhlleo/DeepSegmentor/tree/master/eval) repository.

### Model Checkpoints
  - The model checkpoints are for truedeep are available [here](https://drive.google.com/drive/folders/18Ytylwl37ItO8PQKhccitUeJM8V23m1u?usp=drive_link).

### Workarounds
-  If you're getting the error **ImportError: cannot import name 'MultiHeadAttention' from 'tensorflow.keras.layers'**, comment out the corresponding lines of code from the `keras_unet_collection` source library.

### Acknowledgements

We would like to thank the authors of the DeepCrack, FIVES and BCCD for their contributions to the research community and for making this research possible.

### Reference
If you find the code useful for your research, please cite our paper:

```
@article{pandey2023truedeep,
  title={TrueDeep: A systematic approach of crack detection with less data},
  author={Pandey, Ramkrishna and Achara, Akshit},
  journal={Expert Systems with Applications},
  pages={122785},
  year={2023},
  publisher={Elsevier}
}
```

If you are using the stochastic width augmentations from the code, please cite our paper:

```
@inproceedings{pandey2023coredeep,
  title={CoreDeep: Improving crack detection algorithms using width stochasticity},
  author={Pandey, Ramkrishna and Achara, Akshit},
  booktitle={International Conference on Computer Vision and Image Processing},
  pages={62--73},
  year={2023},
  organization={Springer}
}
```