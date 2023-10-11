## TrueDeep: A Systematic of Crack Detection with Less Data
### Installation
- Python Version: 3.8.10
- Install the requirements mentioned in the `requirements.txt`.

### Datasets

- DeepCrack
  - This is a crack segmentation dataset that can be downloaded from the ![https://github.com/yhlleo/DeepCrack/blob/master/dataset/DeepCrack.zip](DeepCrack) repository.
- FIVES
  - This is a vessel segmentation dataset that can be downloaded from ![https://figshare.com/ndownloader/files/34969398](here).
- BCCD
  - This is a blood cell segmetation dataset that can be downloaded from ![https://www.kaggle.com/datasets/jeetblahiri/bccd-dataset-with-mask](here).

We have used binary segmentation datasets to perform the experiments but feel free to choose any image dataset to subset your overall dataset.

### Workarounds
-  If you're getting the error "ImportError: cannot import name 'MultiHeadAttention' from 'tensorflow.keras.layers'", comment out the corresponding lines of code from the `keras_unet_collection` source library.
