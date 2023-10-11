# Download the DeepCrack.zip file and place it in this folder
unzip DeepCrack.zip

rm -rf test train

# create the data folders in the required format
mkdir -p train/images
mkdir -p train/masks
mkdir -p test/images
mkdir -p test/masks

# move the images and masks into the previously created folders
mv train_img/* train/images*
mv train_lab/* train/masks*
mv test_img/* test/images*
mv test_lab/* test/masks*

rm -rf train_lab train_img test_lab test_img