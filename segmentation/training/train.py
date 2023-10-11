import sys
import numpy as np
import glob
from tensorflow import keras
from tensorflow.python.keras.callbacks import ReduceLROnPlateau
from keras_unet_collection import models, losses
import os
import data_generator
import random
import segmentation_models as sm
sm.set_framework("tf.keras")
sm.framework()
import argparse

n_classes=2
MODEL_SAVE_PATH = os.environ.get("MODEL_SAVE_PATH")


def get_list_of_files(dirName):
    listOfFile = os.listdir(dirName)
    print(dirName)
    files = []
    for file in listOfFile:
        filename =os.fsdecode(file)
        files.append(filename)
    print(files, len(files))
    return files

def get_learningRate_callback():
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                    patience=50,
                                    min_lr=1e-6)
    return reduce_lr

def get_train_val(train_dirpath, val_dirpath):
    train_image_files = get_list_of_files(os.path.join(train_dirpath, 'images'))
    val_image_files = get_list_of_files(os.path.join(val_dirpath, 'images'))
    print(f"The number of training files: {len(train_image_files)}")
    print(f"The number of validation files: {len(val_image_files)}")

    return (train_image_files, val_image_files)

class CustomSaver(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch in [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]:
            self.model.save(MODEL_SAVE_PATH + "/epoch-{}.hd5".format(epoch))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', default="EfficientNetB0")
    parser.add_argument('--batch_size', default=8)
    parser.add_argument('--n_epochs', default=51)
    parser.add_argument('--train_dirpath', default='data/train/')
    parser.add_argument('--val_dirpath', default='data/val/')
    parser.add_argument('--image_size', nargs='+', type=int, default=None, help="The input should be in the format: height<space>width")

    args = parser.parse_args()

    backbone = args.backbone
    batch_size = args.batch_size
    n_epochs = args.n_epochs
    train_dirpath = args.train_dirpath
    val_dirpath = args.val_dirpath
    image_size = args.image_size
    if image_size is not None:
        # size compatible for cv2 resize
        image_size = tuple([image_size[1], image_size[0]])
    

    # get images for training
    train_image_files, val_image_files = get_train_val(train_dirpath=train_dirpath,
                                                val_dirpath=val_dirpath)
    

    # loss functions
    binary_focal_loss = sm.losses.BinaryFocalLoss(alpha=0.5, gamma=3.33)
    dice_loss = sm.losses.DiceLoss()
    binary_focal_dice_loss = binary_focal_loss + dice_loss
    loss_fn = binary_focal_dice_loss

    train_image_datagen = data_generator.imageLoader(img_dir=os.path.join(train_dirpath,'images'),
                        img_list=train_image_files,
                        mask_dir=os.path.join(train_dirpath,'masks'),
                        mask_list=train_image_files,
                        batch_size=batch_size,
                        size=image_size,
                        rand_aug=True)
    
    val_image_datagen = data_generator.imageLoader(img_dir=os.path.join(val_dirpath,'images'),
                        img_list=val_image_files,
                        mask_dir=os.path.join(val_dirpath,'masks'),
                        mask_list=val_image_files,
                        batch_size=batch_size//2,
                        size=image_size,
                        rand_aug=False,
                        val=True)
    
    model = models.unet_2d((None, None, 3), filter_num=[16, 32, 64, 128, 256, 256],
            n_labels=n_classes,
            stack_num_down=2, stack_num_up=2,
            activation='ReLU',
            output_activation='Softmax',
            batch_norm=True, pool=True, unpool=True,
            backbone=backbone, weights='imagenet',
            freeze_backbone=False, freeze_batch_norm=False,
            name='unet')
    
    model.compile(loss=loss_fn, optimizer=keras.optimizers.Adam(lr=1e-3), 
                metrics=['accuracy', losses.dice_coef])
    
    steps_per_epoch = len(train_image_files) // batch_size
    val_steps_per_epoch = len(val_image_files) // (batch_size//2)

    print(model.summary())
    saver = CustomSaver()
    reduce_lr = get_learningRate_callback()

    model.fit(train_image_datagen,
            steps_per_epoch=steps_per_epoch,
            epochs=n_epochs,
            validation_data=val_image_datagen,
            validation_steps=val_steps_per_epoch,
            callbacks=[saver, reduce_lr])



if __name__=="__main__":
    main()