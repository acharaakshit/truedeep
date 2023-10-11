import numpy as np
import cv2
from tensorflow.keras.models import load_model
from keras_unet_collection import losses


def infer(image_path, output_path):
    model = load_model('./training/models/epoch-50.hd5', compile=False,
                    custom_objects={"dice_coef": losses.dice_coef})
    img = cv2.imread(image_path, 1)
    pred = model.predict(np.array([img]))
    pred_argmax = np.argmax(pred, -1)
    cv2.imwrite(output_path, pred_argmax[0,:,:]*255)

if __name__=="__main__":
    infer('training/data/train/images/11116-4.jpg', './prediction.png')



    