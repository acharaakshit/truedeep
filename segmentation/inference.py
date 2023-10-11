import numpy as np
import cv2
from tensorflow.keras.models import load_model
from keras_unet_collection import losses
import argparse


def infer(image_path, output_path, model_path):
    model = load_model(model_path, compile=False,
                    custom_objects={"dice_coef": losses.dice_coef})
    img = cv2.imread(image_path, 1)
    pred = model.predict(np.array([img]))
    pred_argmax = np.argmax(pred, -1)
    cv2.imwrite(output_path, pred_argmax[0,:,:]*255)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path')
    parser.add_argument('--output_path', default='./prediction.png')
    parser.add_argument('--model_path', default='./training/models/epoch-50.hd5')
    args = parser.parse_args()
    image_path = args.image_path
    output_path = args.output_path
    model_path = args.model_path
    infer(image_path, output_path, model_path)

if __name__=="__main__":
    main()



    