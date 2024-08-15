import numpy as np
import cv2
from tensorflow.keras.models import load_model
from keras_unet_collection import losses
import argparse
import os


def infer(image_path, output_path, model_path, size):
    model = load_model(model_path, compile=False,
                    custom_objects={"dice_coef": losses.dice_coef})

    for image_file in os.listdir(image_path):
        print(os.path.join(image_path,image_file))
        img = cv2.imread(os.path.join(image_path,image_file), 1)
        if size is None:
            # keep same size
            size = (img.shape[1], img.shape[0])
        img = cv2.resize(img, size, cv2.INTER_LINEAR)
        pred = model.predict(np.array([img]))
        pred_argmax = np.argmax(pred, -1)
        cv2.imwrite(os.path.join(output_path, image_file.split(".")[0] + ".png"), pred_argmax[0,:,:]*255)
        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path')
    parser.add_argument('--output_path', default='./predictions/')
    parser.add_argument('--model_path', default='../../checkpoints/truecrack.hd5')
    parser.add_argument('--image_size', nargs='+', type=int, default=None, 
                    help="""The image size to be used""")
    args = parser.parse_args()
    image_path = args.image_path
    output_path = args.output_path
    os.makedirs(output_path, exist_ok=True)
    model_path = args.model_path
    image_size = args.image_size
    if image_size is not None:
        # size compatible for cv2 resize
        image_size = tuple([image_size[1], image_size[0]])
    infer(image_path, output_path, model_path, image_size)

if __name__=="__main__":
    main()



    