

from keras.models import Sequential, load_model
from keras.preprocessing.image import load_img, img_to_array
import numpy as np

import argparse
import cv2
from PIL import Image

def faceDetectionFromPath(path, size):
    cvImg = cv2.imread(path)
    cascade_path = "./lib/haarcascade_frontalface_alt.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    facerect = cascade.detectMultiScale(cvImg, scaleFactor=1.1, minNeighbors=1, minSize=(1, 1))
    faceData = []
    for rect in facerect:
        faceImg = cvImg[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]]
        resized = cv2.resize(faceImg,None, fx=float(size/faceImg.shape[0]),fy=float( size/faceImg.shape[1]))
        CV_im_RGB = resized[:, :, ::-1].copy()
        pilImg=Image.fromarray(CV_im_RGB)
        faceData.append(pilImg)

    return faceData

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', default='.\\mymodel.h5', required=True)
    parser.add_argument('--testpath', '-t', default='.\\images\\shiraishi.jpg')
    args = parser.parse_args()

    num_classes = 7
    img_rows, img_cols = 128, 128

    ident = [""] * num_classes
    for line in open("whoiswho.txt", "r"):
        dirname = line.split(",")[0]
        label = line.split(",")[1]
        ident[int(label)] = dirname

    model = load_model(args.model)
    faceImgs = faceDetectionFromPath(args.testpath, img_rows)
    imgarray = []
    for faceImg in faceImgs:
        faceImg.show()
        imgarray.append(img_to_array(faceImg))
    imgarray = np.array(imgarray) / 255.0
    imgarray.astype('float32')

    preds = model.predict(imgarray, batch_size=imgarray.shape[0])
    for pred in preds:
        predR = np.round(pred)
        for pre_i in np.arange(len(predR)):
            if predR[pre_i] == 1:
                print("he/she is {}".format(ident[pre_i]))

if __name__ == '__main__':
    main()
