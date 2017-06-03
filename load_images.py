

import numpy as np
import glob
from keras.preprocessing.image import load_img, img_to_array
import re

def load_images_from_labelFolder(path, img_width, img_height, train_test_ratio=(9,1)):
    pathsAndLabels = []
    label_i = 0
    data_list = glob.glob(path + '\\*')
    datatxt = open('whoiswho.txt' ,'w')
    print('data_list', data_list)
    for dataFolderName in data_list:
        pathsAndLabels.append([dataFolderName, label_i])
        pattern = r".*\\(.*)"
        matchOB = re.finditer(pattern, dataFolderName)
        directoryname = ""
        if matchOB:
            for a in matchOB:
                directoryname += a.groups()[0]
        datatxt.write(directoryname + "," + str(label_i) + "\n")
        label_i = label_i + 1
    datatxt.close()
    allData = []
    for pathAndLabel in pathsAndLabels:
        path = pathAndLabel[0]
        label = pathAndLabel[1]
        imagelist = glob.glob(path + "\\*")
        for imgName in imagelist:
            allData.append((imgName, label))
    allData = np.random.permutation(allData)

    train_x = []
    train_y = []
    for (imgpath, label) in allData:
        img = load_img(imgpath, target_size=(img_width,img_height))
        imgarry = img_to_array(img)
        train_x.append(imgarry)
        train_y.append(label)

    threshold = (train_test_ratio[0]*len(train_x))//(train_test_ratio[0]+train_test_ratio[1])
    test_x = np.array(train_x[threshold:])
    test_y = np.array(train_y[threshold:])
    train_x = np.array(train_x[:threshold])
    train_y = np.array(train_y[:threshold])

    return (train_x, train_y), (test_x, test_y)


if __name__ == '__main__':
    (train_x,train_y),(_,_) = load_images_from_labelFolder('D:\\forwin\\deepLearning\\nogi_images\\images', 128, 128)
    print('trainx.shape:',trainx.shape)
