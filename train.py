
from __future__ import print_function
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, ZeroPadding2D
from keras import backend as K
import argparse
from load_images import load_images_from_labelFolder

batch_size = 20
num_classes = 7
epoch = 30

img_rows, img_cols = 128, 128


parser = argparse.ArgumentParser()
parser.add_argument('--path', '-p', default='D:\\forwin\\deepLearning\\nogi_images\\images')
args = parser.parse_args()


(x_train, y_train), (x_test, y_test) = load_images_from_labelFolder(args.path,img_cols, img_rows, train_test_ratio=(6,1))
print(K.image_data_format())
print(x_train[0].shape)

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)
    input_shape = (img_rows, img_cols, 3)

x_train = x_train.astype('float32')
x_test  = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print('x_test shape:',  x_test.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print('y_test[0]', y_test[0])
print('y_train[0]', y_train[0])

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print('after utils')
print('y_test[0]', y_test[0])
print('y_train[0]', y_train[0])

model = Sequential()

model.add(Conv2D(32, kernel_size=(3,3),
            activation='relu',
            input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(Dropout(0.2))

model.add(ZeroPadding2D(padding=(1, 1)))
model.add(Conv2D(96, kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(Dropout(0.2))

model.add(ZeroPadding2D(padding=(1, 1)))
model.add(Conv2D(96, kernel_size=(3,3)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(Flatten())

model.add(Dense(units=1024, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(units=num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.Adam(),
                metrics=['accuracy'])
print('x_train.shape:', x_train.shape)
print('x_train.shape:', y_train.shape)
print('x_test.shape:', x_test.shape)
print('y_test.shape:', y_test.shape)
model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epoch,
            verbose=1,
            validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

model.save('mymodel.h5')
