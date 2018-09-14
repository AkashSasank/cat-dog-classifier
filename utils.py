import numpy as np
import os
from scipy import misc
from sklearn.model_selection import train_test_split
import keras
import matplotlib.pyplot as plt
import time
from livelossplot import  PlotLossesKeras


keras.backend.set_image_dim_ordering('tf')
keras.backend.set_image_data_format('channels_last')

#random state initializer
seed=5
np.random.seed(seed)

#convert to one hot array
def one_hot(Y,classes):
    Y=np.eye(classes)[Y.reshape(-1)]
    return  Y

# load dataset
def load_dataset(directory,image_size,test_size):
    dataset = []
    labels = os.listdir(directory)
    for image_label in labels:
        images = os.listdir(directory + "/" + image_label)
        for image in images:
            img = misc.imread(directory + "/" + image_label + '/' + image)
            img = misc.imresize(img, (image_size, image_size))
            dataset.append((img, image_label))
    classes = len(labels)
    X = []  # image list
    Y = []  # label list
    for image, label in dataset:
        X.append(image)
        Y.append(labels.index(label))
    # list to array
    X = np.array(X)
    Y = np.array(Y)
    Y=one_hot(Y,classes)
    # normalize the inputs to range 0-1
    X = X.astype('float32')
    X /= 255.0
    #split data to train and test sets
    X_train,X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, shuffle=True,random_state=seed)
    return X_train, Y_train, X_test, Y_test, classes


def define_model(input_shape,num_classes):#input_shape is a tuple (Height,Width,channels)
    model=keras.models.Sequential()
    model.add(keras.layers.Conv2D(32, (3, 3), input_shape=input_shape, padding='same', activation='relu',kernel_initializer='glorot_normal'))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same',kernel_initializer='glorot_normal'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(512, activation='relu',kernel_initializer='glorot_normal'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(num_classes, activation='softmax'))
    model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])
    print(model.summary())
    return  model


def train(X_train,Y_train,validation_split,model,epoch,batch_size):
    tb_callback=keras.callbacks.TensorBoard(log_dir='./graph',write_images=True)
    history = model.fit(X_train, Y_train, validation_split=validation_split, epochs=epoch, batch_size=batch_size,
                        callbacks=[tb_callback])
   # history=model.fit(X_train,Y_train,validation_split= validation_split,epochs=epoch,batch_size=batch_size,callbacks=[PlotLossesKeras()])
    # list all data in history
    # print(history.history.keys())
    # working_dir = os.getcwd()
    # newdir = working_dir + '\\learning_curve\\' + time.asctime(time.localtime()).replace(':','_')
    # if not os.path.exists(newdir):
    #     os.makedirs(newdir)
    # os.chdir(newdir)
    # # summarize history for accuracy
    # plt.figure(1)
    # plt.plot(history.history['acc'])
    # plt.plot(history.history['val_acc'])
    # plt.title('model accuracy')
    # plt.ylabel('accuracy')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.savefig('accuracy.png')
    # plt.show(block=False)
    # # summarize history for loss
    # plt.figure(2)
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('model loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.savefig('loss.png')
    # plt.show(block=False)
    # os.chdir(working_dir)
    return True

def save_model(model):
    working_dir = os.getcwd()
    newdir = working_dir + '\\model\\'+time.asctime(time.localtime()).replace(':','_')
    if not os.path.exists(newdir):
        os.makedirs(newdir)
    os.chdir(newdir)
    # serialize model to JSONx
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")
    os.chdir(working_dir)
    return  True

def test_model(model,X_test,Y_test):
    scores = model.evaluate(X_test, Y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1] * 100))
    return  True

def load_model(model_path):
    print(os.listdir(model_path))
    json_file=model_path+"\\"+([json for _,json in enumerate(os.listdir(model_path)) if json.endswith('.json')])[0]
    print(json_file)
    json_file=open(json_file,'r')
    loaded_model_json=json_file.read()
    json_file.close()
    model=keras.models.model_from_json(loaded_model_json)
    weights = model_path +"\\"+ ([weight for _,weight in  enumerate(os.listdir(model_path)) if weight.endswith('.h5')])[0]
    model.load_weights(weights)
    print("Loaded model from"+model_path)
    return  model
