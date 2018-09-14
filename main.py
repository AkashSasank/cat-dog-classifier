from utils import *
dataser_dir='G:\cat-dog\dataset'#add the path of the image data
image_size=64
input_shape=(64,64,3)
num_classes=2#dog and cat
epoch=10
batch_size=16
X_train,Y_train,X_test,Y_test,classes=cnn.load_dataset(dataser_dir,image_size=image_size,test_size=0.2)
cat_dog_model=define_model(input_shape,num_classes)
train(X_train,Y_train,validation_split=0.1,model= cat_dog_model ,epoch= epoch,batch_size= batch_size)
test_model(cat_dog_model, X_test,Y_test)
save_model(cat_dog_model)
