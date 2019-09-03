import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import optimizers
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import h5py

from skimage import data
from skimage.color import rgb2gray, rgb2lab, lab2rgb
from skimage.transform import resize
from tensorflow.keras.callbacks import LambdaCallback


"""Read data and convert to grayscale"""
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def loadBatch(filename, batch_size=100):
    dict = unpickle(filename)
    Y = dict[b'data']
    Y = Y[:batch_size, :].reshape((batch_size, 32, 32, 3), order='F').swapaxes(1,2)   # all values between 0 and 1.
    X = (rgb2gray(Y)).reshape((batch_size, 32, 32))
    #Y = np.zeros((np.unique(y).size, X.shape[1]))
    #Y[y, range(Y.shape[1])] = 1 # One-hot representation.
    #Plot a test image to see that everything is all right
    """"
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    ax = axes.ravel()
    ax[0].imshow(Y[2,:,:,:])
    ax[1].imshow(X[2,:,:], cmap=plt.cm.gray)
    plt.show()
    """

    return X, Y

def load_all(batch_size=100):
    gray_imgs, color_imgs = loadBatch("../cifar-10-batches-py/data_batch_1",batch_size) #Adjust this to find cifar data_batch
    #lab_img = rgb2lab(color_imgs)
    #plot to seee that everything is allright:
    """
    fig, axes = plt.subplots()
    axes.imshow((lab_img[2, :, :, :] + [0, 128, 128])/[100, 255, 255])
    plt.show()
    """
    ab_imgs = rgb2lab(color_imgs)[:,:,:,1:]
    norm_ab_imgs = (ab_imgs + 128)/255
    return gray_imgs, norm_ab_imgs, color_imgs
    #TO-DO: compute ab_space qunatization.

#cifar sequence to use for multiprocessing:
class CIFAR10Sequence(tf.keras.utils.Sequence): #This should actually work for arbitrary size images as well.
    def __init__(self, x_set, y_set, batch_size=32):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        return batch_x, batch_y



""""predict color: """
def prob_to_color():
    pass

"""Create CNN model in keras."""

model = tf.keras.Sequential()

#Add 8 convolutional layers and a final batchnorm to the model.
# TODO: kernel size and number of filters
in_shape = (32, 32, 1)
k_size = 3
dil_rate = 2
fact=4
pad='same'

#2-3 convolution layers with relu followed by batch norm.

#conv1_1
model.add(layers.Conv2D(activation='relu', filters=int(64/fact), strides=1, name='conv1_1', input_shape=in_shape, kernel_size=k_size, padding=pad,data_format="channels_last"))
#conv2_2
model.add(layers.Conv2D(activation='relu', filters=int(64/fact), strides=2, name='conv1_2', kernel_size=k_size , padding=pad, data_format="channels_last"))
model.add(layers.BatchNormalization())

#conv2_1
model.add(layers.Conv2D(activation='relu', filters=int(128/fact), strides=1, name='conv2_1',  kernel_size=k_size, padding=pad, data_format="channels_last"))
#conv2_2
model.add(layers.Conv2D(activation='relu', filters=int(128/fact), strides=2, name='conv2_2',  kernel_size=k_size, padding=pad, data_format="channels_last"))
model.add(layers.BatchNormalization())

#conv3_1
model.add(layers.Conv2D(activation='relu', filters=int(256/fact), strides=1, name='conv3_1', kernel_size=k_size, padding=pad, data_format="channels_last"))
#conv3_2
model.add(layers.Conv2D(activation='relu', filters=int(256/fact), strides=1, name='conv3_2', kernel_size=k_size, padding=pad, data_format="channels_last"))
#conv3_3
model.add(layers.Conv2D(activation='relu', filters=int(256/fact), strides=2, name='conv3_3', kernel_size=k_size, padding=pad, data_format="channels_last"))
model.add(layers.BatchNormalization())

#conv4_1
model.add(layers.Conv2D(activation='relu', filters=int(512/fact), strides=1, name='conv4_1', kernel_size=k_size, padding=pad, data_format="channels_last"))
#conv4_2
model.add(layers.Conv2D(activation='relu', filters=int(512/fact), strides=1,  name='conv4_2', kernel_size=k_size, padding=pad, data_format="channels_last"))
#conv4_3
model.add(layers.Conv2D(activation='relu', filters=int(512/fact), strides=1,  name='conv4_3', kernel_size=k_size, padding=pad, data_format="channels_last"))
model.add(layers.BatchNormalization())

#conv5_1
model.add(layers.Conv2D(activation='relu', filters=int(512/fact), strides=1, dilation_rate=dil_rate, name='conv5_1', kernel_size=k_size, padding=pad, data_format="channels_last"))
#conv5_2
model.add(layers.Conv2D(activation='relu', filters=int(512/fact), strides=1, dilation_rate=dil_rate, name='conv5_2', kernel_size=k_size, padding=pad, data_format="channels_last"))
#conv5_3
model.add(layers.Conv2D(activation='relu', filters=int(512/fact), strides=1, dilation_rate=dil_rate, name='conv5_3', kernel_size=k_size, padding=pad, data_format="channels_last"))
model.add(layers.BatchNormalization())

#conv6_1
model.add(layers.Conv2D(activation='relu', filters=int(512/fact), strides=1, dilation_rate=dil_rate, name='conv6_1', kernel_size=k_size, padding=pad, data_format="channels_last"))
#conv6_2
model.add(layers.Conv2D(activation='relu', filters=int(512/fact), strides=1, dilation_rate=dil_rate, name='conv6_2', kernel_size=k_size, padding=pad, data_format="channels_last"))
#conv6_3
model.add(layers.Conv2D(activation='relu', filters=int(512/fact), strides=1, dilation_rate=dil_rate, name='conv6_3', kernel_size=k_size, padding=pad, data_format="channels_last"))
model.add(layers.BatchNormalization())

#conv7_1
model.add(layers.UpSampling2D(size=(2,2)))
model.add(layers.Conv2D(activation='relu', filters=int(256/fact), strides=1, name='conv7_1', kernel_size=k_size, padding=pad, data_format="channels_last"))
#model.add(layers.Conv2D(activation='relu', filters=int(512/fact), strides=1, name='conv7_1', kernel_size=k_size, padding='same', data_format="channels_last"))
#conv7_2
model.add(layers.Conv2D(activation='relu', filters=int(256/fact), strides=1, name='conv7_2', kernel_size=k_size, padding=pad, data_format="channels_last"))
#conv7_3
model.add(layers.Conv2D(activation='relu', filters=int(256/fact), strides=1, name='conv7_3', kernel_size=k_size, padding=pad, data_format="channels_last"))
model.add(layers.BatchNormalization())

#conv8_1
model.add(layers.UpSampling2D(size=(2,2)))
model.add(layers.Conv2D(activation='relu', filters=int(128/fact), strides=1, name='conv8_1', kernel_size=k_size, padding=pad, data_format="channels_last"))
#conv8_2
model.add(layers.Conv2D(activation='relu', filters=int(128/fact), strides=1, name='conv8_2', kernel_size=k_size, padding=pad, data_format="channels_last"))
#conv8_3
model.add(layers.Conv2D(activation='relu', filters=int(128/fact), strides=1, name='conv8_3', kernel_size=k_size, padding=pad, data_format="channels_last"))


#1x1 convolution.
model.add(layers.Conv2D(activation='relu', filters=int(128/fact), strides=1, name='conv8_3L', kernel_size=1, padding=pad, data_format="channels_last"))

#1x1 conv with cross-entropy loss TO BE ADDED
# CHANGE "filters=3" TO "filters=2"
#model.add(layers.Conv2DTranspose(activation='relu', filters=3, strides=4, kernel_size=1))
model.add(layers.UpSampling2D(size=(2,2)))
model.add(layers.Conv2D(activation='relu', filters=2, kernel_size=1, padding=pad))
#model.add(layers.UpSampling2D(size=(2,2)))

"""End of cnn creation"""


#Maybe for future comparison?
def euclidean_loss(y_true, y_pred):
    return K.sum(K.square(y_pred - y_true), axis=-1)/2 #??

def loss_classification(y_true, y_pred): #is this cross-entropy though?

    pass


model.compile(loss=euclidean_loss, optimizer=optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.99, decay=0.001))
print("hej")
#Concatenate input with the final output.

# (a,b) prob. dist.


#batchnorm

"""custom loss and other functions here"""


"""load data:"""
#loc = "../cifar-10-batches-py/"
#Xtrain, Yground = loadBatch(loc + "data_batch_1")
#trainin   #ground  #
gray_imgs, ab_imgs, color_imgs = load_all()
Xtrain = gray_imgs
Yground = ab_imgs

model.summary()

#print_weights = LambdaCallback(on_epoch_end=lambda batch, logs: print(model.layers[0].get_weights()))

history = model.fit(Xtrain.reshape(Xtrain.shape[0], Xtrain.shape[1], Xtrain.shape[2], 1), Yground, epochs=600)#, callbacks=[print_weights])
#gen = CIFAR10Sequence(Xtrain.reshape(Xtrain.shape[0], Xtrain.shape[1], Xtrain.shape[2], 1), Yground)
#history = model.fit_generator(gen, epochs=20, workers=16, use_multiprocessing=True)


#Wack testing of images:
for i in range(10):
    pred = ((model.predict(Xtrain[i,:].reshape((1, Xtrain.shape[1], Xtrain.shape[2], 1))))*255 - 128)
    #pred = pred.reshape((32, 32, 2), order='F')
    output = np.empty((Xtrain.shape[1], Xtrain.shape[2], 3))
    output[:,:,0] = ((Xtrain[i,:,:])*100) #the light channel.
    output[:,:,1] = pred[0,:,:,0]
    output[:,:,2] = pred[0,:,:,1]
    #pred = pred.reshape(32, 32, 3)
    #output = output.reshape((32, 32, 3), order='F')


    fig, axes = plt.subplots(1, 3, figsize=(8, 4))
    ax = axes.ravel()
    rgb_out = lab2rgb(output)
    ax[0].imshow(color_imgs[i, :, :, :])
    ax[1].imshow(rgb_out)
    ax[2].imshow(gray_imgs[i,:,:], cmap=plt.cm.gray)
    plt.show()

model.save("colorization_model.h5")