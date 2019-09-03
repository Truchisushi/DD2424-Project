from tensorflow import keras #this might be needed to be changed to import keras depending on version.
import matplotlib.pyplot as plt
import numpy as np

from skimage.color import rgb2gray, rgb2lab, lab2rgb
from main_clean import loss_classification, euclidean_loss, get_gray_rgb_imgs

def load_test_imgs():
    images = load_images('cifar-10-batches-py/test_batch')
    print(images.shape)

    return images

def load_images(filename):
    import pickle
    with open(filename, 'rb') as f:
        dataset = pickle.load(f, encoding='latin1')  # Nxd(3072) (Nx (32x32x3))
        img_data = dataset['data']
        #images = img_data.reshape(img_data.shape[0], 3, 32, 32).transpose(0, 2, 3, 1).astype('uint8')
        return img_data

def test_imgs(model_eucl, model_class, model_class_reb):
    for i in range(20):
        #if loss is euclidean_loss:
        pred_eucl = ((model_eucl.predict(
                gray_imgs[-i, :].reshape((1, gray_imgs.shape[1], gray_imgs.shape[2], 1)))) * 255 - 128)
        #else:
        pred_class = (
            prob_to_color(model_class.predict(gray_imgs[-i, :].reshape((1, gray_imgs.shape[1], gray_imgs.shape[2], 1)))))

        pred_class_reb = (
            prob_to_color(
                model_class_reb.predict(gray_imgs[-i, :].reshape((1, gray_imgs.shape[1], gray_imgs.shape[2], 1)))))

        output_eucl = np.empty((gray_imgs.shape[1], gray_imgs.shape[2], 3))
        output_eucl[:, :, 0] = ((gray_imgs[-i, :, :]) * 100)  # the light channel.
        output_eucl[:, :, 1] = pred_eucl[0, :, :, 0]
        output_eucl[:, :, 2] = pred_eucl[0, :, :, 1]

        output_class = np.empty((gray_imgs.shape[1], gray_imgs.shape[2], 3))
        output_class[:, :, 0] = ((gray_imgs[-i, :, :]) * 100)  # the light channel.
        output_class[:, :, 1] = pred_class[0, :, :, 0]
        output_class[:, :, 2] = pred_class[0, :, :, 1]

        output_class_reb = np.empty((gray_imgs.shape[1], gray_imgs.shape[2], 3))
        output_class_reb[:, :, 0] = ((gray_imgs[-i, :, :]) * 100)  # the light channel.
        output_class_reb[:, :, 1] = pred_class_reb[0, :, :, 0]
        output_class_reb[:, :, 2] = pred_class_reb[0, :, :, 1]
        # pred = pred.reshape(32, 32, 3)
        # output = output.reshape((32, 32, 3), order='F')

        fig, axes = plt.subplots(1, 3, figsize=(8, 4))
        ax = axes.ravel()
        rgb_out_eucl = lab2rgb(output_eucl)
        rgb_out_class = lab2rgb(output_class)
        rgb_out_class_reb = lab2rgb(output_class_reb)
        ax[0].imshow(color_imgs[-i, :, :, :])
        #ax[1].imshow(rgb_out_eucl)
        ax[1].imshow(rgb_out_class)
        ax[2].imshow(rgb_out_class_reb)
        #ax[4].imshow(gray_imgs[-i, :, :], cmap=plt.cm.gray)

        ax[0].set_title("Ground Truth")
        #ax[1].set_title("Regression")
        ax[1].set_title("Classification")
        ax[2].set_title("Classification w/ reb")
        #ax[4].set_title("Gray-scale")
        plt.show()

def prob_to_color(z):  # This part is not part of the CNN. but is the mapping of the CNN output to ab_space
    res = np.argmax(annealed_mean(z), axis=-1)
    return quantized_ab[res]

def annealed_mean(z):
    # res1 = np.sum(np.exp(np.log(z + 0.0000001)/ TEMPERATURE), axis=3 )[:, :, :, None]
    z /= np.sum(z, axis=-1, keepdims=True)
    epsilon = 1e-14
    z = np.clip(z, epsilon, 1.0-epsilon)
    res = np.exp(np.log(z) / TEMPERATURE) / np.maximum(np.sum(np.exp(np.log(z) / TEMPERATURE), axis=-1)[:, :, :, None], epsilon)
    z /= np.sum(z, axis=-1, keepdims=True)
    return res

imgs = load_test_imgs()
num_images = 50000
gray_imgs = np.load('gray_imgs.npy')[:num_images]
color_imgs = np.load('color_imgs.npy')[:num_images]
quantized_ab = np.load('quantized_ab.npy')
# Load Model
model_eucl = keras.models.load_model('euclidean_50k_200i_b40.h5', custom_objects={'euclidean_loss':euclidean_loss})
model_class = keras.models.load_model('classification_50k_200i_b32.h5', custom_objects={'loss_classification':loss_classification})
model_class_reb = keras.models.load_model('classification_weighted_50k_100i_b32.h5', custom_objects={'loss_classification':loss_classification})


TEMPERATURE = 1  # CONSTANT FOR ANNEALED MEAN
test_imgs(model_eucl, model_class, model_class_reb)