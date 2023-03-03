from pycocotools.coco import COCO
import numpy as np
import os
import skimage.io as io
import random
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing import image
from PIL import ImageOps
from keras import layers
from keras.models import load_model
import datetime
from keras.callbacks import CSVLogger
from keras.layers import Input, Flatten, Dense, Reshape
from keras.applications import MobileNetV2
from keras_segmentation.models.unet import vgg_unet
from keras.applications import mobilenet_v2
from keras.applications.vgg16 import VGG16
#from tensorflow.python.saved_model import loader_impl
#from tensorflow.python.keras.saving.saved_model import load as saved_model_load
import cv2
### For visualizing the outputs ###
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime


def filterDataset(folder, classes=None, mode='train'):
    # initialize COCO api for instance annotations
    annFile = '{}/annotations/instances_{}.json'.format(folder, mode)
    coco = COCO(annFile)

    images = []
    if classes != None:
        # iterate for each individual class in the list
        for className in classes:
            # get all images containing given categories
            catIds = coco.getCatIds(catNms=className)
            imgIds = coco.getImgIds(catIds=catIds)
            images += coco.loadImgs(imgIds)

    else:
        imgIds = coco.getImgIds()
        images = coco.loadImgs(imgIds)

    # Now, filter out the repeated images
    unique_images = []
    for i in range(len(images)):
        if images[i] not in unique_images:
            unique_images.append(images[i])

    random.shuffle(unique_images)
    dataset_size = len(unique_images)

    return unique_images, dataset_size, coco


def getClassName(classID, cats):
    for i in range(len(cats)):
        if cats[i]['id'] == classID:
            return cats[i]['name']
    return None


def getImage(imageObj, img_folder, input_image_size):
    # Read and normalize an image
    train_img = io.imread(img_folder + '/' + imageObj['file_name']) / 255.0
    # Resize
    train_img = cv2.resize(train_img, input_image_size)
    if (len(train_img.shape) == 3 and train_img.shape[2] == 3):  # If it is a RGB 3 channel image
        return train_img
    else:  # To handle a black and white image, increase dimensions to 3
        stacked_img = np.stack((train_img,) * 3, axis=-1)
        return stacked_img


def getNormalMask(imageObj, classes, coco, catIds, input_image_size):
    annIds = coco.getAnnIds(imageObj['id'], catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)
    cats = coco.loadCats(catIds)
    train_mask = np.zeros(input_image_size)
    for a in range(len(anns)):
        className = getClassName(anns[a]['category_id'], cats)
        pixel_value = classes.index(className)
        new_mask = cv2.resize(coco.annToMask(anns[a]) * pixel_value, input_image_size)
        train_mask = np.maximum(new_mask, train_mask)

    # Add extra dimension for parity with train_img size [X * X * 3]
    train_mask = train_mask.reshape(input_image_size[0], input_image_size[1], 1)
    return train_mask


def getBinaryMask(imageObj, coco, catIds, input_image_size):
    annIds = coco.getAnnIds(imageObj['id'], catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)
    train_mask = np.zeros(input_image_size)
    for a in range(len(anns)):
        new_mask = cv2.resize(coco.annToMask(anns[a]), input_image_size)

        # Threshold because resizing may cause extraneous values
        new_mask[new_mask >= 0.5] = 1
        new_mask[new_mask < 0.5] = 0

        train_mask = np.maximum(new_mask, train_mask)

    # Add extra dimension for parity with train_img size [X * X * 3]
    train_mask = train_mask.reshape(input_image_size[0], input_image_size[1], 1)
    return train_mask


def dataGeneratorCoco(images, classes, coco, folder,
                      input_image_size=(224, 224), batch_size=4, mode='train', mask_type='binary'):
    img_folder = '{}/images/{}'.format(folder, mode)
    dataset_size = len(images)
    catIds = coco.getCatIds(catNms=classes)

    c = 0
    while (True):
        img = np.zeros((batch_size, input_image_size[0], input_image_size[1], 3)).astype('float')
        mask = np.zeros((batch_size, input_image_size[0], input_image_size[1], 1)).astype('float')

        for i in range(c, c + batch_size):  # initially from 0 to batch_size, when c = 0
            imageObj = images[i]

            ### Retrieve Image ###
            train_img = getImage(imageObj, img_folder, input_image_size)

            ### Create Mask ###
            if mask_type == "binary":
                train_mask = getBinaryMask(imageObj, coco, catIds, input_image_size)

            elif mask_type == "normal":
                train_mask = getNormalMask(imageObj, classes, coco, catIds, input_image_size)

                # Add to respective batch sized arrays
            img[i - c] = train_img
            mask[i - c] = train_mask

        c += batch_size
        if (c + batch_size >= dataset_size):
            c = 0
            random.shuffle(images)
        yield img, mask


def visualizeGenerator(gen,batch_size):
    img, mask = next(gen)

    fig = plt.figure(figsize=(20, 10))
    outerGrid = gridspec.GridSpec(1, 2, wspace=0.1, hspace=0.1)

    for i in range(2):
        innerGrid = gridspec.GridSpecFromSubplotSpec(3, 3,
                                                     subplot_spec=outerGrid[i], wspace=0.05, hspace=0.05)

        for j in range(batch_size):
            ax = plt.Subplot(fig, innerGrid[j])
            if (i == 1):
                ax.imshow(img[j])
            else:
                ax.imshow(mask[j][:, :, 0])

            ax.axis('off')
            fig.add_subplot(ax)
    plt.show()
def display_mask(i,prediction):
    """Quick utility to display a model's prediction."""
    mask = np.argmax(prediction[i], axis=-1)
    mask = np.expand_dims(mask, axis=-1)
    img = ImageOps.autocontrast(keras.preprocessing.image.array_to_img(mask))
    plt.imshow(img)
    plt.show()


#print(device_lib.list_local_devices())


class UpdatedMeanIoU(tf.keras.metrics.MeanIoU):
  def __init__(self,
               y_true=None,
               y_pred=None,
               num_classes=None,
               name=None,
               dtype=None):
    super(UpdatedMeanIoU, self).__init__(num_classes = num_classes,name=name, dtype=dtype)

  def update_state(self, y_true, y_pred, sample_weight=None):
    y_pred = tf.math.argmax(y_pred, axis=-1)
    return super().update_state(y_true, y_pred, sample_weight)




def Inference(path,folder,image_size,batch_size,classes):
    mode_test = 'test'
    model_infer = load_model(path, compile=False)
    model_infer.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),loss="sparse_categorical_crossentropy",metrics=UpdatedMeanIoU(num_classes=4))
    img_folder = '{}/images/{}'.format(folder, mode_test)
    images_test, dataset_size_test, coco_test = filterDataset(folder, classes, mode_test)

    train_img = getImage(images_test[0], img_folder, input_image_size)
    #train_img = io.imread(img_folder + '/' + 'frame8280.jpg') / 255.0
    train_img = cv2.resize(train_img, image_size)
    #train_img = np.stack((train_img,) * 3, axis=-1)
    img = np.zeros((batch_size, image_size[0], image_size[1], 3)).astype('float')
    img[0] = train_img
    plt.imshow(train_img)
    plt.show()
    prediction = model_infer.predict(img)
    display_mask(0,prediction)

def convolution_block(
    block_input,
    num_filters=256,
    kernel_size=3,
    dilation_rate=1,
    padding="same",
    use_bias=False,
):
    x = layers.Conv2D(
        num_filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding="same",
        use_bias=use_bias,
        kernel_initializer=keras.initializers.HeNormal(),
    )(block_input)
    x = layers.BatchNormalization()(x)
    return tf.nn.relu(x)


def DilatedSpatialPyramidPooling(dspp_input):
    dims = dspp_input.shape
    x = layers.AveragePooling2D(pool_size=(dims[-3], dims[-2]))(dspp_input)
    x = convolution_block(x, kernel_size=1, use_bias=True)
    out_pool = layers.UpSampling2D(
        size=(dims[-3] // x.shape[1], dims[-2] // x.shape[2]), interpolation="bilinear",
    )(x)

    out_1 = convolution_block(dspp_input, kernel_size=1, dilation_rate=1)
    out_6 = convolution_block(dspp_input, kernel_size=3, dilation_rate=6)
    out_12 = convolution_block(dspp_input, kernel_size=3, dilation_rate=12)
    out_18 = convolution_block(dspp_input, kernel_size=3, dilation_rate=18)

    x = layers.Concatenate(axis=-1)([out_pool, out_1, out_6, out_12, out_18])
    output = convolution_block(x, kernel_size=1)
    return output



def vgg16(input_shape, num_classes):


    model_input = keras.Input(shape=(input_shape, input_shape, 3))

    # Define the input and output shapes
    input_shape = (input_shape, input_shape, 3)
    output_shape = (num_classes, input_shape, input_shape, 3)

    # Load the VGG16 model with pre-trained ImageNet weights
    vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=model_input)

    # Freeze the convolutional layers
    for layer in vgg16.layers:
        layer.trainable = False

    # Add a classification head to the VGG16 model
    x = Flatten()(vgg16.output)
    x = Dense(512, activation='relu')(x)
    x = Dense(4 * 512 * 512 * 3, activation='softmax')(x)
    x = Reshape(output_shape)(x)

    # Create the model
    model = keras.Model(inputs=vgg16.input, outputs=x)
    return model

def Unet(img_size, num_classes):
    inputs = keras.Input(shape=img_size + (3,))

    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64, 128, 256]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###

    for filters in [256, 128, 64, 32]:
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.UpSampling2D(2)(x)

        # Project residual
        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(filters, 1, padding="same")(residual)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    outputs = layers.Conv2D(num_classes, 3, activation="softmax", padding="same")(x)

    # Define the model
    model = keras.Model(inputs, outputs)
    return model

def DeeplabV3Plus(image_size, num_classes):
    model_input = keras.Input(shape=(image_size, image_size, 3))
    resnet50 = keras.applications.ResNet50(
        weights="imagenet", include_top=False, input_tensor=model_input
    )
    x = resnet50.get_layer("conv4_block6_2_relu").output
    x = DilatedSpatialPyramidPooling(x)

    input_a = layers.UpSampling2D(
        size=(image_size // 4 // x.shape[1], image_size // 4 // x.shape[2]),
        interpolation="bilinear",
    )(x)
    input_b = resnet50.get_layer("conv2_block3_2_relu").output
    input_b = convolution_block(input_b, num_filters=48, kernel_size=1)

    x = layers.Concatenate(axis=-1)([input_a, input_b])
    x = convolution_block(x)
    x = convolution_block(x)
    x = layers.UpSampling2D(
        size=(image_size // x.shape[1], image_size // x.shape[2]),
        interpolation="bilinear",
    )(x)
    model_output = layers.Conv2D(num_classes, kernel_size=(1, 1), padding="same")(x)
    return keras.Model(inputs=resnet50.input, outputs=model_output)


with tf.device("/gpu:0"):
    def training(arch, path,checkpoint_path, folder, input_image_size,batch_size,classes,load):
        mode_train = 'train'
        mode_val = 'val'
        mask_type = 'normal'
        images_val, dataset_size_val, coco_val = filterDataset(folder, classes,  mode_val)
        images_train, dataset_size_train, coco_train = filterDataset(folder, classes,  mode_train)
        val_gen = dataGeneratorCoco(images_val, classes, coco_val, folder,
                                    input_image_size, batch_size, mode_val, mask_type)
        train_gen = dataGeneratorCoco(images_train, classes, coco_train, folder,
                                    input_image_size, batch_size, mode_train, mask_type)
        csv_logger = CSVLogger(arch + "_" + datetime.now().strftime("%m_%d_%Y_%H_%M_%S") + ".log", separator=',', append=False)

        callbacks = [
            keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=0, save_best_only=False,
                                            save_weights_only=False, mode='auto', period=1),csv_logger
        ]

        if load == True:
            train_model = load_model(path, compile=False)
        else:
            if arch == 'DeepLab':
                train_model = DeeplabV3Plus(input_image_size[0], 4)
            elif arch == 'MobileNet':
                train_model = vgg16(input_image_size[0], 4)
            elif arch == 'Unet':
                train_model = Unet(input_image_size[0], 4)

        train_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),loss="sparse_categorical_crossentropy",metrics=UpdatedMeanIoU(num_classes=4))
        history = train_model.fit(train_gen, epochs=60,steps_per_epoch=len(images_train)/batch_size,validation_data=val_gen,validation_steps=len(images_val)/batch_size,  callbacks=callbacks)
        plt.plot(history.history["loss"])
        plt.title("Training Loss")
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.show()

        plt.plot(history.history["updated_mean_io_u"])
        plt.title("Training Accuracy")
        plt.ylabel("updated_mean_io_u")
        plt.xlabel("epoch")
        plt.show()

        plt.plot(history.history["val_loss"])
        plt.title("Validation Loss")
        plt.ylabel("val_loss")
        plt.xlabel("epoch")
        plt.show()

        plt.plot(history.history["val_updated_mean_io_u"])
        plt.title("Validation Accuracy")
        plt.ylabel("val_updated_mean_io_u")
        plt.xlabel("epoch")
        plt.show()



with tf.device("/gpu:0"):
    folder = './Dataset'
    batch_size = 4
    load = False
    arch = 'Unet'
    input_image_size = (512,512)
    classes = ['panel', 'home', 'bridge','background']
    checkpoint_path = "./deeplabv3/model-{epoch:04d}.h5"
    #Inference('final_models/model-0050.h5', folder, input_image_size, batch_size, classes)

    training(arch, "./deeplabv3/model-0001.h5",checkpoint_path,folder,input_image_size,batch_size,classes,False)