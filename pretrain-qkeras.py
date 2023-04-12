# TensorFlow and tf.keras
import tensorflow as tf
import tensorflow_datasets as tfds

# Helper libraries
import os
import numpy as np
import matplotlib.pyplot as plt

from qkeras import *
from qkeras.utils import model_quantize
from keras.optimizers import Adam

import cv2
from keras.layers import *
from keras.models import Model
from keras import losses


labels_path = '/ldisk/dataset/imagenet2012/5.0.0/label.labels.txt'
imagenet_labels = np.array(open(labels_path).read().splitlines())

imagenet_dataset_builder = tfds.builder('imagenet2012', data_dir='/ldisk/dataset')
imagenet_train = imagenet_dataset_builder.as_dataset(split=tfds.Split.VALIDATION,as_supervised=True,batch_size=1)
def resize_with_crop(image, label):
    i = image
    i = tf.image.resize(i, (224,224))
    i = tf.keras.applications.resnet.preprocess_input(i)
    return (i, label)

imagenet_train = imagenet_train.map(resize_with_crop)
config = {
  "QConv2D": {
      "kernel_quantizer": "quantized_bits(8,8,1)",
      "bias_quantizer": "quantized_po2(8)"
  },
  "QActivation": { "ulaw": "quantized_ulaw(8,8)" },
}

pretrained_model = tf.keras.applications.resnet.ResNet50(include_top=True,weights="imagenet")
pretrained_model.trainable = False
qmodel = model_quantize(pretrained_model,config,8,transfer_weights=True)
x1 = qmodel.get_layer('conv3_block4_out').output
x2 = qmodel.get_layer('conv4_block5_out').output
x3 = qmodel.get_layer('conv5_block3_out').output

def make_model(x1,x2,x3):
    # Load the network
    logging.info('Loading Network...')
    x_conv3 = x1 
    x_conv4 = x2 
    x_conv5 = x3 

    y1 = QConv2D(128,1,1,padding="valid",kernel_quantizer=quantized_bits(8,8,1),bias_quantizer=quantized_po2(8))(x_conv3)
    y1 = QActivation("quantized_relu(8,8)")(y1)

    y2 = QConv2D(128,1,1,padding="valid",kernel_quantizer=quantized_bits(8,8,1),bias_quantizer=quantized_po2(8))(x_conv4)
    y2 = QActivation("quantized_relu(8,8)")(y2)

    y3 = QConv2D(128,1,1,padding="valid",kernel_quantizer=quantized_bits(8,8,1),bias_quantizer=quantized_po2(8))(x_conv5)
    y3 = QActivation("quantized_relu(8,8)")(y3)

    #FUM module
    y = QConv2DTranspose(128,3,strides=2,padding="same",kernel_quantizer=quantized_bits(8,8,1),bias_quantizer=quantized_po2(8))(y3)
    y = QActivation("quantized_relu(8,8)")(y)
    y = keras.layers.add([y,y2])
    y = QConv2DTranspose(128,3,strides=2,padding="same",kernel_quantizer=quantized_bits(8,8,1),bias_quantizer=quantized_po2(8))(y)
    y = QActivation("quantized_relu(8,8)")(y)
    y = keras.layers.add([y,y1])

     #grasp angle classification
    y_angle = QConv2D(128,3,1,padding="same",kernel_quantizer=quantized_bits(8,8,1),bias_quantizer=quantized_po2(8))(y)
    y_angle = QActivation("quantized_relu(8,8)")(y_angle)
    y_angle = QConv2D(128,3,1,padding="same",kernel_quantizer=quantized_bits(8,8,1),bias_quantizer=quantized_po2(8))(y_angle)
    y_angle = QActivation("quantized_relu(8,8)")(y_angle)
    y_angle = QConv2D(18,3,1,padding="same",kernel_quantizer=quantized_bits(8,8,1),bias_quantizer=quantized_po2(8))(y_angle)
    y_angle = QActivation("quantized_relu(8,8)")(y_angle)

    #CHV Rectangles Regression
    y_CHV = QConv2D(128,3,1,padding="same",kernel_quantizer=quantized_bits(8,8,1),bias_quantizer=quantized_po2(8))(y)
    y_CHV = QActivation("quantized_relu(8,8)")(y_CHV)
    y_CHV = QConv2D(128,3,1,padding="same",kernel_quantizer=quantized_bits(8,8,1),bias_quantizer=quantized_po2(8))(y_CHV)
    y_CHV = QActivation("quantized_relu(8,8)")(y_CHV)
    y_CHV = QConv2D(4,3,1,padding="same",kernel_quantizer=quantized_bits(8,8,1),bias_quantizer=quantized_po2(8))(y_CHV)
    y_CHV = QActivation("quantized_relu(8,8)")(y_CHV)

    return Model(inputs=[x1,x2,x3],outputs=[y_angle,y_CHV])


model = make_model(x1=x1,x2=x2,x3=x3)
model.compile(optimizer='adam', 
                         loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
                         metrics=['accuracy'])
model.summary()
#decode_predictions = tf.keras.applications.resnet.decode_predictions

# Print Accuracy
#result = qmodel.evaluate(imagenet_train)
#print("Test loss:", result[0])
#print("Test accuracy:", result[1])