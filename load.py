import numpy as np

import tensorflow as tf

def init(CLASSES):
    IMAGE_SIZE = [224, 224]
    img_adjust_layer = tf.keras.layers.Lambda(lambda data: tf.keras.applications.vgg16.preprocess_input(tf.cast(data, tf.float32)), input_shape=[*IMAGE_SIZE, 3])
    pretrained_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False)
    
    pretrained_model.trainable = False # False = transfer learning, True = fine-tuning

    model = tf.keras.Sequential([
        img_adjust_layer,
        pretrained_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(len(CLASSES), activation='softmax')
    ])

    model.load_weights('/home/tecblic/Desktop/CODE/FC_Model/FCmodel.h5')

    return model