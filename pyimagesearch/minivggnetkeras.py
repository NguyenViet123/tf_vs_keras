# import the necessary packages
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model


class MiniVGGNetKeras:
    @staticmethod
    def build(width, height, depth, classes):
        input_shape = (height, width, depth)
        chan_dim = -1

        inputs = Input(shape=input_shape)

        x = Conv2D(32, (3,3), padding='same')(inputs)
        x = Activation('relu')(x)
        x = BatchNormalization(axis=chan_dim)(x)
        x = Conv2D(32, (3,3), padding='same')(x)
        x = Activation('relu')(x)
        x = BatchNormalization(axis=chan_dim)(x)
        x = MaxPooling2D()(x)
        x = Dropout(0.25)(x)

        x = Conv2D(64, (3, 3), padding='same')(x)
        x = Activation('relu')(x)
        x = BatchNormalization(axis=chan_dim)(x)
        x = Conv2D(64, (3, 3), padding='same')(x)
        x = Activation('relu')(x)
        x = BatchNormalization(axis=chan_dim)(x)
        x = MaxPooling2D()(x)
        x = Dropout(0.25)(x)

        x = Flatten()(x)
        x = Dense(512)(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)

        x = Dense(classes)(x)
        x = Activation('softmax')(x)

        model = Model(inputs, x, name='minivggnet_keras')

        return model









