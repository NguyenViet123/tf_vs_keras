import tensorflow as tf

class MiniVGGNetTF:
    @staticmethod
    def build(width, height, depth, classes):
        input_shape = (height, width, depth)
        chan_dim = -1

        inputs = tf.keras.layers.Input(shape=input_shape)

        x = tf.keras.layers.Conv2D(32, (3, 3), padding='same')(inputs)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.BatchNormalization(axis=chan_dim)(x)
        x = tf.keras.layers.Conv2D(32, (3, 3), padding='same')(x)
        x = tf.keras.layers.Lambda(lambda t: tf.nn.crelu(t))(x)
        x = tf.keras.layers.BatchNormalization(axis=chan_dim)(x)
        x = tf.keras.layers.MaxPooling2D()(x)
        x = tf.keras.layers.Dropout(0.25)(x)

        x = tf.keras.layers.Conv2D(64, (3, 3), padding='same')(x)
        x = tf.keras.layers.Lambda(lambda t: tf.nn.crelu(t))(x)
        x = tf.keras.layers.BatchNormalization(axis=chan_dim)(x)
        x = tf.keras.layers.Conv2D(64, (3, 3), padding='same')(x)
        x = tf.keras.layers.Lambda(lambda t: tf.nn.crelu(x))
        x = tf.keras.layers.BatchNormalization(axis=chan_dim)(x)
        x = tf.keras.layers.MaxPooling2D()(x)
        x = tf.keras.layers.Dropout(0.25)(x)

        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(512)(x)
        x = tf.keras.layers.Lambda(lambda t: tf.nn.crelu(t))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.5)(x)

        x = tf.keras.layers.Dense(classes)(x)
        x = tf.keras.layers.Activation('softmax')(x)

        model = tf.keras.models.Model(inputs, x, name='minivggnet_tf')

        return model
