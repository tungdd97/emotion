# from keras import layers
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, \
    AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.utils import layer_utils
# from tensorflow.keras.utils.data_utils import get_file
# from tensorflow.keras.applications.imagenet_utils import preprocess_input
# # import pydot
# # from IPython.display import SVG
# from tensorflow.keras.utils.vis_utils import model_to_dot
# from tensorflow.keras.utils import plot_model

from tensorflow.keras.initializers import glorot_uniform
import scipy.misc

import tensorflow.keras.backend as K

from split_data import data_generator, callbacks, xtrain, xtest, ytrain, ytest, batch_size, num_epochs

K.set_image_data_format('channels_last')
K.set_learning_phase(1)


class ModelTrain:
    def identity_block(self, X, f, filters, stage, block):
        """
        Implementation of the identity block

        Arguments:
        X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
        f -- integer, specifying the shape of the middle CONV's window for the main path
        filters -- python list of integers, defining the number of filters in the CONV layers of the main path
        stage -- integer, used to name the layers, depending on their position in the network
        block -- string/character, used to name the layers, depending on their position in the network

        Returns:
        X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
        """

        # defining name basis
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        # Retrieve Filters
        F1, F2, F3 = filters

        # Save the input value. You'll need this later to add back to the main path.
        X_shortcut = X

        # First component of main path
        X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2a',
                   kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
        X = Activation('relu')(X)

        # Second component of main path
        X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
                   kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
        X = Activation('relu')(X)

        # Third component of main path
        X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c',
                   kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

        # Final step: Add shortcut value to main path, and pass it through a RELU activation
        X = Add()([X_shortcut, X])
        X = Activation("relu")(X)

        return X

    def convolutional_block(self, X, f, filters, stage, block, s=2):
        """
        Implementation of the convolutional block4

        Arguments:
        X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
        f -- integer, specifying the shape of the middle CONV's window for the main path
        filters -- python list of integers, defining the number of filters in the CONV layers of the main path
        stage -- integer, used to name the layers, depending on their position in the network
        block -- string/character, used to name the layers, depending on their position in the network
        s -- Integer, specifying the stride to be used

        Returns:
        X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
        """

        # defining name basis
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        # Retrieve Filters
        F1, F2, F3 = filters

        # Save the input value
        X_shortcut = X

        ##### MAIN PATH #####
        # First component of main path
        X = Conv2D(F1, (1, 1), strides=(s, s), name=conv_name_base + '2a', padding='valid',
                   kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
        X = Activation('relu')(X)

        # Second component of main path
        X = Conv2D(F2, (f, f), strides=(1, 1), name=conv_name_base + '2b', padding='same',
                   kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
        X = Activation('relu')(X)

        # Third component of main path
        X = Conv2D(F3, (1, 1), strides=(1, 1), name=conv_name_base + '2c', padding='valid',
                   kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

        ##### SHORTCUT PATH ####
        X_shortcut = Conv2D(F3, (1, 1), strides=(s, s), name=conv_name_base + '1', padding='valid',
                            kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
        X_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)

        # Final step: Add shortcut value to main path, and pass it through a RELU activation
        X = Add()([X_shortcut, X])
        X = Activation("relu")(X)

        return X

    def Net50(self, input_shape=(48, 48, 1), classes=7):
        """
        Implementation of the popular ResNet50 the following architecture:
        CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
        -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

        Arguments:
        input_shape -- shape of the images of the dataset
        classes -- integer, number of classes

        Returns:
        model -- a Model() instance in Keras
        """

        # Define the input as a tensor with shape input_shape
        X_input = Input(input_shape)

        # Zero-Padding
        # X = ZeroPadding2D((1, 1))(X_input)
        X = X_input
        # Stage 1

        X = Conv2D(8, (3, 3), strides=(1, 1), name='conv1', kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=3, name='bn_conv1')(X)
        X = Activation('relu')(X)
        # removed maxpool
        # X = MaxPooling2D((3, 3), strides=(2, 2))(X)

        # Stage 2
        X = self.convolutional_block(X, f=3, filters=[32, 32, 128], stage=2, block='a', s=1)
        X = self.identity_block(X, 3, [32, 32, 128], stage=2, block='b')
        X = self.identity_block(X, 3, [32, 32, 128], stage=2, block='c')

        # Stage 3
        X = self.convolutional_block(X, f=3, filters=[64, 64, 256], stage=3, block='a', s=2)
        X = self.identity_block(X, 3, [64, 64, 256], stage=3, block='b')
        X = self.identity_block(X, 3, [64, 64, 256], stage=3, block='c')
        X = self.identity_block(X, 3, [64, 64, 256], stage=3, block='d')

        # Stage 4
        X = self.convolutional_block(X, f=3, filters=[128, 128, 512], stage=4, block='a', s=2)
        X = self.identity_block(X, 3, [128, 128, 512], stage=4, block='b')
        X = self.identity_block(X, 3, [128, 128, 512], stage=4, block='c')
        X = self.identity_block(X, 3, [128, 128, 512], stage=4, block='d')
        X = self.identity_block(X, 3, [128, 128, 512], stage=4, block='e')
        X = self.identity_block(X, 3, [128, 128, 512], stage=4, block='f')

        # Stage 5
        X = self.convolutional_block(X, f=3, filters=[256, 256, 1024], stage=5, block='a', s=2)
        X = self.identity_block(X, 3, [256, 256, 1024], stage=5, block='b')
        X = self.identity_block(X, 3, [256, 256, 1024], stage=5, block='c')

        # AVGPOOL .
        X = AveragePooling2D((2, 2), name='avg_pool')(X)

        # output layer
        X = Flatten()(X)
        X = Dense(512, activation='relu', name='fc1024', kernel_initializer=glorot_uniform(seed=0))(X)
        X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer=glorot_uniform(seed=0))(X)

        # Create model
        model = Model(inputs=X_input, outputs=X, name='Net50')

        return model


if __name__ == "__main__":
    model = ModelTrain().Net50(input_shape=(48, 48, 1), classes=7)
    optimizer = Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-7)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    history = model.fit(data_generator.flow(xtrain, ytrain, ),
                        steps_per_epoch=len(xtrain) / batch_size,
                        epochs=num_epochs, verbose=1, callbacks=callbacks,
                        validation_data=(xtest, ytest))

    fer_json = history.to_json()
    with open("json/fer.json", "w") as json_file:
        json_file.write(fer_json)
    model.save_weights("model/fer.h5")