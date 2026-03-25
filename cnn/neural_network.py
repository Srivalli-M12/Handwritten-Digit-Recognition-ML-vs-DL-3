from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dense, Input

class CNN:
    @staticmethod
    def build(width, height, depth, total_classes, Saved_Weights_Path=None):
        model = Sequential()

        # Modern Keras uses an explicit Input layer
        model.add(Input(shape=(depth, height, width)))

        # Block 1: CONV => RELU => POOL (Updated for TensorFlow syntax)
        model.add(Conv2D(20, kernel_size=(5, 5), padding="same", data_format="channels_first"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), data_format="channels_first"))

        # Block 2: CONV => RELU => POOL
        model.add(Conv2D(50, kernel_size=(5, 5), padding="same", data_format="channels_first"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), data_format="channels_first"))

        # Fully Connected Layer
        model.add(Flatten(data_format="channels_first"))
        model.add(Dense(500))
        model.add(Activation("relu"))

        # Output Layer (Softmax Classifier)
        model.add(Dense(total_classes))
        model.add(Activation("softmax"))

        if Saved_Weights_Path:
            try:
                model.load_weights(Saved_Weights_Path)
            except:
                pass

        return model