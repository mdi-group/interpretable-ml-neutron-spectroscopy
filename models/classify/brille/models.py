import sys
sys.path.append('/home/mts87985/ml-ins/')
from utils import *
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Lambda, GlobalAveragePooling2D 
from tensorflow.keras.layers import Activation, Dropout, Flatten, BatchNormalization
from tensorflow import keras

def cnn_global(nlayers):
#add model layers 
    inputs = keras.Input(shape=(240, 400, 1)) 
    x = Conv2D(16, (5, 5), data_format='channels_last', input_shape=(20, 200, 1), activation='relu')(inputs)
    x = Conv2D(16, (3, 3), activation='relu', name='cblock1')(x)
    if nlayers >2:
        x = MaxPooling2D(pool_size=(1, 2))(x)
        x = BatchNormalization()(x)
        x = Conv2D(16, (1, 3), activation='relu')(x)
        x = Conv2D(32, (1, 3), activation='relu', name='cblock2')(x)
    if nlayers > 4:
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = BatchNormalization()(x)
        x = Conv2D(32, (1, 3), activation='relu')(x)
        x = Conv2D(32, (1, 3), activation='relu', name='cblock3')(x)
    x = BatchNormalization()(x)
# Add the global pooling layer, required for activation maps
    x = Flatten()(x)
    #x = GlobalAveragePooling2D(name='gap')(x)
    #x = Dropout(0.3)(x)
    x = Dense(64, activation='relu', name='dense')(x)

    output = Dense(units=2, activation='sigmoid', name='out')(x)
            
    return Model(inputs, output)


def cnn_discrim(input_shape=(240, 400, 1), use_dropout=False, output_dim=2):
#add model layers 
    model = Sequential()
    model.add(Conv2D(16, (5, 5), data_format='channels_last', 
              input_shape=input_shape, activation='relu'))
    model.add(Conv2D(16, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(1, 2)))
    model.add(Conv2D(32, (1, 3), activation='relu'))
    model.add(Conv2D(32, (1, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (1, 3), activation='relu'))
    model.add(Conv2D(32, (1, 3), activation='relu', name='final_conv'))
    
    model.add(Flatten())
    if use_dropout:
        model.add(Dropout(0.4))

    model.add(Dense(32, activation='relu'))
    if use_dropout:
        model.add(Dropout(0.4))
    model.add(Dense(units=output_dim, activation='sigmoid'))

    return model

def cnn_examine(input_shape=(240, 400, 1), output_dim=2):
#add model layers 
    model = Sequential()
    model.add(Conv2D(16, (5, 5), data_format='channels_last', input_shape=input_shape, activation='relu'))
    model.add(Conv2D(16, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(1, 2)))
    model.add(Conv2D(32, (1, 3), activation='relu'))
    model.add(Conv2D(32, (1, 3), activation='relu', name='final_conv'))
    
# Add the global pooling layer, required for activation maps
    model.add(Lambda(global_average_pooling, 
                 output_shape=global_average_pooling_shape))

    model.add(Dense(units=output_dim, activation='sigmoid'))
    
    return model

def cnn_examine_train(data_shape=(240, 400, 1)):
#add model layers 
    model = Sequential()
    model.add(Conv2D(16, (5, 5), data_format='channels_last', input_shape=data_shape, activation='relu'))
    model.add(Conv2D(16, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(1, 2)))
    model.add(Conv2D(16, (1, 3), activation='relu'))
    model.add(Conv2D(32, (1, 3), activation='relu', name='final_conv'))
    
    model.add(Flatten())

    model.add(Dense(32, activation='relu'))
    model.add(Dense(units=2, activation='sigmoid'))
    
    return model

def cnn_simple():
    #create model
    model = Sequential()

    #add model layers
    model = Sequential()
    model.add(Conv2D(32, (5, 5), data_format='channels_last', input_shape=(20, 200, 1)))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(units=2, activation='sigmoid'))
    #model.add(Dense(1, activation='sigmoid'))

    return model
