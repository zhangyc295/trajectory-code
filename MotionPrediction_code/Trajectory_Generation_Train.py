from keras.src.callbacks import EarlyStopping
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Conv2D, MaxPooling2D, AveragePooling2D, Activation,Permute,ConvLSTM2D,Reshape
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import TimeDistributed

import numpy as np
import pandas as pd
import os



def Trajectory_Generation_Train(model_path):
    os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    train_data_path = os.path.join(model_path, "train_data")
    path_x_train = os.path.join(train_data_path, "x_train.csv")
    path_y_train = os.path.join(train_data_path, "y_train.csv")
    path_x_ver = os.path.join(train_data_path, "x_ver.csv")
    path_y_ver = os.path.join(train_data_path, "y_ver.csv")
    save_path = os.path.join(model_path, "trajectory_generation_model.h5")
    if not os.path.exists(os.path.dirname(model_path)):
        os.makedirs(os.path.dirname(model_path))
    n_feature = 14
    n_input = 2
    n_feature_all = n_feature*n_input
    rd = pd.read_csv(path_x_train, iterator=True)
    loop=True
    dflst = []
    i=0
    while loop:
        try:
            i+=1

            df0 = rd.get_chunk(1000)
            dflst.append(df0)
        except StopIteration:
            loop =False

    df = pd.concat(dflst)
    input = df.values
    rd_val=pd.read_csv(path_x_ver, iterator=True)
    loop_val=True
    dflst_val=[]
    i=0
    while loop_val:
        try:
            i+=1

            df0_val = rd_val.get_chunk(1000)
            dflst_val.append(df0_val)
        except StopIteration:
            loop_val =False

    df_val=pd.concat(dflst_val)
    input_val=df_val.values

    print("DataReading done")
    output = np.loadtxt(open(path_y_train, "rb"),delimiter=",", skiprows=0)
    output=np.delete(output,0,0)
    output_val = np.loadtxt(open(path_y_ver, "rb"),delimiter=",", skiprows=0)
    output_val=np.delete(output_val,0,0)
    val_data=(input_val,output_val)

    model = Sequential()
    model.add(Reshape((1, n_feature_all, -1), input_shape=((n_feature_all),)))
    model.add(Conv2D(64, (1, 8), padding='valid'))
    model.add(Activation('relu'))
    model.add(AveragePooling2D(pool_size=(1, 4), strides=None, padding='valid', data_format=None))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.05))
    model.add(Dense(n_feature, activation='tanh'))
    model.compile(loss='mean_absolute_error', optimizer='Adam')
    model.summary()
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    history2 = model.fit(input, output, batch_size=20, epochs=100, verbose=1, validation_data=val_data, callbacks=[early_stopping])
    print('history', history2)
    model.save(save_path)
    with open(os.path.join(model_path, "trajectory_generation_model.csv"), 'w') as ff:
        ff.write(str(history2.history))

    print('done!')

