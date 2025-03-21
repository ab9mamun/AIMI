# -*- coding: utf-8 -*-
import os.path

from modelmanager import lstm, cnn
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from tensorflow.keras.utils import plot_model
from sklearn.metrics import f1_score
import time

def train_and_validate(Xtrain, Ytrain, Xtest, Ytest, peruserTests, arch, num_epochs, just_test, basepath, datapth,
                       output_path, trial, seed, allusers, subfeatures, pretrained_file, left_pivot, right_pivot):
    if arch == 'cnn':
        Xtrain = Xtrain.reshape(Xtrain.shape[0], Xtrain.shape[1], Xtrain.shape[2], 1)
        Xtest = Xtest.reshape(Xtest.shape[0], Xtest.shape[1], Xtest.shape[2], 1)

    else:
        Xtrain = Xtrain.reshape(Xtrain.shape[0], Xtrain.shape[1], Xtrain.shape[2])
        Xtest = Xtest.reshape(Xtest.shape[0], Xtest.shape[1], Xtest.shape[2])

    if arch == 'cnn':
        model = cnn.create_model_cnn(Xtrain.shape[1], Xtrain.shape[2])

    elif arch == 'lstm':
        model = lstm.create_model_lstm(Xtrain.shape[1], Xtrain.shape[2])

    else:
        raise Exception("Unknown architecture")
        
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    
    if pretrained_file is not None:
        model.load_weights(pretrained_file)
    
    Xtrain, Ytrain = shuffle(Xtrain, Ytrain, random_state=seed+trial)

    train_time = 0

    usertext = '_'.join([str(user) for user in allusers])
    new_pretrained_file = f'{output_path}/trial_{trial}_pivots_{left_pivot}_{right_pivot}_model_{subfeatures}_{usertext}_SEED_{seed}.h5'
    history_file = f'{output_path}/trial_{trial}_pivots_{left_pivot}_{right_pivot}_history_{subfeatures}_{usertext}_SEED_{seed}.csv'

    if not just_test:
        before_train= time.time()
        history = model.fit(Xtrain, Ytrain, validation_split=0.2, batch_size = 32, epochs=num_epochs, verbose=1)
        after_train = time.time()
        model.save_weights(new_pretrained_file)
        train_time = after_train - before_train
        history_df = pd.DataFrame(history.history)
        history_df.to_csv(history_file, index=False)
    else:
        pass
    
    before_test = time.time()
    Ypred =  model.predict(Xtest)
    
    after_test = time.time()
    test_time = after_test - before_test
    
    before_test2 = time.time()
    Ypred2 =  model.predict(Xtest[0:1, :, :])
    after_test2 = time.time()
    single_test_time = after_test2 - before_test2
    
    Ypred = np.round(Ypred.reshape(Ypred.shape[0],)).astype(int)
    print(Ytrain.shape, Ytest.shape, Ypred.shape)
    
    finaldf = pd.DataFrame(columns=['real', 'prediction'])
    finaldf.real = Ytest
    finaldf.prediction = Ypred
    acc =  np.count_nonzero(Ypred==Ytest)/len(Ytest)
    f1 =  f1_score(Ytest, Ypred)
    precision = np.sum((Ypred==1) & (Ytest==1)) / np.sum(Ypred==1) if np.sum(Ypred==1) > 0 else 0
    recall = np.sum((Ypred==1) & (Ytest==1)) / np.sum(Ytest==1) if np.sum(Ytest==1) > 0 else 0

    # Save Results
    resuls_file_name = f'{output_path}/results.csv'

    if not os.path.exists(resuls_file_name):
        with open(resuls_file_name, 'w') as f:
            f.write('seed,trial,subfeatures,left_pivot,right_pivot,usertext,arch,epochs,accuracy,f1,precision,recall,train_time,test_time,single_test_time\n')
            f.write(f'{seed},{trial},{subfeatures},{left_pivot},{right_pivot},{usertext},{arch},{num_epochs},{acc},{f1},{precision},{recall},{train_time},{test_time},{single_test_time}\n')
    else:
        with open(resuls_file_name, 'a') as f:
            f.write(f'{seed},{trial},{subfeatures},{left_pivot},{right_pivot},{usertext},{arch},{num_epochs},{acc},{f1},{precision},{recall},{train_time},{test_time},{single_test_time}\n')

    predictions_file = f'{output_path}/trial_{trial}_pivots_{left_pivot}_{right_pivot}_predictions_{subfeatures}_{usertext}_SEED_{seed}.csv'
    finaldf.to_csv(predictions_file, index=False)
    
    if just_test:
        return

    # Plotting learning curves
    loss_plot_file = f'{output_path}/trial_{trial}_pivots_{left_pivot}_{right_pivot}_learning_loss_curve_{subfeatures}_{usertext}_SEED_{seed}.png'
    plt.figure()
    epoch_arr = list(range(1,num_epochs+1))
    plt.plot(epoch_arr, history.history['loss'])
    plt.plot(epoch_arr, history.history['val_loss'])
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Learning curve for training and validation losses")
    plt.legend(['training loss', 'validation loss'])
    plt.savefig(loss_plot_file)
    plt.close()
    
    acc_plot_file = f'{output_path}/trial_{trial}_pivots_{left_pivot}_{right_pivot}_learning_acc_curve_{subfeatures}_{usertext}_SEED_{seed}.png'
    plt.figure()
    epoch_arr = list(range(1,num_epochs+1))
    plt.plot(epoch_arr, history.history['accuracy'])
    plt.plot(epoch_arr, history.history['val_accuracy'])
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Learning curve for training and validation accuracies")
    plt.legend(['training acc', 'validation acc'])
    plt.ylim([0.0, 1.0])
    plt.savefig(acc_plot_file)
    plt.close()

    return new_pretrained_file
    