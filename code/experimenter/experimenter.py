# -*- coding: utf-8 -*-

import numpy as np
from datamanager.datamanager import DataManager
import modelmanager.modelmanager as modelM

def run(basepath, dataset, datapath, output_path, trial, seed, arch, num_epochs, allusers, subfeatures, pretrained_file, left_pivot, right_pivot):
    print('Hello')
    np.set_printoptions(suppress=True)
    dm = DataManager(dataset, basepath, f'{basepath}/_cache')

    traindf, testdf = dm.create_raw_pdframes(allusers[0])
    Xtrain, Ytrain, Xtest, Ytest = dm.get_frames(traindf, testdf, allusers[0], debug=False)


    print('Shape of xtrain: '+ str(Xtrain.shape))

    peruserTests = None
    just_test = False
    
    for user in allusers[1:]:
        traindf, testdf = dm.create_raw_pdframes(user)
        c, d, e, f = dm.get_frames(traindf, testdf, user)
        if not just_test:
            Xtrain = np.concatenate([Xtrain, c], axis=0)
            Ytrain = np.concatenate([Ytrain, d], axis=0)
        
        Xtest = np.concatenate([Xtest, e], axis=0)

        Ytest = np.concatenate([Ytest, f], axis=0)
    
    Xtrain, Xtest = dm.subfeatures(Xtrain, Xtest, 'reorder_knowledge')
    
    
    Xtrain, Xtest = dm.subfeatures(Xtrain, Xtest, subfeatures)
    
    if not just_test:
        Xtrain, Ytrain = dm.balance_data_Adasyn(Xtrain, Ytrain)
    Xtest, Ytest = dm.balance_data_Adasyn(Xtest, Ytest)

    
    return modelM.train_and_validate(Xtrain, Ytrain, Xtest, Ytest, peruserTests, arch, num_epochs, just_test, basepath,
                              datapath, output_path, trial, seed, allusers, subfeatures, pretrained_file, left_pivot, right_pivot)
    
