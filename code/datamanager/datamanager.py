# -*- coding: utf-8 -*-
import os, pickle
import numpy as np
from imblearn.over_sampling import ADASYN


class DataManager:
    def __init__(self, dataset, basepath, pickle_path):
        self.dataset = dataset
        self.basepath = basepath
        self.pickle_path = pickle_path
        
    def create_raw_pdframes(self, user = None):
        dfs = self.load_all_must('user'+str(user)+'/traindf', 'user'+str(user)+'/testdf')
        if dfs is not None:
            return dfs #dfs is [traindf, testdf] in this case

        if self.dataset == 'nih':
            from datamanager.nih_hourly import pdframes

        else:
            print("Dataset not supported now")
            return None, None
            
        traindf, testdf = pdframes(user, self.basepath)
        self.save(traindf, 'user'+str(user)+'/traindf')
        self.save(testdf, 'user'+str(user)+'/testdf')
        
        return traindf, testdf
    
    def save(self, data, filename):
        """
        This function saves a pickled file. Steps ->
        1. Check if the filename contains a folder path
        2. If yes, create the folder path at the folder path if needed
        3. Save the data to the path


        :param data:
        :param filename:
        :return:
        """
        if '/' in filename:
            folderpath = f"{self.pickle_path}/{filename[:filename.rfind('/')]}"
        else:
            folderpath = self.pickle_path

        if not os.path.exists(folderpath):
            os.makedirs(folderpath)
        with open(f'{self.pickle_path}/{filename}', 'wb') as f:
            pickle.dump(data, f)
            
    def load(self, filename):
        """
        This function loads a pickled file. Steps ->
        0. Create directory at the pickle_path if it does not exist
        1. Check if the file exists
        2. If not, return None
        3. If yes, load and return the data

        :param filename:
        :return:
        """
        if not os.path.exists(self.pickle_path):
            os.makedirs(self.pickle_path)
        if not os.path.exists(f'{self.pickle_path}/{filename}'):
            return None
        
        with open(f'{self.pickle_path}/{filename}', 'rb') as f:
            d= pickle.load(f)
        return d
    
    def load_all_must(self, *files):
        """
        This function loads multiple pickled files. If any of the files do not exist, it returns None.
        Otherwise, it returns a list of loaded data.

        :param files:
        :return:
        """
        loaded = []
        for file in files:
            data = self.load(file)
            if data is None:
                return None
            loaded.append(data)
        return loaded

    def get_frames(self, traindf,  testdf, user, debug=False):

        """
        This function gets the frames for training and testing. If debug is True, it loads the frames from disk if they exist.
        Otherwise, it generates the frames and saves them to disk.

        :param traindf:
        :param testdf:
        :param user:
        :param debug:
        :return:
        """
        
        if debug:
            frames = None
        else:
            frames = self.load_all_must('user'+str(user)+'/Xtrain',
                                    'user'+str(user)+'/Ytrain', 
                                    'user'+str(user)+'/Xtest', 
                                    'user'+str(user)+'/Ytest')
        
        if frames is not None:
            return frames
        
        if self.dataset == 'nih':
            from datamanager.nih_hourly import inner_get_frames
        else:
            print("Dataset not supported now")
        
        Xtrain, Ytrain, Xtest, Ytest = inner_get_frames(traindf, testdf)
        
        if not debug:
            self.save(Xtrain, 'user'+str(user)+'/Xtrain')
            self.save(Ytrain, 'user'+str(user)+'/Ytrain')
            self.save(Xtest, 'user'+str(user)+'/Xtest')
            self.save(Ytest, 'user'+str(user)+'/Ytest')
            
        return Xtrain, Ytrain, Xtest, Ytest

    def balance_data_Adasyn(self, X, y):
        """
        This function balances the data using ADASYN. In case, the oversampling fails, it returns the original data.
        :param X:
        :param y:
        :return:
        """
        print('Balancing data..')

        oversample = ADASYN()
        s1, s2, s3 = X.shape
        X = X.reshape(s1, s2*s3)
        failed = False
        try:
            X, y = oversample.fit_resample(X, y)
        except ValueError:
            failed = True

        n = X.shape[0]
        X = X.reshape(n, s2, s3)
        if not failed:
            pass
        return X, y

    def subfeatures(self, X1, X2, case):
        """
        This function selects subfeatures based on the case. To get the correct subfeatures, it is necessary that the
        knowledge features are reordered first.

        :param X1:
        :param X2:
        :param case:
        :return:
        """
        if case == 'reorder_knowledge':
            X1_ = np.concatenate((X1[:, :, 2:50], X1[:,:,74:79], X1[:,:,:2], X1[:,:,50:74]), axis=2)
            X2_ =  np.concatenate((X2[:, :, 2:50], X2[:,:,74:79], X2[:,:,:2], X2[:,:,50:74]), axis=2)
            return X1_, X2_
        
        elif case == 'without_knowledge':
            #after reordering
            return X1[:, :, :53], X2[:, :, :53]
        elif case == 'location_only':
            #after reordering
            return X1[:, :, 9:11], X2[:, :, 9:11]
        
        elif case == 'high_res':
            #after reordering
            return X1[:, :, :14], X2[:, :, :14]
        
        elif case == 'high_and_knowledge':
            #after reordering
            X1_ = np.concatenate((X1[:, :, :14], X1[:,:,53:79]), axis=2)
            X2_ =  np.concatenate((X2[:, :, :14], X2[:,:,53:79]), axis=2)
            return X1_, X2_
        elif case == 'location_and_knowledge':
            #after reordering
            X1_ = np.concatenate((X1[:, :, 9:11], X1[:,:,53:79]), axis=2)
            X2_ =  np.concatenate((X2[:, :, 9:11], X2[:,:,53:79]), axis=2)
            return X1_, X2_
        
        elif case == 'all':
            return X1, X2
        
        else:
            return None, None
