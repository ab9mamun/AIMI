# -*- coding: utf-8 -*-
import pandas as pd
import time, datetime
import numpy as np

def pdframes(user, basepath):

    df = pd.read_csv(f'{basepath}/data/nih/Merged'+str(user)+'_past_n_future_v2.csv')

    df = df[['Sensor Data Window ID', 'Added Time (UTC)', 'Activity Label Provided By User', 'Activity Label Chosen By Model', "Reason User Didn\'t Provide Label", 'Device Type', 'Device ID', 'Sensor Data ID', 'Sensor Data Time (Local)', 'Yaw (rad)', 'Pitch (rad)', 'Roll (rad)', 'Rotation Rate X (rad/s)', 'Rotation Rate Y (rad/s)', 'Rotation Rate Z (rad/s)', 'User Acceleration X (m/s^2)', 'User Acceleration Y (m/s^2)', 'User Acceleration Z (m/s^2)', 'Latitude', 'Longitude', 'Altitude (m)', 'Horizontal Accuracy (m)', 'Vertical Accuracy (m)', 'Course (deg)', 'Speed (m/s)', 'user', 'last_medication', 'next_medication', 'last_reminder', 'next_reminder', 'lasteventTime', 'nexteventTime', 'date_day', 'date_hour', 'dayofweek', 'last_event_day', 'last_event_hour', 'next_event_day', 'next_event_hour', 'medication_this_hr', 'is_day_0', 'is_day_1', 'is_day_2', 'is_day_3', 'is_day_4', 'is_day_5', 'is_day_6', 'is_hr_0', 'is_hr_1', 'is_hr_2', 'is_hr_3', 'is_hr_4', 'is_hr_5', 'is_hr_6', 'is_hr_7', 'is_hr_8', 'is_hr_9', 'is_hr_10', 'is_hr_11', 'is_hr_12', 'is_hr_13', 'is_hr_14', 'is_hr_15', 'is_hr_16', 'is_hr_17', 'is_hr_18', 'is_hr_19', 'is_hr_20', 'is_hr_21', 'is_hr_22', 'is_hr_23', 'prescribed_date', 'prescribed_hour', 'is_prescribed_at_hr_0', 'is_prescribed_at_hr_1', 'is_prescribed_at_hr_2', 'is_prescribed_at_hr_3', 'is_prescribed_at_hr_4', 'is_prescribed_at_hr_5', 'is_prescribed_at_hr_6', 'is_prescribed_at_hr_7', 'is_prescribed_at_hr_8', 'is_prescribed_at_hr_9', 'is_prescribed_at_hr_10', 'is_prescribed_at_hr_11', 'is_prescribed_at_hr_12', 'is_prescribed_at_hr_13', 'is_prescribed_at_hr_14', 'is_prescribed_at_hr_15', 'is_prescribed_at_hr_16', 'is_prescribed_at_hr_17', 'is_prescribed_at_hr_18', 'is_prescribed_at_hr_19', 'is_prescribed_at_hr_20', 'is_prescribed_at_hr_21', 'is_prescribed_at_hr_22', 'is_prescribed_at_hr_23', 'med_last_2hrs', 'med_last_3hrs', 'med_last_6hrs', 'med_last_12hrs', 'med_last_24hrs', 'medication_next_hr']]
    
    df.drop(columns=['Sensor Data Window ID', 'Activity Label Provided By User', 'Activity Label Chosen By Model', 'Device Type', 'Device ID', 'Sensor Data ID', 'Vertical Accuracy (m)', 'Course (deg)'], axis=1, inplace=True)
    df.drop(columns=['Reason User Didn\'t Provide Label'], axis=1, inplace=True)
    #df.drop(columns=['next_medication'], axis=1, inplace=True) It is eventually dropped later, so we can keep it this way.
    df.replace('RECORDED', 1, inplace=True)
    df.replace('MISSED', 0, inplace=True)
    
    df.fillna(0, inplace=True)   #we are essentially adding a bias here. What else could we do here?    
    trainsize = int(df.shape[0]*0.9)
    testsize = df.shape[0] - trainsize
    return df.head(trainsize), df.tail(testsize)

def string_to_stamp(string_):
    if type(string_) is not str:
        return 0
    if string_ == '--':
        return 0
    
    if '.' in string_:
        return datetime.datetime.strptime(string_, '%Y-%m-%d %H:%M:%S.%f').timestamp()
    return datetime.datetime.strptime(string_, '%Y-%m-%d %H:%M:%S').timestamp()


def create_frames(pdframe):
    window_size = 1800
    window_hop = 900
    X = []
    Y = []
    U = []
    for next_reminder in pdframe['next_reminder'].unique():
        filtered = pdframe[pdframe['next_reminder'] == next_reminder]
        #print(filtered.shape)
        n = filtered.shape[0]
        i = 0
        
        while i <= n-window_size:
            windowpd = filtered.iloc[i:i+window_size]
            basetime = windowpd.iloc[0].next_reminder
            windowpd['Added Time (UTC)'] = (windowpd['Added Time (UTC)'] - basetime)/3600
            windowpd['Sensor Data Time (Local)'] = (windowpd['Sensor Data Time (Local)'] - basetime)/3600
            windowpd['last_reminder'] = (windowpd['last_reminder'] - basetime)/3600
            
            y = windowpd.iloc[-1]['medication_next_hr']
            u = windowpd.iloc[-1]['user']
            windowpd.drop(columns=['medication_this_hr','medication_next_hr', 'next_reminder', 'next_medication', 'user', 'nexteventTime', 'date_day',
                                   'date_hour', 'dayofweek', 'prescribed_date', 'prescribed_hour', 'last_event_day', 'next_event_day', 'next_event_hour'], axis=1, inplace=True)


            x = windowpd.values
            i+= window_hop
            X.append(x)
            Y.append(y)
            U.append(u)
    
    return np.array(X).astype('float32'), np.array(Y).astype('float32'), U

def dftoframes(df):
    
    df['Added Time (UTC)'] =  df['Added Time (UTC)'].apply(lambda x: string_to_stamp(x) )
    df['Sensor Data Time (Local)'] =  df['Sensor Data Time (Local)'].apply(lambda x: string_to_stamp(x))
    df['last_reminder'] =  df['last_reminder'].apply(lambda x: string_to_stamp(x))
    df['next_reminder'] =  df['next_reminder'].apply(lambda x: string_to_stamp(x))
    df.drop(columns=['lasteventTime'], axis=1, inplace=True)
    print("df.shape:",df.shape)
    X, Y, u = create_frames(df)
    return X, Y


def inner_get_frames(traindf, testdf):
    Xtrain, Ytrain = dftoframes(traindf)
    Xtest, Ytest = dftoframes(testdf)
    return Xtrain, Ytrain, Xtest, Ytest