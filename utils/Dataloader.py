import numpy as np
from scipy import io 
import os
import torch
import random

def LoadIMU_EPO_simple(load_path, subject_list, num_session=5, num_class=100, eps=1e-8):
    """
    mat 파일 구조:
      load_path/{subject}/epo_session{1~5}.mat
        - epo: 1x1 struct
        - epo.x: (timepoint, channel, class) = (200, 30, 100)

    Returns
    -------
    data  : np.ndarray
        shape (subject, session, class, timepoint, channel)
    label : np.ndarray
        shape (subject, session, class)
    """
    data_list = []

    for subj in subject_list:
        session_list = []

        for sess in range(1, num_session + 1):
            fpath = os.path.join(load_path, subj, f"epo_session{sess}.mat")
            mat = io.loadmat(fpath, struct_as_record=False, squeeze_me=True)

            x = mat["epo"].x          # (200, 30, 100)  time, ch, class
            x = np.asarray(x)

            # (time, ch, class) -> (class, time, ch)
            x = np.transpose(x, (2, 0, 1))   # (100, 200, 30)
            # channel_indexing = [0,3,4,6,7]
            # selected_channels = []
            # for n in channel_indexing:
            #     selected_channels.extend([3*n,3*n+1,3*n+2])
            # x = x[:, :, selected_channels] # (100, 200, 15)

            # Normalize
            mu  = x.mean(axis=1, keepdims=True)   # (class, 1, ch)
            sig = x.std(axis=1, keepdims=True)    # (class, 1, ch)
            x = (x - mu) / (sig + eps)

            session_list.append(x)  # (class, time, ch)

        data_list.append(session_list)  # (session, class, time, ch)

    data = np.asarray(data_list)  # (subject, session, class, time, ch)

    # label: (subject, session, class)
    label = np.zeros((len(subject_list), num_session, num_class), dtype=int)
    for i in range(num_class):
        label[:, :, i] = i

    return data, label

def ReadMatData(load_path,subject_list):
    
    num_class = 100
    
    # Data Load
    
    data_list=[]
    for s in subject_list:
        tmp = io.loadmat(load_path+'/'+s+'.mat')
        data_list.append(tmp['high']['x'].item) #[subject,session,class,channel,timepoint]

    # convert to np
    origin_data = np.stack(data_list, axis=0) #[subject,session,class,channel,timepoint]
    origin_data = origin_data.transpose((0,1,2,4,3)) # [subject,session,class,timepoint,channel]
    
    # make label 
    
    label = np.zeros((origin_data.shape[0],origin_data.shape[1],num_class),dtype=int)
    
    for i in range(0,num_class):
        label[:,:,i] = i # [subject,session,class,class_vector]
        
    return origin_data, label        

def LoadEMGMatData(load_path, subject_list, num_class=100):

    '''
    Load EMG data from .mat files

    load_path: path to the dataset (outside folder of subject folders) 
    subject_list: list of subject folder names to load      ex) ['KJG','SGY',...]

    
    Data structure: numpy array [subject,session,class,timepoint,channel]
    Label structure: numpy array [subject,session,class]
    '''
    data_list = []
    for s in subject_list:
        session_list = []
        for session in np.arange(1,6):
            word_list = []

            for word in np.arange(1,num_class+1):
            
                load_file = load_path + '/' + s + '/EMG/session_{}/word{}.mat'.format(session,word)
                tmp = io.loadmat(load_file)
                word_list.append(tmp['save_data'])        # [word,timepoint,channel]
            #end
            session_list.append(word_list)   # [session,class,timepoint,channel]
        #end
        data_list.append(session_list)        # [subject,session,class,timepoint,channel]
    #end

    # convert to np
    data = np.asarray(data_list)   # [subject,session,class,timepoint,channel]

    # make label
    label = np.zeros((data.shape[0],data.shape[1],num_class),dtype=int)

    for i in range(0,num_class):
        label[:,:,i] = i      # [subject,session,class]
    #end

    return data, label


def LoadIMUMatData(load_path, subject_list, num_class=100):
    '''
    Load IMU data from .mat files

    load_path: path to the dataset (outside folder of subject folders) 
    subject_list: list of subject folder names to load      
    ex) ['KJG','SGY',...]
    
    Data structure: numpy array [subject,session,class,timepoint,channel]
    Label structure: numpy array [subject,session,class]
    
    '''
    data_list = []
    for s in subject_list:
        session_list = []
        for session in np.arange(1,6):
            word_list = []
            stacked = np.zeros((num_class,200, 15), dtype=float)
            
            for word in np.arange(1,num_class+1):            
                load_file = load_path + '/' + s + '/IMU/session_{}/word{}.mat'.format(session,word)
                A = io.loadmat(load_file)
                
                
                if A['save_data'].shape[2] < 400:
                    print("timepoint less than 400\nSub: {} Session: {}, Word: {}".format(s,session,word))
                    T = A['save_data'].shape[2]
                else:
                    T = 400

                # (T x 15)로 펴기
                tmp = np.zeros((T, 15), dtype=float)
                ch = 0
                for sn in range(5):
                    for xyz in range(3):
                        tmp[:, ch] = A['save_data'][sn, xyz, :T]
                        ch += 1
                # [T,15]
                data_resample = resample_to(tmp,200, kind="cubic", allow_extrap=True)
                # [200,15]
                if data_resample.shape != (200, 15):
                    raise ValueError(f"resample_to_200 must return (200,15). Got {data_resample.shape}")
                
                
                
                mean = np.mean(data_resample, axis=0, keepdims=True)   # shape: (30, 1, 15)
                std = np.std(data_resample, axis=0, keepdims=True)     # shape: (30, 1, 15)
                normalized_resampdata = (data_resample - mean) / (std + 1e-8)	
                stacked[word-1,:, :] = normalized_resampdata  # [class,timepoint,channel]

            session_list.append(stacked)   # [session,class,timepoint,channel]    
        data_list.append(session_list)        # [subject,session,class,timepoint,channel]

    # convert to np
    data = np.asarray(data_list)   # [subject,session,class,timepoint,channel]

    # make label
    label = np.zeros((data.shape[0],data.shape[1],num_class),dtype=int)

    for i in range(0,num_class):
        label[:,:,i] = i      # [subject,session,class]

    return data, label


def resample_to(data: np.ndarray, Sample_number: int,kind: str = "cubic", allow_extrap: bool = True) -> np.ndarray:
    """
    data: (N, C)  where N = time samples, C = channels (e.g., 15)
    Returns: (Sample_number, C)
    - MATLAB interp1(...,'spline')에 대응: kind='cubic'으로 근사
    """
    if data.ndim != 2:
        raise ValueError(f"data must be 2D (N, C). Got shape {data.shape}")

    N, C = data.shape
    x = np.arange(N, dtype=float)
    xq = np.linspace(0, N - 1, Sample_number)

    # scipy 없이도 동작하도록 기본은 np.interp(1D)로 채널별 처리
    # (np.interp는 선형보간이므로 cubic이 꼭 필요하면 scipy.interpolate를 권장)
    if kind != "linear":
        try:
            from scipy.interpolate import interp1d
            fill_value = "extrapolate" if allow_extrap else np.nan
            f = interp1d(x, data, kind=kind, axis=0, fill_value=fill_value, bounds_error=False)
            out = f(xq)
        except Exception:
            # scipy가 없거나 실패하면, 선형보간으로 안전 폴백
            out = np.empty((200, C), dtype=float)
            for ch in range(C):
                out[:, ch] = np.interp(xq, x, data[:, ch])
    else:
        out = np.empty((200, C), dtype=float)
        for ch in range(C):
            out[:, ch] = np.interp(xq, x, data[:, ch])

    return out