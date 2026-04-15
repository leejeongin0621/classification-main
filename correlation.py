import numpy as np
import pandas as pd
from utils.Dataloader import *
import os

subject_list = ['250805_KDY','250731_LGE','250806_LJI','250812_WDY','250814_JCM',
                '250818_ICY','250819_PYH','250820_LTG','250822_JSH','250825_JDB']

this_path = os.getcwd()
path_sub_proj = this_path
load_path  = os.path.join(path_sub_proj, 'data_10ch')
save_path  = os.path.join(path_sub_proj, 'results')
model_path = os.path.join(path_sub_proj, 'models')

IMU_data, IMU_label = LoadIMU_EPO_simple(load_path, subject_list)

IMU_data = IMU_data.reshape(len(subject_list), 5, 100, 200, 30)
# IMU_label = IMU_label.reshape(len(subject_list), 5, 100)

def make_sample_corr(x_sample): # 각 세션에 하나씩 corr matrix 만들기 
    """
    x_sample: (200, 30)
    return: (10, 10)
    """
    X = x_sample.reshape(200, 10, 3)

    signals = []
    for i in range(10):

        signal = np.linalg.norm(X, axis=-1)
        signal = signal[:, i]  # (200,)
        signals.append(signal)

    signals = np.array(signals)

    A_corr = np.corrcoef(signals)
    A_corr = np.nan_to_num(A_corr, nan=0.0)
    A_corr = np.abs(A_corr)

    if A_corr.max() > 0:
        A_corr = A_corr / (A_corr.max() + 1e-8)

    np.fill_diagonal(A_corr, 1.0)
    return A_corr.astype(np.float32)

def make_word_word_corr(x_sample): # 각 세션에 하나씩 corr matrix 만들기 

    S, Sess, W, T, D = IMU_data.shape

    # 30 = 10 nodes * 3 axes
    X = IMU_data.reshape(S, Sess, W, T, 10, 3)
    signal_norm = np.linalg.norm(X, axis=-1)

    signals = []
    for w in range(W):
        signal = signal_norm[:, :, w, :, :].reshape(-1)  # flatten all sessions/samples for this word
        signals.append(signal)

    signals = np.array(signals)

    A_corr = np.corrcoef(signals)
    A_corr = np.nan_to_num(A_corr, nan=0.0)
    A_corr = np.abs(A_corr)

    if A_corr.max() > 0:
        A_corr = A_corr / (A_corr.max() + 1e-8)

    np.fill_diagonal(A_corr, 1.0)
    return A_corr.astype(np.float32)

# A_word = make_word_word_corr(IMU_data)

# df_word_corr = pd.DataFrame(
#     A_word,
#     index=[f'word_{i+1}' for i in range(100)],
#     columns=[f'word_{i+1}' for i in range(100)]
# )

# df_word_corr.to_excel('all_subject_all_session_word_to_word_corr.xlsx')


def get_edge_ranking(A):
    edges = []
    for i in range(A.shape[0]):
        for j in range(i + 1, A.shape[1]):
            edges.append((i, j, A[i, j]))

    edges = sorted(edges, key=lambda x: x[2], reverse=True)
    return edges

def make_ranking_df(A, top_k=20):
    edges = get_edge_ranking(A)[:top_k]

    rows = []
    for rank, (i, j, val) in enumerate(edges, start=1):
        rows.append({
            'rank': rank,
            'node_i': i,
            'node_j': j,
            'value': val
        })

    return pd.DataFrame(rows)

def make_word_avg_corr(X_word): # 4세션기반 평균 corr 구하기 
    """
    X_word: (4, 200, 30)
    return: (10, 10)
    """
    corr_list = []

    for i in range(len(X_word)):
        A_corr = make_sample_corr(X_word[i])
        corr_list.append(A_corr)

    corr_list = np.stack(corr_list, axis=0)
    A_mean = corr_list.mean(axis=0)

    np.fill_diagonal(A_mean, 1.0)
    return A_mean.astype(np.float32)

# def compare_subject_fold_word(IMU_data, subject_idx, fold_idx, word, top_k=20):
#     X_test_word = IMU_data[subject_idx, fold_idx, word]   # (200, 30)

#     train_sessions = [s for s in range(5) if s != fold_idx]
#     X_train_word = IMU_data[subject_idx, train_sessions, word]   # (4, 200, 30)

#     A_train = make_word_avg_corr(X_train_word)
#     A_test  = make_sample_corr(X_test_word)

#     df_train = make_ranking_df(A_train, top_k)
#     df_test  = make_ranking_df(A_test, top_k)

#     df_train['type'] = 'train'
#     df_test['type'] = 'test'

#     df_train['subject_idx'] = subject_idx
#     df_test['subject_idx'] = subject_idx

#     df_train['fold_idx'] = fold_idx
#     df_test['fold_idx'] = fold_idx

#     df_train['word'] = word
#     df_test['word'] = word

#     df = pd.concat([df_train, df_test], axis=0, ignore_index=True)
#     return df

# df = compare_subject_fold_word(
#     IMU_data=IMU_data,
#     subject_idx=5,
#     fold_idx=0,
#     word=1,
#     top_k=20
# )

# df.to_excel('subject5_fold0_word1_corr_compare.xlsx', index=False)


def make_global_corr(X_data): # 모든 sample concat → corr
    """
    X_data: (N, 200, 30)
    return: (10, 10) global correlation matrix
    """

    X = X_data.reshape(-1, 200, 10, 3)

    signals = []
    for i in range(10):
        signal = np.linalg.norm(X, axis=-1)
        signal = signal[:, :, i].reshape(-1)  # flatten to 1D
        signals.append(signal)

    signals = np.array(signals)  # (10, total_length)

    A_corr = np.corrcoef(signals)
    A_corr = np.nan_to_num(A_corr, nan=0.0)
    A_corr = np.abs(A_corr)

    if A_corr.max() > 0:
        A_corr = A_corr / (A_corr.max() + 1e-8)

    np.fill_diagonal(A_corr, 1.0)
    return A_corr.astype(np.float32)

import pandas as pd

# global corr 계산
X_data = IMU_data.reshape(-1, 200, 30)  # 전체 데이터를 flatten
A_corr = make_global_corr(X_data)  # (10, 10)

# DataFrame으로 변환
df = pd.DataFrame(
    A_corr,
    index=[f'node_{i}' for i in range(10)],
    columns=[f'node_{i}' for i in range(10)]
)

# 엑셀 저장
df.to_excel('global_corr_matrix.xlsx')

print("엑셀 저장 완료")

def make_word_corr_dict(X_train, y_train):
    """
    X_train: (N, 200, 30)
    y_train: (N,)
    return: {word_label: (10,10)}
    """
    word_corr_dict = {}
    unique_words = np.unique(y_train)

    for word in unique_words:
        X_word = X_train[y_train == word]
        word_corr_dict[int(word)] = make_word_avg_corr(X_word)

    return word_corr_dict

