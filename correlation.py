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


# ============================================================
# Subject-Independent (LOSO) — fold 마다 train/test top-K edge 추출
# ============================================================
def loso_top_edges(IMU_data, subject_list, top_k=20,
                   save_path='LOSO_top_edges.xlsx'):
    
    num_subject = IMU_data.shape[0]
    assert len(subject_list) == num_subject

    sheets = {}

    for test_idx, test_subj in enumerate(subject_list):
        # ---- LOSO split ----
        train_mask = np.ones(num_subject, dtype=bool)
        train_mask[test_idx] = False
        train_subj_idx = np.where(train_mask)[0]

        X_train = IMU_data[train_subj_idx].reshape(-1, 200, 30)   # 4500 word
        X_test  = IMU_data[test_idx].reshape(-1, 200, 30)         # 500 word

        # ---- corr matrix → top-K edge ----
        A_train = make_global_corr(X_train)
        A_test  = make_global_corr(X_test)

        edges_train = get_edge_ranking(A_train)[:top_k]
        edges_test  = get_edge_ranking(A_test )[:top_k]

        # ---- side-by-side DataFrame ----
        rows = []
        for r in range(top_k):
            i_tr, j_tr, v_tr = edges_train[r]
            i_te, j_te, v_te = edges_test [r]
            rows.append({
                'rank':       r + 1,
                'train_edge': f"({i_tr}-{j_tr})",
                'train_corr': v_tr,
                'test_edge':  f"({i_te}-{j_te})",
                'test_corr':  v_te,
            })
        sheets[f"{test_subj}"] = pd.DataFrame(rows)

        print(f"[fold {test_idx+1:>2d}/{num_subject}] test={test_subj}  → top-{top_k} 저장")

    # ---- Excel 저장 ----
    with pd.ExcelWriter(save_path) as writer:
        for sheet_name, df_sheet in sheets.items():
            df_sheet.to_excel(writer, sheet_name=sheet_name[:31], index=False)

    print(f"\n저장: {save_path}")


# 실행: fold 별 train top-20 vs test top-20
loso_top_edges(
    IMU_data     = IMU_data,
    subject_list = subject_list,
    top_k        = 20,
    save_path    = os.path.join(save_path, 'LOSO_top_edges.xlsx'),
)


# ============================================================
# Subject 별 session-wise corr top-K — within-subject 일관성 확인용
# ============================================================
def per_subject_session_top_edges(IMU_data, subject_list, top_k=20,
                                   save_path='per_subject_session_top_edges.xlsx'):
    """
    각 subject 의 각 session 마다 corr matrix 계산 후 top-K edge 추출.
    한 subject 안에서 5 session 들의 top-K 가 얼마나 비슷한지 (= within-subject 일관성).

    Excel: subject 마다 sheet 1장, session 1~5 가 column 으로 나란히.
    """
    num_subject, num_session = IMU_data.shape[0], IMU_data.shape[1]
    sheets = {}

    for s_idx, subj in enumerate(subject_list):
        # 각 session 마다 top-K edge 미리 계산
        per_sess_edges = []
        for sess in range(num_session):
            X_sess = IMU_data[s_idx, sess].reshape(-1, 200, 30)   # (100, 200, 30)
            A_sess = make_global_corr(X_sess)
            edges  = get_edge_ranking(A_sess)[:top_k]
            per_sess_edges.append(edges)

        # rank 1..top_k 행으로 정리, session 별 column
        rows = []
        for r in range(top_k):
            row = {'rank': r + 1}
            for sess in range(num_session):
                i, j, v = per_sess_edges[sess][r]
                row[f'sess{sess+1}_edge'] = f"({i}-{j})"
                row[f'sess{sess+1}_corr'] = v
            rows.append(row)

        sheets[subj] = pd.DataFrame(rows)
        print(f"[{s_idx+1:>2d}/{num_subject}] {subj} → 5 session top-{top_k} 저장")

    with pd.ExcelWriter(save_path) as writer:
        for sheet_name, df_sheet in sheets.items():
            df_sheet.to_excel(writer, sheet_name=sheet_name[:31], index=False)

    print(f"\n저장: {save_path}")


# 실행: subject 별 session-wise top-20
per_subject_session_top_edges(
    IMU_data     = IMU_data,
    subject_list = subject_list,
    top_k        = 20,
    save_path    = os.path.join(save_path, 'per_subject_session_top_edges.xlsx'),
)

