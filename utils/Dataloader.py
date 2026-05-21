import numpy as np
from scipy import io
import os
import math
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


# ============================================================
# (private) z-정렬 / yaw 회전 헬퍼 — 아래 loader 들이 사용
# ============================================================
def _z_align_one_sensor(sig):

    org = sig[0].astype(np.float64).copy()
    flip_z = (org[2] <= 0)
    if flip_z:
        org[2] = -org[2]

    roll  = math.atan2(org[1], math.sqrt(org[2]**2 + org[0]**2))
    pitch = math.atan2(-org[0], math.sqrt(org[2]**2 + org[1]**2))
    c_r, s_r = math.cos(roll),  math.sin(roll)
    c_p, s_p = math.cos(pitch), math.sin(pitch)

    R_roll  = np.array([[1, 0, 0], [0, c_r, -s_r], [0, s_r, c_r]])
    R_pitch = np.array([[ c_p, 0, s_p], [0, 1, 0], [-s_p, 0, c_p]])
    R = R_pitch @ R_roll

    s = sig.astype(np.float64).copy()
    if flip_z:
        s[:, 2] = -s[:, 2]
    return (s @ R.T).astype(sig.dtype)


def _yaw_rotate_one_trial(trial, num_sensor, d_range, rng):

    out = np.empty_like(trial)
    for i in range(num_sensor):
        deg = rng.randint(max(0, d_range - 5) + 1, d_range + 5 + 2)
        if rng.randint(0, 2) == 0:
            deg = -deg
        rad = math.radians(deg)
        c, s = math.cos(rad), math.sin(rad)
        R = np.array([[ c, s, 0],
                      [-s, c, 0],
                      [ 0, 0, 1]], dtype=trial.dtype)
        out[:, 3*i:3*(i+1)] = trial[:, 3*i:3*(i+1)] @ R
    return out


# ============================================================
# Loader 1: z-정렬 + 정규화  (LoadIMU_EPO_simple + z-align)
# ============================================================
def LoadIMU_EPO_zaligned(load_path, subject_list, num_session=5, num_class=100,
                          num_sensor=10, eps=1e-8):

    data_list = []

    for subj in subject_list:
        session_list = []
        for sess in range(1, num_session + 1):
            fpath = os.path.join(load_path, subj, f"epo_session{sess}.mat")
            mat = io.loadmat(fpath, struct_as_record=False, squeeze_me=True)

            x = np.asarray(mat["epo"].x)                  # (200, 30, 100)
            x = np.transpose(x, (2, 0, 1)).astype(np.float32)  # (100, 200, 30)

            # 1) z-정렬 (per sample × per sensor)
            for c in range(num_class):
                for s in range(num_sensor):
                    sig = x[c, :, 3*s:3*(s+1)]
                    x[c, :, 3*s:3*(s+1)] = _z_align_one_sensor(sig)

            # 2) 정규화 (LoadIMU_EPO_simple 과 동일)
            mu  = x.mean(axis=1, keepdims=True)
            sig = x.std (axis=1, keepdims=True)
            x = (x - mu) / (sig + eps)

            session_list.append(x)
        data_list.append(session_list)

    data = np.asarray(data_list)
    label = np.zeros((len(subject_list), num_session, num_class), dtype=int)
    for i in range(num_class):
        label[:, :, i] = i

    return data, label


# ============================================================
# Loader 2: z-정렬 + yaw 증강 + 정규화
# ============================================================
def LoadIMU_EPO_zaligned_yaw(load_path, subject_list, num_session=5, num_class=100,
                              num_sensor=10, amount=2, d_range=25,
                              seed=None, eps=1e-8):

    rng = random.Random(seed) if seed is not None else random

    data_list = []
    for subj in subject_list:
        session_list = []
        for sess in range(1, num_session + 1):
            fpath = os.path.join(load_path, subj, f"epo_session{sess}.mat")
            mat = io.loadmat(fpath, struct_as_record=False, squeeze_me=True)

            x = np.asarray(mat["epo"].x)                  # (200, 30, 100)
            x = np.transpose(x, (2, 0, 1)).astype(np.float32)  # (100, 200, 30)

            # 1) z-정렬
            for c in range(num_class):
                for s in range(num_sensor):
                    sig = x[c, :, 3*s:3*(s+1)]
                    x[c, :, 3*s:3*(s+1)] = _z_align_one_sensor(sig)

            # 2) yaw 증강: 원본 + amount 개의 회전 복사본 concat
            blocks = [x.copy()]
            for _ in range(amount):
                rot = np.empty_like(x)
                for c in range(num_class):
                    rot[c] = _yaw_rotate_one_trial(x[c], num_sensor, d_range, rng)
                blocks.append(rot)
            x_full = np.concatenate(blocks, axis=0)        # (class*(1+amount), 200, 30)

            # 3) 정규화 (per-sample per-channel — LoadIMU_EPO_simple 과 동일)
            mu  = x_full.mean(axis=1, keepdims=True)
            sig = x_full.std (axis=1, keepdims=True)
            x_full = (x_full - mu) / (sig + eps)

            session_list.append(x_full)
        data_list.append(session_list)

    data = np.asarray(data_list)   # (subj, sess, class*(1+amount), time, ch)

    # label: 각 round 마다 0..C-1 반복
    total_per_session = num_class * (1 + amount)
    label = np.zeros((len(subject_list), num_session, total_per_session), dtype=int)
    for k in range(1 + amount):
        for i in range(num_class):
            label[:, :, k * num_class + i] = i

    return data, label

