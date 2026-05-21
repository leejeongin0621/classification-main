"""
LOSO 학습 결과를 불러와서 피험자 *마다* 어디서 틀렸는지 자세히 분석.

전제:
  - IMU_main_LOSO.py 가 fold 마다 best state 를
      models/{run_tag}_test_{subject}.pth
    형태로 저장해뒀음.

분석 내용 (전부 subject 단위로 분해):
  1. per-subject overall acc
  2. per-subject 가장 박살난 단어 top N (worst classes)
  3. per-subject 가장 자주 헷갈리는 (true → pred) 쌍 top N
  4. per-(subject, session) 정확도 분포 (세션 안정성)
  5. subject × class accuracy 매트릭스 (히트맵 용 데이터)

결과: results/per_subject_error_analysis_{run_tag}.xlsx
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import numpy as np
import pandas as pd
from collections import Counter
from torch.utils.data import TensorDataset, DataLoader

from utils.Dataloader import LoadIMU_EPO_simple


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

# =========================
# Settings — IMU_main_LOSO.py 와 일치
# =========================
subject_list = ['250805_KDY','250731_LGE','250806_LJI','250812_WDY','250814_JCM',
                '250818_ICY','250819_PYH','250820_LTG','250822_JSH','250825_JDB']
# left='250804_KTS','250805_SMC','250811_LPR','250811_JHS','250812_HHJ',
#      '250813_YMS','250814_CYJ','250819_CYK','250822_KTH','250827_HJH'

num_subject = len(subject_list)
num_session = 5
num_class   = 100
num_time    = 200
num_channel = 30
batch_size  = 128

# IMU_main_LOSO.py 의 run_tag 와 일치시켜야 함 (.pth 파일명: {run_tag}_test_{subj}.pth)
# TensorBoard 폴더에 붙는 _{timestamp} 는 .pth 에 안 들어감
template_pt_path  = os.path.join('models', "Model_41.pt")   # ST-GCN 템플릿
run_tag           = "LOSO_ST-GCN_30ch_m90"                  # ← 보고 싶은 run
# run_tag         = "LOSO_BiLSTM_30ch_m90"                  # 이전 BiLSTM run 보려면

# (옛 .pt 패치)
fallback_in_channels = 3

load_path  = 'data_10ch'
save_path  = 'results'
model_path = 'models'
word_xlsx  = 'excel/TRT_Word_KOR.xlsx'

top_n_worst_class = 15
top_n_conf        = 20

# =========================
# Word 매핑
# =========================
word_names = None
if os.path.exists(word_xlsx):
    df_word = pd.read_excel(word_xlsx)
    word_names = {int(row['Number']) - 1: str(row['Content'])
                  for _, row in df_word.iterrows()
                  if pd.notna(row.get('Content'))}
    print(f"word mapping: {len(word_names)}개 단어 로드")

def wname(i):
    return word_names.get(int(i), '') if word_names else ''

# =========================
# Data load
# =========================
print("Loading data...")
IMU_data, IMU_label = LoadIMU_EPO_simple(load_path, subject_list)
assert IMU_data.shape == (num_subject, num_session, num_class, num_time, num_channel)

# =========================
# Per-fold inference (저장된 .pth 만)
# =========================
# 결과 저장용: (subject, sample) 단위 + session 정보
all_records = []   # dict list: {subject, session, class_true, class_pred, correct}
per_sc_correct = np.full((num_subject, num_class), np.nan)
per_sc_total   = np.full((num_subject, num_class), np.nan)
overall_acc_per_subj = {}

for test_idx, test_subj in enumerate(subject_list):
    best_state_path = os.path.join(model_path, f"{run_tag}_test_{test_subj}.pth")
    if not os.path.exists(best_state_path):
        print(f"  ⚠ {test_subj}: {best_state_path} 없음 — skip")
        continue

    # 모델 reload
    model = torch.load(template_pt_path, map_location=device, weights_only=False)
    if not hasattr(model, 'in_channels'):
        model.in_channels = fallback_in_channels
    state = torch.load(best_state_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()

    # 이 subject 의 5 session 데이터, session 정보 보존
    X_full = IMU_data[test_idx]              # (5, 100, 200, 30)
    y_full = IMU_label[test_idx]             # (5, 100)

    # session index 도 같이
    X_flat = X_full.reshape(-1, num_time, num_channel)               # (500, 200, 30)
    y_flat = y_full.reshape(-1)                                       # (500,)
    sess_flat = np.repeat(np.arange(num_session), num_class)         # (500,)

    ds = TensorDataset(torch.tensor(X_flat, dtype=torch.float32),
                       torch.tensor(y_flat, dtype=torch.long))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)

    y_pred_list = []
    with torch.no_grad():
        for X, _ in dl:
            X = X.to(device)
            logits = model(X)
            y_pred_list.append(logits.argmax(1).cpu().numpy())
    y_pred = np.concatenate(y_pred_list)

    # 기록
    for i in range(len(y_flat)):
        all_records.append({
            'subject':    test_subj,
            'session':    int(sess_flat[i]),
            'class_true': int(y_flat[i]),
            'class_pred': int(y_pred[i]),
            'correct':    bool(y_pred[i] == y_flat[i]),
        })

    # per-(subject, class) 통계
    sc_total = np.zeros(num_class, dtype=int)
    sc_correct = np.zeros(num_class, dtype=int)
    for c in range(num_class):
        mask = (y_flat == c)
        sc_total[c]   = mask.sum()
        sc_correct[c] = (y_pred[mask] == c).sum()
    per_sc_total[test_idx]   = sc_total
    per_sc_correct[test_idx] = sc_correct

    acc = (y_pred == y_flat).mean()
    overall_acc_per_subj[test_subj] = acc
    print(f"  {test_subj}: acc = {acc:.4f}  ({(y_pred == y_flat).sum()}/{len(y_flat)})")

if not all_records:
    raise SystemExit("처리된 fold 없음 — run_tag / 경로 확인 필요")

df_all = pd.DataFrame(all_records)

# =========================
# Sheet 1: per-subject 요약 (overall acc + 세션별 acc)
# =========================
subj_summary_rows = []
for subj in subject_list:
    if subj not in overall_acc_per_subj:
        continue
    sub = df_all[df_all['subject'] == subj]
    row = {
        'subject': subj,
        'overall_acc': overall_acc_per_subj[subj],
        'total': len(sub),
        'correct': int(sub['correct'].sum()),
    }
    # 세션별 정확도
    for sess in range(num_session):
        s = sub[sub['session'] == sess]
        if len(s) > 0:
            row[f'sess{sess}_acc']   = s['correct'].mean()
            row[f'sess{sess}_total'] = len(s)
        else:
            row[f'sess{sess}_acc']   = np.nan
            row[f'sess{sess}_total'] = 0
    subj_summary_rows.append(row)
df_summary = pd.DataFrame(subj_summary_rows)

# =========================
# Sheet 2: per-subject worst classes (어떤 단어를 가장 못 맞췄나)
# =========================
per_sc_acc = np.where(per_sc_total > 0, per_sc_correct / np.maximum(per_sc_total, 1), np.nan)
worst_rows = []
for s_idx, subj in enumerate(subject_list):
    sc_acc = per_sc_acc[s_idx]
    if np.all(np.isnan(sc_acc)):
        continue
    valid = ~np.isnan(sc_acc)
    order = np.argsort(sc_acc + (~valid) * 999)
    for rank, c in enumerate(order[:top_n_worst_class], start=1):
        if not valid[c]:
            continue
        worst_rows.append({
            'subject': subj,
            'rank':    rank,
            'class':   int(c),
            'word':    wname(c),
            'acc':     float(sc_acc[c]),
            'correct': int(per_sc_correct[s_idx, c]),
            'total':   int(per_sc_total  [s_idx, c]),
        })
df_worst = pd.DataFrame(worst_rows)

# =========================
# Sheet 3: per-subject confusion pairs (어떤 단어를 어떤 단어로 헷갈렸나)
# =========================
conf_rows = []
for subj in subject_list:
    sub = df_all[(df_all['subject'] == subj) & (~df_all['correct'])]
    if len(sub) == 0:
        continue
    pairs = Counter(zip(sub['class_true'].tolist(), sub['class_pred'].tolist()))
    for rank, ((t, p), c) in enumerate(pairs.most_common(top_n_conf), start=1):
        conf_rows.append({
            'subject':    subj,
            'rank':       rank,
            'true_class': t,
            'true_word':  wname(t),
            'pred_class': p,
            'pred_word':  wname(p),
            'count':      c,
        })
df_per_subj_conf = pd.DataFrame(conf_rows)

# =========================
# Sheet 4: subject × class accuracy matrix (히트맵 만들기 좋게)
# =========================
df_sc_matrix = pd.DataFrame(per_sc_acc,
                            index=subject_list,
                            columns=[f"class_{c}" for c in range(num_class)])

# =========================
# Sheet 5: 개별 sample 단위 정답/오답 (장 길지만 가장 raw 한 형태)
# =========================
df_all_with_words = df_all.copy()
df_all_with_words['true_word'] = df_all_with_words['class_true'].apply(wname)
df_all_with_words['pred_word'] = df_all_with_words['class_pred'].apply(wname)

# =========================
# 저장
# =========================
out = os.path.join(save_path, f"per_subject_error_analysis_{run_tag}.xlsx")
with pd.ExcelWriter(out) as writer:
    df_summary.to_excel        (writer, sheet_name='subj_summary',     index=False)
    df_worst.to_excel          (writer, sheet_name='subj_worst_class', index=False)
    df_per_subj_conf.to_excel  (writer, sheet_name='subj_confusions',  index=False)
    df_sc_matrix.to_excel      (writer, sheet_name='subj_class_acc_mat')
    df_all_with_words.to_excel (writer, sheet_name='raw_samples',      index=False)

print(f"\n저장: {out}")

# =========================
# 콘솔 요약
# =========================
print("\n=== Per-subject overall acc ===")
for subj in subject_list:
    if subj in overall_acc_per_subj:
        a = overall_acc_per_subj[subj]
        print(f"  {subj}: {a:.4f}")
    else:
        print(f"  {subj}: (skip)")
