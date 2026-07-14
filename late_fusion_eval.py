#%%
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from utils.Dataloader import LoadIMU_EPO_simple
from scipy import io

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# =========================
# Settings
# =========================
subject_list = ['250805_KDY','250731_LGE','260709_LJS','250812_WDY','250814_JCM',
                '250818_ICY','250819_PYH','250820_LTG','250822_JSH','250825_JDB']

num_session     = 5
num_class       = 100
num_time_IMU    = 200
num_channel_IMU = 30

load_path = os.path.join(os.getcwd(), 'data_10ch')
save_path = os.path.join(os.getcwd(), 'results')
os.makedirs(save_path, exist_ok=True)

# 아키텍처 .pt 파일
BILSTM_ARCH_PT = os.path.join('models', 'Model_127.pt')
STGCN_ARCH_PT  = os.path.join('models', 'Model_122.pt')

# 가중치 .pth 파일 경로 패턴
BILSTM_DIR = os.path.join('models', 'BILSTM', '학습률 0.001')
STGCN_DIR  = os.path.join('models', 'ST-GCN')
BILSTM_TAG = 'LOSO_30ch_Bilstm_260709_test'
STGCN_TAG  = 'LOSO_30ch_ST-GCN_260709_test'

# fusion weight (0.0~1.0, 0.5 = 단순 평균)
W_BILSTM = 0.5
W_STGCN  = 0.5

# =========================
# Evaluation
# =========================
acc_bilstm = np.full(len(subject_list), np.nan)
acc_stgcn  = np.full(len(subject_list), np.nan)
acc_fused  = np.full(len(subject_list), np.nan)

for test_idx, test_subj in enumerate(subject_list):
    print(f"\n===== LOSO fold {test_idx+1}/10  test = {test_subj} =====")

    # ---- 테스트 데이터 로드 ----
    X_data, y_data = LoadIMU_EPO_simple(
        load_path, [test_subj],
        num_session=num_session, num_class=num_class,
    )
    X_test = torch.tensor(X_data.reshape(-1, num_time_IMU, num_channel_IMU), dtype=torch.float32)
    y_test = torch.tensor(y_data.reshape(-1), dtype=torch.long)
    loader = DataLoader(TensorDataset(X_test, y_test), batch_size=256, shuffle=False)

    # ---- 모델 로드 (아키텍처 + 가중치) ----
    bilstm_pth = os.path.join(BILSTM_DIR, f'{BILSTM_TAG}_{test_subj}.pth')
    stgcn_pth  = os.path.join(STGCN_DIR,  f'{STGCN_TAG}_{test_subj}.pth')

    model_bilstm = torch.load(BILSTM_ARCH_PT, map_location=device, weights_only=False)
    model_bilstm.load_state_dict(torch.load(bilstm_pth, map_location=device, weights_only=True))
    model_bilstm.eval().to(device)

    model_stgcn = torch.load(STGCN_ARCH_PT, map_location=device, weights_only=False)
    model_stgcn.load_state_dict(torch.load(stgcn_pth, map_location=device, weights_only=True))
    model_stgcn.eval().to(device)

    # ---- Inference ----
    logits_a_list, logits_b_list, label_list = [], [], []

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            logits_a_list.append(model_bilstm(X).cpu())
            logits_b_list.append(model_stgcn(X).cpu())
            label_list.append(y)

    logits_a = torch.cat(logits_a_list)   # (N, 100)
    logits_b = torch.cat(logits_b_list)   # (N, 100)
    labels   = torch.cat(label_list)

    # ---- Accuracy ----
    acc_a = (logits_a.argmax(1) == labels).float().mean().item()
    acc_b = (logits_b.argmax(1) == labels).float().mean().item()

    logits_f = W_BILSTM * logits_a + W_STGCN * logits_b
    acc_f    = (logits_f.argmax(1) == labels).float().mean().item()

    acc_bilstm[test_idx] = acc_a
    acc_stgcn[test_idx]  = acc_b
    acc_fused[test_idx]  = acc_f

    print(f"  BiLSTM : {acc_a:.4f}")
    print(f"  ST-GCN : {acc_b:.4f}")
    print(f"  Fused  : {acc_f:.4f}  (w={W_BILSTM}/{W_STGCN})")

# =========================
# Summary
# =========================
print("\n========== Late Fusion Summary ==========")
print(f"{'Subject':<20} {'BiLSTM':>8} {'ST-GCN':>8} {'Fused':>8}")
for i, s in enumerate(subject_list):
    print(f"  {s:<18} {acc_bilstm[i]:.4f}   {acc_stgcn[i]:.4f}   {acc_fused[i]:.4f}")
print(f"\n  BiLSTM  mean ± std : {np.nanmean(acc_bilstm):.4f} ± {np.nanstd(acc_bilstm):.4f}")
print(f"  ST-GCN  mean ± std : {np.nanmean(acc_stgcn):.4f}  ± {np.nanstd(acc_stgcn):.4f}")
print(f"  Fused   mean ± std : {np.nanmean(acc_fused):.4f}  ± {np.nanstd(acc_fused):.4f}")

# =========================
# Save
# =========================
io.savemat(
    os.path.join(save_path, 'late_fusion_bilstm_stgcn_result.mat'),
    {
        'subject_list': subject_list,
        'acc_bilstm':   acc_bilstm,
        'acc_stgcn':    acc_stgcn,
        'acc_fused':    acc_fused,
        'W_BILSTM':     W_BILSTM,
        'W_STGCN':      W_STGCN,
    }
)
print("\nresult saved: results/late_fusion_bilstm_stgcn_result.mat")
