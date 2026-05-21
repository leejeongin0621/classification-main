"""
한 (subject, fold) 쌍에 대해 학습된 모델의 layer별 feature를 dump.

저장 형식: features/feat_{SUBJECT}_fold{FOLD}.pt
  {
    "gcn":   [Tensor(N, C_i)] * num_layers,   # global avg pool 후
    "tcn":   [Tensor(N, C_i)] * num_layers,
    "block": [Tensor(N, C_i)] * num_layers,
    "pre_fc": Tensor(N, 256),
    "y": Tensor(N,),
    "subject": str,
    "fold": int,
  }
"""
import os
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

from utils.Dataloader import LoadIMU_EPO_simple

# =========================
# Config — 필요하면 여기 수정
# =========================
SUBJECT = "250805_KDY"
FOLD = 3

SUBJECT_LIST = [
    '250805_KDY', '250731_LGE', '250806_LJI', '250812_WDY', '250814_JCM',
    '250818_ICY', '250819_PYH', '250820_LTG', '250822_JSH', '250825_JDB',
]
NUM_TIME_IMU = 200
NUM_CHANNEL_IMU = 30
NUM_SESSION = 5
BATCH_SIZE = 50

# =========================
# Paths
# =========================
this_path = os.getcwd()
load_path = os.path.join(this_path, 'data_10ch')
model_path = os.path.join(this_path, 'models')
out_path = os.path.join(this_path, 'features')
os.makedirs(out_path, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device = {device}")

# =========================
# Data
# =========================
IMU_data_patient, IMU_label_patient = LoadIMU_EPO_simple(load_path, SUBJECT_LIST)
s = SUBJECT_LIST.index(SUBJECT)

sessions = np.arange(NUM_SESSION)
train_session = np.delete(sessions, FOLD)

X_train = IMU_data_patient[s, train_session].reshape(-1, NUM_TIME_IMU, NUM_CHANNEL_IMU)
y_train = IMU_label_patient[s, train_session].reshape(-1)

print(f"subject {SUBJECT} fold {FOLD}: X_train {X_train.shape}, y_train {y_train.shape}")

loader = DataLoader(
    TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    ),
    batch_size=BATCH_SIZE, shuffle=False
)

# =========================
# Model: 아키텍처 + 학습된 weight
# =========================
model = torch.load(
    os.path.join(model_path, "Model_54.pt"),
    map_location=device, weights_only=False,
)
state = torch.load(
    os.path.join(model_path, f"best_model_{SUBJECT}_fold_{FOLD}.pth"),
    map_location=device,
)
model.load_state_dict(state)
model.to(device).eval()

n_layers = len(model.st_gcn_networks)
print(f"num st_gcn layers = {n_layers}")

# =========================
# Forward & collect (GAP over T, V → (N, C))
# =========================
buf_gcn = [[] for _ in range(n_layers)]
buf_tcn = [[] for _ in range(n_layers)]
buf_blk = [[] for _ in range(n_layers)]
buf_prefc = []
buf_y = []
buf_logits = []

with torch.no_grad():
    for x, y in loader:
        x = x.to(device)
        logits, gcn_outs, tcn_outs, block_outs, pre_fc = model.extract_all(x)
        for i in range(n_layers):
            # (N, C, T, V) → (N, C)
            buf_gcn[i].append(gcn_outs[i].mean(dim=(2, 3)).cpu())
            buf_tcn[i].append(tcn_outs[i].mean(dim=(2, 3)).cpu())
            buf_blk[i].append(block_outs[i].mean(dim=(2, 3)).cpu())
        buf_prefc.append(pre_fc.cpu())
        buf_logits.append(logits.cpu())
        buf_y.append(y)

gcn_feat = [torch.cat(b, dim=0) for b in buf_gcn]
tcn_feat = [torch.cat(b, dim=0) for b in buf_tcn]
blk_feat = [torch.cat(b, dim=0) for b in buf_blk]
pre_fc_feat = torch.cat(buf_prefc, dim=0)
logits_all = torch.cat(buf_logits, dim=0)
y_all = torch.cat(buf_y, dim=0)

# 학습 정확도 확인 (dump가 제대로 됐는지 sanity check)
pred = logits_all.argmax(dim=1)
acc = (pred == y_all).float().mean().item()
print(f"train acc on dumped features: {acc:.4f}")

save_file = os.path.join(out_path, f"feat_{SUBJECT}_fold{FOLD}.pt")
torch.save({
    "gcn": gcn_feat,
    "tcn": tcn_feat,
    "block": blk_feat,
    "pre_fc": pre_fc_feat,
    "y": y_all,
    "subject": SUBJECT,
    "fold": FOLD,
}, save_file)

print(f"\nsaved to {save_file}")
print(f"shapes:")
for i in range(n_layers):
    print(f"  layer {i}: gcn {tuple(gcn_feat[i].shape)}, tcn {tuple(tcn_feat[i].shape)}, blk {tuple(blk_feat[i].shape)}")
print(f"  pre_fc: {tuple(pre_fc_feat.shape)}")
