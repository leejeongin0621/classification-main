#%%
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from utils.Dataloader import LoadIMU_EPO_simple
from utils.graph import compute_fold_A
from utils.Model import PureGCN
import random
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# =========================
# Settings
# =========================
all_subjects  = ['250805_KDY','250731_LGE','260709_LJS','250812_WDY','250814_JCM',
                 '250818_ICY','250819_PYH','250820_LTG','250822_JSH','250825_JDB']
test_subjects = ['250731_LGE', '250814_JCM']

LAYER_CONFIGS = {
    3: (64, 128, 256),
    5: (64, 64, 128, 128, 256),
    7: (64, 64, 64, 128, 128, 128, 256),
}

num_session      = 5
num_class        = 100
num_time_IMU     = 200
num_channel_IMU  = 30
num_feat_per_sensor = 3

max_epochs  = 400
min_epochs  = 300
patience    = 20
batch_size  = 128
lr          = 1e-3
base_seed   = 2023

load_path      = os.path.join(os.getcwd(), 'data_10ch')
model_save_dir = os.path.join(os.getcwd(), 'models', 'PureGCN')
os.makedirs(model_save_dir, exist_ok=True)

# =========================
# Sweep
# =========================
results = {}   # {num_layers: {subject: best_acc}}

for num_layers, hidden_dims in LAYER_CONFIGS.items():
    print(f"\n{'='*60}")
    print(f"  PureGCN  layers={num_layers}  dims={hidden_dims}")
    print(f"{'='*60}")

    accs = {}

    for test_subj in test_subjects:
        test_idx = all_subjects.index(test_subj)
        seed = base_seed + test_idx * 100
        random.seed(seed); np.random.seed(seed)
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark     = False
        torch.backends.cudnn.deterministic = True

        train_subjects = [s for s in all_subjects if s != test_subj]

        X_tr_np, y_tr_np = LoadIMU_EPO_simple(load_path, train_subjects, num_session, num_class)
        X_te_np, y_te_np = LoadIMU_EPO_simple(load_path, [test_subj],    num_session, num_class)

        X_tr_flat = X_tr_np.reshape(-1, num_time_IMU, num_channel_IMU)
        y_tr_flat = y_tr_np.reshape(-1)
        X_te_flat = X_te_np.reshape(-1, num_time_IMU, num_channel_IMU)
        y_te_flat = y_te_np.reshape(-1)

        train_loader = DataLoader(
            TensorDataset(torch.tensor(X_tr_flat, dtype=torch.float32),
                          torch.tensor(y_tr_flat, dtype=torch.long)),
            batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(
            TensorDataset(torch.tensor(X_te_flat, dtype=torch.float32),
                          torch.tensor(y_te_flat, dtype=torch.long)),
            batch_size=batch_size, shuffle=False)

        model = PureGCN(
            in_channels=num_feat_per_sensor,
            num_class=num_class,
            graph_args={'max_hop': 1, 'dilation': 1},
            hidden_dims=hidden_dims,
            dropout=0.2,
        ).to(device)

        # fold별 F-stat 그래프
        A_fold = compute_fold_A(X_tr_flat, y_tr_flat, top_k=20).to(device)
        model.A.data.copy_(A_fold)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        best_val_acc = -1.0
        best_state   = None
        no_improve   = 0

        for epoch in tqdm(range(max_epochs),
                          desc=f"L={num_layers} {test_subj}", leave=False):
            # train
            model.train()
            for X, y in train_loader:
                X, y = X.to(device), y.to(device)
                optimizer.zero_grad()
                criterion(model(X), y).backward()
                optimizer.step()

            # eval
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for X, y in val_loader:
                    X, y = X.to(device), y.to(device)
                    correct += model(X).argmax(1).eq(y).sum().item()
                    total   += y.size(0)
            val_acc = correct / total

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state   = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                no_improve   = 0
            else:
                if epoch >= min_epochs - 1:
                    no_improve += 1
                    if no_improve >= patience:
                        print(f"    early stop @ epoch {epoch}")
                        break

        # save best model
        pth_path = os.path.join(model_save_dir, f'PureGCN_L{num_layers}_test_{test_subj}.pth')
        if best_state is not None:
            torch.save(best_state, pth_path)

        accs[test_subj] = best_val_acc
        print(f"  ▶ {test_subj}: {best_val_acc:.4f}  saved -> {pth_path}")

    results[num_layers] = accs

# =========================
# Summary
# =========================
print("\n\n========== PureGCN Depth Sweep Summary ==========")
header = f"{'Layers':<8}"
for s in test_subjects:
    header += f"  {s:>15}"
header += f"  {'mean':>8}"
print(header)
print("-" * len(header))

for nl in [2, 3, 5, 7]:
    line = f"  {nl:<6}"
    vals = [results[nl][s] for s in test_subjects]
    for v in vals:
        line += f"  {v:>15.4f}"
    line += f"  {np.mean(vals):>8.4f}"
    print(line)

# =========================
# Save
# =========================
from scipy import io
os.makedirs('results', exist_ok=True)
save_dict = {'test_subjects': test_subjects}
for nl in [2, 3, 5, 7]:
    save_dict[f'layer_{nl}'] = [results[nl][s] for s in test_subjects]
io.savemat(os.path.join('results', 'puregcn_depth_sweep.mat'), save_dict)
print("\nSaved -> results/puregcn_depth_sweep.mat")
