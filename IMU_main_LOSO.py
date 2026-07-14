import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from utils.Dataloader import LoadIMU_EPO_simple, LoadIMU_EPO_zaligned_fix_yaw, LoadIMU_EPO_zaligned_fix
from utils.graph import compute_fold_A
from scipy import io
import random
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# =========================
# Settings
# =========================
subject_list = ['250805_KDY','250731_LGE','260709_LJS','250812_WDY','250814_JCM','250818_ICY','250819_PYH','250820_LTG','250822_JSH','250825_JDB']
#left='250804_KTS','250805_SMC','250811_LPR','250811_JHS','250812_HHJ','250813_YMS','250814_CYJ','250819_CYK','260713_LSW', '250827_HJH'
#right='250805_KDY','250731_LGE','260709_LJS','250812_WDY','250814_JCM','250818_ICY','250819_PYH','250820_LTG','250822_JSH','250825_JDB'

model_pt_path  = os.path.join('models', 'Model_136.pt')
run_tag        = 'LOSO_30ch_ST-GCN_model35'

num_subject = len(subject_list)
num_session = 5

fs = 50
num_time_IMU = fs * 4         # 200
num_class = 100
num_channel_IMU = 30          # raw IMU (10 sensor × 3 axis)
num_sensor = 10
num_feat_per_sensor = 3       # 3 raw axes only (no magnitude)

this_path = os.getcwd()
path_sub_proj = this_path

load_path  = os.path.join(path_sub_proj, 'data_10ch')
save_path  = os.path.join(path_sub_proj, 'results')
model_path = os.path.join(path_sub_proj, 'models')

os.makedirs(save_path, exist_ok=True)
os.makedirs(model_path, exist_ok=True)

max_epochs = 600
min_epochs = 500
patience   = 20
batch_size = 128
lr         = 1e-3

base_seed = 2023

# =========================
# LOSO loop
# =========================
run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
tb_logdir = os.path.join(save_path, "run", f"{run_tag}_{run_id}")
writer = SummaryWriter(log_dir=tb_logdir)
print("TensorBoard logdir:", tb_logdir)

# 결과 저장
best_acc_per_subject = np.full(num_subject, np.nan)
best_loss_per_subject = np.full(num_subject, np.nan)
train_curves = np.full((num_subject, 2, max_epochs), np.nan)  # [subj, train/val, epoch] loss
acc_curves   = np.full((num_subject, 2, max_epochs), np.nan)  # [subj, train/val, epoch] acc

for test_idx, test_subj in enumerate(subject_list):
    print(f"\n========== LOSO fold {test_idx+1}/{num_subject}  test = {test_subj} ==========")

    # ---- seed (test subject 마다 다르게) ----
    seed = base_seed + test_idx * 100
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.enabled = False

    # ---- split + train-only augmentation ----
    train_subjects = [s for i, s in enumerate(subject_list) if i != test_idx]
    test_subjects  = [test_subj]

    X_train_data, y_train_data = LoadIMU_EPO_simple(
        load_path,
        train_subjects,
        num_session=num_session,
        num_class=num_class,
    )

    X_test_data, y_test_data = LoadIMU_EPO_simple(
        load_path,
        test_subjects,
        num_session=num_session,
        num_class=num_class,
    )

    X_train = X_train_data.reshape(-1, num_time_IMU, num_channel_IMU)
    y_train = y_train_data.reshape(-1)

    X_test = X_test_data.reshape(-1, num_time_IMU, num_channel_IMU)
    y_test = y_test_data.reshape(-1)

    # ---- 30ch ----
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    X_test_t  = torch.tensor(X_test,  dtype=torch.float32)

    print(f"  train: {tuple(X_train_t.shape)} (9 subjects × 5 sessions × 100 class, 200, 30)")
    print(f"  test : {tuple(X_test_t.shape)}  (1 subject × 5 sessions × 100 class, 200, 30)")

    # ---- DataLoader ----
    train_ds = TensorDataset(
        X_train_t,
        torch.tensor(y_train, dtype=torch.long),
    )
    test_ds = TensorDataset(
        X_test_t,
        torch.tensor(y_test, dtype=torch.long),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    model = torch.load(model_pt_path, map_location=device, weights_only=False)
    if not hasattr(model, 'in_channels'):
        model.in_channels = num_feat_per_sensor
    model.to(device)

    #fold별 F-statistic 그래프 계산 후 A 교체
    A_fold = compute_fold_A(X_train, y_train, top_k=20).to(device)
    model.A.data.copy_(A_fold)
    for ei in model.edge_importance:
        nn.init.ones_(ei)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_acc  = -1.0
    best_val_loss = np.inf
    best_epoch    = -1
    no_improve    = 0
    best_state    = None
    best_train_acc_at_best = -1.0   # best val 시점의 train acc 도 기록

    for epoch in tqdm(range(max_epochs),
                      desc=f"{test_subj}",
                      leave=False):
        # ---- Train ----
        model.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            tr_loss    += loss.item() * X.size(0)
            tr_correct += logits.argmax(1).eq(y).sum().item()
            tr_total   += y.size(0)

        train_loss = tr_loss / tr_total
        train_acc  = tr_correct / tr_total

        # ---- Eval ----
        model.eval()
        va_loss, va_correct, va_total = 0.0, 0, 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                logits = model(X)
                loss = criterion(logits, y)
                va_loss    += loss.item() * X.size(0)
                va_correct += logits.argmax(1).eq(y).sum().item()
                va_total   += y.size(0)

        val_loss = va_loss / va_total
        val_acc  = va_correct / va_total

        # ---- logging ----
        train_curves[test_idx, 0, epoch] = train_loss
        train_curves[test_idx, 1, epoch] = val_loss
        acc_curves[test_idx, 0, epoch] = train_acc
        acc_curves[test_idx, 1, epoch] = val_acc

        writer.add_scalar(f"{test_subj}/train_loss", train_loss, epoch)
        writer.add_scalar(f"{test_subj}/val_loss",   val_loss,   epoch)
        writer.add_scalar(f"{test_subj}/train_acc",  train_acc,  epoch)
        writer.add_scalar(f"{test_subj}/val_acc",    val_acc,    epoch)

        # ---- best tracking ----
        improved = val_acc > best_val_acc
        if improved:
            best_val_acc  = val_acc
            best_val_loss = val_loss
            best_epoch    = epoch
            best_train_acc_at_best = train_acc
            best_state    = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve    = 0
        else:
            if epoch >= min_epochs - 1:
                no_improve += 1
                if no_improve >= patience:
                    print(f"  early stop at epoch {epoch}, best epoch was {best_epoch}")
                    break

    # ---- save best ----
    best_acc_per_subject[test_idx]  = best_val_acc
    best_loss_per_subject[test_idx] = best_val_loss

    best_model_file = os.path.join(model_path, f"{run_tag}_test_{test_subj}.pth")
    if best_state is not None:
        torch.save(best_state, best_model_file)
    print(f"  ▶ {test_subj}: best val acc = {best_val_acc:.4f}  "
          f"(train_acc at best = {best_train_acc_at_best:.4f}, epoch {best_epoch})")


# =========================
# Summary
# =========================
print(f"\n========== {run_tag} Summary ==========")
for i, s in enumerate(subject_list):
    print(f"  {s}: {best_acc_per_subject[i]:.4f}")
print(f"\n  mean ± std : {np.mean(best_acc_per_subject):.4f} ± {np.std(best_acc_per_subject):.4f}")

# =========================
# Save .mat
# =========================
io.savemat(
    os.path.join(save_path, f"{run_tag}_{run_id}_result.mat"),
    {
        'best_acc_per_subject':  best_acc_per_subject,
        'best_loss_per_subject': best_loss_per_subject,
        'train_curves':          train_curves,
        'acc_curves':            acc_curves,
        'subject_list':          subject_list,
    }
)
print(f"\nresult saved: {run_tag}_{run_id}_result.mat")

writer.flush()
writer.close()
