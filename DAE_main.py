import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from scipy import io
import random
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from utils.DAE_Model import DAE_STGCN, DAE_BiLSTM


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# =========================
# Settings
# =========================
NUM_SENSOR = 5
NUM_AXIS   = 3
NUM_CH     = NUM_SENSOR * NUM_AXIS   # 15
NUM_TIME   = 200
NUM_CLASS  = 30

base_dir   = 'DAE_npydata'
save_path  = 'results'
model_path = 'models'
os.makedirs(save_path, exist_ok=True)
os.makedirs(model_path, exist_ok=True)

# Train: sess 1 + 2 (clean) 고정
train_session_files = [
    'sess1_clean_X.npy',
    'sess2_clean_X.npy',
]
train_label_files = [
    'sess1_clean_Y.npy',
    'sess2_clean_Y.npy',
]

# Test conditions — 각각 별도 실험
test_conditions = [
    # (조건 이름, X 파일, Y 파일)
    ('sess4_hori', 'sess4_motion_hori_X.npy', 'sess4_motion_hori_Y.npy'),
    ('sess5_vert', 'sess5_motion_vert_X.npy', 'sess5_motion_vert_Y.npy'),
    ('sess6_rand', 'sess6_motion_rand_X.npy', 'sess6_motion_rand_Y.npy'),
]
num_test = len(test_conditions)

# Train hyperparameters
max_epochs = 250
min_epochs = 200
patience   = 20
batch_size = 32
lr         = 1e-3

run_tag    = "DAE_STGCN"
base_seed  = 2023

# =========================
# Subject list
# =========================
subject_list = sorted([d for d in os.listdir(base_dir)
                       if os.path.isdir(os.path.join(base_dir, d))])
num_subject = len(subject_list)
print(f"subjects ({num_subject}): {subject_list}")


# =========================
# Helpers
# =========================
def augment_batch(X, noise_std=0.05, shift_max=10):
    """
    Train batch 에 적용하는 가벼운 augmentation.
    - Gaussian noise: 시간축에 noise 추가 → motion-like 변형 학습
    - Time shift: 전체 신호를 -shift_max ~ +shift_max 만큼 이동
    """
    B = X.size(0)
    # Gaussian noise (channel-wise std 기반 비례)
    noise = torch.randn_like(X) * (X.std(dim=1, keepdim=True) * noise_std)
    X = X + noise
    # Random time shift (sample 마다 다르게)
    if shift_max > 0:
        out = torch.empty_like(X)
        for i in range(B):
            s = torch.randint(-shift_max, shift_max + 1, (1,)).item()
            out[i] = torch.roll(X[i], shifts=s, dims=0)
        X = out
    return X


def mixup(X, y, alpha=0.4):
    """
    Mixup augmentation.
    두 sample 을 선형 보간한 가상 sample 생성 → overfit 강력하게 억제.
    Loss: lam * CE(pred, y_a) + (1-lam) * CE(pred, y_b)
    """
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(X.size(0), device=X.device)
    X_mix = lam * X + (1.0 - lam) * X[idx]
    y_a, y_b = y, y[idx]
    return X_mix, y_a, y_b, lam


def load_subject(subj_dir, x_files, y_files):
    """Concat multiple sessions of one subject. Returns (X, y_int)."""
    Xs, Ys = [], []
    for xf, yf in zip(x_files, y_files):
        X = np.load(os.path.join(subj_dir, xf))   # (trial, 200, 15)
        Y = np.load(os.path.join(subj_dir, yf))   # (trial, 30) one-hot
        assert X.shape[1:] == (NUM_TIME, NUM_CH), f"bad X shape {X.shape} in {xf}"
        assert Y.shape == (X.shape[0], NUM_CLASS), f"bad Y shape {Y.shape} in {yf}"
        Xs.append(X)
        Ys.append(Y)
    X = np.concatenate(Xs, axis=0)
    Y = np.concatenate(Ys, axis=0)
    y_int = np.argmax(Y, axis=1).astype(np.int64)
    return X.astype(np.float32), y_int


# =========================
# Training loop
# =========================
run_id    = datetime.now().strftime("%Y%m%d-%H%M%S")
tb_logdir = os.path.join(save_path, "runboard", f"{run_tag}_{run_id}")
writer    = SummaryWriter(log_dir=tb_logdir)
print("TensorBoard logdir:", tb_logdir)

# 결과: [subject, test_condition]
best_acc  = np.full((num_subject, num_test), np.nan)
best_loss = np.full((num_subject, num_test), np.nan)
train_curves = np.full((num_subject, num_test, 2, max_epochs), np.nan)
acc_curves   = np.full((num_subject, num_test, 2, max_epochs), np.nan)

for s_idx, subj in enumerate(subject_list):
    print(f"\n========== {s_idx+1}/{num_subject}  {subj} ==========")
    subj_dir = os.path.join(base_dir, subj)

    # Train 데이터는 sess1+2 로 고정 → subject 마다 한 번만 로드
    X_train, y_train = load_subject(subj_dir, train_session_files, train_label_files)
    print(f"  train: {X_train.shape}  (sess 1+2 clean)")

    for t_idx, (test_name, test_x, test_y) in enumerate(test_conditions):
        print(f"  --- test condition: {test_name} ---")

        # seed (subject + test_condition 마다 다르게)
        seed = base_seed + s_idx * 100 + t_idx
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = False

        # Test 데이터 로드
        X_test, y_test = load_subject(subj_dir, [test_x], [test_y])
        print(f"    test : {X_test.shape}  ({test_name})")

        # DataLoader
        train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
        test_ds  = TensorDataset(torch.tensor(X_test),  torch.tensor(y_test))
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

        # Fresh model — DAE_STGCN (5-node FC graph, regularized for small data)
        model = DAE_STGCN(
            num_class=NUM_CLASS,
            in_channels=NUM_AXIS,
            num_node=NUM_SENSOR,
            max_hop=1, dilation=1,
            edge_importance_weighting=True,
            dropout=0.4,                       # 작은 데이터(60 sample) overfit 방지
        ).to(device)

        # Label smoothing + weight decay 도 같이 (30-class 에선 label smoothing 이 안정성 도움)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

        best_val_acc  = -1.0
        best_val_loss = np.inf
        best_epoch    = -1
        no_improve    = 0
        best_state    = None
        best_train_acc_at_best = -1.0

        for epoch in tqdm(range(max_epochs),
                          desc=f"{subj}/{test_name}",
                          leave=False):
            # Train
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

            # Eval
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

            # log
            train_curves[s_idx, t_idx, 0, epoch] = train_loss
            train_curves[s_idx, t_idx, 1, epoch] = val_loss
            acc_curves[s_idx, t_idx, 0, epoch] = train_acc
            acc_curves[s_idx, t_idx, 1, epoch] = val_acc

            writer.add_scalar(f"{subj}/{test_name}/train_loss", train_loss, epoch)
            writer.add_scalar(f"{subj}/{test_name}/val_loss",   val_loss,   epoch)
            writer.add_scalar(f"{subj}/{test_name}/train_acc",  train_acc,  epoch)
            writer.add_scalar(f"{subj}/{test_name}/val_acc",    val_acc,    epoch)

            # 학습 진행 가시화 (50 epoch 마다)
            if (epoch + 1) % 50 == 0 or epoch == 0:
                print(f"    epoch {epoch+1:>3d}  train_acc={train_acc:.3f}  val_acc={val_acc:.3f}  "
                      f"train_loss={train_loss:.3f}  val_loss={val_loss:.3f}")

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
                        print(f"    early stop @ epoch {epoch}, best={best_epoch}")
                        break

        best_acc[s_idx, t_idx]  = best_val_acc
        best_loss[s_idx, t_idx] = best_val_loss

        if best_state is not None:
            torch.save(best_state,
                       os.path.join(model_path, f"{run_tag}_{subj}_{test_name}.pth"))

        print(f"    ▶ best val acc = {best_val_acc:.4f}  "
              f"(train_acc at best = {best_train_acc_at_best:.4f}, epoch {best_epoch})")


# =========================
# Summary
# =========================
test_names = [c[0] for c in test_conditions]

print(f"\n========== {run_tag} Summary ==========")
print(f"{'subject':<20s}  " + "  ".join(f"{n:<12s}" for n in test_names))
for i, s in enumerate(subject_list):
    print(f"{s:<20s}  " + "  ".join(f"{best_acc[i, t]:.4f}      " for t in range(num_test)))

print(f"\n{'mean':<20s}  " + "  ".join(f"{np.mean(best_acc[:, t]):.4f}      " for t in range(num_test)))
print(f"{'std':<20s}  " + "  ".join(f"{np.std(best_acc[:, t]):.4f}      "  for t in range(num_test)))

io.savemat(
    os.path.join(save_path, f"{run_tag}_{run_id}_result.mat"),
    {
        'best_acc':     best_acc,         # (subject, test_cond)
        'best_loss':    best_loss,
        'train_curves': train_curves,
        'acc_curves':   acc_curves,
        'subject_list': subject_list,
        'test_names':   test_names,
    }
)
print(f"\nresult saved: {run_tag}_{run_id}_result.mat")

writer.flush()
writer.close()
