import time, os, subprocess, sys

print("Waiting for PID 8832 to finish...", flush=True)

while True:
    result = subprocess.run(['tasklist', '/FI', 'PID eq 8832'], capture_output=True, text=True)
    if '8832' not in result.stdout:
        print("PID 8832 finished. Starting BiLSTM...", flush=True)
        break
    time.sleep(60)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch, torch.nn as nn, torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from utils.Dataloader import LoadIMU_EPO_zaligned_fix
from utils.Model import BiLSTM
from scipy import io
from datetime import datetime

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"device: {device}", flush=True)

subject_list = ['250805_KDY','250731_LGE','250806_LJI','250812_WDY','250814_JCM',
                '250818_ICY','250819_PYH','250820_LTG','250822_JSH','250825_JDB']
num_class = 100
num_session = 5
max_epochs = 600
min_epochs = 500
patience = 20
batch_size = 128
lr = 0.0005
run_tag = 'LOSO_30ch_bilstm_lr5e4_right_zaligned'
run_id = datetime.now().strftime('%Y%m%d-%H%M%S')

best_acc_per_subject = np.full(len(subject_list), np.nan)
best_loss_per_subject = np.full(len(subject_list), np.nan)

for test_idx, test_subj in enumerate(subject_list):
    print(f'fold {test_idx+1}/10  test={test_subj}', flush=True)
    train_subjs = [s for i,s in enumerate(subject_list) if i != test_idx]

    X_tr, y_tr = LoadIMU_EPO_zaligned_fix('data_10ch', train_subjs, num_session, num_class)
    X_te, y_te = LoadIMU_EPO_zaligned_fix('data_10ch', [test_subj],  num_session, num_class)

    X_tr = X_tr.reshape(-1, 200, 30)
    y_tr = y_tr.reshape(-1)
    X_te = X_te.reshape(-1, 200, 30)
    y_te = y_te.reshape(-1)

    model = BiLSTM(num_classes=num_class, nCh=30, nFilters=250, kernel_size=8,
                   hidden_dim=250, gap_dropout=0.5).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(TensorDataset(torch.tensor(X_tr, dtype=torch.float32),
                                            torch.tensor(y_tr, dtype=torch.long)),
                              batch_size=batch_size, shuffle=True)
    X_te_t = torch.tensor(X_te, dtype=torch.float32).to(device)
    y_te_t  = torch.tensor(y_te, dtype=torch.long).to(device)

    best_val_acc = -1.0
    best_val_loss = np.inf
    no_improve = 0

    for epoch in range(max_epochs):
        model.train()
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            criterion(model(Xb), yb).backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(X_te_t)
            val_loss = criterion(logits, y_te_t).item()
            val_acc  = logits.argmax(1).eq(y_te_t).float().mean().item()

        if val_acc > best_val_acc:
            best_val_acc  = val_acc
            best_val_loss = val_loss
            no_improve = 0
        else:
            if epoch >= min_epochs - 1:
                no_improve += 1
                if no_improve >= patience:
                    print(f'  early stop at epoch {epoch}', flush=True)
                    break

    best_acc_per_subject[test_idx] = best_val_acc
    best_loss_per_subject[test_idx] = best_val_loss
    print(f'  >> best_acc={best_val_acc:.4f}', flush=True)

print(f'\nmean={np.mean(best_acc_per_subject):.4f}  std={np.std(best_acc_per_subject):.4f}')
print('per subj:', np.round(best_acc_per_subject, 3))

os.makedirs('results', exist_ok=True)
io.savemat(f'results/{run_tag}_{run_id}_result.mat',
           {'best_acc_per_subject': best_acc_per_subject,
            'best_loss_per_subject': best_loss_per_subject,
            'subject_list': subject_list})
print('saved.')
