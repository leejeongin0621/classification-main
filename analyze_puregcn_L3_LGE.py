#%%
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torch.utils.data import TensorDataset, DataLoader
from utils.Dataloader import LoadIMU_EPO_simple
from utils.Model import PureGCN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =========================
# Load model
# =========================
PTH_PATH  = os.path.join('models', 'PureGCN', 'PureGCN_L3_test_250731_LGE.pth')
DATA_DIR  = os.path.join(os.getcwd(), 'data_10ch')
TEST_SUBJ = '250731_LGE'
HIDDEN_DIMS = (64, 128, 256)

model = PureGCN(in_channels=3, num_class=100,
                graph_args={'max_hop': 1, 'dilation': 1},
                hidden_dims=HIDDEN_DIMS, dropout=0.2).to(device)
state = torch.load(PTH_PATH, map_location=device, weights_only=True)
model.load_state_dict(state)
model.eval()
print(f"Loaded: {PTH_PATH}")
print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

# =========================
# Test data
# =========================
X_np, y_np = LoadIMU_EPO_simple(DATA_DIR, [TEST_SUBJ], num_session=5, num_class=100)
X_t = torch.tensor(X_np.reshape(-1, 200, 30), dtype=torch.float32)
y_t = torch.tensor(y_np.reshape(-1), dtype=torch.long)
loader = DataLoader(TensorDataset(X_t, y_t), batch_size=256, shuffle=False)

# =========================
# Linear CKA
# =========================
def linear_CKA(X, Y):
    X = X - X.mean(0); Y = Y - Y.mean(0)
    num = np.linalg.norm(X.T @ Y, 'fro') ** 2
    den = np.linalg.norm(X.T @ X, 'fro') * np.linalg.norm(Y.T @ Y, 'fro')
    return float(num / (den + 1e-12))

# =========================
# Forward hooks
# =========================
layer_bufs = [[] for _ in range(len(model.gcn_layers))]
hooks = []
for i, blk in enumerate(model.gcn_layers):
    def hook(m, inp, out, idx=i):
        layer_bufs[idx].append(out.mean(dim=(2, 3)).detach().cpu().numpy())
    hooks.append(blk.register_forward_hook(hook))

all_logits, all_labels = [], []
with torch.no_grad():
    for X, y in loader:
        X = X.to(device)
        out = model(X)
        all_logits.append(out.cpu())
        all_labels.append(y)

for h in hooks:
    h.remove()

all_logits = torch.cat(all_logits)
all_labels = torch.cat(all_labels)
layer_feats = [np.concatenate(b, axis=0) for b in layer_bufs]

# =========================
# 1) Accuracy
# =========================
preds = all_logits.argmax(1)
acc = (preds == all_labels).float().mean().item()
print(f"\n[Accuracy] {TEST_SUBJ}: {acc:.4f}  ({int(acc*500)}/500)")

# =========================
# 2) Per-class accuracy
# =========================
per_class_acc = np.zeros(100)
for c in range(100):
    mask = (all_labels == c)
    per_class_acc[c] = (preds[mask] == all_labels[mask]).float().mean().item()

worst5  = np.argsort(per_class_acc)[:5]
best5   = np.argsort(per_class_acc)[-5:][::-1]
print(f"\n[Worst 5 classes] {[(int(c), f'{per_class_acc[c]:.2f}') for c in worst5]}")
print(f"[Best  5 classes] {[(int(c), f'{per_class_acc[c]:.2f}') for c in best5]}")

# =========================
# 3) Layer-wise CKA
# =========================
# input feature: (N, T, V, C) → mean over T → (N, V*C=30)
X_feat = X_t.numpy().reshape(-1, 200, 10, 3).mean(axis=1).reshape(-1, 30)
all_feats = [X_feat] + layer_feats   # input + 3 layers

labels_str = ['Input(30)', 'GCN-L1(64)', 'GCN-L2(128)', 'GCN-L3(256)']
N = len(all_feats)
cka_mat = np.array([[linear_CKA(all_feats[a], all_feats[b]) for b in range(N)] for a in range(N)])

chain_cka = [linear_CKA(all_feats[i], all_feats[i+1]) for i in range(N-1)]
print(f"\n[Layer-chain CKA]")
for i, v in enumerate(chain_cka):
    print(f"  {labels_str[i]:15s} → {labels_str[i+1]:15s} : {v:.4f}")

# =========================
# 4) Graph A analysis
# =========================
A = model.A.cpu().numpy()   # (2, 10, 10)
sensor_names = [f'S{i+1}' for i in range(10)]

# =========================
# Plot
# =========================
fig = plt.figure(figsize=(20, 14))
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

# (A) per-class accuracy bar
ax0 = fig.add_subplot(gs[0, :2])
colors = ['tomato' if v < 0.4 else ('gold' if v < 0.7 else 'steelblue') for v in per_class_acc]
ax0.bar(range(100), per_class_acc, color=colors, width=0.8)
ax0.axhline(acc, color='black', lw=1.5, linestyle='--', label=f'mean={acc:.3f}')
ax0.set_xlabel('Class index', fontsize=11)
ax0.set_ylabel('Accuracy', fontsize=11)
ax0.set_title(f'(A) Per-class accuracy — PureGCN L3  {TEST_SUBJ}', fontsize=11)
ax0.set_ylim(0, 1.05); ax0.legend()
from matplotlib.patches import Patch
ax0.legend(handles=[
    Patch(color='steelblue', label='>=0.7'),
    Patch(color='gold',      label='0.4~0.7'),
    Patch(color='tomato',    label='<0.4'),
    plt.Line2D([0],[0], color='black', lw=1.5, linestyle='--', label=f'mean={acc:.3f}'),
], fontsize=9)

# (B) CKA heatmap
ax1 = fig.add_subplot(gs[0, 2])
im = ax1.imshow(cka_mat, vmin=0, vmax=1, cmap='RdYlGn')
ax1.set_xticks(range(N)); ax1.set_xticklabels(labels_str, fontsize=8, rotation=15)
ax1.set_yticks(range(N)); ax1.set_yticklabels(labels_str, fontsize=8)
ax1.set_title('(B) Layer CKA\n(Input + 3 GCN layers)', fontsize=10)
plt.colorbar(im, ax=ax1, fraction=0.046)
for a in range(N):
    for b in range(N):
        ax1.text(b, a, f'{cka_mat[a,b]:.2f}', ha='center', va='center', fontsize=8)

# (C) Chain CKA line
ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(range(N-1), chain_cka, 'o-', color='steelblue', lw=2, ms=9)
for i, v in enumerate(chain_cka):
    ax2.annotate(f'{v:.3f}', (i, v), textcoords='offset points', xytext=(0, 8), ha='center', fontsize=9)
ax2.set_xticks(range(N-1))
ax2.set_xticklabels([f'{labels_str[i]}\n→\n{labels_str[i+1]}' for i in range(N-1)], fontsize=7)
ax2.set_ylabel('CKA', fontsize=11)
ax2.set_title('(C) Layer-to-layer CKA drift', fontsize=10)
ax2.set_ylim(0, 1.05); ax2.grid(alpha=0.3)

# (D) Graph A[1] heatmap (1-hop adjacency)
ax3 = fig.add_subplot(gs[1, 1])
im2 = ax3.imshow(A[1], cmap='Blues')
ax3.set_xticks(range(10)); ax3.set_xticklabels(sensor_names, fontsize=8)
ax3.set_yticks(range(10)); ax3.set_yticklabels(sensor_names, fontsize=8)
ax3.set_title('(D) Learned graph A[1]\n(1-hop adjacency weight)', fontsize=10)
plt.colorbar(im2, ax=ax3, fraction=0.046)
for i in range(10):
    for j in range(10):
        if A[1, i, j] > 0.01:
            ax3.text(j, i, f'{A[1,i,j]:.2f}', ha='center', va='center', fontsize=6)

# (E) Per-class acc distribution
ax4 = fig.add_subplot(gs[1, 2])
ax4.hist(per_class_acc, bins=20, color='steelblue', edgecolor='white')
ax4.axvline(acc, color='red', lw=1.5, linestyle='--', label=f'mean={acc:.3f}')
ax4.set_xlabel('Class accuracy', fontsize=11)
ax4.set_ylabel('# classes', fontsize=11)
ax4.set_title('(E) Per-class accuracy distribution', fontsize=10)
ax4.legend(fontsize=9)

os.makedirs('results', exist_ok=True)
save_path = os.path.join('results', f'puregcn_L3_{TEST_SUBJ}_analysis.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"\nSaved -> {save_path}")
plt.show()
