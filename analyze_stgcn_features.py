#%%
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torch.utils.data import TensorDataset, DataLoader
from utils.Dataloader import LoadIMU_EPO_simple

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# =========================
# Settings
# =========================
ARCH_PT   = os.path.join('models', 'Model_122.pt')
MODEL_DIR = os.path.join('models', 'ST-GCN')
DATA_DIR  = os.path.join(os.getcwd(), 'data_10ch')
TAG       = 'LOSO_30ch_ST-GCN_260709_test'

subject_list    = ['250805_KDY','250731_LGE','260709_LJS','250812_WDY','250814_JCM',
                   '250818_ICY','250819_PYH','250820_LTG','250822_JSH','250825_JDB']
num_session     = 5
num_class       = 100
num_time_IMU    = 200
num_channel_IMU = 30

# =========================
# Linear CKA
# =========================
def linear_CKA(X, Y):
    """X, Y: (N, D) numpy — mean-centred linear CKA"""
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    num = np.linalg.norm(X.T @ Y, 'fro') ** 2
    den = np.linalg.norm(X.T @ X, 'fro') * np.linalg.norm(Y.T @ Y, 'fro')
    return float(num / (den + 1e-12))

# =========================
# Feature extraction (hooks)
# =========================
def extract_features(model, loader):
    """
    Captures per-layer outputs with forward hooks.
    gcn_feats[i]   : (N, C_i)  — after ConvTemporalGraphical, mean-pooled (T, V)
    block_feats[i] : (N, C_i)  — after full st_gcn block (relu out), mean-pooled (T, V)
    """
    L = len(model.st_gcn_networks)
    gcn_buf   = [[] for _ in range(L)]
    block_buf = [[] for _ in range(L)]
    hooks = []

    for i, blk in enumerate(model.st_gcn_networks):
        def gcn_hook(m, inp, out, idx=i):
            # ConvTemporalGraphical returns (x, A)
            gcn_buf[idx].append(out[0].mean(dim=(2, 3)).detach().cpu().numpy())
        def block_hook(m, inp, out, idx=i):
            # st_gcn returns (x, A)
            block_buf[idx].append(out[0].mean(dim=(2, 3)).detach().cpu().numpy())

        hooks.append(blk.gcn.register_forward_hook(gcn_hook))
        hooks.append(blk.register_forward_hook(block_hook))

    model.eval()
    with torch.no_grad():
        for X, _ in loader:
            model(X.to(device))

    for h in hooks:
        h.remove()

    gcn_feats   = [np.concatenate(b, axis=0) for b in gcn_buf]
    block_feats = [np.concatenate(b, axis=0) for b in block_buf]
    return gcn_feats, block_feats

# =========================
# Main loop
# =========================
all_intra_cka  = []   # (n_subj, 7) — CKA(gcn_i, block_i)
all_gcn_chain  = []   # (n_subj, 6) — CKA(gcn_i, gcn_{i+1})
all_blk_chain  = []   # (n_subj, 6) — CKA(block_i, block_{i+1})
all_matrix     = []   # (n_subj, 14, 14)
valid_subjects = []

for test_subj in subject_list:
    pth = os.path.join(MODEL_DIR, f'{TAG}_{test_subj}.pth')
    if not os.path.exists(pth):
        print(f"[skip] {test_subj}")
        continue
    print(f"[{test_subj}] loading ...", end=' ', flush=True)

    model = torch.load(ARCH_PT, map_location=device, weights_only=False)
    model.load_state_dict(torch.load(pth, map_location=device, weights_only=True))
    model.to(device)

    X_np, y_np = LoadIMU_EPO_simple(DATA_DIR, [test_subj], num_session, num_class)
    X_t = torch.tensor(X_np.reshape(-1, num_time_IMU, num_channel_IMU), dtype=torch.float32)
    y_t = torch.tensor(y_np.reshape(-1), dtype=torch.long)
    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=256, shuffle=False)

    gcn_feats, block_feats = extract_features(model, loader)
    L = len(gcn_feats)

    # 1) intra-layer: CKA(gcn_i, block_i) — TCN이 공간 feature를 얼마나 바꾸나
    intra = [linear_CKA(gcn_feats[i], block_feats[i]) for i in range(L)]
    all_intra_cka.append(intra)

    # 2) gcn chain: CKA(gcn_i, gcn_{i+1}) — GCN feature가 layer 간 얼마나 변하나
    gcn_chain = [linear_CKA(gcn_feats[i], gcn_feats[i+1]) for i in range(L-1)]
    all_gcn_chain.append(gcn_chain)

    # 3) block chain: CKA(block_i, block_{i+1})
    blk_chain = [linear_CKA(block_feats[i], block_feats[i+1]) for i in range(L-1)]
    all_blk_chain.append(blk_chain)

    # 4) 14×14 pairwise: [G1..G7, T1..T7]
    all_feats = gcn_feats + block_feats
    mat = np.array([[linear_CKA(all_feats[a], all_feats[b])
                     for b in range(14)] for a in range(14)])
    all_matrix.append(mat)
    valid_subjects.append(test_subj)

    print(f"intra-CKA = {[f'{v:.2f}' for v in intra]}")

# =========================
# Aggregate
# =========================
mean_intra  = np.mean(all_intra_cka, axis=0)   # (7,)
mean_gcnc   = np.mean(all_gcn_chain,  axis=0)   # (6,)
mean_blkc   = np.mean(all_blk_chain,  axis=0)   # (6,)
mean_matrix = np.mean(all_matrix,    axis=0)   # (14, 14)

# =========================
# Plot
# =========================
fig = plt.figure(figsize=(20, 12))
gs  = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.35)

layers  = np.arange(1, 8)
layers6 = np.arange(1, 7)          # x-axis for chain plots (between layers)

# ── (A) 14×14 Heatmap ──
ax0 = fig.add_subplot(gs[:, 0])
im = ax0.imshow(mean_matrix, vmin=0, vmax=1, cmap='RdYlGn', aspect='auto')
tick_labels = [f'G{i+1}' for i in range(7)] + [f'T{i+1}' for i in range(7)]
ax0.set_xticks(range(14)); ax0.set_xticklabels(tick_labels, fontsize=9)
ax0.set_yticks(range(14)); ax0.set_yticklabels(tick_labels, fontsize=9)
ax0.set_title('(A) Pairwise Linear CKA — mean over all subjects\nG=after GCN, T=after TCN block', fontsize=11)
plt.colorbar(im, ax=ax0, fraction=0.03, pad=0.02)
ax0.axhline(6.5, color='white', lw=2.5)
ax0.axvline(6.5, color='white', lw=2.5)
for a in range(14):
    for b in range(14):
        v = mean_matrix[a, b]
        ax0.text(b, a, f'{v:.2f}', ha='center', va='center', fontsize=6,
                 color='black' if 0.25 < v < 0.85 else 'white')

# ── (B) Intra-layer CKA: GCN vs TCN block ──
ax1 = fig.add_subplot(gs[0, 1])
ax1.plot(layers, mean_intra, 'o-', color='steelblue', lw=2, ms=8, label='mean')
for i, v in enumerate(mean_intra):
    ax1.annotate(f'{v:.3f}', (i+1, v), textcoords='offset points', xytext=(0, 8),
                 ha='center', fontsize=8)
for subj_row in all_intra_cka:
    ax1.plot(layers, subj_row, '-', alpha=0.2, color='gray')
ax1.set_xlabel('Layer index', fontsize=11)
ax1.set_ylabel('CKA(GCN_out, Block_out)', fontsize=11)
ax1.set_title('(B) Intra-layer: how much TCN changes GCN features\n(1=unchanged, 0=completely different)', fontsize=10)
ax1.set_xticks(list(layers)); ax1.set_ylim(0, 1.05); ax1.grid(alpha=0.3)

# ── (C) Inter-layer chain similarity ──
ax2 = fig.add_subplot(gs[1, 1])
ax2.plot(layers6, mean_gcnc, 's--', color='tomato',    lw=2, ms=7, label='GCN chain')
ax2.plot(layers6, mean_blkc, 'o-',  color='steelblue', lw=2, ms=7, label='Block chain')
for i, (gv, bv) in enumerate(zip(mean_gcnc, mean_blkc)):
    ax2.annotate(f'{gv:.2f}', (i+1, gv), textcoords='offset points', xytext=(-8, 6),
                 ha='center', fontsize=7, color='tomato')
    ax2.annotate(f'{bv:.2f}', (i+1, bv), textcoords='offset points', xytext=(8, -12),
                 ha='center', fontsize=7, color='steelblue')
ax2.set_xlabel('Layer i → i+1', fontsize=11)
ax2.set_ylabel('CKA(layer_i, layer_{i+1})', fontsize=11)
ax2.set_title('(C) Layer-to-layer feature drift\n(낮을수록 다음 layer에서 representation 많이 변화)', fontsize=10)
ax2.set_xticks(list(layers6))
ax2.set_xticklabels([f'L{i}→L{i+1}' for i in range(1, 7)], fontsize=8)
ax2.set_ylim(0, 1.05); ax2.legend(fontsize=9); ax2.grid(alpha=0.3)

# ── save ──
os.makedirs('results', exist_ok=True)
save_path = os.path.join('results', 'stgcn_feature_similarity.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"\nSaved → {save_path}")
plt.show()

# =========================
# Print table
# =========================
print("\n===== (B) Intra-layer CKA(GCN_out, Block_out) =====")
print(f"{'Subject':<20}", end="")
for i in range(7): print(f"  L{i+1}  ", end="")
print()
for subj, row in zip(valid_subjects, all_intra_cka):
    print(f"{subj:<20}", end="")
    for v in row: print(f" {v:.3f}", end="")
    print()
print(f"\n{'mean':<20}", end="")
for v in mean_intra: print(f" {v:.3f}", end="")
print()

print("\n===== (C) Layer-to-layer CKA =====")
print(f"{'':20}", end="")
for i in range(6): print(f"  L{i+1}→{i+2}", end="")
print()
print(f"{'GCN chain':<20}", end="")
for v in mean_gcnc: print(f"  {v:.3f}", end="")
print()
print(f"{'Block chain':<20}", end="")
for v in mean_blkc: print(f"  {v:.3f}", end="")
print()
