import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils.Dataloader import LoadIMU_EPO_simple

# =========================
# Settings
# =========================
subject_list = [
    '250805_KDY', '250731_LGE', '260709_LJS', '250812_WDY', '250814_JCM',
    '250818_ICY', '250819_PYH', '250820_LTG', '250822_JSH', '250825_JDB'
]

load_path = os.path.join(os.getcwd(), 'data_10ch')
save_path = os.path.join(os.getcwd(), 'results', 'fstat_loso')
os.makedirs(save_path, exist_ok=True)

num_session = 5
num_class   = 100
top_k       = 20

NODE_LABELS = [f'S{i}' for i in range(10)]
edges_all   = [(i, j) for i in range(10) for j in range(i + 1, 10)]  # 45 edges
edge_labels = [f'({i},{j})' for i, j in edges_all]

# =========================
# F-statistic computation
# =========================
def compute_fstat(X_np, y_np):
    """X_np: (N, 200, 30), y_np: (N,)  →  F: (45,)"""
    norm = np.linalg.norm(X_np.reshape(-1, 200, 10, 3), axis=-1)  # (N, 200, 10)
    N = norm.shape[0]
    classes = np.unique(y_np)

    ec = np.zeros((N, 45))
    for idx, (i, j) in enumerate(edges_all):
        si = norm[:, :, i]
        sj = norm[:, :, j]
        si_c = si - si.mean(1, keepdims=True)
        sj_c = sj - sj.mean(1, keepdims=True)
        r = np.clip(
            (si_c * sj_c).sum(1) / (np.sqrt((si_c**2).sum(1) * (sj_c**2).sum(1)) + 1e-8),
            -0.9999, 0.9999
        )
        ec[:, idx] = np.arctanh(r)

    gm = ec.mean(0)
    bw = np.zeros(45)
    wt = np.zeros(45)
    for c in classes:
        m  = (y_np == c)
        cm = ec[m].mean(0)
        bw += m.sum() * (cm - gm) ** 2
        wt += ((ec[m] - cm) ** 2).sum(0)

    F = (bw / (len(classes) - 1)) / (wt / (N - len(classes)) + 1e-10)
    return F

# =========================
# LOSO loop
# =========================
F_matrix  = np.zeros((len(subject_list), 45))   # (10, 45)
top20_mask = np.zeros((len(subject_list), 45), dtype=int)  # 1 if in top-20

for test_idx, test_subj in enumerate(subject_list):
    print(f"[{test_idx+1}/{len(subject_list)}] test={test_subj}", end=' ... ')

    train_subjects = [s for i, s in enumerate(subject_list) if i != test_idx]

    X_train, y_train = LoadIMU_EPO_simple(
        load_path, train_subjects,
        num_session=num_session, num_class=num_class
    )
    X_train = X_train.reshape(-1, 200, 30)
    y_train = y_train.reshape(-1)

    F = compute_fstat(X_train, y_train)
    F_matrix[test_idx] = F

    top20_idx = np.argsort(-F)[:top_k]
    top20_mask[test_idx, top20_idx] = 1

    print(f"top-20: {[edge_labels[k] for k in top20_idx]}")

# =========================
# Helpers: 45-vec → 10×10 mat
# =========================
def vec_to_mat(vec, fill_diag=np.nan):
    mat = np.full((10, 10), np.nan)
    for idx, (i, j) in enumerate(edges_all):
        mat[i, j] = vec[idx]
        mat[j, i] = vec[idx]
    if fill_diag is not None:
        np.fill_diagonal(mat, fill_diag)
    return mat

def draw_heatmap(ax, mat, title, cmap, fmt='.1f', vmin=None, vmax=None, center=None):
    mask = np.isnan(mat)
    sns.heatmap(
        mat, ax=ax, cmap=cmap, center=center,
        vmin=vmin, vmax=vmax,
        annot=True, fmt=fmt, annot_kws={'size': 8},
        mask=mask, square=True,
        linewidths=0.4, linecolor='#cccccc',
        xticklabels=NODE_LABELS, yticklabels=NODE_LABELS,
        cbar_kws={'shrink': 0.8},
    )
    ax.set_title(title, fontsize=11, pad=6)
    ax.tick_params(labelsize=8)

# =========================
# Heatmap 1: 빈도수 (Top-20 등장 횟수)
# =========================
freq_vec = top20_mask.sum(axis=0)   # (45,)  0~10
freq_mat = vec_to_mat(freq_vec, fill_diag=0)

fig, ax = plt.subplots(figsize=(9, 7))
draw_heatmap(
    ax, freq_mat,
    title='Top-20 Edge Frequency  (LOSO, n=10 subjects)',
    cmap='YlOrRd', fmt='.0f', vmin=0, vmax=10
)
plt.tight_layout()
out1 = os.path.join(save_path, 'heatmap_top20_frequency.png')
plt.savefig(out1, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved: {out1}")

# =========================
# Heatmap 2: 평균 F-statistic
# =========================
mean_F_vec = F_matrix.mean(axis=0)  # (45,)
mean_F_mat = vec_to_mat(mean_F_vec, fill_diag=np.nan)

fig, ax = plt.subplots(figsize=(9, 7))
draw_heatmap(
    ax, mean_F_mat,
    title='Mean F-statistic per Edge  (LOSO, n=10 subjects)',
    cmap='Blues', fmt='.2f'
)
plt.tight_layout()
out2 = os.path.join(save_path, 'heatmap_mean_fstat.png')
plt.savefig(out2, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {out2}")

# =========================
# Excel: fold별 F-statistic
# =========================
df_fstat = pd.DataFrame(
    F_matrix,
    index=subject_list,
    columns=edge_labels
)
df_fstat.index.name = 'test_subject'

df_freq = pd.DataFrame(
    [freq_vec],
    index=['frequency'],
    columns=edge_labels
)
df_freq.index.name = 'test_subject'

df_mean = pd.DataFrame(
    [mean_F_vec],
    index=['mean_F'],
    columns=edge_labels
)
df_mean.index.name = 'test_subject'

df_top20 = pd.DataFrame(
    top20_mask,
    index=subject_list,
    columns=edge_labels
)
df_top20.index.name = 'test_subject'

excel_path = os.path.join(save_path, 'fstat_loso_results.xlsx')
with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
    df_fstat.to_excel(writer, sheet_name='F_statistic_per_fold')
    df_top20.to_excel(writer, sheet_name='Top20_mask')
    pd.concat([df_mean, df_freq]).to_excel(writer, sheet_name='Summary')

print(f"Saved: {excel_path}")
print("\nDone.")
