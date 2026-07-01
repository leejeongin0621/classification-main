import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

model_dir = 'models/right/st-gcn'
num_layers = 7

# ── 1. 모든 피험자 PA 로드 ──────────────────────────────────────────────
subject_PAs = {}   # {subject: (7, 2, 10, 10)}
PA_mask = None

for fname in sorted(os.listdir(model_dir)):
    if not fname.endswith('.pth'):
        continue
    subj = fname.replace('LOSO_30ch_bilstm_test_', '').replace('.pth', '')
    sd   = torch.load(os.path.join(model_dir, fname), map_location='cpu', weights_only=False)

    if PA_mask is None:
        PA_mask = sd['PA_mask'].numpy()   # (2, 10, 10)  — 0인 위치만 1

    pa_stack = np.stack([sd[f'PA.{i}'].numpy() for i in range(num_layers)])  # (7,2,10,10)
    subject_PAs[subj] = pa_stack

subjects = sorted(subject_PAs.keys())
print(f"로드된 피험자: {len(subjects)}명\n")

# ── 2. 전체 평균 PA (mask 적용, 레이어 평균) ──────────────────────────────
all_pa = np.stack([subject_PAs[s] for s in subjects])  # (S, 7, 2, 10, 10)
mean_pa = all_pa.mean(axis=(0, 1))                      # (2, 10, 10) — 피험자·레이어 평균
mean_pa_masked = mean_pa * PA_mask                      # 새 연결 위치만

# partition 합산 (2, 10, 10) → (10, 10)
mean_pa_flat = mean_pa_masked.sum(axis=0)

# ── 3. 상위 엣지 랭킹 출력 ───────────────────────────────────────────────
print("=== 전체 평균 PA 상위 20 엣지 (새 연결만) ===")
edges = []
for i in range(10):
    for j in range(i + 1, 10):
        val = (mean_pa_masked[:, i, j] + mean_pa_masked[:, j, i]).mean()
        edges.append((i, j, val))
edges.sort(key=lambda x: x[2], reverse=True)

for rank, (i, j, v) in enumerate(edges[:20], 1):
    print(f"  {rank:>2}. 센서{i} ↔ 센서{j}  PA={v:.5f}")

# ── 4. 피험자별 PA 히트맵 ────────────────────────────────────────────────
fig, axes = plt.subplots(2, 5, figsize=(18, 7))
axes = axes.flatten()

for ax, subj in zip(axes, subjects):
    pa = subject_PAs[subj]                   # (7, 2, 10, 10)
    pa_m = (pa * PA_mask).mean(axis=(0, 1))  # (10, 10) 레이어 평균 + mask
    pa_sym = (pa_m + pa_m.T) / 2            # 대칭화
    np.fill_diagonal(pa_sym, 0)

    im = ax.imshow(pa_sym, cmap='hot', vmin=0)
    ax.set_title(subj, fontsize=8)
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    plt.colorbar(im, ax=ax, fraction=0.046)

fig.suptitle('Subject-wise learned PA (new connections only, layer avg)', fontsize=12)
plt.tight_layout()
plt.savefig('results/PA_heatmap_per_subject.png', dpi=120)
print("\n저장: results/PA_heatmap_per_subject.png")

# ── 5. 레이어별 평균 PA 히트맵 ───────────────────────────────────────────
fig2, axes2 = plt.subplots(1, num_layers, figsize=(20, 3))

for layer_idx, ax in enumerate(axes2):
    pa_layer = all_pa[:, layer_idx]          # (S, 2, 10, 10)
    pa_m = (pa_layer * PA_mask).mean(axis=(0, 1))   # (10, 10)
    pa_sym = (pa_m + pa_m.T) / 2
    np.fill_diagonal(pa_sym, 0)

    im = ax.imshow(pa_sym, cmap='hot', vmin=0)
    ax.set_title(f'Layer {layer_idx}', fontsize=9)
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))

fig2.suptitle('Layer-wise learned PA (all subjects avg)', fontsize=11)
plt.tight_layout()
plt.savefig('results/PA_heatmap_per_layer.png', dpi=120)
print("저장: results/PA_heatmap_per_layer.png")

# ── 6. 피험자별 상위 5 엣지 출력 ─────────────────────────────────────────
print("\n=== 피험자별 상위 5 새 연결 ===")
for subj in subjects:
    pa = subject_PAs[subj]
    pa_m = (pa * PA_mask).mean(axis=(0, 1))
    edges_s = []
    for i in range(10):
        for j in range(i + 1, 10):
            val = (pa_m[i, j] + pa_m[j, i]) / 2
            edges_s.append((i, j, val))
    edges_s.sort(key=lambda x: x[2], reverse=True)
    top5 = ', '.join([f"({i}-{j})" for i, j, _ in edges_s[:5]])
    print(f"  {subj}: {top5}")

plt.show()
