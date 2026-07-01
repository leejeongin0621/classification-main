import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

model_dir = 'models/right/fully connected'
num_layers = 7

# 현재 사용 중인 graph.py right 20개 엣지
original_edges = set(tuple(sorted(e)) for e in [
    [6,7],[4,8],[3,4],[8,9],[7,8],
    [4,9],[7,9],[3,6],[4,7],[2,7],
    [6,8],[1,5],[2,3],[4,6],[5,7],
    [6,9],[1,3],[2,4],[3,7],[0,1]
])

# ── 1. 로드 ──────────────────────────────────────────────────────────────
subject_data = {}
for fname in sorted(os.listdir(model_dir)):
    if not fname.endswith('.pth'): continue
    subj = fname.replace('LOSO_30ch_st-gcn_yaw_test_', '').replace('.pth', '')
    sd = torch.load(os.path.join(model_dir, fname), map_location='cpu', weights_only=False)
    A = sd['A'].numpy()  # (2,10,10)
    imp_stack = np.stack([sd[f'edge_importance.{i}'].numpy() for i in range(num_layers)])  # (7,2,10,10)
    subject_data[subj] = {'A': A, 'imp': imp_stack}

subjects = sorted(subject_data.keys())
print(f"로드: {len(subjects)}명\n")

# ── 2. 피험자별 effective weight 계산 ──────────────────────────────────
# effective = A * importance, 레이어 평균, 파티션 합산 → (10,10)
def get_eff(sd):
    A   = sd['A']                          # (2,10,10)
    imp = sd['imp']                        # (7,2,10,10)
    eff = (A * imp).mean(axis=0)           # (2,10,10) 레이어 평균
    eff = eff.sum(axis=0)                  # (10,10) 파티션 합산
    eff_sym = (eff + eff.T) / 2
    np.fill_diagonal(eff_sym, 0)
    return eff_sym

# ── 3. 피험자별 히트맵 ────────────────────────────────────────────────
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
axes = axes.flatten()

for ax, subj in zip(axes, subjects):
    eff = get_eff(subject_data[subj])

    # 마스크: original=1, new=0
    mask_orig = np.zeros((10,10))
    mask_new  = np.zeros((10,10))
    for i in range(10):
        for j in range(i+1, 10):
            if (i,j) in original_edges:
                mask_orig[i,j] = mask_orig[j,i] = 1
            else:
                mask_new[i,j]  = mask_new[j,i]  = 1

    eff_orig = eff * mask_orig
    eff_new  = eff * mask_new

    im = ax.imshow(eff, cmap='hot', vmin=0, vmax=eff.max())

    # original 엣지: 파란 테두리 표시
    for i in range(10):
        for j in range(10):
            if mask_orig[i,j] == 1:
                ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1,
                             fill=False, edgecolor='cyan', linewidth=1.2))

    ax.set_title(subj, fontsize=8)
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    plt.colorbar(im, ax=ax, fraction=0.046)

orig_patch = mpatches.Patch(edgecolor='cyan', facecolor='none', label='original 20 edges')
fig.legend(handles=[orig_patch], loc='lower center', fontsize=10)
fig.suptitle('Fully-connected: effective edge weight (A × importance, layer avg)\ncyan border = original 20 edges', fontsize=12)
plt.tight_layout()
plt.savefig('results/FC_importance_heatmap.png', dpi=120)
print("저장: results/FC_importance_heatmap.png")

# ── 4. original 20 vs 나머지 25: 전체 평균 비교 ──────────────────────
orig_vals, new_vals = [], []
for subj in subjects:
    eff = get_eff(subject_data[subj])
    for i in range(10):
        for j in range(i+1, 10):
            v = eff[i,j]
            if (i,j) in original_edges:
                orig_vals.append(v)
            else:
                new_vals.append(v)

print("=== 전체 평균 비교 ===")
print(f"original 20: mean={np.mean(orig_vals):.5f}  max={np.max(orig_vals):.5f}  min={np.min(orig_vals):.5f}")
print(f"new      25: mean={np.mean(new_vals):.5f}  max={np.max(new_vals):.5f}  min={np.min(new_vals):.5f}")
print(f"ratio (original / new): {np.mean(orig_vals)/np.mean(new_vals):.3f}x")

# ── 5. 전체 평균 엣지 랭킹 출력 ──────────────────────────────────────
all_eff = np.mean([get_eff(subject_data[s]) for s in subjects], axis=0)
edges_ranked = []
for i in range(10):
    for j in range(i+1, 10):
        tag = 'orig' if (i,j) in original_edges else 'new'
        edges_ranked.append((i, j, all_eff[i,j], tag))
edges_ranked.sort(key=lambda x: x[2], reverse=True)

print("\n=== 전체 피험자 평균 엣지 랭킹 (상위 20) ===")
for rank, (i, j, v, tag) in enumerate(edges_ranked[:20], 1):
    marker = '★' if tag == 'orig' else '  '
    print(f"  {rank:>2}. {marker} 센서{i}↔센서{j}  {v:.5f}  [{tag}]")

# ── 6. 박스플롯 비교 ──────────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(6, 5))
ax2.boxplot([orig_vals, new_vals], labels=['original 20', 'new 25'], patch_artist=True,
            boxprops=dict(facecolor='skyblue'), medianprops=dict(color='red', linewidth=2))
ax2.set_ylabel('effective edge weight (A × importance)')
ax2.set_title('original 20 edges vs new 25 edges\n(fully connected model)')
ax2.grid(axis='y', alpha=0.4)
plt.tight_layout()
plt.savefig('results/FC_importance_boxplot.png', dpi=120)
print("\n저장: results/FC_importance_boxplot.png")

plt.show()
