"""
Edge Class Separability — F-statistic (z-transform) & 2D Fisher Discriminant Ratio
====================================================================================

[방법 1] F-statistic with Fisher z-transform  ← 권장 (edge-native 지표)
---------------------------------------------------------------------------
두 센서 사이의 "동기화 정도(temporal correlation)"가 단어마다 얼마나 달라지는지를 측정.
Pearson r은 정의상 두 변수가 있어야 존재하므로, 단일 노드 속성으로 분해 불가 → 진짜 edge 지표.

Step 1. 신호 추출
    raw IMU (N, 200, 30) → L2 norm → (N, 200, 10)  각 센서의 크기 신호

Step 2. 샘플별 Pearson correlation 계산
    각 샘플(단어 1회)에 대해 센서 i와 j의 시계열(200 timestep) 간 Pearson r 계산
    → scalar 1개 per sample.  결과: edge_corr (N, 45)

Step 3. Fisher z-transform 적용
    r은 -1~1로 bounded되어 분산이 r 값에 따라 달라지는 비등질적 분포 → F-test 가정 위반
    z = atanh(r) 로 변환하면 근사 정규분포, 분산 안정화됨 (Fisher 1915)

Step 4. F-statistic 계산
    각 edge의 z값에 대해:
        between = Σ_c n_c*(μ_c - μ)²  / (C-1)   클래스 간 분산
        within  = Σ_c Σ_x (x - μ_c)² / (N-C)   클래스 내 분산
        F = between / within
    F가 클수록 → 두 센서의 동기화 패턴이 단어마다 크게 다름 → 분류에 유용한 edge

[방법 2] 2D Fisher Discriminant Ratio  ← 참고용 (node importance 혼입 위험)
---------------------------------------------------------------------------
두 센서의 amplitude feature([mean, std, max])를 6D vector로 묶고
between/within scatter matrix 비율로 class separability 측정.

주의: 센서 i나 j 하나만으로도 단어 구별이 잘 된다면,
      그 센서가 들어간 모든 쌍이 자동으로 높은 점수를 받아 hub 구조 발생.
      → "edge 관계"가 아닌 "node 중요도"가 반영될 수 있음.

Step 1~2. 동일
Step 3. 센서 쌍 (i,j)의 6D feature: concat([mean_i,std_i,max_i, mean_j,std_j,max_j])
Step 4. S_B = Σ_c n_c*(μ_c-μ)(μ_c-μ)^T / (C-1)   [6×6 between-class scatter]
        S_W = Σ_c Σ_x (x-μ_c)(x-μ_c)^T / (N-C)   [6×6 within-class scatter]
Step 5. J = trace(S_W^{-1} S_B) / 6
"""

import numpy as np
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from utils.Dataloader import LoadIMU_EPO_simple


# ── 설정 ─────────────────────────────────────────────────────────────────────
subject_list = [
    '250805_KDY','250731_LGE','250806_LJI','250812_WDY','250814_JCM',
    '250818_ICY','250819_PYH','250820_LTG','250822_JSH','250825_JDB'
]
load_path = os.path.join(os.path.dirname(__file__), 'data_10ch')
save_path = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(save_path, exist_ok=True)


# ── Step 1–2: 데이터 로드 & feature 추출 ─────────────────────────────────────
print('데이터 로드 중...')
IMU_data, IMU_label = LoadIMU_EPO_simple(load_path, subject_list)

X_raw = IMU_data.reshape(-1, 200, 10, 3)   # (N, 200, 10, 3)
y     = IMU_label.reshape(-1)               # (N,)
norm  = np.linalg.norm(X_raw, axis=-1)      # (N, 200, 10)  L2 norm per sensor
N     = norm.shape[0]

feat_mean = norm.mean(axis=1)  # (N, 10)
feat_std  = norm.std(axis=1)   # (N, 10)
feat_max  = norm.max(axis=1)   # (N, 10)
sensor_feat = np.stack([feat_mean, feat_std, feat_max], axis=2)  # (N, 10, 3)

classes = np.unique(y)
n_class = len(classes)
print(f'  N={N}, classes={n_class}')


# ── 비교용 edge set ───────────────────────────────────────────────────────────
fc_edges = set(map(tuple, [sorted(e) for e in [
    (0,6),(1,6),(5,6),(3,6),(6,9),(2,6),(6,8),(0,1),(0,5),(4,6),
    (1,5),(0,2),(6,7),(0,8),(0,3),(0,4),(1,8),(0,9),(5,8),(1,2)
]]))
pr_edges = set(map(tuple, [sorted(e) for e in [
    (6,7),(4,8),(3,4),(8,9),(7,8),(4,9),(7,9),(3,6),(4,7),(2,7),
    (6,8),(1,5),(2,3),(4,6),(5,7),(6,9),(1,3),(2,4),(3,7),(0,1)
]]))


# ── [방법 1] F-statistic with Fisher z-transform ─────────────────────────────
print('\n[방법 1] F-statistic (Fisher z-transform) 계산 중...')

edges = [(i, j) for i in range(10) for j in range(i + 1, 10)]

# Step 2: 샘플별 Pearson r
edge_corr = np.zeros((N, len(edges)))
for idx, (i, j) in enumerate(edges):
    si = norm[:, :, i]; sj = norm[:, :, j]
    si_c = si - si.mean(axis=1, keepdims=True) # 각 샘플에서 시간 평균을 뺌 
    sj_c = sj - sj.mean(axis=1, keepdims=True)
    num  = (si_c * sj_c).sum(axis=1)
    den  = np.sqrt((si_c**2).sum(axis=1) * (sj_c**2).sum(axis=1)) + 1e-8
    r    = np.clip(num / den, -0.9999, 0.9999)
    # Step 3: Fisher z-transform  z = atanh(r)
    edge_corr[:, idx] = np.arctanh(r)

# Step 4: F-statistic
grand_mean_z = edge_corr.mean(axis=0)
between_z = np.zeros(len(edges))
within_z  = np.zeros(len(edges))
for c in classes:
    mask = (y == c); nc = mask.sum()
    cm = edge_corr[mask].mean(axis=0)
    between_z += nc * (cm - grand_mean_z) ** 2
    within_z  += ((edge_corr[mask] - cm) ** 2).sum(axis=0)
between_z /= (n_class - 1)
within_z  /= (N - n_class)
F_z = between_z / (within_z + 1e-10)

ranked_fz = sorted(range(len(edges)), key=lambda k: -F_z[k])

print('\n=== F-statistic (z-transform) top-20 ===')
top20_fz = []
for rank, idx in enumerate(ranked_fz[:20], 1):
    i, j = edges[idx]
    e  = tuple(sorted((i, j)))
    fc = 'FC' if e in fc_edges else '  '
    pr = 'PR' if e in pr_edges else '  '
    print(f'  {rank:2d}. ({i},{j})  F={F_z[idx]:.2f}  [{fc}][{pr}]')
    top20_fz.append(e)
top20_fz_set = set(top20_fz)
print(f'FC overlap: {len(top20_fz_set & fc_edges)}/20  |  Pearson overlap: {len(top20_fz_set & pr_edges)}/20')


# ── [방법 2] 2D Fisher ratio 계산 함수 ───────────────────────────────────────
print('\n[방법 2] 2D Fisher (amplitude) 계산 중...')
def fisher_2d(X2d: np.ndarray, y: np.ndarray):
    """
    X2d : (N, d)  — 두 센서의 feature를 concat한 행렬 (d=6)
    y   : (N,)    — 클래스 레이블

    Returns
    -------
    J       : trace(S_W^{-1} S_B) / d  — Fisher criterion (scalar)
    max_eig : S_W^{-1} S_B 의 최대 고유값
    S_B     : between-class scatter matrix (d×d)
    S_W     : within-class scatter matrix  (d×d)
    """
    N_local, d = X2d.shape
    grand_mean = X2d.mean(axis=0)          # (d,)

    S_B = np.zeros((d, d))
    S_W = np.zeros((d, d))

    for c in classes:
        mask = (y == c)
        nc   = mask.sum()
        Xc   = X2d[mask]                   # (nc, d)
        mu_c = Xc.mean(axis=0)             # (d,)

        diff = (mu_c - grand_mean).reshape(-1, 1)
        S_B += nc * (diff @ diff.T)

        Xc_centered = Xc - mu_c
        S_W += Xc_centered.T @ Xc_centered

    S_B /= (n_class - 1)
    S_W /= (N_local - n_class)

    try:
        SW_inv  = np.linalg.inv(S_W + np.eye(d) * 1e-8)
        M       = SW_inv @ S_B
        J       = np.trace(M) / d
        max_eig = np.max(np.linalg.eigvals(M).real)
    except np.linalg.LinAlgError:
        J, max_eig = 0.0, 0.0

    return J, max_eig, S_B, S_W


# ── [방법 2] 45개 edge 전체 계산 ─────────────────────────────────────────────
results = []
for i, j in edges:
    X2d = np.concatenate([sensor_feat[:, i, :], sensor_feat[:, j, :]], axis=1)
    J, max_eig, S_B, S_W = fisher_2d(X2d, y)
    results.append({'i': i, 'j': j, 'J': J, 'max_eig': max_eig})

results.sort(key=lambda x: -x['J'])


# ── 결과 출력 ─────────────────────────────────────────────────────────────────
print('\n=== 2D Fisher top-20 edges ===')
top20 = []
for rank, r in enumerate(results[:20], 1):
    i, j = r['i'], r['j']
    e  = tuple(sorted((i, j)))
    fc = 'FC' if e in fc_edges else '  '
    pr = 'PR' if e in pr_edges else '  '
    print(f"  {rank:2d}. ({i},{j})  J={r['J']:.3f}  [{fc}][{pr}]")
    top20.append(e)

top20_set = set(top20)
print(f'\nFC overlap:      {len(top20_set & fc_edges)}/20')
print(f'Pearson overlap: {len(top20_set & pr_edges)}/20')


# ── Excel 저장 ────────────────────────────────────────────────────────────────
from collections import Counter

# F-statistic sheet
fz_rank = {edges[idx]: r for r, idx in enumerate(ranked_fz, 1)}
rows_fz = []
for rank, idx in enumerate(ranked_fz, 1):
    i, j = edges[idx]
    e = tuple(sorted((i, j)))
    rows_fz.append({
        'rank':            rank,
        'edge':            f'({i},{j})',
        'node_i':          i,
        'node_j':          j,
        'F_statistic':     round(F_z[idx], 4),
        'in_FC_graph':     'O' if e in fc_edges else '',
        'in_Pearson_graph':'O' if e in pr_edges else '',
        'top20':           'O' if rank <= 20 else '',
    })
df_fz = pd.DataFrame(rows_fz)

# 2D Fisher sheet
rows_2d = []
for rank, r in enumerate(results, 1):
    i, j = r['i'], r['j']
    e = tuple(sorted((i, j)))
    rows_2d.append({
        'rank':            rank,
        'edge':            f'({i},{j})',
        'node_i':          i,
        'node_j':          j,
        'J_trace':         round(r['J'],       4),
        'max_eigenvalue':  round(r['max_eig'], 4),
        'in_FC_graph':     'O' if e in fc_edges else '',
        'in_Pearson_graph':'O' if e in pr_edges else '',
        'top20':           'O' if rank <= 20 else '',
    })
df_2d = pd.DataFrame(rows_2d)

# node degree sheet
deg_fz = Counter(); deg_2d = Counter()
for e in top20_fz_set: deg_fz[e[0]] += 1; deg_fz[e[1]] += 1
for e in top20_set:    deg_2d[e[0]] += 1; deg_2d[e[1]] += 1
df_deg = pd.DataFrame([{
    'node':               n,
    'F_stat_degree':      deg_fz[n],
    '2D_Fisher_degree':   deg_2d[n],
    'FC_degree':          sum(1 for a,b in fc_edges if a==n or b==n),
    'Pearson_degree':     sum(1 for a,b in pr_edges if a==n or b==n),
} for n in range(10)])

save_file = os.path.join(save_path, 'edge_separability_ranking.xlsx')
with pd.ExcelWriter(save_file) as writer:
    df_fz.to_excel(writer, sheet_name='F_stat_ztransform',    index=False)
    df_2d.to_excel(writer, sheet_name='2D_Fisher_amplitude',  index=False)
    df_deg.to_excel(writer, sheet_name='node_degree',         index=False)

print(f'\n저장: {save_file}')


# ── LOSO fold별 공통 엣지 분석 ────────────────────────────────────────────────
print('\n' + '='*60)
print('LOSO F-statistic 공통 엣지 분석')
print('='*60)

left_subjects = [
    '250804_KTS','250805_SMC','250811_LPR','250811_JHS','250812_HHJ',
    '250813_YMS','250814_CYJ','250819_CYK','250822_KTH','250827_HJH'
]

def compute_fstat_ranked_from_norm(norm_all, y_all, exclude_idx):
    # norm_all: (n_subj, n_samples_per_subj, 200, 10)
    idxs   = np.arange(norm_all.shape[0]) != exclude_idx
    norm   = norm_all[idxs].reshape(-1, 200, 10)
    y_l    = y_all[idxs].reshape(-1)
    N_l    = norm.shape[0]
    classes_l = np.unique(y_l)

    ec = np.zeros((N_l, 45))
    for idx, (i, j) in enumerate(edges):
        si = norm[:, :, i]; sj = norm[:, :, j]
        si_c = si - si.mean(1, keepdims=True)
        sj_c = sj - sj.mean(1, keepdims=True)
        r = np.clip((si_c * sj_c).sum(1) /
                    (np.sqrt((si_c**2).sum(1) * (sj_c**2).sum(1)) + 1e-8),
                    -0.9999, 0.9999)
        ec[:, idx] = np.arctanh(r)

    gm = ec.mean(0)
    bw = np.zeros(45); wt = np.zeros(45)
    for c in classes_l:
        m = (y_l == c); cm = ec[m].mean(0)
        bw += m.sum() * (cm - gm) ** 2
        wt += ((ec[m] - cm) ** 2).sum(0)
    F = (bw / (len(classes_l) - 1)) / (wt / (N_l - len(classes_l)) + 1e-10)
    return [edges[i] for i in np.argsort(-F)]

# ── RIGHT / LEFT 전체 피험자 top-20 엣지 → Excel ─────────────────────────────
print('\n' + '='*60)
print('RIGHT / LEFT 전체 피험자 F-statistic top-20')
print('='*60)

group_dfs = {}
for group_name, subj_list in [('RIGHT', subject_list), ('LEFT', left_subjects)]:
    print(f'\n--- {group_name} (데이터 로드 중...) ---', flush=True)
    data_all, label_all = LoadIMU_EPO_simple(load_path, subj_list)
    n_subj   = len(subj_list)
    norm_all = np.linalg.norm(data_all.reshape(n_subj, -1, 200, 10, 3), axis=-1)
    y_all    = label_all.reshape(n_subj, -1)

    # fold별 top-20 계산
    fold_top20 = {}
    for i, test_subj in enumerate(subj_list):
        ranked = compute_fstat_ranked_from_norm(norm_all, y_all, i)
        fold_top20[test_subj] = [f'({a},{b})' for a, b in ranked[:20]]
        print(f'  fold {i+1:2d} ({test_subj}): {fold_top20[test_subj]}')

    # 피험자별 top-20을 열로 저장
    df = pd.DataFrame(
        {'rank': list(range(1, 21))} |
        {subj: fold_top20[subj] for subj in subj_list}
    )
    group_dfs[group_name] = df

rl_save = os.path.join(save_path, 'edge_fstat_right_left_top20.xlsx')
with pd.ExcelWriter(rl_save) as writer:
    group_dfs['RIGHT'].to_excel(writer, sheet_name='RIGHT', index=False)
    group_dfs['LEFT'].to_excel(writer,  sheet_name='LEFT',  index=False)
print(f'\n저장: {rl_save}')


# ── LOSO fold별 공통 엣지 분석 ────────────────────────────────────────────────
print('\n' + '='*60)
print('LOSO F-statistic 공통 엣지 분석')
print('='*60)

for group_name, subj_list in [('RIGHT', subject_list), ('LEFT', left_subjects)]:
    print(f'\n--- {group_name} (데이터 로드 중...) ---', flush=True)
    data_all, label_all = LoadIMU_EPO_simple(load_path, subj_list)
    # (n_subj, n_session, n_class, 200, 30)
    n_subj = len(subj_list)
    norm_all = np.linalg.norm(
        data_all.reshape(n_subj, -1, 200, 10, 3), axis=-1)  # (n_subj, S*C, 200, 10)
    y_all = label_all.reshape(n_subj, -1)                   # (n_subj, S*C)

    for top_k in [13, 15, 17, 20]:
        fold_sets = []
        for i in range(n_subj):
            ranked = compute_fstat_ranked_from_norm(norm_all, y_all, i)
            fold_sets.append(set(tuple(sorted(e)) for e in ranked[:top_k]))
        common = fold_sets[0]
        for s in fold_sets[1:]:
            common &= s
        print(f'  top-{top_k}: 공통 {len(common):2d}개  {sorted(common)}')
