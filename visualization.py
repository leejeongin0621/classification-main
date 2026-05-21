import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import imageio.v2 as imageio
from utils.Dataloader import LoadIMU_EPO_simple
from correlation import make_word_corr_dict
 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
subject_list = [
    '250805_KDY','250731_LGE','250806_LJI','250812_WDY','250814_JCM',
    '250818_ICY','250819_PYH','250820_LTG','250822_JSH','250825_JDB'
]
 
# 센서 위치 (픽셀 좌표 기준, 얼굴 이미지 위에 매핑)
sensor_pos = {
    0: (203, 72),
    1: (160, 100),
    2: (130, 174),
    3: (198, 149),
    4: (174, 215),
    5: (315, 101),
    6: (245, 115),
    7: (285, 173),
    8: (216, 201),
    9: (278, 271),
}
 
this_path = os.getcwd()
load_path = os.path.join(this_path, 'data_10ch')
model_path = os.path.join(this_path, 'models')
save_root = os.path.join(this_path, 'visualization_results')
os.makedirs(save_root, exist_ok=True)
 
TARGET_SUBJECT = 0
TARGET_WORD    = 3
SAMPLE_INDEX_WITHIN_WORD = 0
 
BASE_MODEL_PATH = os.path.join(model_path, 'Model_53.pt')
BEST_MODEL_PATH = os.path.join(model_path, 'best_model_250822_JSH_fold_0.pth')

SAVE_DIR = os.path.join(
    save_root,
    f"subject{TARGET_SUBJECT}_word{TARGET_WORD}_sample{SAMPLE_INDEX_WITHIN_WORD}"
)
 
# ── 데이터 로드 ──────────────────────────────────────────────────
IMU_data_patient, IMU_label_patient = LoadIMU_EPO_simple(load_path, subject_list)
X_data = IMU_data_patient[TARGET_SUBJECT].reshape(-1, 200, 30)
y_data = IMU_label_patient[TARGET_SUBJECT].reshape(-1)
A_dict = make_word_corr_dict(X_data, y_data)
 
# ── 모델 로드 ────────────────────────────────────────────────────
model = torch.load(BASE_MODEL_PATH, map_location=device, weights_only=False)
model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
model.to(device)
model.eval()
 
 
# ── 유틸 함수 ────────────────────────────────────────────────────
def get_adjacency_edges(A_word, threshold=0.25):
    """adjacency matrix → edge list"""
    if A_word.ndim == 3:
        A_word = A_word.sum(axis=0)
    A_word = (A_word.astype(np.float32) >= threshold).astype(np.float32)
    np.fill_diagonal(A_word, 1.0)
    edges = [
        (i, j)
        for i in range(A_word.shape[0])
        for j in range(i + 1, A_word.shape[0])
        if A_word[i, j] > 0
    ]
    return edges
 
 
def extract_intensity(model, x_tensor):
    """
    마지막 레이어 feature → 시간별 노드 활성화 강도 (T, V)
    extract_feature는 A_corr 없이 호출 (모델 내부 self.A 사용)
    """
    model.eval()
    with torch.no_grad():
        output, feature = model.extract_feature(x_tensor)
    # feature: (N, C, T, V, M) 또는 (N, M, C, T, V) — 코드에 따라 다름
    # extract_feature 반환: feature = x.view(N,M,c,t,v).permute(0,2,3,4,1) → (N,C,T,V,M)
    feat = feature[0]          # (C, T, V, M) or (C, T, V)
    if feat.dim() == 4:
        feat = feat[..., 0]    # M 차원 제거 → (C, T, V)
    # L2 norm across channel → (T, V)
    intensity = torch.sqrt((feat ** 2).sum(dim=0))
    return intensity.cpu().numpy()   # (T, V)
 
 
def smooth_intensity(intensity, window=5):
    """temporal smoothing으로 flickering 줄이기"""
    T, V = intensity.shape
    smoothed = np.zeros_like(intensity)
    for t in range(T):
        t0 = max(0, t - window // 2)
        t1 = min(T, t + window // 2 + 1)
        smoothed[t] = intensity[t0:t1].mean(axis=0)
    return smoothed
 
 
def draw_frame(node_scores, edges, sensor_pos, save_path,
               score_min, score_max, t, T, word_idx):
    """
    얼굴 위 센서 위치에 노드 활성화를 heatmap으로 그리기
    - 노드 크기 + 색상(냉→온) 으로 활성화 표현
    - 엣지는 활성화 평균으로 투명도 조절
    """
    fig, ax = plt.subplots(figsize=(5, 6))
    ax.set_facecolor('#0a0a0a')
    fig.patch.set_facecolor('#0a0a0a')
 
    norm = mcolors.Normalize(vmin=score_min, vmax=score_max)
    cmap = plt.cm.plasma   # 보라→빨강→노랑: 활성화 잘 보임
 
    # ── 엣지 그리기 ──
    for i, j in edges:
        x1, y1 = sensor_pos[i]
        x2, y2 = sensor_pos[j]
        avg_act = (node_scores[i] + node_scores[j]) / 2.0
        alpha = 0.15 + 0.5 * norm(avg_act)
        ax.plot([x1, x2], [y1, y2],
                color='white', linewidth=1.2, alpha=float(alpha))
 
    # ── 노드 그리기 ──
    for idx, (x, y) in sensor_pos.items():
        s = norm(node_scores[idx])
        size   = 80 + s * 1400       # 활성화 강할수록 크게
        color  = cmap(s)
        alpha  = 0.4 + 0.6 * s
 
        # glow 효과: 큰 반투명 원 뒤에 작은 선명 원
        ax.scatter(x, y, s=size * 2.5, c=[color], alpha=float(alpha) * 0.3,
                   edgecolors='none')
        ax.scatter(x, y, s=size, c=[color], alpha=float(alpha),
                   edgecolors='white', linewidths=0.5)
 
        ax.text(x + 5, y - 5, str(idx),
                color='white', fontsize=7, alpha=0.8)
 
    # ── 타임바 (하단) ──
    bar_x = np.linspace(0.05, 0.95, T)
    ax.axhline(y=310, xmin=0.05, xmax=0.95,
               color='white', linewidth=1, alpha=0.2,
               transform=ax.transData, clip_on=False)
    progress = t / max(T - 1, 1)
    ax.annotate('', xy=(80 + progress * 280, 312), xytext=(80, 312),
                arrowprops=dict(arrowstyle='-', color='cyan', lw=2))
    ax.scatter(80 + progress * 280, 312, s=60, c='cyan', zorder=5)
 
    # ── 타이틀 ──
    ax.set_title(f'Word {word_idx}  |  t = {t:3d} / {T}',
                 color='white', fontsize=11, pad=6)
 
    # ── colorbar ──
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white', fontsize=7)
    cbar.set_label('Activation', color='white', fontsize=8)
 
    ax.set_xlim(80, 360)
    ax.set_ylim(320, 40)   # y 반전 (위쪽이 작은 y)
    ax.axis('off')
 
    fig.savefig(save_path, dpi=130, facecolor='#0a0a0a',
                bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
 
 
# ── 메인 시각화 함수 ─────────────────────────────────────────────
def visualize_word_sample(model, X_data, y_data, A_dict,
                          target_word, sample_index,
                          sensor_pos, save_dir,
                          gif_fps=20, smooth_window=5, device='cuda'):
    os.makedirs(save_dir, exist_ok=True)
 
    # 샘플 선택
    candidate_idx = np.where(y_data == target_word)[0]
    if len(candidate_idx) == 0:
        raise ValueError(f'단어 {target_word} 샘플 없음')
    if sample_index >= len(candidate_idx):
        raise ValueError(f'sample_index={sample_index} 범위 초과 (총 {len(candidate_idx)}개)')
 
    selected_idx = candidate_idx[sample_index]
    print(f'[INFO] word={target_word}, global_idx={selected_idx}')
 
    # 엣지 구성
    A_word = A_dict[target_word]
    edges  = get_adjacency_edges(A_word)
 
    # 입력 준비
    x_np = X_data[selected_idx]   # (200, 30)
    x    = torch.tensor(x_np, dtype=torch.float32).unsqueeze(0).to(device)
 
    # feature → intensity (T, V)
    intensity = extract_intensity(model, x)
    intensity = smooth_intensity(intensity, window=smooth_window)
    T, V = intensity.shape
    print(f'[INFO] intensity shape: {intensity.shape}')
 
    g_min, g_max = intensity.min(), intensity.max()
 
    # 프레임 생성
    image_paths = []
    for t in range(T):
        path = os.path.join(save_dir, f'frame_{t:03d}.png')
        draw_frame(
            node_scores=intensity[t],
            edges=edges,
            sensor_pos=sensor_pos,
            save_path=path,
            score_min=g_min,
            score_max=g_max,
            t=t, T=T,
            word_idx=target_word,
        )
        image_paths.append(path)
        if t % 50 == 0:
            print(f'  frame {t}/{T}')
 
    # GIF 저장
    gif_path = os.path.join(save_dir, f'word_{target_word}.gif')
    duration  = 1.0 / gif_fps   # seconds per frame
    images    = [imageio.imread(p) for p in image_paths]
    imageio.mimsave(gif_path, images, duration=duration, loop=0)
    print(f'[INFO] GIF 저장: {gif_path}')
 
    # 요약 플롯 (시간 평균 활성화)
    mean_act = intensity.mean(axis=0)   # (V,)
    fig, ax = plt.subplots(figsize=(5, 6))
    ax.set_facecolor('#0a0a0a'); fig.patch.set_facecolor('#0a0a0a')
    norm = mcolors.Normalize(vmin=mean_act.min(), vmax=mean_act.max())
    cmap = plt.cm.plasma
    for i, j in edges:
        x1,y1 = sensor_pos[i]; x2,y2 = sensor_pos[j]
        ax.plot([x1,x2],[y1,y2],color='white',lw=1,alpha=0.3)
    for idx,(x,y) in sensor_pos.items():
        s = norm(mean_act[idx])
        ax.scatter(x,y,s=150+s*1200,c=[cmap(s)],alpha=0.85,
                   edgecolors='white',linewidths=0.5)
        ax.text(x+5,y-5,str(idx),color='white',fontsize=8)
    ax.set_xlim(80,360); ax.set_ylim(320,40); ax.axis('off')
    ax.set_title(f'Word {target_word} — Mean Activation',color='white',fontsize=12)
    sm = plt.cm.ScalarMappable(cmap=cmap,norm=norm); sm.set_array([])
    cbar = fig.colorbar(sm,ax=ax,fraction=0.03,pad=0.02)
    cbar.set_label('Mean Activation',color='white',fontsize=8)
    plt.setp(cbar.ax.yaxis.get_ticklabels(),color='white',fontsize=7)
    fig.savefig(os.path.join(save_dir,'mean_activation.png'),
                dpi=150,facecolor='#0a0a0a',bbox_inches='tight')
    plt.close(fig)
    print(f'[INFO] 평균 활성화 이미지 저장 완료')
 
 
# ── 실행 ─────────────────────────────────────────────────────────
visualize_word_sample(
    model=model,
    X_data=X_data,
    y_data=y_data,
    A_dict=A_dict,
    target_word=TARGET_WORD,
    sample_index=SAMPLE_INDEX_WITHIN_WORD,
    sensor_pos=sensor_pos,
    save_dir=SAVE_DIR,
    gif_fps=20,
    smooth_window=5,
    device=device,
)