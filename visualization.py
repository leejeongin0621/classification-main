import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from utils.Dataloader import LoadIMU_EPO_simple
from correlation import make_word_corr_dict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

subject_list = [
    '250805_KDY','250731_LGE','250806_LJI','250812_WDY','250814_JCM','250818_ICY','250819_PYH','250820_LTG','250822_JSH','250825_JDB']

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
TARGET_WORD = 3
SAMPLE_INDEX_WITHIN_WORD = 0

BASE_MODEL_PATH = os.path.join(model_path, 'Model_32.pt')
BEST_MODEL_PATH = os.path.join(model_path, 'best_model_250822_JSH_fold_0.pth')
SAVE_DIR = os.path.join(
    save_root,
    f"subject{TARGET_SUBJECT}_word{TARGET_WORD}_sample{SAMPLE_INDEX_WITHIN_WORD}"
)

IMU_data_patient, IMU_label_patient = LoadIMU_EPO_simple(load_path, subject_list)

X_data = IMU_data_patient[TARGET_SUBJECT].reshape(-1, 200, 30)
y_data = IMU_label_patient[TARGET_SUBJECT].reshape(-1)

A_dict = make_word_corr_dict(X_data, y_data)

model = torch.load(BASE_MODEL_PATH, map_location=device, weights_only=False)
model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
model.to(device)
model.eval()


def get_word_adjacency(word_idx, A_dict, threshold=0.25):
    A_word = A_dict[word_idx]

    if A_word.ndim == 3:
        A_word = A_word.sum(axis=0)

    A_word = A_word.astype(np.float32)
    A_word = (A_word >= threshold).astype(np.float32)
    np.fill_diagonal(A_word, 1.0)
    return A_word


def extract_last_feature(model, x, A_corr=None):
    model.eval()
    with torch.no_grad():
        output, feature = model.extract_feature(x, A_corr=A_corr)
    return output, feature


def compute_intensity(feature):
    # feature: (N, C, T, V, M)
    feature = feature[0, :, :, :, 0]   # (C, T, V)
    intensity = torch.sqrt((feature ** 2).sum(dim=0))  # (T, V)
    return intensity.cpu().numpy()


def adjacency_to_edges(A_word):
    edges = []
    V = A_word.shape[0]
    for i in range(V):
        for j in range(i + 1, V):
            if A_word[i, j] > 0 or A_word[j, i] > 0:
                edges.append((i, j))
    return edges


def draw_frame(node_scores, edges, sensor_pos, save_path,
               score_min=None, score_max=None):
    plt.figure(figsize=(6, 6))
    ax = plt.gca()
    ax.set_facecolor('black')
    plt.gcf().patch.set_facecolor('black')

    for i, j in edges:
        x1, y1 = sensor_pos[i]
        x2, y2 = sensor_pos[j]
        plt.plot([x1, x2], [y1, y2], color='white', linewidth=1.5, alpha=0.5)

    scores = node_scores.copy()

    if score_min is None:
        score_min = scores.min()
    if score_max is None:
        score_max = scores.max()

    denom = (score_max - score_min) + 1e-8
    norm_scores = (scores - score_min) / denom

    for i, (x, y) in sensor_pos.items():
        s = norm_scores[i]
        size = 100 + s * 1200
        alpha = 0.35 + 0.65 * s

        plt.scatter(x, y, s=size, c='white', alpha=alpha, edgecolors='none')
        plt.text(x + 4, y - 4, f'{i}', color='red', fontsize=8)

    plt.xlim(80, 360)
    plt.ylim(320, 40)
    # plt.gca().invert_yaxis()  # Y축 반전 제거 - 실제 센서 위치에 맞게 표시
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, facecolor='black',
                bbox_inches='tight', pad_inches=0)
    plt.close()


def visualize_one_word_sample(model, X_data, y_data, A_dict,
                              target_word, sample_index_within_word,
                              sensor_pos,
                              save_dir, make_gif=True, gif_name='result.gif',
                              device='cuda'):
    os.makedirs(save_dir, exist_ok=True)

    candidate_idx = np.where(y_data == target_word)[0]

    if len(candidate_idx) == 0:
        raise ValueError(f'단어 {target_word} 에 해당하는 샘플이 없음')

    if sample_index_within_word >= len(candidate_idx):
        raise ValueError(
            f'단어 {target_word} 샘플 수는 {len(candidate_idx)}개인데 '
            f'SAMPLE_INDEX_WITHIN_WORD={sample_index_within_word} 는 범위를 넘음'
        )

    selected_idx = candidate_idx[sample_index_within_word]
    print(f'[INFO] target_word={target_word}, selected_idx={selected_idx}')

    A_word = get_word_adjacency(target_word, A_dict)
    edges = adjacency_to_edges(A_word)

    x = X_data[selected_idx]   # (200, 30)
    x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)  # (1,200,30)

    _, feature = extract_last_feature(model, x, A_corr=A_word)

    intensity = compute_intensity(feature)
    T, V = intensity.shape

    global_min = intensity.min()
    global_max = intensity.max()

    image_paths = []
    for t in range(T):
        frame_path = os.path.join(save_dir, f'frame_{t:03d}.png')
        draw_frame(
            node_scores=intensity[t],
            edges=edges,
            sensor_pos=sensor_pos,
            save_path=frame_path,
            score_min=global_min,
            score_max=global_max
        )
        image_paths.append(frame_path)

    print(f'[INFO] {T}개 프레임 저장 완료: {save_dir}')

    if make_gif:
        gif_path = os.path.join(save_dir, gif_name)
        images = [imageio.imread(p) for p in image_paths]
        imageio.mimsave(gif_path, images, duration=0.06)
        print(f'[INFO] GIF 저장 완료: {gif_path}')


# ===== 실제 실행 =====
visualize_one_word_sample(
    model=model,
    X_data=X_data,
    y_data=y_data,
    A_dict=A_dict,
    target_word=TARGET_WORD,
    sample_index_within_word=SAMPLE_INDEX_WITHIN_WORD,
    sensor_pos=sensor_pos,
    save_dir=SAVE_DIR,
    make_gif=True,
    gif_name=f"word_{TARGET_WORD}.gif",
    device=device
)