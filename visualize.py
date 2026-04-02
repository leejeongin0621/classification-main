# visualize_word_importance.py

import numpy as np
import matplotlib.pyplot as plt
import os

# 설정
save_path = 'image'
bg_image_path = '센서이미지.png'  

subject_list = ['250805_KDY','250731_LGE','250806_LJI','250812_WDY','250814_JCM','250818_ICY','250819_PYH','250820_LTG','250822_JSH','250825_JDB']

num_class = 100
num_node = 10

sensor_pos = {
    0: (250, 80),   # N0 코아래
    1: (170, 130),  # N1 왼뺨위
    2: (120, 210),  # N2 왼뺨아래
    3: (220, 200),  # N3 턱중앙위
    4: (150, 280),  # N4 왼턱아래
    5: (320, 130),  # N5 오른뺨위
    6: (270, 200),  # N6 턱중앙
    7: (330, 220),  # N7 오른턱위
    8: (250, 330),  # N8 목중앙
    9: (350, 360),  # N9 오른목
}

# 전체 피험자 평균
all_subject_importance = np.zeros((num_class, num_node))
subject_count = 0

for s_name in subject_list:
    word_importance_total = np.zeros((num_class, num_node))
    fold_count = 0
    
    for fold in range(5):
        path = os.path.join(save_path, f"word_importance_{s_name}_fold{fold}.npy")
        if os.path.exists(path):
            word_importance_total += np.load(path)
            fold_count += 1
    
    if fold_count > 0:
        all_subject_importance += word_importance_total / fold_count
        subject_count += 1

# 피험자 평균
all_subject_importance /= subject_count  # (100, 10)

# 저장 폴더
vis_path = os.path.join(save_path, 'word_vis_avg')
os.makedirs(vis_path, exist_ok=True)

img = plt.imread(bg_image_path)

# 단어별 시각화
for word_idx in range(num_class):
    imp = all_subject_importance[word_idx]  # (10,)
    imp_norm = (imp - imp.min()) / (imp.max() - imp.min() + 1e-8)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img)

    for node_idx, (x, y) in sensor_pos.items():
        radius = 10 + imp_norm[node_idx] * 30
        circle = plt.Circle((x, y), radius,
                             color=plt.cm.YlOrRd(imp_norm[node_idx]),
                             alpha=0.7)
        ax.add_patch(circle)
        ax.text(x, y, f'N{node_idx}', ha='center', va='center', fontsize=7, color='black')

    ax.set_title(f'Word {word_idx} - Average Node Importance')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(vis_path, f"word_avg_{word_idx:03d}.png"), dpi=100)
    plt.close()

print(f"저장 완료: {vis_path}")