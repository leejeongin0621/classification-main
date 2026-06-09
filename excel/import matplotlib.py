import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

n_subjects = 10
x = np.arange(n_subjects)

group_gap = 0.15
width = 0.18

o1 = -1.5*width - group_gap/2
o2 = -0.5*width - group_gap/2
o3 = +0.5*width + group_gap/2
o4 = +1.5*width + group_gap/2

cnn_base  = [0.5600, 0.8880, 0.8200, 0.5860, 0.7340, 0.6660, 0.5680, 0.7060, 0.1480, 0.6260]
stgcn_base= [0.6160, 0.9380, 0.9100, 0.7880, 0.8180, 0.7500, 0.6940, 0.7000, 0.1100, 0.6480]
cnn_z     = [0.6180, 0.8620, 0.8520, 0.6980, 0.7700, 0.7860, 0.6460, 0.7840, 0.7240, 0.6160]
stgcn_z   = [0.6160, 0.9160, 0.9160, 0.7860, 0.7960, 0.8620, 0.7480, 0.7020, 0.7880, 0.7220]

cnn_base_mean,  cnn_base_std  = 0.6302, 0.1907
stgcn_base_mean,stgcn_base_std= 0.6972, 0.2197
cnn_z_mean,     cnn_z_std     = 0.7356, 0.0856
stgcn_z_mean,   stgcn_z_std   = 0.7852, 0.0901

color_cnn_base = '#FDDBC7'
color_st_base  = '#EF8A62'
color_cnn_z    = '#D1E5F0'
color_st_z     = '#2166AC'
edge_c         = '#2C3E50'

scale = 1.6
x_pos = x * scale

fig, ax = plt.subplots(figsize=(20, 8))

ax.bar(x_pos + o1, cnn_base,  width, color=color_cnn_base, edgecolor='white', linewidth=0.5)
ax.bar(x_pos + o2, stgcn_base,width, color=color_st_base,  edgecolor='white', linewidth=0.5)
ax.bar(x_pos + o3, cnn_z,     width, color=color_cnn_z,    edgecolor='white', linewidth=0.5)
ax.bar(x_pos + o4, stgcn_z,   width, color=color_st_z,     edgecolor='white', linewidth=0.5)

tx = x_pos[-1] + scale * 1.2

ax.bar(tx + o1, cnn_base_mean,  width, yerr=cnn_base_std,   color=color_cnn_base, edgecolor=edge_c, linewidth=1.2, capsize=5, error_kw=dict(elinewidth=1.2, ecolor='#555'))
ax.bar(tx + o2, stgcn_base_mean,width, yerr=stgcn_base_std, color=color_st_base,  edgecolor=edge_c, linewidth=1.2, capsize=5, error_kw=dict(elinewidth=1.2, ecolor='#555'))
ax.bar(tx + o3, cnn_z_mean,     width, yerr=cnn_z_std,      color=color_cnn_z,    edgecolor=edge_c, linewidth=1.2, capsize=5, error_kw=dict(elinewidth=1.2, ecolor='#555'))
ax.bar(tx + o4, stgcn_z_mean,   width, yerr=stgcn_z_std,    color=color_st_z,     edgecolor=edge_c, linewidth=1.2, capsize=5, error_kw=dict(elinewidth=1.2, ecolor='#555'))

def p_to_stars(p):
    if p >= 0.05:    return 'ns'
    elif p < 0.0001: return '****'
    elif p < 0.001:  return '***'
    elif p < 0.01:   return '**'
    else:            return '*'

def format_threshold(p):
    if p < 0.0001:  return '0.0001'
    elif p < 0.001: return '0.001'
    elif p < 0.01:  return '0.01'
    else:           return '0.05'

def draw_bracket(ax, x1, x2, y, p_val, lw=1.2, fontsize=10):
    h = 0.012
    color = '#2C3E50'
    stars = p_to_stars(p_val)
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], color=color, lw=lw,
            clip_on=False)
    if stars == 'ns':
        label = '$\\it{ns}$'
    else:
        label = f'$\\it{{p}}$ < {format_threshold(p_val)}  {stars}'
    ax.text((x1+x2)/2, y+h+0.006, label, ha='center', va='bottom',
            fontsize=fontsize, color=color, clip_on=False)

lvl1_base = 1.02
lvl1_z    = 1.08

draw_bracket(ax, tx+o1, tx+o2, lvl1_base, 0.0137)
draw_bracket(ax, tx+o3, tx+o4, lvl1_z,    0.0137)

lvl2_cnn   = 1.17
lvl2_stgcn = 1.26

draw_bracket(ax, tx+o1, tx+o3, lvl2_cnn,   0.0098)
draw_bracket(ax, tx+o2, tx+o4, lvl2_stgcn, 0.1544)

# 세로 구분선
for i in range(n_subjects - 1):
    mid = (x_pos[i] + x_pos[i+1]) / 2
    ax.axvline(mid, color='#E0E0E0', linewidth=0.8, linestyle='--')
ax.axvline((x_pos[-1] + scale*0.6 + tx - scale*0.6) / 2 + scale*0.3,
           color='#AAAAAA', linewidth=1.2, linestyle='-')

# 그룹 레이블
for xi in list(x_pos) + [tx]:
    ax.annotate('Base', xy=(xi + (o1+o2)/2, -0.04), xycoords=('data','axes fraction'),
                ha='center', fontsize=7, color='#777')
    ax.annotate('Z-Ali', xy=(xi + (o3+o4)/2, -0.04), xycoords=('data','axes fraction'),
                ha='center', fontsize=7, color='#555')

# X축 레이블
all_x = list(x_pos) + [tx]
all_labels = [f'Subject {i}' for i in range(1, 11)] + ['Mean ± Std']
ax.set_xticks(all_x)
ax.set_xticklabels(all_labels, fontsize=9.5)
ax.tick_params(axis='x', pad=18)

# 범례
patches = [
    mpatches.Patch(color=color_cnn_base, label='Baseline: CNN-biLSTM'),
    mpatches.Patch(color=color_st_base,  label='Baseline: ST-GCN'),
    mpatches.Patch(color=color_cnn_z,    label='Z-Align: CNN-biLSTM'),
    mpatches.Patch(color=color_st_z,     label='Z-Align: ST-GCN'),
]
ax.legend(handles=patches, fontsize=10, loc='upper left', bbox_to_anchor=(1.01, 1.0))

ax.set_title('Model Performance Comparison (Baseline vs Z-Axis Alignment)',
             fontsize=14, fontweight='bold', pad=15)
ax.set_ylabel('Accuracy', fontsize=12)

# y축 최대 1.0, 테두리 박스
ax.set_ylim(0, 1.0)
ax.set_xlim(x_pos[0] - 0.6, tx + 0.6)

for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_color('#2C3E50')
    spine.set_linewidth(1.2)

ax.grid(axis='y', linestyle=':', alpha=0.4)

plt.tight_layout()
plt.savefig('performance_chart.png', dpi=180, bbox_inches='tight')
plt.close()