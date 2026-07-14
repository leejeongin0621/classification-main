#%%
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
from utils.Model import PureTCN, PureGCN, IMU_Conformer, IMU_ConformerBiLSTM, IMU_Transformer,IMU_conso_processing_Transformer,IMU_STGCN,BiLSTM,IMU_GASTNet,IMU_DualBranch,IMU_GateFusion
from os import path
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# 모델 초기화
num_classes = 100   # 단어 수 (클래스 수)
hidden_dim = 250
num_layers = 2
gap_dropout = 0.2
nhead = 4
nCh = 30
IMU_conv_channel = 16
conso_layers = 16 
k=0

#num_layers = 16      # Transformer 인코더 레이어 수
#hidden_dim = 64    # Transformer 모델의 숨겨진 차원 크기
#nhead = 4        # 멀티헤드 어텐션의 헤드 수
#IMU_conv_channel= 16     # IMU 첫 CNN 출력 채널 수
#conso_layers= 16      # conso Transformer 레이어 수


# model = IMU_STGCN(
#     num_class=num_classes,
#     in_channels=3,
#     graph_args={'max_hop': 1, 'dilation': 1},
#     edge_importance_weighting=True,
#     dropout=gap_dropout,
# ).to(device)

# model = BiLSTM(
#     num_classes=num_classes,
#     nCh=nCh,              # 30
#     nFilters=250,
#     kernel_size=8,
#     hidden_dim=250,
#     gap_dropout=0.5,
# ).to(device)


# model = IMU_Transformer(
#     num_classes=num_classes,
#     num_layers=num_layers,
#     d_model=64,
#     nhead=nhead
# ).to(device)

# model = IMU_ConformerBiLSTM(
#     num_classes=100,
#     d_model=128,
#     nhead=4,
#     num_conformer_layers=4,   
#     ff_dim=512,
#     conv_kernel_size=31,
#     dropout=0.1,
# ).to(device)


# ── 133: PureTCN ──────────────────────────────────────────────
# model = PureTCN(
#     num_classes=100,
#     in_channels=30,
#     hidden_dim=128,
#     num_layers=7,
#     kernel_size=11,
#     dropout=0.2,
# ).to(device)

#── 134: PureGCN ──────────────────────────────────────────────
# model = PureGCN(
#     in_channels=3,
#     num_class=100,
#     graph_args={'max_hop': 1, 'dilation': 1},
#     hidden_dims=(64, 64, 64, 128, 128, 128, 256),
#     dropout=0.2,
# ).to(device)

# ── 135: IMU_GASTNet (GAST-Net 스타일: TCN↔GAttn 교차, Local F-stat + Global Bk)
# model = IMU_GASTNet(
#     in_channels=3,
#     num_class=100,
#     graph_args={'max_hop': 1, 'dilation': 1},
#     edge_importance_weighting=True,
#     dropout=0.2,
# ).to(device)

# ── 136: IMU_STGCN_TPP (ST-GCN + Temporal Pyramid Pooling, scales=[1,2,4])
# model = IMU_STGCN_TPP(
#     in_channels=3,
#     num_class=100,
#     graph_args={'max_hop': 1, 'dilation': 1},
#     edge_importance_weighting=True,
#     dropout=0.2,
#     tpp_scales=(1, 2, 4),
# ).to(device)

# ── 136: IMU_DualBranch (GCN branch 7겹 + TCN branch 7겹 병렬 → cross-attn → FC)
# model = IMU_DualBranch(
#     in_channels=3,
#     num_class=100,
#     graph_args={'max_hop': 1, 'dilation': 1},
#     gcn_dims=(64, 64, 64, 128, 128, 128, 256),
#     tcn_hidden=128,
#     tcn_kernel=11,
#     edge_importance_weighting=True,
#     num_heads=4,
#     dropout=0.2,
# ).to(device)

# ── 138: IMU_GateFusion (SharedLocalCNN → GCN+TCN → BranchGate → FusionFC)
model = IMU_GateFusion(
    in_channels=3,
    num_class=100,
    num_sensor=10,
    graph_args={'max_hop': 1, 'dilation': 1},
    dropout=0.2,
).to(device)

# 모델 전체 저장
Path("models").mkdir(exist_ok=True)
model_num = 140
save_path = f"./models/Model_{model_num}.pt"


# 모델에 더미 입력 전달하여 정상 작동 확인
dummy_input_IMU = torch.randn(20, 200, 30).to(device)   # 10 sensor × 3 axis

output = model(dummy_input_IMU)  # 더미 입력으로 테스트


if not path.exists(save_path):
    torch.save(model, save_path)
    print("Model saved to {}".format(save_path))
else:
    print("File already exists at {}".format(save_path))
#end


#%%
# 모델 전체 불러오기
import torch

model_num = 140

save_path = "./models/Model_{}.pt".format(model_num)
model_loaded = torch.load(save_path, weights_only=False, map_location=device)
print("model num : {}".format(model_num))
print(model_loaded)
# model_loaded.eval()  # 평가 모드로 전환 (필요시)

#101 : IMU_STGCN (edge importance top20 기반 graph) #102 : left graph #103 : edge importance 기반 그래프 # 104: ?이거 그냥 fc 다시 한거 # 105 : 섞은거 #106 : fully connected graph #107 :separability 
#108 : 17개 edge로 바꿈-이게 제일 나옴 #109 : left graph(15) #110 : right (15) #111 : RIGHT KERENEL SIZE 9 #112 : RIGHT SEPARABILITY BASED 13개 +2개  #113 모델 구조 depthwise/mutiscale #114 multiscale+max_hop2 
# #115 ms_stgcn (k=3∥k=11 multi-scale TCN)/gatedtcn/gcn3겹+tcn7겹 #116 gcn 3겹 +tcn 5겹 #117 gated tcn/ 지금은 edge feature 이용한거 symmetric하지 않는거 #118 edge feature 이용한 symmetric한거 #119 edge TCN + line-graph 
# 120 : 119 수정본 ### 122 : LOG로 WEIGHTED GRAPH 만들어줌 
# #123 : linear weighted graph (top20 edge) #124 : 첫단꺼 residual 반영(잘 안나옴) #125 : tcn->gcn-> tcn #126 #127 : bilstm(lr=0.0005 돌린거 ) #128 : dropout 비율을 달리해봄 
#131 : Conformer+bilstm