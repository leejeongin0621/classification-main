#%%
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
from utils.Model import IMU_Transformer,IMU_BiLSTM,IMU_conso_processing_Transformer,IMU_STGCN
from os import path
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# 모델 초기화
num_classes = 100   # 단어 수 (클래스 수)
hidden_dim = 128
num_layers = 2
gap_dropout = 0.2
nhead = 4
nCh = 30
IMU_conv_channel = 16
conso_layers = 16 

#num_layers = 16      # Transformer 인코더 레이어 수
#hidden_dim = 64    # Transformer 모델의 숨겨진 차원 크기
#nhead = 4        # 멀티헤드 어텐션의 헤드 수
#IMU_conv_channel= 16     # IMU 첫 CNN 출력 채널 수
#conso_layers= 16      # conso Transformer 레이어 수


model = IMU_STGCN(
    num_class=num_classes,
    in_channels=3,
    graph_args={'max_hop': 1, 'dilation': 1},
    edge_importance_weighting=True,
    dropout=gap_dropout
).to(device)

# model = AGCN(
#     num_class=num_classes,
#     in_channels=3,
#     graph_args={'max_hop': 1, 'dilation': 1},
#     dropout=gap_dropout
# ).to(device)

# model = IMU_STGCNINCEPTION(
#     num_class=num_classes,
#     in_channels=3,
#     graph_args={'max_hop': 1, 'dilation': 1},
#     edge_importance_weighting=True,
#     dropout=gap_dropout
# ).to(device)

# model = EMG_IMU_BiLSTM(
#     num_classes=num_classes,
#     num_layers=num_layers,
#     hidden_dim=hidden_dim
# ).to(device)

# model = IMU_Transformer(
#     num_classes=num_classes,
#     num_layers=num_layers,
#     d_model=hidden_dim,
#     nhead=nhead
# ).to(device)

# model = IMU_conso_processing_Transformer(
#     num_classes=num_classes,
#     num_layers=num_layers,
#     d_model=hidden_dim,
#     nhead=nhead,
#     IMU_conv_channel=IMU_conv_channel,
#   conso_layers=conso_layers
# ).to(device)

#  model = IMU_BiLSTM(
#    num_classes=num_classes,
#    hidden_dim=hidden_dim,
#    num_layers=num_layers,
#    gap_dropout=gap_dropout,
#    nCh=nCh
# ).to(device)

# 모델 전체 저장
Path("models").mkdir(exist_ok=True)
model_num = 37
save_path = "./models/Model_37.pt"


# 모델에 더미 입력 전달하여 정상 작동 확인
dummy_input_IMU = torch.randn(20, 200, 30).to(device)

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

model_num = 37

save_path = "./models/Model_{}.pt".format(model_num)
model_loaded = torch.load(save_path, weights_only=False, map_location=device)
print("model num : {}".format(model_num))
print(model_loaded)
# model_loaded.eval()  # 평가 모드로 전환 (필요시)