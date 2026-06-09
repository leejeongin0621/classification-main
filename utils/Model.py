import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.transformer import _get_clones
from utils.tgcn import ConvTemporalGraphical
from utils.graph import Graph
import numpy as np


#----------------------------
#ST-GCN blocks (원본 그대로 사용)
#----------------------------
class IMU_STGCN(nn.Module):
    def __init__(self, in_channels, num_class, graph_args,
                 edge_importance_weighting, dropout, ):
        super().__init__()

        self.in_channels = in_channels

        # load graph
        self.graph = Graph(**graph_args) # graph.py의 Graph 클래스에서 정의한 그래프 구조를 불러옴
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A) 

        #build networks
        # 분리 ST-GCN baseline: GCN(graph) + TCN(temporal) 분리 처리
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        self.st_gcn_networks = nn.ModuleList((
            st_gcn(in_channels, 64,  kernel_size, 1, residual=False, dropout=0),
            st_gcn(64,  64,  kernel_size, 1, dropout),
            st_gcn(64,  64,  kernel_size, 1, dropout),
            st_gcn(64,  128, kernel_size, 2, dropout),  # 200→100
            st_gcn(128, 128, kernel_size, 1, dropout),
            st_gcn(128, 128, kernel_size, 1, dropout),
            st_gcn(128, 256, kernel_size, 2, dropout),  # 100→50
        ))

        # temporal_kernel_size = 11
        # kernel_size = (temporal_kernel_size, spatial_kernel_size)
        # self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        # self.st_gcn_networks = nn.ModuleList((
        #     st_gcn(in_channels, 64,  kernel_size, 1, residual=False, dropout=0),
        #     st_gcn(64,  64,  kernel_size, 1, dropout),
        #     st_gcn(64,  64,  kernel_size, 1, dropout),
        #     st_gcn(64,  128, kernel_size, 2, dropout),  # 200→100
        #     st_gcn(128, 128, kernel_size, 1, dropout),
        #     st_gcn(128, 128, kernel_size, 1, dropout),
        #     st_gcn(128, 256, kernel_size, 2, dropout),  # 100→50
        # ))
        

        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

        # fcn for prediction
        # fcn for prediction
        self.fcn = nn.Conv2d(256, num_class, kernel_size=1)

    def forward(self, x_IMU):

        # data normalization
        B, T, D = x_IMU.shape
        x = x_IMU.view(B, T, 10, self.in_channels)  
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.unsqueeze(-1)

        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # forward
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        # global pooling
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, M, -1, 1, 1).mean(dim=1)

        # prediction
        x = self.fcn(x)
        x = x.view(x.size(0), -1)

        return x

    def extract_feature(self, x_IMU):

        # data normalization
        B, T, D = x_IMU.shape
        x = x_IMU.view(B, T, 10, self.in_channels)   
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.unsqueeze(-1) # (B,3,200,10,1) 형태로 변환하여 ST-GCN 입력에 맞춤

        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # forwad
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        _, c, t, v = x.size()
        feature = x.view(N, M, c, t, v).permute(0, 2, 3, 4, 1)

        # prediction
        x = self.fcn(x)
        output = x.view(N, M, -1, t, v).permute(0, 2, 3, 4, 1)

        return output, feature
    
    
class Zero(nn.Module):
    def forward(self, x):
        return 0

class st_gcn(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dropout=0.2,
                 residual=True):

        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                         kernel_size[1])

        # 단일 TCN (baseline)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = Zero()
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):
        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res
        return self.relu(x), A


#biLSTM 수정본 
import torch
import torch.nn as nn


class BiLSTM(nn.Module):
    def __init__(self, num_classes, nCh=30, nFilters=250, kernel_size=8,
                 hidden_dim=250, gap_dropout=0.5):
        super().__init__()

        # TensorFlow Conv1D 기본 padding='valid'와 맞추려면 padding=0
        self.IMU_conv1 = nn.Sequential(
            nn.Conv1d(nCh, nFilters, kernel_size=kernel_size, padding=0),
            nn.BatchNorm1d(nFilters),
            nn.ReLU(),
            nn.AvgPool1d(2)
        )

        self.IMU_conv2 = nn.Sequential(
            nn.Conv1d(nFilters, nFilters, kernel_size=kernel_size, padding=0),
            nn.BatchNorm1d(nFilters),
            nn.ReLU(),
            nn.AvgPool1d(2)
        )

        self.IMU_conv3 = nn.Sequential(
            nn.Conv1d(nFilters, nFilters, kernel_size=kernel_size, padding=0),
            nn.BatchNorm1d(nFilters),
            nn.ReLU(),
            nn.AvgPool1d(2)
        )

        self.IMU_conv4 = nn.Sequential(
            nn.Conv1d(nFilters, nFilters, kernel_size=kernel_size, padding=0),
            nn.BatchNorm1d(nFilters),
            nn.ReLU(),
            nn.AvgPool1d(2)
        )

        self.lstm = nn.LSTM(
            input_size=nFilters,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.0
        )

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(p=gap_dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x_IMU, return_feat=False):
        # x_IMU: (B, T, C) = (B, 200, 30)
        x = x_IMU.transpose(1, 2)   # (B, C, T)

        x = self.IMU_conv1(x)
        x = self.IMU_conv2(x)
        x = self.IMU_conv3(x)
        x = self.IMU_conv4(x)

        x = x.transpose(1, 2)       # (B, T', nFilters)

        x, _ = self.lstm(x)         # (B, T', 2*hidden_dim)

        x = x.transpose(1, 2)       # (B, 2*hidden_dim, T')

        feat = self.gap(x).squeeze(-1)
        feat = self.dropout(feat)

        logits = self.fc(feat)

        return (logits, feat) if return_feat else logits

class PreNormTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, src):
        output = src

        for mod in self.layers:
            output = mod(output)

        return output

class PreNormTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, src):
        # Self-attention with pre-norm
        src2 = self.norm1(src)
        attn_output, _ = self.self_attn(src2, src2, src2)
        src = src + self.dropout1(attn_output)

        # Feedforward with pre-norm
        src2 = self.norm2(src)
        ff_output = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(ff_output)

        return src

# ============================================================
# 2) IMU only Transformer (EMG_IMU_Transformer 에서 EMG 제거) 
# ============================================================
class IMU_Transformer(nn.Module):
    """
    입력: x_IMU (B, 200, 15)
    처리: Conv1d 3단 -> (B, 64, 25) -> (B,25,64) + pos -> Transformer -> mean pool -> FC
    """
    def __init__(self, num_classes, num_layers=2, nhead=4, d_model=64, max_len=100, dropout=0.1):
        super().__init__()

        # Conv stack (출력 채널을 d_model로 맞춤)
        self.IMU_conv1 = nn.Sequential(
            nn.Conv1d(in_channels=30, out_channels=64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AvgPool1d(2) 
        )
        self.IMU_conv2 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AvgPool1d(2)  
        )
        self.IMU_conv3 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AvgPool1d(2)  
        )
        self.IMU_conv4 = nn.Sequential(
        nn.Conv1d(64, d_model, kernel_size=7, padding=3),
        nn.BatchNorm1d(d_model),
        nn.ReLU(),
        nn.AvgPool1d(2)  
        )

        self.pos_embedding = nn.Embedding(max_len, d_model)

        encoder_layer = PreNormTransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="relu"
        )
        self.transformer = PreNormTransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x_IMU, return_feat: bool=False):
        # x_IMU: (B, 200, 30)
        x = x_IMU.transpose(1, 2)   # (B, 30, 200)
        x = self.IMU_conv1(x)       # (B, 64, 100)
        x = self.IMU_conv2(x)       # (B, 64, 50)
        x = self.IMU_conv3(x)       # (B, d_model, 25)
        x = self.IMU_conv4(x)       # (B, d_model, 12)

        x = x.permute(0, 2, 1)      # (B, 12, d_model)
        B, T, D = x.shape

        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        x = x + self.pos_embedding(pos)

        # PreNormTransformerEncoderLayer는 batch_first=True로 MultiheadAttention을 쓰므로 (B,T,D) 그대로 넣어도 됨.
        out = self.transformer(x)   # (B, 12, d_model)

        feat = out.mean(dim=1)      # (B, d_model)
        logits = self.classifier(feat)
        return (logits, feat) if return_feat else logits


# ============================================================
# 3) IMU only conso_processing Transformer
#    (원본은 EMG+IMU cross-attn인데, EMG 제거 → IMU 그룹 간 cross로 대체)
# ============================================================
class IMU_conso_processing_Transformer(nn.Module):
    """
    입력: x_IMU (B, 200, 15)
    - 15채널을 (5 그룹, 각 3채널)로 reshape
    - 각 그룹별 Conv1d -> (B, IMU_conv_channel, 100)
    - 마지막 그룹을 'conso'로 보고, 다른 그룹과 CrossModalityTransformerEncoder로 상호 참조
    - concat -> Conv2/Conv3 -> Transformer -> FC
    """
    def __init__(
        self,
        num_classes,
        num_layers=2,
        nhead=4,
        IMU_conv_channel=16,
        conso_layers=2,
        max_len=100,
        d_model=64,
        dropout=0.1
    ):
        super().__init__()

        # 그룹별 conv1 (각 그룹 in_channels=3)
        self.IMU_conv1_channels = _get_clones(
            nn.Sequential(
                nn.Conv1d(in_channels=3, out_channels=IMU_conv_channel, kernel_size=7, padding=3),
                nn.BatchNorm1d(IMU_conv_channel),
                nn.ReLU(),
                nn.AvgPool1d(2)  # 200 -> 100
            ),
            10
        )

        # 그룹 feature들을 concat 해서 줄이는 conv
        # 10개 그룹 + (cross 결과 2개) 정도를 붙일 예정 → 채널 수를 유연하게 계산
        # 아래는: 10 그룹 + 2개 cross_out = 12개 => 12 * IMU_conv_channel
        in_ch_conv2 = IMU_conv_channel * 12

        self.IMU_conv2 = nn.Sequential(
            nn.Conv1d(in_ch_conv2, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AvgPool1d(2)  # 100 -> 50
        )
        self.IMU_conv3 = nn.Sequential(
            nn.Conv1d(64, d_model, kernel_size=7, padding=3),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.AvgPool1d(2)  # 50 -> 25
        )

        # 그룹/시간 위치 임베딩 (conso attention에 사용)
        self.channel_pos_embedding = nn.Embedding(max_len, IMU_conv_channel)
        layer_for_cross = PreNormTransformerEncoderLayer(
            d_model=IMU_conv_channel,
            nhead=nhead,
            dim_feedforward=IMU_conv_channel * 4,
            dropout=dropout,
            activation="relu"
        )
        self.channel_transformer = CrossModalityTransformerEncoder(layer_for_cross, num_layers=conso_layers)

        # 최종 transformer
        self.pos_embedding = nn.Embedding(max_len, d_model)
        encoder_layer = PreNormTransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="relu"
        )
        self.transformer = PreNormTransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x_IMU, return_feat: bool=False):
        # x_IMU: (B, 200, 15)
        x = x_IMU.transpose(1, 2)                 # (B, 15, 200)
        x = x.reshape(x.size(0), 10, 3, x.size(2)) # (B, 10, 3, 200)


        # 그룹별 conv1
        outs = []
        for g in range(10):
            outs.append(self.IMU_conv1_channels[g](x[:, g, :, :]))  # (B, IMU_conv_channel, 100)

        # conso 그룹(마지막)을 다른 그룹 집계(out_other)와 cross-attn
        conso = outs[-1].transpose(1, 2)  # (B,100,Cc)
        other = torch.stack(outs[:-1], dim=0).mean(dim=0).transpose(1, 2)  # (B,100,Cc)

        B, T, Cc = conso.shape
        pos = torch.arange(T, device=conso.device).unsqueeze(0).expand(B, T)
        conso = conso + self.channel_pos_embedding(pos)
        other = other + self.channel_pos_embedding(pos)

        # CrossModalityTransformerEncoder는 (B,T,C) batch_first=True로 동작
        out1, out2 = self.channel_transformer(conso, other)  # (B,100,Cc) 두 개

        out1 = out1.transpose(1, 2)  # (B,Cc,100)
        out2 = out2.transpose(1, 2)  # (B,Cc,100)

        # 5개 그룹 + cross 결과 2개 => 7개를 concat
        x_cat = torch.cat(outs + [out1, out2], dim=1)  # (B, 7*Cc, 100)

        x = self.IMU_conv2(x_cat)  # (B,64,50)
        x = self.IMU_conv3(x)      # (B,d_model,25)

        x = x.permute(0, 2, 1)     # (B,25,d_model)
        B, T, D = x.shape
        pos2 = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        x = x + self.pos_embedding(pos2)

        out = self.transformer(x)  # (B,25,d_model)
        feat = out.mean(dim=1)     # (B,d_model)
        logits = self.classifier(feat)
        return (logits, feat) if return_feat else logits