"""
DAE_npydata 전용 모델 파일 — IMU 쪽 (utils/Model.py) 과 분리하여 관리.

- DAE_STGCN: 5 sensor × 3 axis (15ch) ST-GCN, 5-node fully-connected graph
- DAE_BiLSTM: 15ch 입력 BiLSTM

두 모델 모두 입력은 (B, T=200, 15) 형태로 받음.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.tgcn import ConvTemporalGraphical
from utils.graph import Graph_FC


# ============================================================
# Helpers (utils/Model.py 의 Zero / st_gcn 와 동일한 구조 — 의존성 없이 자체 정의)
# ============================================================
class Zero(nn.Module):
    def forward(self, x):
        return 0


class st_gcn(nn.Module):
    """단일 ST-GCN 블록 (GCN + TCN). utils/Model.py 의 st_gcn 과 동일."""
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, dropout=0.2, residual=True):
        super().__init__()
        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = ConvTemporalGraphical(in_channels, out_channels, kernel_size[1])

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels,
                      (kernel_size[0], 1), (stride, 1), padding),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = Zero()
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):
        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res
        return self.relu(x), A


# ============================================================
# DAE_STGCN
# ============================================================
class DAE_STGCN(nn.Module):
    """
    DAE_npydata (5 sensor × 3 axis = 15ch) 용 ST-GCN.
    - num_node = 5 (fully-connected graph)
    - in_channels = 3 (raw XYZ per sensor)
    - 입력: (B, T=200, 15)
    """
    def __init__(self,
                 num_class,
                 in_channels=3,
                 num_node=5,
                 max_hop=1, dilation=1,
                 edge_importance_weighting=True,
                 dropout=0.2):
        super().__init__()
        self.in_channels = in_channels
        self.num_node    = num_node

        # 5-node FC graph
        self.graph = Graph_FC(num_node=num_node, max_hop=max_hop, dilation=dilation)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        # ST-GCN blocks
        # 60 train sample × 30 class 환경에 맞춰 점진적 stride, 작은 채널, 충분한 dropout.
        # - layer 1 은 stride=1 + residual=False (입력 직후 정보 유지)
        # - 점진적으로 stride=2 두 번 (200→100→50) 으로만 압축
        # - 채널 64→128 까지만 (1.6M → ~400K params)
        spatial_kernel_size  = A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)

        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        self.st_gcn_networks = nn.ModuleList((
            st_gcn(in_channels, 64,  kernel_size, 1, residual=False, dropout=0),
            st_gcn(64,  64,  kernel_size, 1, dropout),
            st_gcn(64,  128, kernel_size, 2, dropout),   # 200 → 100
            st_gcn(128, 128, kernel_size, 1, dropout),
            st_gcn(128, 128, kernel_size, 2, dropout),   # 100 → 50
        ))

        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for _ in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

        self.fcn = nn.Conv2d(128, num_class, kernel_size=1)

    def forward(self, x_IMU):
        # x_IMU: (B, T, num_node * in_channels) = (B, 200, 15)
        B, T, D = x_IMU.shape
        x = x_IMU.view(B, T, self.num_node, self.in_channels)

        x = x.permute(0, 3, 1, 2).contiguous()    # (B, C, T, V)
        x = x.unsqueeze(-1)                       # (B, C, T, V, M=1)

        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        # global pooling
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, M, -1, 1, 1).mean(dim=1)

        x = self.fcn(x)
        return x.view(x.size(0), -1)


# ============================================================
# DAE_BiLSTM
# ============================================================
class DAE_BiLSTM(nn.Module):
    """
    DAE_npydata (15ch) 용 BiLSTM.
    - 입력: (B, T=200, 15)
    - 구조: Conv1d × 4 (valid padding, pool/2) → BiLSTM → GAP → FC
    """
    def __init__(self, num_classes,
                 nCh=15, nFilters=64, kernel_size=7,
                 hidden_dim=128, gap_dropout=0.5):
        super().__init__()

        self.IMU_conv1 = nn.Sequential(
            nn.Conv1d(nCh, nFilters, kernel_size=kernel_size, padding=0),
            nn.BatchNorm1d(nFilters), nn.ReLU(), nn.AvgPool1d(2),
        )
        self.IMU_conv2 = nn.Sequential(
            nn.Conv1d(nFilters, nFilters, kernel_size=kernel_size, padding=0),
            nn.BatchNorm1d(nFilters), nn.ReLU(), nn.AvgPool1d(2),
        )
        self.IMU_conv3 = nn.Sequential(
            nn.Conv1d(nFilters, nFilters, kernel_size=kernel_size, padding=0),
            nn.BatchNorm1d(nFilters), nn.ReLU(), nn.AvgPool1d(2),
        )
        self.IMU_conv4 = nn.Sequential(
            nn.Conv1d(nFilters, nFilters, kernel_size=kernel_size, padding=0),
            nn.BatchNorm1d(nFilters), nn.ReLU(), nn.AvgPool1d(2),
        )

        self.lstm = nn.LSTM(
            input_size=nFilters,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.0,
        )

        self.gap     = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(p=gap_dropout)
        self.fc      = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x_IMU, return_feat=False):
        x = x_IMU.transpose(1, 2)   # (B, C, T)

        x = self.IMU_conv1(x)
        x = self.IMU_conv2(x)
        x = self.IMU_conv3(x)
        x = self.IMU_conv4(x)

        x = x.transpose(1, 2)       # (B, T', F)
        x, _ = self.lstm(x)         # (B, T', 2*H)
        x = x.transpose(1, 2)       # (B, 2*H, T')

        feat = self.gap(x).squeeze(-1)
        feat = self.dropout(feat)
        logits = self.fc(feat)
        return (logits, feat) if return_feat else logits
