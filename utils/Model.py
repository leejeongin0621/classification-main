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
                 edge_importance_weighting, dropout):
        super().__init__()

        self.in_channels = in_channels

        # load graph
        self.graph = Graph(**graph_args) # graph.py의 Graph 클래스에서 정의한 그래프 구조를 불러옴
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        #build networks
        # 분리 ST-GCN baseline: GCN(graph) + TCN(temporal) 분리 처리
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 11
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

        # forward
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

        self.gcn = ConvTemporalGraphical(in_channels, out_channels, kernel_size[1])

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


# ============================================================
# GAST-Net 스타일 IMU 분류 모델
# TCN block (시간만) + Graph Attn block (공간만) 교차 배치
# ============================================================

class _TemporalBlock(nn.Module):
    """순수 시간 방향 Conv: kernel (k,1) — 센서 간 관계 무관"""
    def __init__(self, in_ch, out_ch, kernel_size=9, stride=1, dropout=0.2):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, (kernel_size, 1), stride=(stride, 1), padding=(pad, 0)),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, 1),
            nn.BatchNorm2d(out_ch),
        )
        self.res  = (nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, stride=(stride, 1)),
                                   nn.BatchNorm2d(out_ch))
                     if in_ch != out_ch or stride != 1 else nn.Identity())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.net(x) + self.res(x))


class _GraphAttnBlock(nn.Module):
    """Local GCN (F-stat 그래프) + Global Bk attention 병렬 → concat
       시간 방향은 건드리지 않음 (공간 처리 전용)
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        half = out_ch // 2
        dk   = max(in_ch // 4, 1)

        # Local branch: F-stat adjacency 기반 GCN
        self.local_conv = nn.Conv2d(in_ch, half, 1)
        self.local_bn   = nn.BatchNorm2d(half)

        # Global branch: Bk attention (입력 기반 동적 adjacency)
        self.dk         = dk
        self.theta      = nn.Conv1d(in_ch, dk, 1)
        self.phi        = nn.Conv1d(in_ch, dk, 1)
        self.global_conv = nn.Conv2d(in_ch, half, 1)
        self.global_bn   = nn.BatchNorm2d(half)

        self.out_bn = nn.BatchNorm2d(out_ch)
        self.res    = (nn.Sequential(nn.Conv2d(in_ch, out_ch, 1), nn.BatchNorm2d(out_ch))
                       if in_ch != out_ch else nn.Identity())
        self.relu   = nn.ReLU(inplace=True)

    def forward(self, x, A):
        # x: (B, C, T, V),  A: (K, V, V)
        res = self.res(x)

        # A: (K,V,V) → 합산해서 (V,V)
        A_sum = A.sum(0)  # (V, V)

        # Local: 이웃 집계 → conv
        x_loc = torch.einsum('vw,bctw->bctv', A_sum, x)
        x_loc = self.local_bn(self.local_conv(x_loc))    # (B, half, T, V)

        # Global: Bk attention
        xv    = x.mean(dim=2)                            # (B, C, V)  temporal avg
        theta = self.theta(xv)                           # (B, dk, V)
        phi   = self.phi(xv)                             # (B, dk, V)
        attn  = torch.einsum('bdi,bdj->bij', theta, phi) / math.sqrt(self.dk)
        attn  = F.softmax(attn, dim=-1)                  # (B, V, V)
        x_glo = torch.einsum('bij,bctj->bcti', attn, x) # (B, C, T, V)
        x_glo = self.global_bn(self.global_conv(x_glo)) # (B, half, T, V)

        out = torch.cat([x_loc, x_glo], dim=1)          # (B, out_ch, T, V)
        return self.relu(self.out_bn(out) + res)


class IMU_GASTNet(nn.Module):
    """
    GAST-Net 스타일 IMU 100클래스 분류 모델
    구조: TCN → GAttn → TCN → GAttn → TCN → GAttn → GAP → FC

    TCN block : 시간 방향만 처리 (stride로 T 축소)
    GAttn block: 공간 방향만 처리 (Local F-stat 그래프 + Global Bk attention)
    """
    def __init__(self, in_channels, num_class, graph_args,
                 edge_importance_weighting=True, dropout=0.2):
        super().__init__()
        self.in_channels = in_channels
        V = 10

        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        self.data_bn = nn.BatchNorm1d(in_channels * V)

        # TCN blocks (시간 처리)                  in  out  stride  T 변화
        self.tcn1 = _TemporalBlock( in_channels,  64, stride=1, dropout=dropout)  # 200→200
        self.tcn2 = _TemporalBlock(128, 128,       stride=2, dropout=dropout)      # 200→100
        self.tcn3 = _TemporalBlock(256, 256,       stride=2, dropout=dropout)      # 100→50

        # Graph Attention blocks (공간 처리)       in   out
        self.gattn1 = _GraphAttnBlock( 64, 128)   # 64→128
        self.gattn2 = _GraphAttnBlock(128, 256)   # 128→256
        self.gattn3 = _GraphAttnBlock(256, 256)   # 256→256

        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(A.size())) for _ in range(3)
            ])
        else:
            self.edge_importance = [1, 1, 1]

        self.fcn  = nn.Linear(256, num_class)
        self.drop = nn.Dropout(dropout)

    def forward(self, x_IMU):
        B, T, D = x_IMU.shape
        V = 10

        # (B, T, 30) → (B, C, T, V)
        x = x_IMU.view(B, T, V, self.in_channels).permute(0, 3, 1, 2).contiguous()

        # BatchNorm: (B, C*V, T)
        x = x.permute(0, 1, 3, 2).contiguous().view(B, self.in_channels * V, T)
        x = self.data_bn(x)
        x = x.view(B, self.in_channels, V, T).permute(0, 1, 3, 2).contiguous()  # (B, C, T, V)

        # Block 1: TCN → GAttn
        x = self.tcn1(x)
        x = self.gattn1(x, self.A * self.edge_importance[0])

        # Block 2: TCN → GAttn
        x = self.tcn2(x)
        x = self.gattn2(x, self.A * self.edge_importance[1])

        # Block 3: TCN → GAttn
        x = self.tcn3(x)
        x = self.gattn3(x, self.A * self.edge_importance[2])

        # GAP over (T, V) → (B, 256)
        x = x.mean(dim=[2, 3])
        x = self.drop(x)
        x = self.fcn(x)
        return x


# ============================================================
# Pure TCN (그래프 없음, 시간만)
# ============================================================
class _TCNResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=11, stride=1, dropout=0.2):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding)
        self.bn   = nn.BatchNorm1d(out_channels)
        self.act  = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout)
        if in_channels != out_channels or stride != 1:
            self.res = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.res = nn.Identity()

    def forward(self, x):
        return self.drop(self.act(self.bn(self.conv(x)) + self.res(x)))


class PureTCN(nn.Module):
    """7겹 TCN. 입력 (N, T, 30). 그래프 없음.
    stride=2 를 3회(layer 2,4,6) 사용 → 200→100→50→25 후 GAP."""
    def __init__(self, num_classes=100, in_channels=30, hidden_dim=128,
                 num_layers=7, kernel_size=11, dropout=0.2):
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_ch  = in_channels if i == 0 else hidden_dim
            stride = 2 if i in (2, 4, 6) else 1
            layers.append(_TCNResBlock(in_ch, hidden_dim, kernel_size, stride=stride, dropout=dropout))
        self.net  = nn.Sequential(*layers)
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = x.transpose(1, 2)      # (N, 30, T)
        x = self.net(x)             # (N, hidden, ~25)
        x = x.mean(dim=2)          # GAP
        return self.fc(self.drop(x))


# ============================================================
# Pure GCN (TCN 없음, 공간만)
# ============================================================
class _SpatialGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, K=2, dropout=0.0):
        super().__init__()
        self.K  = K
        self.fc  = nn.Conv2d(in_channels, out_channels * K, kernel_size=1)
        self.bn  = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout)
        self.res  = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, A):
        # x: (B, C, T, V),  A: (K, V, V)
        res = self.res(x)
        x = self.fc(x)                              # (B, K*C_out, T, V)
        B, KC, T, V = x.size()
        x = x.view(B, self.K, KC // self.K, T, V)
        x = torch.einsum('bkctv,kvw->bctw', x, A)  # partition별 별도 집계
        return self.drop(self.act(self.bn(x) + res))


class PureGCN(nn.Module):
    """Pure spatial GCN. 시간축 처리 없음 — GCN 후 T·V 방향 GAP.
    입력 (N, T, 30). 그래프 구조만 학습."""
    def __init__(self, in_channels=3, num_class=100,
                 graph_args=None, hidden_dims=(64, 64, 64, 128, 128, 128, 256), dropout=0.2):
        super().__init__()
        if graph_args is None:
            graph_args = {'max_hop': 1, 'dilation': 1}
        self.in_channels = in_channels

        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))

        K = A.size(0)
        dims = [in_channels] + list(hidden_dims)
        self.gcn_layers = nn.ModuleList([
            _SpatialGCNBlock(dims[i], dims[i+1], K=K, dropout=(0.0 if i == 0 else dropout))
            for i in range(len(dims) - 1)
        ])
        self.fcn = nn.Conv2d(hidden_dims[-1], num_class, kernel_size=1)

    def forward(self, x_IMU):
        B, T, D = x_IMU.shape
        x = x_IMU.view(B, T, 10, self.in_channels)
        x = x.permute(0, 3, 1, 2).contiguous().unsqueeze(-1)

        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        for gcn in self.gcn_layers:
            x = gcn(x, self.A)

        x = x.mean(dim=2, keepdim=True)   # T 방향 GAP → (N, C, 1, V)
        x = x.mean(dim=3, keepdim=True)   # V 방향 GAP → (N, C, 1, 1)
        x = self.fcn(x)
        return x.view(x.size(0), -1)


# ============================================================
# IMU_DualBranch : GCN branch (공간) + TCN branch (시간) 병렬
# ============================================================
class IMU_DualBranch(nn.Module):
    """
    GCN branch : 7× _SpatialGCNBlock → T 평균 → (B, V=10,  256) 센서 토큰
    TCN branch : 7× _TCNResBlock     →          (B, T'=25, 128) 시간 토큰

    양방향 Cross-Attention:
      G→T : 각 센서가 시간 패턴 중 어디에 attend → (B, V,  256)
      T→G : 각 시간 step이 어느 센서에 attend   → (B, T', 128)
    mean → concat(384) → Dropout → FC → 100
    """
    def __init__(self, in_channels=3, num_class=100,
                 graph_args=None,
                 gcn_dims=(64, 64, 64, 128, 128, 128, 256),
                 tcn_hidden=128, tcn_kernel=11,
                 edge_importance_weighting=True,
                 num_heads=4, dropout=0.2):
        super().__init__()
        self.in_channels = in_channels
        if graph_args is None:
            graph_args = {'max_hop': 1, 'dilation': 1}

        # ── GCN branch ──
        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)
        V = A.size(1)

        K = A.size(0)
        self.gcn_bn = nn.BatchNorm1d(in_channels * V)
        dims = [in_channels] + list(gcn_dims)
        self.gcn_layers = nn.ModuleList([
            _SpatialGCNBlock(dims[i], dims[i+1], K=K, dropout=(0.0 if i == 0 else dropout))
            for i in range(len(dims) - 1)
        ])
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(A.size()))
                for _ in self.gcn_layers
            ])
        else:
            self.edge_importance = [1] * len(self.gcn_layers)
        gcn_out = gcn_dims[-1]   # 256

        # ── TCN branch ──
        tcn_layers = []
        for i in range(7):
            in_ch  = in_channels * V if i == 0 else tcn_hidden
            stride = 2 if i in (0, 2, 4) else 1
            tcn_layers.append(_TCNResBlock(in_ch, tcn_hidden, tcn_kernel,
                                           stride=stride, dropout=dropout))
        self.tcn_layers = nn.Sequential(*tcn_layers)
        tcn_out = tcn_hidden   # 128

        # ── Cross-Attention (양방향) ──
        # G→T : Q=GCN(256), KV=TCN(128)
        self.cross_g2t = nn.MultiheadAttention(
            embed_dim=gcn_out, num_heads=num_heads,
            kdim=tcn_out, vdim=tcn_out,
            batch_first=True, dropout=dropout)
        # T→G : Q=TCN(128), KV=GCN(256)
        self.cross_t2g = nn.MultiheadAttention(
            embed_dim=tcn_out, num_heads=num_heads,
            kdim=gcn_out, vdim=gcn_out,
            batch_first=True, dropout=dropout)

        self.drop = nn.Dropout(dropout)
        self.fcn  = nn.Linear(gcn_out + tcn_out, num_class)

    def forward(self, x_IMU):
        B, T, D = x_IMU.shape
        V = 10

        # ── GCN branch → (B, V, gcn_out) ──
        xg = x_IMU.view(B, T, V, self.in_channels).permute(0, 3, 1, 2).contiguous()
        xg = xg.view(B, self.in_channels * V, T)
        xg = self.gcn_bn(xg)
        xg = xg.view(B, self.in_channels, T, V)
        for gcn, imp in zip(self.gcn_layers, self.edge_importance):
            xg = gcn(xg, self.A * imp)
        z_graph_seq = xg.mean(dim=2).permute(0, 2, 1)  # (B, V=10, 256)

        # ── TCN branch → (B, T', tcn_out) ──
        xt = x_IMU.transpose(1, 2)       # (B, 30, T)
        xt = self.tcn_layers(xt)         # (B, 128, T'=25)
        z_temp_seq = xt.permute(0, 2, 1) # (B, T'=25, 128)

        # ── 양방향 Cross-Attention ──
        # 각 센서가 시간 패턴 어디에 집중할지
        z_g2t, _ = self.cross_g2t(z_graph_seq, z_temp_seq, z_temp_seq)  # (B, V,  256)
        # 각 시간 step이 어느 센서에 집중할지
        z_t2g, _ = self.cross_t2g(z_temp_seq, z_graph_seq, z_graph_seq) # (B, T', 128)

        z_graph = z_g2t.mean(dim=1)   # (B, 256)
        z_temp  = z_t2g.mean(dim=1)   # (B, 128)

        z = torch.cat([z_graph, z_temp], dim=1)   # (B, 384)
        return self.fcn(self.drop(z))


# ============================================================
# IMU_GateFusion : SharedLocalCNN → GCN branch + TCN branch
#                  → concat → fusion FC → 100 classes
# ============================================================
class IMU_GateFusion(nn.Module):
    """
    (B,200,30) → Shared LocalCNN (sensor-wise, stride=2) → (B,64,10,100)
        GCN branch : _SpatialGCNBlock x3 → node-attn pool → BiLSTM(128,64,bidir) → mean+max → (B,256)
        TCN branch : flatten C*V → Conv1d(640,64,1) → Conv1d x3 (k=7,9,11) → mean+max → (B,256)
    concat (B,512) → fusion FC (512→256→100)
    Always returns (main_logits, gcn_logits, tcn_logits).
    Loss = main + 0.2*gcn + 0.2*tcn
    """
    def __init__(self, in_channels=3, num_class=100, num_sensor=10,
                 graph_args=None, dropout=0.2):
        super().__init__()
        self.in_channels = in_channels
        self.num_sensor  = num_sensor

        # ── Graph ──
        if graph_args is None:
            graph_args = {'max_hop': 1, 'dilation': 1}
        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32)
        self.register_buffer('A', A)
        K = int(A.size(0))

        # ── Shared sensor-wise local CNN ──
        # (B, 3, 10, 200) → (B, 64, 10, 100)
        self.local_cnn = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=(1, 5), stride=(1, 2), padding=(0, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # ── GCN branch (5 layers) ──
        self.gcn_layers = nn.ModuleList([
            _SpatialGCNBlock(64,  64,  K=K, dropout=dropout),
            _SpatialGCNBlock(64,  64,  K=K, dropout=dropout),
            _SpatialGCNBlock(64,  96,  K=K, dropout=dropout),
            _SpatialGCNBlock(96,  96,  K=K, dropout=dropout),
            _SpatialGCNBlock(96,  128, K=K, dropout=dropout),
        ])
        self.edge_importance = nn.ParameterList([
            nn.Parameter(torch.ones(A.size())) for _ in self.gcn_layers
        ])

        self.node_attn  = nn.Linear(128, 1)
        self.gcn_bilstm = nn.LSTM(128, 64, batch_first=True, bidirectional=True)

        # ── TCN branch (5 layers, kernels 7,7,9,9,11) ──
        self.tcn_proj = nn.Conv1d(64 * num_sensor, 64, kernel_size=1)
        self.tcn = nn.Sequential(               # (B,64,100) → (B,128,50)
            nn.Conv1d(64,  128, kernel_size=7,  padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=7,  padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=9,  padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=9,  padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=11, stride=2, padding=5),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )
        # ── Pooling & norm ──  branch 출력: 64*2=128 → mean+max=256
        self.gcn_ln = nn.LayerNorm(256)
        self.tcn_ln = nn.LayerNorm(256)

        # ── Auxiliary classifiers ──
        self.gcn_aux = nn.Linear(256, num_class)
        self.tcn_aux = nn.Linear(256, num_class)

        # ── Branch gate: (B,512) → α_g, α_t ──
        # ── Fusion classifier ──
        self.fusion_fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_class),
        )

    def _mean_max(self, x):
        # x: (B, T, C) → (B, 2C)
        return torch.cat([x.mean(dim=1), x.max(dim=1).values], dim=1)

    def forward(self, x_IMU):
        B, T, D = x_IMU.shape

        # (B,200,30) → (B,3,10,200)
        x = x_IMU.view(B, T, self.num_sensor, self.in_channels)
        x = x.permute(0, 3, 2, 1).contiguous()

        # Shared local CNN → (B,64,10,100)
        x = self.local_cnn(x)

        # ── GCN branch ──
        xg = x.permute(0, 1, 3, 2).contiguous()       # (B,64,100,10) = (B,C,T,V)
        for gcn, imp in zip(self.gcn_layers, self.edge_importance):
            xg = gcn(xg, self.A * imp)                 # (B,C,100,10)

        # Node attention pooling over V=10
        xg_t = xg.permute(0, 2, 3, 1)                 # (B,T,V,C)
        score = torch.softmax(self.node_attn(xg_t), dim=2)    # (B,T,V,1)
        xg = (xg_t * score).sum(dim=2).permute(0, 2, 1)       # (B,C=128,T=100)

        xg, _ = self.gcn_bilstm(xg.permute(0, 2, 1))  # (B,100,128)
        h_g = self.gcn_ln(self._mean_max(xg))          # (B,256)

        # ── TCN branch ──
        xt = x.reshape(B, -1, x.shape[-1])      # (B, 640, 100) — flatten C×V
        xt = self.tcn_proj(xt)                   # (B, 64, 100)
        xt = self.tcn(xt)                        # (B,128,50)
        h_t = self.tcn_ln(self._mean_max(xt.permute(0, 2, 1)))    # (B,256)

        # ── Auxiliary logits ──
        gcn_logits = self.gcn_aux(h_g)
        tcn_logits = self.tcn_aux(h_t)

        fused = torch.cat([h_g, h_t], dim=1)   # (B,512)
        main_logits = self.fusion_fc(fused)

        return main_logits, gcn_logits, tcn_logits


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


# ============================================================
# IMU Conformer
# ============================================================
class _ConformerConvModule(nn.Module):
    def __init__(self, d_model, kernel_size, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.pw1  = nn.Linear(d_model, d_model * 2)
        self.dw   = nn.Conv1d(d_model, d_model, kernel_size,
                              padding=kernel_size // 2, groups=d_model)
        self.bn   = nn.BatchNorm1d(d_model)
        self.pw2  = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, d_model)
        x = self.norm(x)
        x = self.pw1(x)                     # (B, T, d_model*2)
        x = F.glu(x, dim=-1)               # (B, T, d_model)
        x = x.transpose(1, 2)              # (B, d_model, T)
        x = self.dw(x)
        x = self.bn(x)
        x = F.silu(x)
        x = x.transpose(1, 2)              # (B, T, d_model)
        x = self.pw2(x)
        return self.drop(x)


class _ConformerBlock(nn.Module):
    def __init__(self, d_model, nhead, ff_dim, kernel_size, dropout):
        super().__init__()
        self.norm_ff1  = nn.LayerNorm(d_model)
        self.ff1       = nn.Sequential(
            nn.Linear(d_model, ff_dim), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model), nn.Dropout(dropout),
        )
        self.norm_attn = nn.LayerNorm(d_model)
        self.attn      = nn.MultiheadAttention(d_model, nhead,
                                               dropout=dropout, batch_first=True)
        self.drop_attn = nn.Dropout(dropout)
        self.conv      = _ConformerConvModule(d_model, kernel_size, dropout)
        self.norm_ff2  = nn.LayerNorm(d_model)
        self.ff2       = nn.Sequential(
            nn.Linear(d_model, ff_dim), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model), nn.Dropout(dropout),
        )
        self.norm_out  = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + 0.5 * self.ff1(self.norm_ff1(x))
        x_n = self.norm_attn(x)
        x = x + self.drop_attn(self.attn(x_n, x_n, x_n)[0])
        x = x + self.conv(x)
        x = x + 0.5 * self.ff2(self.norm_ff2(x))
        return self.norm_out(x)


class IMU_Conformer(nn.Module):
    def __init__(self, num_classes=100, in_channels=30, d_model=128, nhead=4,
                 num_layers=4, ff_dim=512, conv_kernel_size=15, dropout=0.1): #conv_kernel_size 변경
        super().__init__()
        self.proj = nn.Linear(in_channels, d_model)
        self.pe   = nn.Parameter(torch.zeros(1, 200, d_model))
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            _ConformerBlock(d_model, nhead, ff_dim, conv_kernel_size, dropout)
            for _ in range(num_layers)
        ])
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x: (B, T, C) = (B, 200, 30)
        x = self.proj(x) + self.pe[:, :x.size(1)]
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = x.mean(dim=1)   # GAP over T
        return self.fc(x)


class IMU_ConformerBiLSTM(nn.Module):
    """
    Kwon et al. (2024) 구조:
    Linear → Conformer × 4 → BiLSTM × 2 → mean pool → FC
    """
    def __init__(self, num_classes=100, in_channels=30, d_model=128, nhead=4,
                 num_conformer_layers=4, ff_dim=384, conv_kernel_size=31,
                 lstm_hidden=128, num_lstm_layers=2, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(in_channels, d_model)
        self.pe   = nn.Parameter(torch.zeros(1, 200, d_model))
        self.drop = nn.Dropout(dropout)
        self.conformer_blocks = nn.ModuleList([
            _ConformerBlock(d_model, nhead, ff_dim, conv_kernel_size, dropout)
            for _ in range(num_conformer_layers)
        ])
        self.bilstm = nn.LSTM(
            input_size=d_model,
            hidden_size=lstm_hidden,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_lstm_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(lstm_hidden * 2, num_classes)

    def forward(self, x):
        # x: (B, T, C) = (B, 200, 30)
        x = self.proj(x) + self.pe[:, :x.size(1)]
        x = self.drop(x)
        for block in self.conformer_blocks:
            x = block(x)                 # (B, T, d_model)
        x, _ = self.bilstm(x)           # (B, T, 2*lstm_hidden)
        x = x.mean(dim=1)               # mean pool over T
        return self.fc(x)
