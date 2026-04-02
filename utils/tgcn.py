# The based unit of graph convolutional networks.

import math

import torch
import torch.nn as nn
import numpy as np


class ConvTemporalGraphical(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True):
        super().__init__()

        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias)

    def forward(self, x, A):
        assert A.size(0) == self.kernel_size

        x = self.conv(x)

        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc//self.kernel_size, t, v)
        x = torch.einsum('nkctv,kvw->nctw', (x, A))

        return x.contiguous(), A

class DynamicGraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, num_node=10):
        super().__init__()
        self.num_node = num_node
        
        # 동적 adjacency 생성
        self.theta = nn.Conv2d(in_channels, in_channels, 1)
        self.phi = nn.Conv2d(in_channels, in_channels, 1)
        
        # 기존 conv
        self.conv = nn.Conv2d(in_channels, out_channels * 2, 1)
        self.out_channels = out_channels

    def forward(self, x, A):
        # x: (B, C, T, V)
        
        # 동적 A 생성
        theta = self.theta(x) # (B, C, V)
        phi = self.phi(x)     # (B, C, V)
        A_dynamic = torch.softmax(
            torch.einsum('bcv,bcw->bvw', theta, phi) / (theta.size(1) ** 0.5),
            dim=-1
        )  # (B, V, V)
        
        # 고정 A + 동적 A 합치기
        x_conv = self.conv(x)
        n, kc, t, v = x_conv.size()
        x_conv = x_conv.view(n, 2, kc//2, t, v)
        
        # partition 0: 고정 A
        x0 = torch.einsum('nctv,vw->nctw', x_conv[:,0], A[0])
        # partition 1: 동적 A
        x1 = torch.einsum('nctv,nvw->nctw', x_conv[:,1], A_dynamic)
        
        x = x0 + x1
        return x.contiguous(), A


#learnable adjacency
class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, coff_embedding=8):
        super(unit_gcn, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        # self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
        # nn.init.constant_(self.PA, 1e-6)
        self.register_buffer('A', torch.from_numpy(A.astype(np.float32)))
        #self.num_subset = num_subset

        self.conv_a = nn.Conv2d(in_channels, inter_channels, 1)
        self.conv_b = nn.Conv2d(in_channels, inter_channels, 1)
        self.conv_d = nn.Conv2d(in_channels, out_channels, 1)

        # for i in range(self.num_subset): #branch=num_subset=2, subset 여러개일 때 각 branch마다 conv layer 하나씩, conv_a,b는 attention 계산용, conv_d는 최종 output 계산용
        #     self.conv_a.append(nn.Conv2d(in_channels, inter_channels, 1))
        #     self.conv_b.append(nn.Conv2d(in_channels, inter_channels, 1))
        #     self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
        # for i in range(self.num_subset):
        #     conv_branch_init(self.conv_d[i], self.num_subset)

    def forward(self, x):
        N, C, T, V = x.size()
        A = self.A.to(x.device)
        #A = A + self.PA

        #y = None
        #for i in range(self.num_subset):
        A1 = self.conv_a(x).permute(0, 3, 1, 2).contiguous().view(N, V, self.inter_c * T) # (N, inter_c, T, V)-> (N, V, inter_c*T)
        A2 = self.conv_b(x).view(N, self.inter_c * T, V) # (N, inter_c*T, V)
        # A1 = torch.softmax(torch.matmul(A1, A2) / A1.size(-1), dim=-1)  # N V V ← 동적 attention
        A_dyn = torch.softmax(torch.matmul(A1, A2) / A1.size(-1), dim=-1)
        A1 = A[0] + A_dyn
        #A1 = A1 + A[0] # 동적 attention + 고정 그래프[i번째 partition]
        A2 = x.view(N, C * T, V)
        z = self.conv_d(torch.matmul(A2, A1).view(N, C, T, V))  # (N, C*T, V) × (N, V, V) → (N, C*T, V)
        #y = z + y if y is not None else z
        return z.contiguous()

def conv_branch_init(conv, branches): #gcn 초기화용, branches 수만큼 나눠서 초기화, subset 여러개일때 각 branch 균등하게 시작
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    nn.init.constant_(conv.bias, 0)

def conv_init(conv): #conv layer 초기화
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    nn.init.constant_(conv.bias, 0)
