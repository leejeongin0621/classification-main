import numpy as np
import torch

class Graph():

    def __init__(self,
                 max_hop=1,
                 dilation=1):
        self.max_hop = max_hop
        self.dilation = dilation

        self.get_edge()
        self.hop_dis = get_hop_distance(
            self.num_node, self.edge, max_hop=max_hop)
        self.get_adjacency()

    def __str__(self):
        return self.A
    
    # def get_edge(self):
    #     self.num_node = 10
    #     self_link = [(i, i) for i in range(self.num_node)]
    #     neighbor_link = [(0,1), (0,6), (0,5),
    #                         (1,2), (1,3), (1,6),
    #                         (2,3), (2,4),
    #                         (3,4), (3,6), (3,8), (3,7),
    #                         (4,8), (4,9),
    #                         (5,6), (5,7),
    #                         (6,7), 
    #                         (7,8), (7,9),
    #                         (8,9)]
    #     self.edge = self_link + neighbor_link

    # total
    # def get_edge(self):
    #         self.num_node = 10
    #         self_link = [(i, i) for i in range(self.num_node)]
    #         neighbor_link = [(8,9), (7,8), (6,8), (5,7),
    #                         (2,3), (1,2), (7,9), (2,7),
    #                         (3,4), (6,9), (6,7), (4,9),
    #                         (4,9),(2,6), (2,9), (1,7),
    #                         (2,8), (2,4), (2,5),(3,8),(0,9)]
    #         self.edge = self_link + neighbor_link

    #right
    def get_edge(self):
            self.num_node = 10
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [[6,7], [4,8],[3,4], [8,9],[7,8],
                             [4,9],[7,9], [3,6], [4,7], [2,7],
                             [6,8], [1,5], [2,3], [4,6],[5,7],
                             [6,9], [1,3], [2,4], [3,7],[0,1]]
            self.edge = self_link + neighbor_link

    # #left
    # def get_edge(self):
    #         self.num_node = 10
    #         self_link = [(i, i) for i in range(self.num_node)]
    #         neighbor_link = [[6,7], [4,8],[7,8], [3,4],[2,7],
    #                          [5,7],[4,7], [1,3], [1,5], [7,9],
    #                          [8,9], [2,4], [3,6], [6,8],[1,2],
    #                          [4,9], [2,6], [3,8], [4,6],[0,6]]
    #         self.edge = self_link + neighbor_link

    # # fully connected graph — 모든 sensor 가 모든 sensor 와 연결 (free mixing)
    # def get_edge(self):
    #         self.num_node = 10
    #         self_link = [(i, i) for i in range(self.num_node)]
    #         neighbor_link = []
    #         for i in range(self.num_node):
    #             for j in range(i + 1, self.num_node):
    #                 neighbor_link.append((i, j))
    #         self.edge = self_link + neighbor_link

    def get_adjacency(self):
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        
        normalize_adjacency = normalize_digraph(adjacency)

        A = np.zeros((len(valid_hop), self.num_node, self.num_node))
        for i, hop in enumerate(valid_hop):
            A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis == hop]
        self.A = A

def get_hop_distance(num_node, edge, max_hop):
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1

    # compute hop steps
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis


def normalize_digraph(A):
    Dl = np.sum(A, 0) #열 방향 합
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-0.5) # D^-1/2
    DAD = np.dot(np.dot(Dn,A), Dn) # D^-1/2 * A * D^-1/2
    return DAD

class Graph_FC():
    """
    Fully-connected graph with configurable num_node.
    모든 node 가 다른 모든 node 와 연결됨 → DAE (5 sensor) 처럼 작은 그래프에 적합.
    """
    def __init__(self, num_node=5, max_hop=1, dilation=1):
        self.max_hop = max_hop
        self.dilation = dilation
        self.num_node = num_node
        self.get_edge()
        self.hop_dis = get_hop_distance(
            self.num_node, self.edge, max_hop=max_hop)
        self.get_adjacency()

    def __str__(self):
        return self.A

    def get_edge(self):
        self_link = [(i, i) for i in range(self.num_node)]
        neighbor_link = [(i, j)
                         for i in range(self.num_node)
                         for j in range(i + 1, self.num_node)]
        self.edge = self_link + neighbor_link

    def get_adjacency(self):
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1

        normalize_adjacency = normalize_digraph(adjacency)

        A = np.zeros((len(valid_hop), self.num_node, self.num_node))
        for i, hop in enumerate(valid_hop):
            A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis == hop]
        self.A = A


def make_laplacian_pe(A, k=2, use_abs=True):
    """
    A: (V, V) adjacency matrix
    k: 사용할 eigenvector 개수
    return: (V, k)
    """
    A = np.asarray(A, dtype=np.float32)

    # degree matrix
    D = np.diag(A.sum(axis=1))

    # graph Laplacian
    L = D - A

    # eigen decomposition
    eigvals, eigvecs = np.linalg.eigh(L)

    # 첫 번째 eigenvector 제외하고 두 번째부터 사용
    pe = eigvecs[:, 1:k+1]

    # sign ambiguity 완화
    if use_abs:
        pe = np.abs(pe)

    return pe.astype(np.float32)
