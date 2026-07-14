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

    # strong top-10 + weak bottom-10 Pearson (single partition)
    # def get_edge(self):
    #         self.num_node = 10
    #         self_link = [(i, i) for i in range(self.num_node)]
    #         neighbor_link = [(6,7),(4,8),(8,9),(3,4),(7,8),(7,9),(4,9),(4,7),(2,7),(3,6),
    #                          (0,8),(0,9),(0,4),(0,2),(1,8),(1,9),(0,5),(0,7),(0,3),(4,5)]
    #         self.edge = self_link + neighbor_link

    #right (top-20 Pearson)
    def get_edge(self):
            self.num_node = 10
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [[6,7], [4,8],[3,4], [8,9],[7,8],
                             [4,9],[7,9], [3,6], [4,7], [2,7],
                             [6,8], [1,5], [2,3], [4,6],[5,7],
                             [6,9], [1,3], [2,4], [3,7],[0,1]]
            self.edge = self_link + neighbor_link

    #right (class separability) [0,2],[2,6]
    # def get_edge(self):
    #         self.num_node = 10
    #         self_link = [(i, i) for i in range(self.num_node)]
    #         neighbor_link = [[1,8],[6,8],[5,8],[0,8],[3,5],
    #                         [0,9],[0,7],[1,3],[0,3],[1,4],
    #                         [4,5],[0,6],[1,9],[0,2],[2,6]]
    #         self.edge = self_link + neighbor_link

    # #left(class separability)
    # def get_edge(self):
    #         self.num_node = 10
    #         self_link = [(i, i) for i in range(self.num_node)]
    #         neighbor_link = [[0,6],[0,3],[6,8],[0,7],[0,2],
    #                          [0,8],[0,4],[4,6],[7,8],[0,9],
    #                          [1,6],[0,1],[1,3],[0,5],[5,6]]
    #         self.edge = self_link + neighbor_link        


    # # #left
    # def get_edge(self):
    #         self.num_node = 10
    #         self_link = [(i, i) for i in range(self.num_node)]
    #         neighbor_link = [[6,7], [4,8],[7,8], [3,4],[2,7],
    #                          [5,7],[4,7], [1,3], [1,5], [7,9],
    #                          [8,9], [2,4], [3,6], [6,8],[1,2],
    #                          [4,9], [2,6], [3,8], [4,6],[0,6]]
    #         self.edge = self_link + neighbor_link

    #fc 기반 edge importance top20개의 edge
    # def get_edge(self):
    #     self.num_node = 10
    #     self_link = [(i, i) for i in range(self.num_node)]
    #     neighbor_link = [
    #                     [0,6], [1,6], [5,6], [3,6], [6,9], 
    #                     [2,6], [6,8], [0,1], [0,5], [4,6], 
    #                     [1,5], [0,2], [6,7], [0,8], [0,3], 
    #                     [0,4], [1,8], [0,9], [5,8], [1,2]]
    #     self.edge = self_link + neighbor_link

    # fully connected graph — 모든 sensor 가 모든 sensor 와 연결 (free mixing)
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


def compute_fold_A(X_train_np, y_train_np, top_k=20, max_hop=1):

    import torch
    edges_all = [(i, j) for i in range(10) for j in range(i + 1, 10)]
    norm = np.linalg.norm(X_train_np.reshape(-1, 200, 10, 3), axis=-1) #유클리드 거리로 3축 하나로 합치기
    N = norm.shape[0]
    classes = np.unique(y_train_np)

    ec = np.zeros((N, 45))
    for idx, (i, j) in enumerate(edges_all):
        si = norm[:, :, i]; sj = norm[:, :, j] #시간축 평균 제거
        si_c = si - si.mean(1, keepdims=True)
        sj_c = sj - sj.mean(1, keepdims=True)
        r = np.clip((si_c * sj_c).sum(1) / #PEARSON R (한번에 처리)
                    (np.sqrt((si_c**2).sum(1) * (sj_c**2).sum(1)) + 1e-8),
                    -0.9999, 0.9999)
        ec[:, idx] = np.arctanh(r) #fisher z-transform

    gm = ec.mean(0); bw = np.zeros(45); wt = np.zeros(45)
    for c in classes:
        m = (y_train_np == c); cm = ec[m].mean(0)
        bw += m.sum() * (cm - gm) ** 2 #between 계산
        wt += ((ec[m] - cm) ** 2).sum(0) #within 계산
    F = (bw / (len(classes) - 1)) / (wt / (N - len(classes)) + 1e-10) # 엣지별 F값
    ranked = [edges_all[k] for k in np.argsort(-F)] #F 높은 순으로 정렬
    F_map = {edges_all[k]: F[k] for k in range(45)}
    selected = list(ranked[:top_k]) #top 20 선택

    # 고립 노드 보완: 고립 노드가 있으면 해당 노드의 최고 F 엣지로 20번째 교체
    num_node = 10
    connected = set(n for e in selected for n in e)
    for node in range(num_node):
        if node not in connected:
            candidates = [(i, j) for (i, j) in edges_all if (i == node or j == node) and (i, j) not in selected]
            if candidates:
                best = max(candidates, key=lambda e: F_map[e])
                selected[-1] = best  # 20번째(최하위) 엣지와 교체
                connected = set(n for e in selected for n in e)

    # Graph 클래스와 동일한 방식으로 adjacency 구성 (max_hop 채널 수 지원)
    num_node = 10
    all_links = [(i, i) for i in range(num_node)] + selected #self-loop +선택된 엣지
    hop_dis   = get_hop_distance(num_node, all_links, max_hop=max_hop)

    # # log 압축 가중치 버전
    w_min = 0.3
    f_vals = np.array([F_map.get(e, F_map.get((e[1], e[0]), 0.0)) for e in selected])
    log_f = np.log(f_vals + 1)
    lo, hi = log_f.min(), log_f.max()
    w_norm = (log_f - lo) / (hi - lo + 1e-8)
    w_edge = w_min + (1 - w_min) * w_norm

    adj_full = np.zeros((num_node, num_node))
    for k, (i, j) in enumerate(selected):
        adj_full[i, j] = w_edge[k]
        adj_full[j, i] = w_edge[k]
    for i in range(num_node):
        adj_full[i, i] = 1.0
    norm_adj = normalize_digraph(adj_full)

    A = np.zeros((max_hop + 1, num_node, num_node))
    for hop in range(max_hop + 1):
        A[hop][hop_dis == hop] = norm_adj[hop_dis == hop]

    return torch.tensor(A, dtype=torch.float32)



