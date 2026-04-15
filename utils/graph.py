import numpy as np

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

    # def get_edge(self):
    #         self.num_node = 10
    #         self_link = [(i, i) for i in range(self.num_node)]
    #         neighbor_link = [(8,9), (7,8), (6,8), (5,7),
    #                         (2,3), (1,2), (7,9), (2,7),
    #                         (3,4), (6,9), (6,7), (4,9),
    #                         (4,9),(2,6), (2,9), (1,7),
    #                         (2,8), (2,4), (2,5),(3,8),(0,9)]
    #         self.edge = self_link + neighbor_link

    def get_edge(self):
            self.num_node = 10
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [(4,7),(1,5),(3,9),(2,8),(5,7)
                             ,(1,4),(0,6),(2,9),(3,7),(6,8)
                             ,(0,5),(1,7),(4,9),(2,6),(3,8)
                             ,(0,7),(5,9),(1,6),(4,8),(2,7)]
            self.edge = self_link + neighbor_link

    #fully connected graph
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
