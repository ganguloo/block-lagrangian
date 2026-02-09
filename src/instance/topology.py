
from dataclasses import dataclass
from typing import List, Dict, Tuple

@dataclass
class EdgeInfo:
    u: int
    v: int
    vars_u: List[int]
    vars_v: List[int]

class TopologyManager:
    def __init__(self, num_blocks: int):
        self.num_blocks = num_blocks
        self.edges: Dict[Tuple[int, int], EdgeInfo] = {}
        self.adj: Dict[int, List[int]] = {i: [] for i in range(num_blocks)}

    def add_coupling(self, u: int, v: int, indices_u: List[int], indices_v: List[int]):
        key = tuple(sorted((u, v)))
        self.edges[key] = EdgeInfo(u, v, indices_u, indices_v)
        self.adj[u].append(v)
        self.adj[v].append(u)

    def get_edge(self, u: int, v: int) -> EdgeInfo:
        return self.edges.get(tuple(sorted((u, v))))

    def get_neighbors(self, u: int) -> List[int]:
        return self.adj[u]

    def create_star(self, center: int, n_nodes: int, coupling_size: int):
        indices = list(range(coupling_size))
        for i in range(self.num_blocks):
            if i != center:
                self.add_coupling(center, i, indices, indices)

    def create_path(self, n_nodes: int, coupling_size: int):
        for i in range(self.num_blocks - 1):
            indices_u = list(range(n_nodes - coupling_size, n_nodes))
            indices_v = list(range(coupling_size))
            self.add_coupling(i, i+1, indices_u, indices_v)

    def create_bintree(self, n_nodes: int, coupling_size: int):
        if n_nodes < coupling_size:
             raise ValueError(f"n_nodes ({n_nodes}) debe ser al menos coupling ({coupling_size})")

        for i in range(self.num_blocks):
            left = 2 * i + 1
            right = 2 * i + 2

            # Logic: Stochastic Multistage Style
            # Parent outputs same state to both children (Non-anticipativity implicit)
            # Input: [0, C), Output: [N-C, N)
            indices_u = list(range(n_nodes - coupling_size, n_nodes)) # Output vars
            indices_v = list(range(coupling_size)) # Input vars

            if left < self.num_blocks:
                self.add_coupling(i, left, indices_u, indices_v)

            if right < self.num_blocks:
                self.add_coupling(i, right, indices_u, indices_v)
