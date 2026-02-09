import gurobipy as gp
import numpy as np
from typing import List, Tuple, Dict, Set
from .base_block import AbstractBlock

class StableSetBlock(AbstractBlock):
    def __init__(self, block_id: int, num_nodes: int, density: float = 0.15,
                 seed: int = 42, weights: List[float] = None):
        super().__init__(block_id, name=f"StableSet_{block_id}")
        self.num_nodes = num_nodes
        self.edges = self._generate_graph(num_nodes, density, seed + block_id)
        self.edge_set: Set[Tuple[int, int]] = set(tuple(sorted(e)) for e in self.edges)

        rng = np.random.default_rng(seed + block_id)
        if weights is None:
            self.weights = list(rng.integers(1, 11, size=num_nodes))
        else:
            self.weights = weights

    def _generate_graph(self, n, p, seed):
        rng = np.random.default_rng(seed)
        edges = []
        all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        num_edges = int(np.ceil(p * len(all_pairs)))
        if num_edges > 0:
            indices = rng.choice(len(all_pairs), size=num_edges, replace=False)
            edges = [all_pairs[i] for i in indices]
        return edges

    def inherit_conflicts(self, neighbor: 'StableSetBlock',
                          my_indices: List[int], neighbor_indices: List[int]) -> bool:
        neigh_to_me = {n_idx: m_idx for m_idx, n_idx in zip(my_indices, neighbor_indices)}
        changed = False
        for u_neigh, v_neigh in neighbor.edges:
            if u_neigh in neigh_to_me and v_neigh in neigh_to_me:
                u_me = neigh_to_me[u_neigh]
                v_me = neigh_to_me[v_neigh]
                new_edge = tuple(sorted((u_me, v_me)))
                if new_edge not in self.edge_set:
                    self.edges.append(new_edge)
                    self.edge_set.add(new_edge)
                    changed = True
        return changed

    def build_model(self, parent_model: gp.Model = None, prefix: str = None):
        if parent_model:
            self.model = parent_model
            pfx = prefix if prefix else f"B{self.block_id}"
        else:
            self.model = gp.Model(self.name)
            self.model.Params.OutputFlag = 0
            pfx = "x"

        self.vars = {}
        created_vars = self.model.addVars(self.num_nodes, vtype=gp.GRB.BINARY, name=f"{pfx}")
        for i in range(self.num_nodes):
            self.vars[i] = created_vars[i]

        for u, v in self.edges:
            self.model.addConstr(self.vars[u] + self.vars[v] <= 1, name=f"{pfx}_edge_{u}_{v}")

        self.local_objective_expr = gp.LinExpr()
        for i in range(self.num_nodes):
            self.local_objective_expr.add(self.vars[i], self.weights[i])

        if parent_model is None:
            self.model.setObjective(self.local_objective_expr, gp.GRB.MAXIMIZE)
            self.model.update()

    def get_vars_by_index(self, indices: List[int]) -> List[gp.Var]:
        return [self.vars[i] for i in indices]