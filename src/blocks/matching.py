
import gurobipy as gp
import numpy as np
from typing import List, Tuple, Dict, Set
from .base_block import AbstractBlock

class MatchingBlock(AbstractBlock):
    def __init__(self, block_id: int, num_nodes: int, num_edges: int,
                 seed: int = 42, weights: List[float] = None):
        super().__init__(block_id, name=f"Matching_{block_id}")
        self.num_nodes = num_nodes
        # num_edges is the REQUESTED number, but self.edges contains the ACTUAL list
        self.edges, self.weights = self._generate_graph(num_nodes, num_edges, seed + block_id, weights)
        self.num_edges = len(self.edges) # Actual size

    def _generate_graph(self, n, m, seed, weights):
        rng = np.random.default_rng(seed)
        all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        actual_m = min(m, len(all_pairs))
        indices = rng.choice(len(all_pairs), size=actual_m, replace=False)
        edges = [all_pairs[i] for i in indices]
        if weights is None:
            w = list(rng.integers(1, 11, size=actual_m))
        else:
            w = weights
        return edges, w

    def build_model(self, parent_model: gp.Model = None, prefix: str = None):
        if parent_model:
            self.model = parent_model
            pfx = prefix if prefix else f"B{self.block_id}"
        else:
            self.model = gp.Model(self.name)
            self.model.Params.OutputFlag = 0
            pfx = "x"

        self.vars = {}
        created_vars = self.model.addVars(len(self.edges), vtype=gp.GRB.BINARY, name=f"{pfx}")
        for i in range(len(self.edges)):
            self.vars[i] = created_vars[i]

        node_incidences = {i: [] for i in range(self.num_nodes)}
        for idx, (u, v) in enumerate(self.edges):
            node_incidences[u].append(idx)
            node_incidences[v].append(idx)

        for i in range(self.num_nodes):
            if node_incidences[i]:
                self.model.addConstr(
                    gp.quicksum(self.vars[e_idx] for e_idx in node_incidences[i]) <= 1,
                    name=f"{pfx}_deg_{i}"
                )

        self.local_objective_expr = gp.LinExpr()
        for i in range(len(self.edges)):
            self.local_objective_expr.add(self.vars[i], self.weights[i])

        if parent_model is None:
            self.model.setObjective(self.local_objective_expr, gp.GRB.MAXIMIZE)
            self.model.update()

    def get_vars_by_index(self, indices: List[int]) -> List[gp.Var]:
        return [self.vars[i] for i in indices]
