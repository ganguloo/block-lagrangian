import gurobipy as gp
from typing import List, Tuple
from .base_block import AbstractBlock

class LinearOrderingBlock(AbstractBlock):
    def __init__(self, block_id: int, global_nodes: List[int], weights_matrix: List[List[float]]):
        """
        block_id: ID of the block
        global_nodes: List of global node indices belonging to this block.
        weights_matrix: Global preference matrix W[i][j].
        """
        super().__init__(block_id, name=f"LOP_{block_id}")
        self.global_nodes = sorted(list(set(global_nodes))) # Ensure unique and sorted
        self.weights = weights_matrix
        self.n_local = len(self.global_nodes)

    def build_model(self, parent_model: gp.Model = None, prefix: str = None):
        if parent_model:
            self.model = parent_model
            pfx = prefix if prefix else f"B{self.block_id}"
        else:
            self.model = gp.Model(self.name)
            self.model.Params.OutputFlag = 0
            pfx = "x"

        self.vars = {} 
        
        # 1. Variables x_uv = 1 if u precedes v
        # Create variables for ALL distinct pairs (u, v) in global_nodes
        for u in self.global_nodes:
            for v in self.global_nodes:
                if u != v:
                    var = self.model.addVar(vtype=gp.GRB.BINARY, name=f"{pfx}_{u}_{v}")
                    self.vars[(u, v)] = var

        # 2. Constraints
        
        # Antisymmetry: x_uv + x_vu = 1
        # Iterate only u < v to add constraint once per pair
        for i in range(self.n_local):
            for j in range(i + 1, self.n_local):
                u, v = self.global_nodes[i], self.global_nodes[j]
                self.model.addConstr(
                    self.vars[u, v] + self.vars[v, u] == 1, 
                    name=f"anti_{u}_{v}"
                )

        # 3-Cycle Elimination (Transitivity): x_uv + x_vw + x_wu <= 2
        # Equivalent to: if u->v and v->w then u->w
        for i in range(self.n_local):
            for j in range(self.n_local):
                if i == j: continue
                for k in range(self.n_local):
                    if k == i or k == j: continue
                    
                    # To avoid O(n^3) redundancy, we can just add for i < j < k
                    # and rely on the fact that antisymmetry covers permutations
                    if i < j and j < k:
                        u, v, w = self.global_nodes[i], self.global_nodes[j], self.global_nodes[k]
                        
                        # Cycle u->v->w->u forbidden
                        self.model.addConstr(
                            self.vars[u, v] + self.vars[v, w] + self.vars[w, u] <= 2,
                            name=f"cycle_{u}_{v}_{w}"
                        )
                        # Also forbid reverse cycle u<-v<-w<-u (implied but good for LP relaxation)
                        self.model.addConstr(
                            self.vars[u, w] + self.vars[w, v] + self.vars[v, u] <= 2,
                            name=f"cycle_rev_{u}_{v}_{w}"
                        )

        # 3. Objective: Maximize sum w_uv * x_uv
        self.local_objective_expr = gp.LinExpr()
        for (u, v), var in self.vars.items():
            w_val = self.weights[u][v]
            if abs(w_val) > 1e-6:
                self.local_objective_expr.add(var, w_val)

        if parent_model is None:
            self.model.setObjective(self.local_objective_expr, gp.GRB.MAXIMIZE)
            self.model.update()

    def get_vars_by_index(self, indices: List[Tuple[int, int]]) -> List[gp.Var]:
        """
        indices: List of tuples (u, v) representing arcs.
        Returns the corresponding Gurobi variables.
        """
        return [self.vars[uv] for uv in indices]