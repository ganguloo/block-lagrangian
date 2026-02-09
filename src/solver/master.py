import gurobipy as gp
from typing import Dict, Tuple, Any, Set

class MasterProblem:
    def __init__(self, num_blocks: int):
        self.model = gp.Model("RMP")
        self.model.Params.OutputFlag = 0
        self.num_blocks = num_blocks
        self.lambda_vars: Dict[int, Dict[Tuple, gp.Var]] = {i: {} for i in range(num_blocks)}
        self.column_registry: Dict[int, Set[int]] = {i: set() for i in range(num_blocks)}
        self.ctr_convexity = {}
        self.ctr_linear = {}
        self.ctr_cuts = {}
        self._init_model()

    def _init_model(self):
        for i in range(self.num_blocks):
            self.ctr_convexity[i] = self.model.addConstr(gp.LinExpr() == 1, name=f"cvx_{i}")
        self.model.setAttr("ModelSense", gp.GRB.MAXIMIZE)

    def register_linear_constraints(self, topology):
        for (u, v), edge in topology.edges.items():
            size = len(edge.vars_u)
            for k in range(size):
                if (u, v, k) not in self.ctr_linear:
                    self.ctr_linear[u, v, k] = self.model.addConstr(gp.LinExpr() == 0.0, name=f"lin_{u}_{v}_{k}")

    def _compute_column_hash(self, obj_val, x_boundary, w_sigs):
        x_key = tuple(sorted([(nid, tuple(val)) for nid, val in x_boundary.items()]))
        w_key = tuple(sorted([(nid, sig) for nid, sig in w_sigs.items()]))
        return hash((round(obj_val, 6), x_key, w_key))

    def add_column(self, block_id: int, obj_val: float,
                   x_boundary: Dict[int, list],
                   w_sigs: Dict[int, Tuple],
                   active_cuts_by_edge: Dict,
                   strategy) -> bool:
        col_hash = self._compute_column_hash(obj_val, x_boundary, w_sigs)
        if col_hash in self.column_registry[block_id]:
            return False
        self.column_registry[block_id].add(col_hash)

        col = gp.Column()
        col.addTerms(1.0, self.ctr_convexity[block_id])
        for nid, vals in x_boundary.items():
            u, v = sorted((block_id, nid))
            is_u = (block_id == u)
            factor = 1.0 if is_u else -1.0
            for k, val in enumerate(vals):
                if val != 0 and (u,v,k) in self.ctr_linear:
                    col.addTerms(factor * val, self.ctr_linear[u,v,k])

        for nid, col_w_sig in w_sigs.items():
            u, v = sorted((block_id, nid))
            cuts = active_cuts_by_edge.get((u, v), [])
            for cut_id, cut_sig in cuts:
                if cut_id in self.ctr_cuts:
                    w_val = strategy.evaluate_cut(col_w_sig, cut_sig)
                    if abs(w_val) > 1e-6:
                        coeff = w_val * (1.0 if block_id == u else -1.0)
                        col.addTerms(coeff, self.ctr_cuts[cut_id])

        full_sig = (round(obj_val, 6), tuple(sorted([(k,tuple(v)) for k,v in x_boundary.items()])), tuple(sorted([(k,v) for k,v in w_sigs.items()])))
        var = self.model.addVar(obj=obj_val, column=col, lb=0.0)
        self.lambda_vars[block_id][full_sig] = var
        return True

    def get_duals(self):
        pi = {k: c.Pi for k, c in self.ctr_linear.items()}
        alpha = {i: c.Pi for i, c in self.ctr_convexity.items()}
        mu = {k: c.Pi for k, c in self.ctr_cuts.items()}
        return alpha, pi, mu

    def solve(self):
        self.model.optimize()

    def switch_to_binary(self):
        for cols in self.lambda_vars.values():
            for var in cols.values():
                var.VType = gp.GRB.BINARY
        self.model.update()