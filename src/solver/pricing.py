
import gurobipy as gp
from typing import Dict, Tuple, List, Any

class PricingSolver:
    def __init__(self, block, strategy, topology):
        self.block = block
        self.strategy = strategy
        self.topology = topology
        self.rebuild_model()

    def rebuild_model(self):
        self.block.build_model()
        self.model = self.block.model
        self.model.Params.NonConvex = 2
        self.model.Params.Threads = 2
        self.boundary_vars = {}
        for nid in self.topology.get_neighbors(self.block.block_id):
            u, v = sorted((self.block.block_id, nid))
            edge = self.topology.get_edge(u, v)
            indices = edge.vars_u if self.block.block_id == u else edge.vars_v
            self.boundary_vars[nid] = self.block.get_vars_by_index(indices)

    def solve(self, alpha, pi, mu, active_cuts):
        obj = self.block.local_objective_expr.copy()
        obj.addConstant(-alpha)

        for nid, vars_list in self.boundary_vars.items():
            u, v = sorted((self.block.block_id, nid))
            is_u = (self.block.block_id == u)
            factor = -1.0 if is_u else 1.0
            for k, var in enumerate(vars_list):
                if (u,v,k) in pi:
                    obj.add(var, factor * pi[u,v,k])

        penalty_inputs = []
        for nid, vars_list in self.boundary_vars.items():
            u, v = sorted((self.block.block_id, nid))
            cuts = active_cuts.get((u, v), [])
            is_u = (self.block.block_id == u)
            for cut_id, sig in cuts:
                if cut_id in mu:
                    real_factor = -1.0 if is_u else 1.0
                    penalty_inputs.append((cut_id, sig, real_factor))
            if penalty_inputs:
                pen = self.strategy.apply_pricing_penalty(
                    self.model, vars_list,
                    [(c, s, f) for c,s,f in penalty_inputs],
                    mu
                )
                obj = obj + pen
            penalty_inputs = []

        self.model.setObjective(obj, gp.GRB.MAXIMIZE)
        self.model.optimize()

        if self.model.Status != gp.GRB.OPTIMAL: return None

        x_bounds = {}
        w_sigs = {}
        for nid, vars_list in self.boundary_vars.items():
            vals = [int(round(v.X)) for v in vars_list]
            x_bounds[nid] = vals
            w_sigs[nid] = self.strategy.get_w_signature(vals)

        return self.model.ObjVal, self.block.local_objective_expr.getValue(), x_bounds, w_sigs
