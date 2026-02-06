
import gurobipy as gp
import time
from typing import List, Dict, Any
from ..instance.topology import TopologyManager
from ..blocks.base_block import AbstractBlock

class MonolithicSolver:
    def __init__(self, topology: TopologyManager, blocks: List[AbstractBlock]):
        self.topology = topology
        self.blocks = blocks
        self.model = gp.Model("MonolithicProblem")
        self.model.Params.OutputFlag = 1

    def build_and_solve(self, time_limit=None) -> Dict[str, Any]:
        total_obj = gp.LinExpr()
        for block in self.blocks:
            block.build_model(parent_model=self.model)
            if hasattr(block, 'local_objective_expr'):
                total_obj.add(block.local_objective_expr)

        for (u, v), edge_info in self.topology.edges.items():
            block_u = next(b for b in self.blocks if b.block_id == u)
            block_v = next(b for b in self.blocks if b.block_id == v)
            vars_u = block_u.get_vars_by_index(edge_info.vars_u)
            vars_v = block_v.get_vars_by_index(edge_info.vars_v)
            for i, (vu, vv) in enumerate(zip(vars_u, vars_v)):
                self.model.addConstr(vu == vv, name=f"link_{u}_{v}_{i}")

        self.model.setObjective(total_obj, gp.GRB.MAXIMIZE)
        self.model.update()

        metrics = {
            "root_lp_val": None,
            "root_lp_presolved_val": None,
            "primal_bound": None,
            "dual_bound": None,
            "gap": None,
            "node_count": 0,
            "status": "Unknown",
            "total_time": 0.0
        }

        try:
            m_copy = self.model.copy()
            m_relax = m_copy.relax()
            m_relax.Params.OutputFlag = 0
            m_relax.optimize()
            if m_relax.Status == gp.GRB.OPTIMAL:
                metrics["root_lp_val"] = m_relax.ObjVal
        except: pass

        try:
            m_pre = self.model.presolve()
            if m_pre:
                m_pre_relax = m_pre.relax()
                m_pre_relax.Params.OutputFlag = 0
                m_pre_relax.optimize()
                if m_pre_relax.Status == gp.GRB.OPTIMAL:
                    metrics["root_lp_presolved_val"] = m_pre_relax.ObjVal
        except: pass

        if time_limit:
            self.model.Params.TimeLimit = time_limit

        start_t = time.time()
        self.model.optimize()
        end_t = time.time()

        metrics["total_time"] = end_t - start_t
        metrics["node_count"] = self.model.NodeCount

        if self.model.SolCount > 0:
            metrics["primal_bound"] = self.model.ObjVal
            metrics["dual_bound"] = self.model.ObjBound
            metrics["gap"] = self.model.MIPGap

        if self.model.Status == gp.GRB.OPTIMAL: metrics["status"] = "Optimal"
        elif self.model.Status == gp.GRB.TIME_LIMIT: metrics["status"] = "TimeLimit"
        else: metrics["status"] = f"Code_{self.model.Status}"

        return metrics

    def get_block_solution(self, block_id: int) -> List[int]:
        if self.model.SolCount == 0: return []
        block = next(b for b in self.blocks if b.block_id == block_id)
        vals = []
        for idx in sorted(block.vars.keys()):
            vals.append(int(round(block.vars[idx].X)))
        return vals
