
import time
import gurobipy as gp
from typing import List, Dict, Any
from ..blocks.base_block import AbstractBlock
from ..instance.topology import TopologyManager

class IntegerLShapedSolver:
    def __init__(self, topology: TopologyManager, blocks: List[AbstractBlock]):
        self.topology = topology
        self.blocks = blocks
        self.center_block = blocks[0]
        self.leaf_blocks = blocks[1:]

        self.time_start = 0.0
        self.time_limit = float('inf')
        self.global_upper_bound_U = 0.0

        self.leaf_models_lp = []
        self.leaf_models_mip = []
        self.leaf_link_constrs_lp = []
        self.leaf_link_constrs_mip = []

        self._prepare_leaves()
        self._build_master()

    def _prepare_leaves(self):
        total_max_possible = 0.0

        for leaf in self.leaf_blocks:
            m_relaxed = gp.Model(f"Leaf_{leaf.block_id}_Relaxed")
            m_relaxed.Params.OutputFlag = 0
            leaf.build_model(parent_model=m_relaxed)
            m_relaxed.setObjective(leaf.local_objective_expr, gp.GRB.MAXIMIZE)
            m_relaxed.update()
            m_relaxed.optimize()
            if m_relaxed.Status == gp.GRB.OPTIMAL:
                total_max_possible += m_relaxed.ObjVal

            m_temp = gp.Model()
            leaf.build_model(parent_model=m_temp)
            m_temp.update()

            edge = self.topology.get_edge(self.center_block.block_id, leaf.block_id)
            temp_vars = leaf.get_vars_by_index(edge.vars_v)
            var_names = [v.VarName for v in temp_vars]

            m_lp = gp.Model(f"Leaf_{leaf.block_id}_LP")
            m_lp.Params.OutputFlag = 0
            m_lp.Params.InfUnbdInfo = 1
            leaf.build_model(parent_model=m_lp)
            m_lp.setObjective(leaf.local_objective_expr, gp.GRB.MAXIMIZE)
            m_lp.update()
            m_lp = m_lp.relax()
            m_lp.update()

            link_constrs = []
            for name in var_names:
                v_in_lp = m_lp.getVarByName(name)
                c = m_lp.addConstr(v_in_lp == 0.0, name=f"link_{name}")
                link_constrs.append(c)
                v_in_lp.ub = gp.GRB.INFINITY

            self.leaf_models_lp.append(m_lp)
            self.leaf_link_constrs_lp.append(link_constrs)

            m_mip = gp.Model(f"Leaf_{leaf.block_id}_MIP")
            m_mip.Params.OutputFlag = 0
            leaf.build_model(parent_model=m_mip)
            m_mip.setObjective(leaf.local_objective_expr, gp.GRB.MAXIMIZE)
            m_mip.update()

            link_constrs_mip = []
            for name in var_names:
                v_in_mip = m_mip.getVarByName(name)
                c = m_mip.addConstr(v_in_mip == 0.0, name=f"link_{name}")
                link_constrs_mip.append(c)

            self.leaf_models_mip.append(m_mip)
            self.leaf_link_constrs_mip.append(link_constrs_mip)

        self.global_upper_bound_U = total_max_possible + 100.0

    def _build_master(self):
        self.master = gp.Model("Master_LShaped")
        self.master.Params.OutputFlag = 1
        self.master.Params.LazyConstraints = 1

        self.center_block.build_model(parent_model=self.master)

        self.theta = self.master.addVar(lb=0.0, ub=self.global_upper_bound_U,
                                        obj=1.0, name="theta")

        full_obj = self.center_block.local_objective_expr.copy()
        full_obj.add(self.theta, 1.0)
        self.master.setObjective(full_obj, gp.GRB.MAXIMIZE)

        self.center_edge_vars_indices = []
        for leaf in self.leaf_blocks:
            edge = self.topology.get_edge(self.center_block.block_id, leaf.block_id)
            self.center_edge_vars_indices.append(edge.vars_u)

    def _get_hamming_distance_expr(self, model, indices, x_sol):
        expr = gp.LinExpr()
        center_vars = self.center_block.get_vars_by_index(indices)
        for idx, var in zip(indices, center_vars):
            val = x_sol[idx]
            if val > 0.5:
                expr.addConstant(1.0)
                expr.add(var, -1.0)
            else:
                expr.add(var, 1.0)
        return expr

    def solve(self, time_limit=None) -> Dict[str, Any]:
        if time_limit:
            self.master.Params.TimeLimit = time_limit
            self.time_limit = time_limit

        self.time_start = time.time()

        def cb(model, where):
            if where == gp.GRB.Callback.MIPSOL:
                center_vars_list = list(self.center_block.vars.values())
                center_vals = model.cbGetSolution(center_vars_list)
                x_center_sol = dict(zip(self.center_block.vars.keys(), center_vals))

                theta_sol = model.cbGetSolution(self.theta)

                total_lp_obj = 0.0
                cut_expr_lp = 0.0
                possible_benders = True

                for i, leaf_lp in enumerate(self.leaf_models_lp):
                    indices = self.center_edge_vars_indices[i]
                    vals = [x_center_sol[idx] for idx in indices]

                    for constr, val in zip(self.leaf_link_constrs_lp[i], vals):
                        constr.RHS = val

                    leaf_lp.optimize()

                    if leaf_lp.Status == gp.GRB.OPTIMAL:
                        total_lp_obj += leaf_lp.ObjVal
                        duals = [c.Pi for c in self.leaf_link_constrs_lp[i]]

                        term_const = leaf_lp.ObjVal
                        term_var = gp.LinExpr()
                        center_vars = self.center_block.get_vars_by_index(indices)

                        for d, v, v_val in zip(duals, center_vars, vals):
                            term_const -= d * v_val
                            term_var.add(v, d)

                        cut_expr_lp += (term_const + term_var)

                    elif leaf_lp.Status == gp.GRB.INF_OR_UNBD or leaf_lp.Status == gp.GRB.INFEASIBLE:
                        # FIX V47: Negative FarkasProof for maximization feasibility cut
                        farkas_val = -leaf_lp.FarkasProof
                        duals = [c.FarkasDual for c in self.leaf_link_constrs_lp[i]]

                        term_const_feas = farkas_val
                        term_var_feas = gp.LinExpr()
                        center_vars = self.center_block.get_vars_by_index(indices)

                        for d, v, v_val in zip(duals, center_vars, vals):
                            term_const_feas -= d * v_val
                            term_var_feas.add(v, d)

                        # FIX V47: Direction is >= 0.0
                        model.cbLazy(term_var_feas + term_const_feas >= 0.0)
                        return

                    else:
                        possible_benders = False
                        break

                if possible_benders:
                    if theta_sol > total_lp_obj + 1e-4:
                        model.cbLazy(self.theta <= cut_expr_lp)
                        return

                total_mip_obj = 0.0
                all_coupling_indices = set()

                for i, leaf_mip in enumerate(self.leaf_models_mip):
                    indices = self.center_edge_vars_indices[i]
                    all_coupling_indices.update(indices)

                    vals = [x_center_sol[idx] for idx in indices]

                    for constr, val in zip(self.leaf_link_constrs_mip[i], vals):
                        constr.RHS = val

                    leaf_mip.optimize()

                    if leaf_mip.Status == gp.GRB.OPTIMAL:
                        total_mip_obj += leaf_mip.ObjVal
                    elif leaf_mip.Status == gp.GRB.INF_OR_UNBD or leaf_mip.Status == gp.GRB.INFEASIBLE:
                        hamming_dist = self._get_hamming_distance_expr(model, indices, x_center_sol)
                        model.cbLazy(hamming_dist >= 1)
                        return
                    else:
                        total_mip_obj += -1e9

                if theta_sol > total_mip_obj + 1e-4:
                    sorted_all_indices = sorted(list(all_coupling_indices))
                    hamming_dist_union = self._get_hamming_distance_expr(model, sorted_all_indices, x_center_sol)

                    Q_val = total_mip_obj
                    U_val = self.global_upper_bound_U

                    cut_rhs = (U_val - Q_val) * hamming_dist_union + Q_val
                    model.cbLazy(self.theta <= cut_rhs)

        self.master.optimize(cb)

        metrics = {
            "method": "IntegerLShaped",
            "status": "Unknown",
            "total_time": time.time() - self.time_start,
            "primal_bound": -float('inf'),
            "dual_bound": float('inf'),
            "gap": 0.0,
            "node_count": self.master.NodeCount
        }

        if self.master.SolCount > 0:
            metrics["primal_bound"] = self.master.ObjVal
            metrics["dual_bound"] = self.master.ObjBound
            metrics["gap"] = self.master.MIPGap

        if self.master.Status == gp.GRB.OPTIMAL: metrics["status"] = "Optimal"
        elif self.master.Status == gp.GRB.TIME_LIMIT: metrics["status"] = "TimeLimit"
        else: metrics["status"] = f"Code_{self.master.Status}"

        return metrics
