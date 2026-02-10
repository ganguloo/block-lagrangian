import time
import gurobipy as gp
from typing import List, Dict, Any
from ..blocks.base_block import AbstractBlock
from ..instance.topology import TopologyManager
from datetime import datetime
import concurrent.futures

class ScenarioDecompositionSolver:
    def __init__(self, topology: TopologyManager, blocks: List[AbstractBlock]):
        self.topology = topology
        self.blocks = blocks
        self.center_block = blocks[0]
        self.leaf_blocks = blocks[1:]

        self.K = len(self.leaf_blocks)
        self.scenario_models = []
        self.scenario_center_vars = []

        self.boundary_indices = set()
        for leaf in self.leaf_blocks:
            edge = self.topology.get_edge(self.center_block.block_id, leaf.block_id)
            self.boundary_indices.update(edge.vars_u)
        self.sorted_boundary_indices = sorted(list(self.boundary_indices))

        self._build_scenarios()

    def _build_scenarios(self):
        for k, leaf in enumerate(self.leaf_blocks):
            model = gp.Model(f"Scenario_{k}")
            model.Params.OutputFlag = 0

            self.center_block.build_model(parent_model=model, prefix="C")

            all_center_vars = self.center_block.vars
            center_vars_k = {idx: all_center_vars[idx] for idx in self.sorted_boundary_indices}
            self.scenario_center_vars.append(center_vars_k)

            leaf.build_model(parent_model=model, prefix="L")
            model.update()

            edge = self.topology.get_edge(self.center_block.block_id, leaf.block_id)
            c_vars = [center_vars_k[i] for i in edge.vars_u]
            l_vars = [leaf.vars[i] for i in edge.vars_v]

            for vc, vl in zip(c_vars, l_vars):
                model.addConstr(vc == vl, name=f"couple_{vc.VarName}_{vl.VarName}")

            total_obj = gp.LinExpr()
            if self.center_block.local_objective_expr:
                total_obj.add(self.center_block.local_objective_expr, 1.0 / self.K)
            if leaf.local_objective_expr:
                total_obj.add(leaf.local_objective_expr, 1.0)

            model.setObjective(total_obj, gp.GRB.MAXIMIZE)
            self.scenario_models.append(model)

    def _solve_scenario(self, k):
        model = self.scenario_models[k]
        model.optimize()
        if model.Status == gp.GRB.OPTIMAL:
            x_sol = {}
            for idx, var in self.scenario_center_vars[k].items():
                x_sol[idx] = int(round(var.X))
            return k, model.ObjVal, x_sol
        else:
            return k, -1e9, None

    def _evaluate_candidate(self, x_cand):
        total_val = 0.0
        feasible = True

        for k, model in enumerate(self.scenario_models):
            original_bounds = {}
            c_vars = self.scenario_center_vars[k]

            for idx, val in x_cand.items():
                v = c_vars[idx]
                original_bounds[v] = (v.LB, v.UB)
                v.LB = val
                v.UB = val

            model.update()
            model.optimize()

            if model.Status == gp.GRB.OPTIMAL:
                total_val += model.ObjVal
            else:
                feasible = False

            for v, (lb, ub) in original_bounds.items():
                v.LB = lb
                v.UB = ub
            model.update()

            if not feasible: break

        return total_val if feasible else -1e9

    def _add_cut(self, x_cand):
        for k, model in enumerate(self.scenario_models):
            c_vars = self.scenario_center_vars[k]
            lhs = gp.LinExpr()

            for idx, val in x_cand.items():
                var = c_vars[idx]
                if val > 0.5:
                    lhs.addConstant(1.0)
                    lhs.add(var, -1.0)
                else:
                    lhs.add(var, 1.0)

            model.addConstr(lhs >= 1.0, name="NoGood")

    def solve(self, time_limit=300):
        start_time = time.time()
        best_lb = -1e9

        metrics = {
            "method": "ScenarioDecomp",
            "status": "Running",
            "total_time": 0.0,
            "primal_bound": -float('inf'),
            "dual_bound": float('inf'),
            "gap": 1.0,
            "iter": 0
        }

        evaluated_candidates = set()

        print(f"\\n--- Starting Scenario Decomposition --- {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        while (time.time() - start_time) < time_limit:
            metrics["iter"] += 1

            current_ub = 0.0
            candidates = []
            iteration_feasible = True

            for k in range(self.K):
                _, obj, x_sol = self._solve_scenario(k)
                if x_sol is None:
                    iteration_feasible = False
                    break
                current_ub += obj
                candidates.append(x_sol)

            if not iteration_feasible:
                if best_lb > -1e8:
                    metrics["status"] = "Optimal"
                    metrics["gap"] = 0.0
                    metrics["dual_bound"] = best_lb
                else:
                    metrics["status"] = "Infeasible"
                break

            metrics["dual_bound"] = current_ub

            # FIX V52: Force bounds equal if crossed
            if current_ub <= best_lb + 1e-4:
                metrics["status"] = "Optimal"
                metrics["gap"] = 0.0
                metrics["dual_bound"] = best_lb
                metrics["primal_bound"] = best_lb
                print(f"  SD Iter {metrics['iter']}: UB {metrics['dual_bound']:.2f} <= LB {metrics['primal_bound']:.2f}, Converged.")
                break

            unique_cands = []
            seen_sigs = set()
            for x in candidates:
                sig = tuple(sorted(x.items()))
                if sig not in seen_sigs:
                    seen_sigs.add(sig)
                    if sig not in evaluated_candidates:
                        unique_cands.append(x)

            if not unique_cands:
                if abs(current_ub - best_lb) < 1e-4:
                    metrics["status"] = "Optimal"
                else:
                    metrics["status"] = "Stalled"

                elapsed = time.time() - start_time
                if best_lb > -1e8:
                    metrics["gap"] = max(0.0, (metrics["dual_bound"] - best_lb) / abs(metrics["dual_bound"]))
                print(f"  SD Iter {metrics['iter']}: UB {metrics['dual_bound']:.2f}, LB {metrics['primal_bound']:.2f}, Gap {metrics['gap']:.4%} (Elapsed: {elapsed:.2f}s)")
                break

            for x_cand in unique_cands:
                val = self._evaluate_candidate(x_cand)
                if val > best_lb:
                    best_lb = val
                    metrics["primal_bound"] = best_lb

                self._add_cut(x_cand)
                evaluated_candidates.add(tuple(sorted(x_cand.items())))

            if best_lb > -1e8:
                # FIX V52: Check crossing again after LB update
                if metrics["dual_bound"] <= best_lb + 1e-4:
                    metrics["status"] = "Optimal"
                    metrics["gap"] = 0.0
                    metrics["dual_bound"] = best_lb
                    metrics["primal_bound"] = best_lb
                    elapsed = time.time() - start_time
                    print(f"  SD Iter {metrics['iter']}: UB {metrics['dual_bound']:.2f} <= LB {metrics['primal_bound']:.2f}, Converged. (Elapsed: {elapsed:.2f}s)")
                    break

                if metrics["dual_bound"] > 1e-9:
                    diff = metrics["dual_bound"] - best_lb
                    if diff < 0: diff = 0
                    metrics["gap"] = diff / abs(metrics["dual_bound"])
                else:
                    metrics["gap"] = 0.0

            elapsed = time.time() - start_time
            print(f"  SD Iter {metrics['iter']}: UB {metrics['dual_bound']:.2f}, LB {metrics['primal_bound']:.2f}, Gap {metrics['gap']:.4%} (Elapsed: {elapsed:.2f}s)")

            if metrics["gap"] < 1e-4:
                metrics["status"] = "Optimal"
                break

        if (time.time() - start_time) >= time_limit and metrics["status"] == "Running":
             metrics["status"] = "TimeLimit"

        metrics["total_time"] = time.time() - start_time
        print(f"--- Finished Scenario Decomposition: {metrics['status']} ---")
        return metrics
