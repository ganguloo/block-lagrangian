import gurobipy as gp
from typing import List, Dict, Tuple, Any
from itertools import combinations
from .base_strategy import SeparationStrategy

class MLagrangianStrategy(SeparationStrategy):
    def __init__(self, mindeg=2, maxdeg=2, factor=1.0, max_cuts=50, tol=1e-4):
        self.mindeg = mindeg
        self.maxdeg = maxdeg
        self.factor = factor
        self.max_cuts = max_cuts
        self.tol = tol

    def get_w_signature(self, x_values: List[int]) -> Tuple:
        return tuple(x_values)

    def separate(self, w_sol_u: Dict[Tuple, float], w_sol_v: Dict[Tuple, float]) -> List[Tuple[Tuple[int, ...], int]]:
        if not w_sol_u and not w_sol_v: return []
        first_sig = next(iter(w_sol_u)) if w_sol_u else next(iter(w_sol_v))
        n_vars = len(first_sig)
        violations = []

        for deg in range(self.mindeg, self.maxdeg + 1):
            relevant_subsets = set()
            def collect_subsets(w_sol):
                for sig, weight in w_sol.items():
                    if weight < 1e-6: continue
                    ones_indices = [i for i, val in enumerate(sig) if val > 0.5]
                    if len(ones_indices) >= deg:
                        for subset in combinations(ones_indices, deg):
                            relevant_subsets.add(subset)
            collect_subsets(w_sol_u)
            collect_subsets(w_sol_v)

            for subset in relevant_subsets:
                eu = 0.0
                for sig, weight in w_sol_u.items():
                    if all(sig[i] > 0.5 for i in subset): eu += weight
                ev = 0.0
                for sig, weight in w_sol_v.items():
                    if all(sig[i] > 0.5 for i in subset): ev += weight

                diff = abs(eu - ev)
                if diff > self.tol:
                    violations.append((subset, diff))

        violations.sort(key=lambda x: x[1], reverse=True)
        limit = int(self.factor * n_vars) if self.factor > 0 else self.max_cuts
        limit = max(limit, self.max_cuts)
        return [v[0] for v in violations[:limit]]

    def apply_pricing_penalty(self, model: gp.Model, vars_list: List[gp.Var],
                              cuts: List[Any], duals: Dict) -> gp.LinExpr:
        penalty_expr = gp.LinExpr()
        for cut_id, indices_s, sign_factor in cuts:
            if cut_id not in duals: continue
            mu = duals[cut_id]
            coeff = sign_factor * mu
            if abs(coeff) < 1e-9: continue

            w_name = f"w_cut_{cut_id}"
            w_var = model.getVarByName(w_name)

            if w_var is None:
                w_var = model.addVar(vtype=gp.GRB.BINARY, name=w_name)
                for k in indices_s:
                    if k < len(vars_list):
                        model.addConstr(w_var <= vars_list[k], name=f"mc_le_{w_name}_{k}")
                s_size = len(indices_s)
                sum_expr = gp.LinExpr()
                valid_indices_count = 0
                for k in indices_s:
                    if k < len(vars_list):
                        sum_expr.add(vars_list[k])
                        valid_indices_count += 1
                if valid_indices_count == s_size:
                    model.addConstr(w_var >= sum_expr - (s_size - 1), name=f"mc_ge_{w_name}")
            penalty_expr.add(w_var, coeff)
        return penalty_expr

    def evaluate_cut(self, column_signature: Tuple, cut_signature: Any) -> float:
        for idx in cut_signature:
            if idx >= len(column_signature) or column_signature[idx] < 0.5:
                return 0.0
        return 1.0