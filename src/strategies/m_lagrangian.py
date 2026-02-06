
import gurobipy as gp
from typing import List, Dict, Tuple, Any
from .base_strategy import SeparationStrategy

class MLagrangianStrategy(SeparationStrategy):
    def __init__(self, max_cuts=50, tol=1e-4):
        self.max_cuts = max_cuts
        self.tol = tol

    def get_w_signature(self, x_values: List[int]) -> Tuple:
        return tuple(x_values)

    def separate(self, w_sol_u: Dict[Tuple, float], w_sol_v: Dict[Tuple, float]) -> List[Tuple[int, int]]:
        if not w_sol_u and not w_sol_v: return []
        moments_u, moments_v = {}, {}

        def compute_moments(w_sol, moments_dict):
            for x_tuple, weight in w_sol.items():
                if weight < 1e-6: continue
                ones = [i for i, val in enumerate(x_tuple) if val > 0.5]
                for i in range(len(ones)):
                    for j in range(i + 1, len(ones)):
                        p, q = ones[i], ones[j]
                        key = (p, q)
                        moments_dict[key] = moments_dict.get(key, 0.0) + weight

        compute_moments(w_sol_u, moments_u)
        compute_moments(w_sol_v, moments_v)

        violations = []
        all_pairs = set(moments_u.keys()) | set(moments_v.keys())
        for p, q in all_pairs:
            val_u = moments_u.get((p, q), 0.0)
            val_v = moments_v.get((p, q), 0.0)
            diff = abs(val_u - val_v)
            if diff > self.tol:
                violations.append(((p, q), diff))

        violations.sort(key=lambda x: x[1], reverse=True)
        return [v[0] for v in violations[:self.max_cuts]]

    def apply_pricing_penalty(self, model: gp.Model, vars_list: List[gp.Var],
                              cuts: List[Any], duals: Dict) -> gp.QuadExpr:
        penalty_expr = gp.QuadExpr()
        for cut_id, signature, sign_factor in cuts:
            if cut_id not in duals: continue
            p, q = signature
            mu = duals[cut_id]
            coeff = sign_factor * mu
            if abs(coeff) < 1e-9: continue
            if p < len(vars_list) and q < len(vars_list):
                penalty_expr.add(vars_list[p] * vars_list[q], coeff)
        return penalty_expr

    def evaluate_cut(self, column_signature: Tuple, cut_signature: Any) -> float:
        p, q = cut_signature
        if p < len(column_signature) and q < len(column_signature):
            return float(column_signature[p] * column_signature[q])
        return 0.0
