
import gurobipy as gp
from typing import List, Dict, Tuple, Any
from .base_strategy import SeparationStrategy

class VLagrangianStrategy(SeparationStrategy):
    def get_w_signature(self, x_values: List[int]) -> Tuple:
        return tuple(x_values)

    def separate(self, w_sol_u: Dict[Tuple, float], w_sol_v: Dict[Tuple, float]) -> List[Tuple]:
        violations = set()
        all_sigs = set(w_sol_u.keys()) | set(w_sol_v.keys())
        for sig in all_sigs:
            val_u = w_sol_u.get(sig, 0.0)
            val_v = w_sol_v.get(sig, 0.0)
            if abs(val_u - val_v) > 1e-5:
                violations.add(sig)
        return list(violations)

    def apply_pricing_penalty(self, model: gp.Model, vars_list: List[gp.Var],
                              cuts: List[Any], duals: Dict) -> gp.LinExpr:
        penalty_expr = gp.LinExpr()
        for cut_id, signature, sign_factor in cuts:
            if cut_id not in duals: continue
            mu = duals[cut_id]
            coeff = sign_factor * mu
            if abs(coeff) < 1e-9: continue

            # FIX V32: Use cut_id in variable name to ensure uniqueness per interface/cut
            # Avoids collision when multiple neighbors have same signature
            w_name = f"w_v_{cut_id}"
            w_var = model.getVarByName(w_name)

            if w_var is None:
                w_var = model.addVar(vtype=gp.GRB.BINARY, name=w_name)

                # Add indicator constraints: w=1 <=> x matches signature
                # This formulation enforces w=0 if mismatch.
                delta_expr = gp.LinExpr()
                n = len(signature)
                for i, bit in enumerate(signature):
                    if bit == 1:
                        # If bit is 1, we want (1 - x_i) to be 0
                        delta_expr.addConstant(1.0)
                        delta_expr.add(vars_list[i], -1.0)
                    else:
                        # If bit is 0, we want x_i to be 0
                        delta_expr.add(vars_list[i], 1.0)

                # If delta_expr > 0 (mismatch), then n*(1-w) must be >= delta, so (1-w) > 0 => w=0.
                model.addConstr(delta_expr <= n * (1 - w_var), name=f"H_le_{w_name}")

                model.addConstr(delta_expr >= 1 - w_var, name=f"H_ge_{w_name}")

            penalty_expr.add(w_var, coeff)
        return penalty_expr

    def evaluate_cut(self, column_signature: Tuple, cut_signature: Any) -> float:
        return 1.0 if column_signature == cut_signature else 0.0
