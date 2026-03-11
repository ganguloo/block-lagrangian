
import gurobipy as gp
from typing import List, Dict, Tuple, Any
from .base_strategy import SeparationStrategy
import math

class VLagrangianStrategy(SeparationStrategy):
    def __init__(self, radius_factor: float = 0.0, tolerance: float = 1e-6, factor: float = 0.5, single_threaded: bool = False):
        super().__init__(single_threaded=single_threaded)
        self.radius_factor = radius_factor
        self.tolerance = tolerance
        self.factor = factor

    def _hamming_distance(self, s1, s2):
        return sum(1 for a, b in zip(s1, s2) if a != b)

    def separate(self, w_sol_u: Dict[Tuple, float], w_sol_v: Dict[Tuple, float]) -> List[Tuple]:
        """
        Identifica violaciones sumando masas en bolas de Hamming y devuelve las más violadas.
        """
        filter_tol = self.tolerance * 0.1
        sigs_u = [(sig, val) for sig, val in w_sol_u.items() if val > filter_tol]
        sigs_v = [(sig, val) for sig, val in w_sol_v.items() if val > filter_tol]

        if not sigs_u and not sigs_v:
            return []

        n_vars = 0
        if sigs_u: n_vars = len(sigs_u[0][0])
        elif sigs_v: n_vars = len(sigs_v[0][0])

        # Cambiamos 'violations' de un set a una lista para guardar (firma, magnitud)
        violations = []
        all_signatures = {s[0] for s in sigs_u} | {s[0] for s in sigs_v}
      
        for sig_center in all_signatures:
            val_u = 0.0
            val_v = 0.0
            
            # Sumar masas de configuraciones dentro del radio R
            for sig_other in all_signatures:
                # Optimización para R=0
                if self.radius_factor == 0:
                    if sig_other == sig_center:
                        val_u += w_sol_u.get(sig_other, 0.0)
                        val_v += w_sol_v.get(sig_other, 0.0)
                else:
                    if self._hamming_distance(sig_other, sig_center) <= math.floor(self.radius_factor * n_vars):
                        val_u += w_sol_u.get(sig_other, 0.0)
                        val_v += w_sol_v.get(sig_other, 0.0)
            
            # Calcular la magnitud de la violación
            violation_magnitude = abs(val_u - val_v)
            if violation_magnitude > self.tolerance:
                violations.append((sig_center, violation_magnitude))

        max_cuts = int(round(self.factor * n_vars))

        # 1. Ordenar la lista de mayor a menor basándose en la magnitud de la violación (índice 1)
        violations.sort(key=lambda x: x[1], reverse=True)

        # 2. Extraer solo las firmas de los 'max_cuts' más grandes
        top_violations = [sig for sig, mag in violations[:max_cuts]]

        return top_violations

    def apply_pricing_penalty(self, model: gp.Model, vars_list: List[gp.Var],
                              cuts: List[Any], duals: Dict) -> gp.LinExpr:
        penalty_expr = gp.LinExpr()
        for cut_id, signature, sign_factor in cuts:
            if cut_id not in duals: continue
            mu = duals[cut_id]
            coeff = sign_factor * mu
            if abs(coeff) < 1e-9: continue
            w_name = f"w_v_{cut_id}"
            w_var = model.getVarByName(w_name)
            if w_var is None:
                w_var = model.addVar(vtype=gp.GRB.BINARY, name=w_name)
                delta_expr = gp.LinExpr()
                n = len(signature)
                radius = math.floor(self.radius_factor*n)
                for i, bit in enumerate(signature):
                    if bit == 1:
                        delta_expr.addConstant(1.0)
                        delta_expr.add(vars_list[i], -1.0)
                    else:
                        delta_expr.add(vars_list[i], 1.0)
                model.addConstr(delta_expr <= n * (1 - w_var) + radius*w_var, name=f"H_le_{w_name}")
                model.addConstr(delta_expr >= (radius + 1)*(1-w_var), name=f"H_ge_{w_name}")
            penalty_expr.add(w_var, coeff)
        return penalty_expr

    def evaluate_cut(self, column_signature: Tuple, cut_signature: Any) -> float:
        """
        Evalúa si una columna (patrón) cae dentro del corte (bola).
        Usado por el Maestro para calcular coeficientes de la columna.
        """
        if self.radius_factor == 0:
            return 1.0 if column_signature == cut_signature else 0.0
        
        dist = self._hamming_distance(column_signature, cut_signature)
        return 1.0 if dist <= math.floor(self.radius_factor*len(column_signature)) else 0.0
