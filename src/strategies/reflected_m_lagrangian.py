import gurobipy as gp
from typing import List, Dict, Tuple, Any
from .base_strategy import SeparationStrategy

class ReflectedMLagrangianStrategy(SeparationStrategy):
    def __init__(self, tolerance: float = 1e-6, factor: float = 0.5, single_threaded: bool = False):
        super().__init__(single_threaded=single_threaded)
        self.tolerance = tolerance
        self.factor = factor

    def separate(self, w_sol_u: Dict[Tuple, float], w_sol_v: Dict[Tuple, float]) -> List[Tuple]:
        # 1. Filtrar columnas relevantes
        filter_tol = self.tolerance * 0.1
        sigs_u = [(sig, val) for sig, val in w_sol_u.items() if val > filter_tol]
        sigs_v = [(sig, val) for sig, val in w_sol_v.items() if val > filter_tol]
        
        if not sigs_u and not sigs_v:
            return []

        n_vars = 0
        if sigs_u: n_vars = len(sigs_u[0][0])
        elif sigs_v: n_vars = len(sigs_v[0][0])
        
        found_cuts = []

        # 2. Resolver ambos lados (U - V y V - U)
        cuts_u = self._solve_separation_mip(n_vars, sigs_u, sigs_v)
        found_cuts.extend(cuts_u)

        cuts_v = self._solve_separation_mip(n_vars, sigs_v, sigs_u)
        found_cuts.extend(cuts_v)
            
        return list(set(found_cuts))

    def _solve_separation_mip(self, n_vars, pos_sigs, neg_sigs) -> List[Tuple]:
        m = gp.Model("ReflectedMLag_Sep")
        m.Params.OutputFlag = 0
        m.Params.PoolSearchMode = 2 
        m.Params.PoolSolutions = max(1, int(round(self.factor * n_vars)))
        
        z = m.addVars(n_vars, vtype=gp.GRB.BINARY, name="z")
        w_pos = m.addVars(len(pos_sigs), vtype=gp.GRB.BINARY, name="w_pos")
        w_neg = m.addVars(len(neg_sigs), vtype=gp.GRB.BINARY, name="w_neg")
        
        obj = gp.LinExpr()
        
        # Maximizar (Masa Positiva - Masa Negativa)
        for k, (sig, lam) in enumerate(pos_sigs):
            obj.add(w_pos[k], lam)
            for p, bit in enumerate(sig):
                # REFLEJO: Si la columna tiene 1, w_pos DEBE ser 0.
                if bit == 1:
                    m.addConstr(w_pos[k] + z[p] <= 1)
        
        for k, (sig, lam) in enumerate(neg_sigs):
            obj.add(w_neg[k], -lam)
            lhs = gp.LinExpr()
            ones_count = 0
            for p, bit in enumerate(sig):
                # REFLEJO: Si la columna tiene 1, elegir ese p rompe la condición.
                if bit == 1:
                    lhs.add(z[p], 1.0)
                    ones_count += 1
            if ones_count > 0:
                # Si elegimos AL MENOS UN p donde x_p=1, entonces w_neg puede ser 0.
                m.addConstr(lhs >= 1 - w_neg[k])
            else:
                # Si la columna es toda 0s, cumple cualquier S que elijamos. w_neg=1 siempre.
                m.addConstr(w_neg[k] == 1)

        m.setObjective(obj, gp.GRB.MAXIMIZE)
        m.optimize()
        
        cuts = []
        n_solutions = m.SolCount
        
        if n_solutions > 0:
            for i in range(n_solutions):
                m.setParam(gp.GRB.Param.SolutionNumber, i)
                if m.PoolObjVal > self.tolerance:
                    s_signature = []
                    for p in range(n_vars):
                        s_signature.append(int(round(z[p].Xn)))
                    
                    if sum(s_signature) > 0:
                        cuts.append(tuple(s_signature))
        
        return cuts

    def apply_pricing_penalty(self, model: gp.Model, vars_list: List[gp.Var],
                              cuts: List[Any], duals: Dict) -> gp.LinExpr:
        """
        Aplica penalización Reflected M-Lagrangian al subproblema (Pricing).
        Define w_S = AND(1 - x_p for p in S).
        """
        penalty_expr = gp.LinExpr()
        
        for cut_id, signature, sign_factor in cuts:
            if cut_id not in duals: continue
            
            mu = duals[cut_id]
            coeff = sign_factor * mu
            
            if abs(coeff) < 1e-9: continue
            
            S_indices = [i for i, bit in enumerate(signature) if bit == 1]
            if not S_indices: continue
            
            w_name = f"w_refl_{cut_id}"
            w_var = model.getVarByName(w_name)
            
            if w_var is None:
                w_var = model.addVar(vtype=gp.GRB.BINARY, name=w_name)
                
                # --- GRUPO 1: Upper Bound (w <= 1 - x_i) ---
                # Si algún x_i es 1, w debe ser 0.
                for idx in S_indices:
                    if idx < len(vars_list):
                        model.addConstr(w_var <= 1 - vars_list[idx], name=f"rmc_le_{w_name}_{idx}")
                
                # --- GRUPO 2: Lower Bound ---
                # w >= Sum(1 - x_p) - |S| + 1
                # Simplificado a: w >= 1 - Sum(x_p)
                # Si todos los x_p son 0 (Suma = 0), w >= 1.
                expr = gp.LinExpr()
                valid_count = 0
                for idx in S_indices:
                    if idx < len(vars_list):
                        expr.add(vars_list[idx], 1.0)
                        valid_count += 1
                
                if valid_count > 0:
                    model.addConstr(w_var >= 1 - expr, name=f"rmc_ge_{w_name}")

            penalty_expr.add(w_var, coeff)
            
        return penalty_expr

    def evaluate_cut(self, column_signature: Tuple, cut_signature: Any) -> float:
        """
        Retorna 1.0 si column_signature cumple el corte Reflejado.
        Es decir, si para todo p en S (cut=1), la columna tiene un 0 (col=0).
        """
        for s_bit, c_bit in zip(cut_signature, column_signature):
            # Si el corte EXIGE un 0 (s_bit=1), pero la columna tiene un 1 (c_bit=1), falla.
            if s_bit == 1 and c_bit == 1:
                return 0.0
        return 1.0