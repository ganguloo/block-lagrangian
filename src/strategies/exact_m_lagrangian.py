import gurobipy as gp
from typing import List, Dict, Tuple, Any
from .base_strategy import SeparationStrategy

class ExactMLagrangianStrategy(SeparationStrategy):
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
        m = gp.Model("MLag_Sep_Exact")
        m.Params.OutputFlag = 0
        m.Params.PoolSearchMode = 2 
        m.Params.PoolSolutions = int(round(self.factor * n_vars))

        if self.single_threaded:
            m.Params.Threads = 1

        z = m.addVars(n_vars, vtype=gp.GRB.BINARY, name="z")
        w_pos = m.addVars(len(pos_sigs), vtype=gp.GRB.BINARY, name="w_pos")
        w_neg = m.addVars(len(neg_sigs), vtype=gp.GRB.BINARY, name="w_neg")
        
        obj = gp.LinExpr()
        
        # Maximizar (Masa Positiva - Masa Negativa)
        for k, (sig, lam) in enumerate(pos_sigs):
            obj.add(w_pos[k], lam)
            for p, bit in enumerate(sig):
                if bit == 0:
                    m.addConstr(w_pos[k] + z[p] <= 1)
        
        for k, (sig, lam) in enumerate(neg_sigs):
            obj.add(w_neg[k], -lam)
            lhs = gp.LinExpr()
            zeros_count = 0
            for p, bit in enumerate(sig):
                if bit == 0:
                    lhs.add(z[p], 1.0)
                    zeros_count += 1
            if zeros_count > 0:
                m.addConstr(lhs >= 1 - w_neg[k])
            else:
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
        Aplica penalización M-Lagrangian al subproblema (Pricing).
        Define w_S = AND(x_p for p in S) usando AMBOS grupos de restricciones
        para asegurar la integridad lógica de la variable auxiliar.
        """
        penalty_expr = gp.LinExpr()
        
        for cut_id, signature, sign_factor in cuts:
            if cut_id not in duals: continue
            
            mu = duals[cut_id]
            coeff = sign_factor * mu
            
            if abs(coeff) < 1e-9: continue
            
            S_indices = [i for i, bit in enumerate(signature) if bit == 1]
            if not S_indices: continue
            
            w_name = f"w_exact_{cut_id}"
            w_var = model.getVarByName(w_name)
            
            if w_var is None:
                w_var = model.addVar(vtype=gp.GRB.BINARY, name=w_name)
                
                # --- GRUPO 1: Upper Bound (w <= x_i) ---
                # Si algún x_i es 0, w debe ser 0.
                for idx in S_indices:
                    if idx < len(vars_list):
                        model.addConstr(w_var <= vars_list[idx], name=f"mc_le_{w_name}_{idx}")
                
                # --- GRUPO 2: Lower Bound (w >= Sum(x) - |S| + 1) ---
                # Si todos los x_i son 1, w debe ser 1.
                expr = gp.LinExpr()
                valid_count = 0
                for idx in S_indices:
                    if idx < len(vars_list):
                        expr.add(vars_list[idx], 1.0)
                        valid_count += 1
                
                if valid_count > 0:
                    model.addConstr(w_var >= expr - (valid_count - 1), name=f"mc_ge_{w_name}")

            penalty_expr.add(w_var, coeff)
            
        return penalty_expr

    def evaluate_cut(self, column_signature: Tuple, cut_signature: Any) -> float:
        for s_bit, c_bit in zip(cut_signature, column_signature):
            if s_bit == 1 and c_bit == 0:
                return 0.0
        return 1.0