import gurobipy as gp
from typing import List, Dict, Tuple, Any
from .base_strategy import SeparationStrategy

class GeneralizedMLagrangianStrategy(SeparationStrategy):
    def __init__(self, tolerance: float = 1e-6, factor: float = 0.5, single_threaded: bool = False):
        super().__init__(single_threaded=single_threaded)
        self.tolerance = tolerance
        self.factor = factor

    def separate(self, w_sol_u: Dict[Tuple, float], w_sol_v: Dict[Tuple, float]) -> List[Tuple]:
        filter_tol = self.tolerance * 0.1
        sigs_u = [(sig, val) for sig, val in w_sol_u.items() if val > filter_tol]
        sigs_v = [(sig, val) for sig, val in w_sol_v.items() if val > filter_tol]
        
        if not sigs_u and not sigs_v:
            return []

        n_vars = 0
        if sigs_u: n_vars = len(sigs_u[0][0])
        elif sigs_v: n_vars = len(sigs_v[0][0])
        
        found_cuts = []

        # Lado A: U - V
        cuts_u = self._solve_separation_mip(n_vars, sigs_u, sigs_v)
        found_cuts.extend(cuts_u)

        # Lado B: V - U
        cuts_v = self._solve_separation_mip(n_vars, sigs_v, sigs_u)
        found_cuts.extend(cuts_v)
            
        return list(set(found_cuts))

    def _solve_separation_mip(self, n_vars, pos_sigs, neg_sigs) -> List[Tuple]:
        m = gp.Model("Gen_MLag_Sep")
        m.Params.OutputFlag = 0
        m.Params.PoolSearchMode = 2 
        m.Params.PoolSolutions = int(round(self.factor * n_vars))
        
        # z_plus[p] = 1 si p está en S+
        z_plus = m.addVars(n_vars, vtype=gp.GRB.BINARY, name="z_plus")
        # z_minus[p] = 1 si p está en S-
        z_minus = m.addVars(n_vars, vtype=gp.GRB.BINARY, name="z_minus")
        
        # Exclusión mutua: un elemento no puede estar en S+ y S- al mismo tiempo
        for p in range(n_vars):
            m.addConstr(z_plus[p] + z_minus[p] <= 1, name=f"mutex_{p}")
            
        w_pos = m.addVars(len(pos_sigs), vtype=gp.GRB.BINARY, name="w_pos")
        w_neg = m.addVars(len(neg_sigs), vtype=gp.GRB.BINARY, name="w_neg")
        
        obj = gp.LinExpr()
        
        # --- Término Positivo: Queremos w_pos = 1 ---
        for k, (sig, lam) in enumerate(pos_sigs):
            obj.add(w_pos[k], lam)
            for p, bit in enumerate(sig):
                if bit == 0:
                    # Si la columna tiene 0, no podemos exigir que sea 1 (z_plus)
                    m.addConstr(w_pos[k] + z_plus[p] <= 1)
                else: # bit == 1
                    # Si la columna tiene 1, no podemos exigir que sea 0 (z_minus)
                    m.addConstr(w_pos[k] + z_minus[p] <= 1)
        
        # --- Término Negativo: Queremos w_neg = 0 ---
        for k, (sig, lam) in enumerate(neg_sigs):
            obj.add(w_neg[k], -lam)
            lhs = gp.LinExpr()
            violation_conditions = 0
            for p, bit in enumerate(sig):
                if bit == 0:
                    # Si pedimos S+, esta columna lo rompe
                    lhs.add(z_plus[p], 1.0)
                    violation_conditions += 1
                else: # bit == 1
                    # Si pedimos S-, esta columna lo rompe
                    lhs.add(z_minus[p], 1.0)
                    violation_conditions += 1
            
            if violation_conditions > 0:
                m.addConstr(lhs >= 1 - w_neg[k])
            else:
                m.addConstr(w_neg[k] == 1)

        m.setObjective(obj, gp.GRB.MAXIMIZE)
        m.optimize()
        
        cuts = []
        if m.SolCount > 0:
            for i in range(m.SolCount):
                m.setParam(gp.GRB.Param.SolutionNumber, i)
                if m.PoolObjVal > self.tolerance:
                    # Construir la firma ternaria: 1 (S+), -1 (S-), 0 (Ninguno)
                    s_signature = []
                    for p in range(n_vars):
                        val_plus = int(round(z_plus[p].Xn))
                        val_minus = int(round(z_minus[p].Xn))
                        if val_plus == 1:
                            s_signature.append(1)
                        elif val_minus == 1:
                            s_signature.append(-1)
                        else:
                            s_signature.append(0)
                    
                    # Ignorar el corte vacío trivial
                    if any(val != 0 for val in s_signature):
                        cuts.append(tuple(s_signature))
        
        return cuts

    def apply_pricing_penalty(self, model: gp.Model, vars_list: List[gp.Var],
                              cuts: List[Any], duals: Dict) -> gp.LinExpr:
        """
        Aplica penalización G-MLagrangian al Pricing.
        Aquí construimos AMBOS lados (LB y UB) sin importar el signo de 'coeff',
        porque el Gurobi Pricing es un problema de decisión donde las x varían.
        """
        penalty_expr = gp.LinExpr()
        
        for cut_id, signature, sign_factor in cuts:
            if cut_id not in duals: continue
            
            mu = duals[cut_id]
            coeff = sign_factor * mu
            
            if abs(coeff) < 1e-9: continue
            
            # Decodificar el corte ternario
            S_plus = [i for i, val in enumerate(signature) if val == 1]
            S_minus = [i for i, val in enumerate(signature) if val == -1]
            
            if not S_plus and not S_minus: continue
            
            w_name = f"w_gmlag_{cut_id}"
            w_var = model.getVarByName(w_name)
            
            if w_var is None:
                w_var = model.addVar(vtype=gp.GRB.BINARY, name=w_name)
                
                # --- GRUPO 1: Upper Bounds (Condiciones necesarias para w=1) ---
                for idx in S_plus:
                    if idx < len(vars_list):
                        # w <= x_p  =>  w=1 exige x_p=1
                        model.addConstr(w_var <= vars_list[idx], name=f"gmc_le_p_{w_name}_{idx}")
                
                for idx in S_minus:
                    if idx < len(vars_list):
                        # w <= 1 - x_p  =>  w=1 exige x_p=0
                        model.addConstr(w_var <= 1 - vars_list[idx], name=f"gmc_le_m_{w_name}_{idx}")
                        
                # --- GRUPO 2: Lower Bound (Condición suficiente para w=1) ---
                # w >= Sum(x_p para S+) + Sum(1 - x_p para S-) - (|S+| + |S-| - 1)
                expr = gp.LinExpr()
                valid_plus = 0
                valid_minus = 0
                
                for idx in S_plus:
                    if idx < len(vars_list):
                        expr.add(vars_list[idx], 1.0)
                        valid_plus += 1
                        
                for idx in S_minus:
                    if idx < len(vars_list):
                        expr.add(vars_list[idx], -1.0) # Sum(1 - x) = |S-| - Sum(x)
                        valid_minus += 1
                        
                # RHS = (Sum(x_+) - Sum(x_-) + |S-|) - (|S+| + |S-| - 1)
                #     = Sum(x_+) - Sum(x_-) - |S+| + 1
                total_valid = valid_plus + valid_minus
                if total_valid > 0:
                    rhs_adjust = valid_plus - 1
                    model.addConstr(w_var >= expr - rhs_adjust, name=f"gmc_ge_{w_name}")

            penalty_expr.add(w_var, coeff)
            
        return penalty_expr

    def evaluate_cut(self, column_signature: Tuple, cut_signature: Any) -> float:
        """
        Retorna 1.0 si la columna satisface todo el patrón S+ y S-.
        """
        for cut_val, c_bit in zip(cut_signature, column_signature):
            if cut_val == 1 and c_bit == 0:
                return 0.0
            if cut_val == -1 and c_bit == 1:
                return 0.0
        return 1.0