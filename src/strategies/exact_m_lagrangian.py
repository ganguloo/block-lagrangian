import gurobipy as gp
from typing import List, Dict, Tuple, Any
from .base_strategy import SeparationStrategy

class ExactMLagrangianStrategy(SeparationStrategy):
    def __init__(self, tolerance: float = 1e-6, max_cuts_per_side: int = 1):
        """
        Estrategia M-Lagrangian Exacta usando separación MIP.
        
        Sustituye la heurística de combinaciones (mindeg/maxdeg) por un modelo exacto
        que busca el subconjunto S que maximiza la violación Lagrangiana.
        
        :param tolerance: Umbral para considerar una lambda > 0 y para violaciones.
        :param max_cuts_per_side: Cuántos cortes (soluciones del MIP) extraer de cada lado (U->V y V->U).
        """
        # No llamamos a super().__init__ porque usamos parámetros distintos
        # y no dependemos de mindeg/maxdeg.
        self.tolerance = tolerance
        self.max_cuts_per_side = max_cuts_per_side

    def separate(self, w_sol_u: Dict[Tuple, float], w_sol_v: Dict[Tuple, float]) -> List[Tuple]:
        """
        Separación EXACTA buscando subsets S que maximicen la diferencia de masa.
        Retorna una lista de 'firmas' (máscaras binarias) de los conjuntos S violados.
        """
        # 1. Filtrar columnas relevantes para reducir el tamaño del MIP
        sigs_u = [(sig, val) for sig, val in w_sol_u.items() if val > self.tolerance]
        sigs_v = [(sig, val) for sig, val in w_sol_v.items() if val > self.tolerance]
        
        if not sigs_u and not sigs_v:
            return []

        # Detectar tamaño de la frontera
        n_vars = 0
        if sigs_u: n_vars = len(sigs_u[0][0])
        elif sigs_v: n_vars = len(sigs_v[0][0])
        
        found_cuts = []

        # 2. Lado A: Buscar S donde (Masa U - Masa V) sea máxima
        # Esto genera cortes donde el bloque U "pesa más" que V en S
        cuts_u = self._solve_separation_mip(n_vars, sigs_u, sigs_v)
        found_cuts.extend(cuts_u)

        # 3. Lado B: Buscar S donde (Masa V - Masa U) sea máxima
        # Invertimos los argumentos para buscar el dominio opuesto
        cuts_v = self._solve_separation_mip(n_vars, sigs_v, sigs_u)
        found_cuts.extend(cuts_v)
            
        # Eliminar duplicados de cortes idénticos encontrados en ambos lados
        return list(set(found_cuts))

    def _solve_separation_mip(self, n_vars, pos_sigs, neg_sigs) -> List[Tuple]:
        """
        Resuelve: Max Sum(pos_lambda * w_pos) - Sum(neg_lambda * w_neg)
        Retorna lista de firmas (tuplas 0/1) que representan los conjuntos S.
        """
        m = gp.Model("MLag_Sep_Exact")
        m.Params.OutputFlag = 0
        #m.Params.Threads = 1
        
        # Configuración para retornar múltiples cortes violados
        if self.max_cuts_per_side > 1:
            m.Params.PoolSearchMode = 2 
            m.Params.PoolSolutions = self.max_cuts_per_side
        
        # Variables z_p: 1 si el elemento p está en el conjunto S
        z = m.addVars(n_vars, vtype=gp.GRB.BINARY, name="z")
        
        # Variables indicadoras w: 1 si la columna 'k' contiene al conjunto S
        w_pos = m.addVars(len(pos_sigs), vtype=gp.GRB.BINARY, name="w_pos")
        w_neg = m.addVars(len(neg_sigs), vtype=gp.GRB.BINARY, name="w_neg")
        
        obj = gp.LinExpr()

        m.addConstr(gp.quicksum(z[p] for p in range(n_vars)) >= 2)
        
        # --- Término Positivo (Queremos w=1 si S es subset de la Columna) ---
        for k, (sig, lam) in enumerate(pos_sigs):
            obj.add(w_pos[k], lam)
            # Restricción: w_pos[k] <= 1 - z[p]  para todo p donde sig[p] == 0
            # (Si elijo p en S, pero la columna no tiene p, entonces w muere)
            for p, bit in enumerate(sig):
                if bit == 0:
                    m.addConstr(w_pos[k] + z[p] <= 1)
        
        # --- Término Negativo (Queremos w=1 para restar la penalización) ---
        for k, (sig, lam) in enumerate(neg_sigs):
            obj.add(w_neg[k], -lam)
            # Restricción: Sum(z[p] donde sig[p]==0) >= 1 - w_neg[k]
            # (Si w=0, debo haber elegido al menos un p que la columna no tiene)
            lhs = gp.LinExpr()
            zeros_count = 0
            for p, bit in enumerate(sig):
                if bit == 0:
                    lhs.add(z[p], 1.0)
                    zeros_count += 1
            
            if zeros_count > 0:
                m.addConstr(lhs >= 1 - w_neg[k])
            else:
                # Si la columna tiene todo 1s, siempre contiene a cualquier S
                m.addConstr(w_neg[k] == 1)

        m.setObjective(obj, gp.GRB.MAXIMIZE)
        m.optimize()
        
        cuts = []
        n_solutions = m.SolCount
        if n_solutions > 0:
            limit = min(n_solutions, self.max_cuts_per_side)
            for i in range(limit):
                m.setParam(gp.GRB.Param.SolutionNumber, i)
                
                # Solo nos interesan cortes con violación positiva significativa
                if m.PoolObjVal > self.tolerance:
                    s_signature = []
                    for p in range(n_vars):
                        val = z[p].Xn # Usar Xn para soluciones del pool
                        s_signature.append(int(round(val)))
                    
                    cuts.append(tuple(s_signature))
        
        return cuts

    def apply_pricing_penalty(self, model: gp.Model, vars_list: List[gp.Var],
                              cuts: List[Any], duals: Dict) -> gp.LinExpr:
        """
        Aplica penalización M-Lagrangian al subproblema (Pricing).
        Crea variables auxiliares z_S = AND(x_p for p in S) y penaliza en la objetivo.
        """
        penalty_expr = gp.LinExpr()
        
        for cut_id, signature, sign_factor in cuts:
            if cut_id not in duals: continue
            
            mu = duals[cut_id]
            coeff = sign_factor * mu
            
            # Ignorar duales numéricamente cero para no inflar el modelo
            if abs(coeff) < 1e-9: continue
            
            # Identificar índices del conjunto S (donde signature es 1)
            S_indices = [i for i, bit in enumerate(signature) if bit == 1]
            if not S_indices: continue
            
            # Nombre consistente con tu nomenclatura
            w_name = f"w_exact_{cut_id}"
            w_var = model.getVarByName(w_name)
            
            if w_var is None:
                w_var = model.addVar(vtype=gp.GRB.BINARY, name=w_name)
                
                # Construcción inteligente de restricciones AND según el signo del objetivo
                # para minimizar variables y constraints en el pricing
                
                # Caso A: Coeficiente Positivo (Queremos MAXIMIZAR w)
                # El solver "quiere" w=1. Debemos prohibirlo si no cumple la lógica.
                # w <= x_p para todo p en S
                if coeff > 0:
                    for idx in S_indices:
                        if idx < len(vars_list):
                            model.addConstr(w_var <= vars_list[idx], name=f"mc_le_{w_name}_{idx}")
                        
                # Caso B: Coeficiente Negativo (Queremos MINIMIZAR w)
                # El solver "quiere" w=0. Debemos forzarlo a 1 si cumple la lógica.
                # w >= Sum(x_p) - (|S| - 1)
                else:
                    expr = gp.LinExpr()
                    valid_count = 0
                    for idx in S_indices:
                        if idx < len(vars_list):
                            expr.add(vars_list[idx], 1.0)
                            valid_count += 1
                    
                    # Ajuste por si el corte referencia variables fuera de rango (boundary mismatch)
                    rhs_adjust = valid_count - 1
                    if valid_count > 0:
                        model.addConstr(w_var >= expr - rhs_adjust, name=f"mc_ge_{w_name}")

            penalty_expr.add(w_var, coeff)
            
        return penalty_expr

    def evaluate_cut(self, column_signature: Tuple, cut_signature: Any) -> float:
        """
        Retorna 1.0 si column_signature contiene al conjunto S (cut_signature).
        Maneja firmas densas (tuplas 0/1).
        """
        # cut_signature es la máscara binaria de S
        # column_signature es la máscara binaria de la columna
        # S subset Col <==> (S & Col) == S
        # Implementación bit a bit:
        for s_bit, c_bit in zip(cut_signature, column_signature):
            if s_bit == 1 and c_bit == 0:
                return 0.0
        return 1.0