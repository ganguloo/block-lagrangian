import time
import gurobipy as gp
import threading
import queue
import copy
from typing import List, Dict, Any
from ..blocks.base_block import AbstractBlock
from ..instance.topology import TopologyManager

class LeafWorker(threading.Thread):
    """
    Patrón Actor: Un Worker persistente que posee sus propios modelos de Gurobi
    y su propio entorno. Escucha comandos por colas y devuelve datos puros de Python.
    """
    def __init__(self, leaf_idx, leaf_block_copy, var_names, in_q, out_q):
        super().__init__()
        self.leaf_idx = leaf_idx
        self.leaf_block = leaf_block_copy
        self.var_names = var_names
        self.in_q = in_q
        self.out_q = out_q
        
        self.env = None
        self.m_lp = None
        self.m_mip = None
        self.link_constrs_lp = []
        self.link_constrs_mip = []

    def run(self):
        # 1. Entorno Dedicado
        self.env = gp.Env(empty=True)
        self.env.setParam("OutputFlag", 0)
        self.env.setParam("Threads", 1)
        self.env.start()
        
        # 2. Modelo LP (Benders)
        m_lp_temp = gp.Model(f"Leaf_{self.leaf_idx}_LP_temp", env=self.env)
        self.leaf_block.build_model(parent_model=m_lp_temp)
        m_lp_temp.setObjective(self.leaf_block.local_objective_expr, gp.GRB.MAXIMIZE)
        m_lp_temp.update()
        
        self.m_lp = m_lp_temp.relax()
        self.m_lp.ModelName = f"Leaf_{self.leaf_idx}_LP"
        
        # Estabilidad de Rayos Farkas
        self.m_lp.Params.Method = 1  # Forzar Dual Simplex
        self.m_lp.Params.InfUnbdInfo = 1 
        self.m_lp.Params.DualReductions = 0 
        self.m_lp.update()
        
        for name in self.var_names:
            v_in_lp = self.m_lp.getVarByName(name)
            c = self.m_lp.addConstr(v_in_lp == 0.0, name=f"link_{name}")
            self.link_constrs_lp.append(c)
            # Quitar cota superior para que los duales representen el acople correctamente
            v_in_lp.UB = gp.GRB.INFINITY 
        self.m_lp.update()
        
        # 3. Modelo MIP (Integer L-Shaped)
        self.m_mip = gp.Model(f"Leaf_{self.leaf_idx}_MIP", env=self.env)
        self.leaf_block.build_model(parent_model=self.m_mip)
        self.m_mip.setObjective(self.leaf_block.local_objective_expr, gp.GRB.MAXIMIZE)
        self.m_mip.update()
        
        for name in self.var_names:
            v_in_mip = self.m_mip.getVarByName(name)
            c = self.m_mip.addConstr(v_in_mip == 0.0, name=f"link_{name}")
            self.link_constrs_mip.append(c)
        self.m_mip.update()

        # 4. Bucle de Eventos (Actor Pattern)
        while True:
            cmd, payload = self.in_q.get()
            
            try:
                if cmd == "STOP":
                    self.out_q.put((self.leaf_idx, "STOP_ACK", True))
                    break
                    
                elif cmd == "SOLVE_LP":
                    for c, val in zip(self.link_constrs_lp, payload):
                        c.RHS = val
                        
                    self.m_lp.optimize()
                    
                    status = self.m_lp.Status
                    obj_val = 0.0
                    duals = []
                    farkas = []
                    
                    if status == gp.GRB.OPTIMAL:
                        obj_val = self.m_lp.ObjVal
                        duals = [c.Pi for c in self.link_constrs_lp]
                    elif status == gp.GRB.INF_OR_UNBD or status == gp.GRB.INFEASIBLE:
                        obj_val = - self.m_lp.FarkasProof
                        farkas = [c.FarkasDual for c in self.link_constrs_lp]
                        
                    self.out_q.put((self.leaf_idx, "LP", (status, obj_val, duals, farkas)))
                    
                elif cmd == "SOLVE_MIP":
                    for c, val in zip(self.link_constrs_mip, payload):
                        c.RHS = val
                        
                    self.m_mip.optimize()
                    
                    status = self.m_mip.Status
                    obj_val = self.m_mip.ObjVal if status == gp.GRB.OPTIMAL else -1e9
                    self.out_q.put((self.leaf_idx, "MIP", (status, obj_val)))
            
            except Exception as e:
                self.out_q.put((self.leaf_idx, "ERROR", None))
                print(f"LeafWorker {self.leaf_idx} Error: {e}")

        # Limpieza al terminar
        self.env.dispose()


class IntegerLShapedSolver:
    def __init__(self, topology: TopologyManager, blocks: List[AbstractBlock]):
            self.topology = topology
            self.blocks = blocks          # <-- Asegúrate de tener guardado self.blocks
            self.center_block = blocks[0]
            self.leaf_blocks = blocks[1:]
            
            self.time_start = 0.0
            self.time_limit = float('inf')
            self.global_upper_bound_U = 0.0
            
            self.K = len(self.leaf_blocks)
            self.in_queues = [queue.Queue() for _ in range(self.K)]
            self.out_queue = queue.Queue()
            self.workers = []
            
            # --- PROPAGACIÓN DE CONFLICTOS ---
            if all(hasattr(b, 'inherit_conflicts') for b in self.blocks):
                self._propagate_conflicts()
                
            self._prepare_workers()
            self._build_master()

    def _propagate_conflicts(self):
            print("Pre-procesamiento: Propagando conflictos Stable Set en L-Shaped...")
            changed = True
            while changed:
                changed = False
                for (u_id, v_id), edge_info in self.topology.edges.items():
                    blk_u = next(b for b in self.blocks if b.block_id == u_id)
                    blk_v = next(b for b in self.blocks if b.block_id == v_id)
                    
                    # Propagar de u hacia v
                    if blk_v.inherit_conflicts(blk_u, edge_info.vars_v, edge_info.vars_u): 
                        changed = True
                        
                    # Propagar de v hacia u
                    if blk_u.inherit_conflicts(blk_v, edge_info.vars_u, edge_info.vars_v): 
                        changed = True

    def _prepare_workers(self):
        total_max_possible = 0.0
        
        for i, leaf in enumerate(self.leaf_blocks):
            # 1. Cota superior precalculada sincrónicamente
            m_temp = gp.Model()
            m_temp.Params.OutputFlag = 0
            leaf.build_model(parent_model=m_temp)
            m_temp.update()
            
            # Extraer Nombres de Variables para el Acople
            edge = self.topology.get_edge(self.center_block.block_id, leaf.block_id)
            temp_vars = leaf.get_vars_by_index(edge.vars_v)
            var_names = [v.VarName for v in temp_vars] 
            
            # Optimizar para el Big-M (Fijando el bug de U=0)
            m_temp.setObjective(leaf.local_objective_expr, gp.GRB.MAXIMIZE)
            m_temp.update() # <-- CRÍTICO
            m_relax = m_temp.relax()
            m_relax.optimize()
            if m_relax.Status == gp.GRB.OPTIMAL:
                total_max_possible += m_relax.ObjVal
            else:
                total_max_possible += 1000.0 # Fallback de seguridad
                
            # 2. Desenganchar objetos de C++ (Gurobi) antes del deepcopy
            orig_model = leaf.model
            orig_vars = leaf.vars
            orig_obj = leaf.local_objective_expr
            
            leaf.model = None
            leaf.vars = {}
            leaf.local_objective_expr = None
            
            leaf_copy = copy.deepcopy(leaf)
            
            leaf.model = orig_model
            leaf.vars = orig_vars
            leaf.local_objective_expr = orig_obj
            
            # 3. Levantar Worker Persistente aislado
            w = LeafWorker(i, leaf_copy, var_names, self.in_queues[i], self.out_queue)
            w.start()
            self.workers.append(w)
            
        # Amplio margen para garantizar que Big-M no limite la búsqueda
        self.global_upper_bound_U = total_max_possible + 1e5
        print(f"[*] Cota Big-M (U) inicializada en: {self.global_upper_bound_U}")

    def _build_master(self):
        self.master = gp.Model("Master_LShaped")
        self.master.Params.OutputFlag = 1
        self.master.Params.LazyConstraints = 1
        
        self.center_block.build_model(parent_model=self.master)
        
        self.theta = self.master.addVar(lb=-gp.GRB.INFINITY, ub=self.global_upper_bound_U, 
                                        obj=1.0, name="theta")
        
        full_obj = self.center_block.local_objective_expr.copy()
        full_obj.add(self.theta, 1.0)
        self.master.setObjective(full_obj, gp.GRB.MAXIMIZE)
        self.master.update()
        
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

    def _broadcast_and_wait(self, cmd, payloads):
        for i in range(self.K):
            self.in_queues[i].put((cmd, payloads[i]))
            
        results = [None] * self.K
        for _ in range(self.K):
            i, _, data = self.out_queue.get()
            results[i] = data
        return results

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
                
                # --- PHASE 1: LP Cuts (Benders) Paralelos ---
                lp_payloads = [
                    [x_center_sol[idx] for idx in self.center_edge_vars_indices[i]]
                    for i in range(self.K)
                ]
                
                lp_results = self._broadcast_and_wait("SOLVE_LP", lp_payloads)
                
                total_lp_obj = 0.0
                cut_expr_lp = 0.0
                possible_benders = True
                
                for i, result in enumerate(lp_results):
                    status, obj_val, duals, farkas = result
                    indices = self.center_edge_vars_indices[i]
                    vals = lp_payloads[i]
                    
                    if status == gp.GRB.OPTIMAL:
                        total_lp_obj += obj_val
                        term_const = obj_val
                        term_var = gp.LinExpr()
                        center_vars = self.center_block.get_vars_by_index(indices)
                        
                        for d, v, v_val in zip(duals, center_vars, vals):
                            term_const -= d * v_val
                            term_var.add(v, d)
                        
                        cut_expr_lp += (term_const + term_var)
                        
                    elif status == gp.GRB.INF_OR_UNBD or status == gp.GRB.INFEASIBLE:
                        term_const_feas = obj_val # FarkasProof positivo ajustado
                        term_var_feas = gp.LinExpr()
                        center_vars = self.center_block.get_vars_by_index(indices)
                        
                        for d, v, v_val in zip(farkas, center_vars, vals):
                            term_const_feas -= d * v_val
                            term_var_feas.add(v, d)
                            
                        model.cbLazy(term_var_feas + term_const_feas >= 0.0)
                        possible_benders = False
                    else:
                        possible_benders = False

                if possible_benders:
                    if theta_sol > total_lp_obj + 1e-4:
                        model.cbLazy(self.theta <= cut_expr_lp)
                        return 
                else:
                    return # Hemos agregado cortes de factibilidad, salir para evaluarlos
                
                # --- PHASE 2: MIP Cuts (Integer L-Shaped) Paralelos ---
                mip_payloads = lp_payloads
                mip_results = self._broadcast_and_wait("SOLVE_MIP", mip_payloads)
                
                total_mip_obj = 0.0
                all_coupling_indices = set()
                must_add_integer_cut = False
                
                for i, result in enumerate(mip_results):
                    status, obj_val = result
                    indices = self.center_edge_vars_indices[i]
                    all_coupling_indices.update(indices)
                    
                    if status == gp.GRB.OPTIMAL:
                        total_mip_obj += obj_val
                    elif status == gp.GRB.INF_OR_UNBD or status == gp.GRB.INFEASIBLE:
                        hamming_dist = self._get_hamming_distance_expr(model, indices, x_center_sol)
                        model.cbLazy(hamming_dist >= 1)
                        must_add_integer_cut = True
                    else:
                        total_mip_obj += -1e9 

                if must_add_integer_cut:
                    return

                if theta_sol > total_mip_obj + 1e-4:
                    sorted_all_indices = sorted(list(all_coupling_indices))
                    hamming_dist_union = self._get_hamming_distance_expr(model, sorted_all_indices, x_center_sol)
                    
                    Q_val = total_mip_obj
                    U_val = self.global_upper_bound_U
                    
                    cut_rhs = (U_val - Q_val) * hamming_dist_union + Q_val
                    model.cbLazy(self.theta <= cut_rhs)

        try:
            self.master.optimize(cb)
        finally:
            # Apagado limpio de Workers
            for q in self.in_queues:
                q.put(("STOP", None))
            for w in self.workers:
                w.join()
        
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
            if metrics["dual_bound"] < float('inf') and metrics["primal_bound"] > -float('inf'):
                denom = abs(metrics["dual_bound"])
                if denom < 1e-10: denom = 1.0
                metrics["gap"] = abs(metrics["dual_bound"] - metrics["primal_bound"]) / denom
                        
        if self.master.Status == gp.GRB.OPTIMAL: metrics["status"] = "Optimal"
        elif self.master.Status == gp.GRB.TIME_LIMIT: metrics["status"] = "TimeLimit"
        else: metrics["status"] = f"Code_{self.master.Status}"
        
        return metrics