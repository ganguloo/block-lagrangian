import gurobipy as gp
import threading
from typing import Dict, Tuple, List, Any

class PricingWorker(threading.Thread):
    def __init__(self, p_idx, block, strategy, topology, num_threads, in_q, out_q, semaphore):
        super().__init__()
        self.p_idx = p_idx
        self.block = block
        self.strategy = strategy
        self.topology = topology
        self.num_threads = num_threads
        self.in_q = in_q
        self.out_q = out_q
        self.semaphore = semaphore

        self.env = None
        self.model = None
        self.boundary_vars = {}

    def run(self):
        # 1. Inicialización de Entorno y Modelo estrictamente dentro del hilo
        self.env = gp.Env(empty=True)
        self.env.setParam("OutputFlag", 0)
        self.env.setParam("Threads", self.num_threads)
        self.env.start()
        
        self.model = gp.Model(self.block.name, env=self.env)
        self.model.Params.NonConvex = 2
        
        self.block.build_model(parent_model=self.model)
        self.model.update()
        
        for nid in self.topology.get_neighbors(self.block.block_id):
            u, v = sorted((self.block.block_id, nid))
            edge = self.topology.get_edge(u, v)
            indices = edge.vars_u if self.block.block_id == u else edge.vars_v
            self.boundary_vars[nid] = self.block.get_vars_by_index(indices)

        # 2. Bucle de Eventos (Patrón Actor)
        while True:
            cmd, payload = self.in_q.get()
            
            try:
                if cmd == "STOP":
                    self.out_q.put((self.p_idx, "STOP_ACK", True))
                    break
                    
                elif cmd == "INIT_COLUMN":
                    # Inicialización usando la solución monolítica fraccionaria
                    x_full = payload
                    orig_bounds = []
                    for idx, var in self.block.vars.items():
                        orig_bounds.append((var, var.LB, var.UB))
                        var.LB = x_full[idx]
                        var.UB = x_full[idx]
                    
                    self.model.update()
                    obj = self.block.local_objective_expr.copy()
                    self.model.setObjective(obj, gp.GRB.MAXIMIZE)
                    
                    with self.semaphore:
                        self.model.optimize()
                        
                    res = None
                    if self.model.Status == gp.GRB.OPTIMAL:
                        x_bounds = {}
                        w_sigs = {}
                        for nid, vars_list in self.boundary_vars.items():
                            vals = [int(round(v.X)) for v in vars_list]
                            x_bounds[nid] = vals
                            w_sigs[nid] = self.strategy.get_w_signature(vals)
                        res = (self.model.ObjVal, self.block.local_objective_expr.getValue(), x_bounds, w_sigs)
                        
                    # Restaurar las cotas originales
                    for var, lb, ub in orig_bounds:
                        var.LB = lb
                        var.UB = ub
                    self.model.update()
                    
                    self.out_q.put((self.p_idx, "INIT_RESULT", res))
                    
                elif cmd == "SOLVE":
                    # Iteración normal de Pricing
                    alpha, pi, mu, active_cuts = payload
                    
                    obj = self.block.local_objective_expr.copy()
                    obj.addConstant(-alpha)

                    for nid, vars_list in self.boundary_vars.items():
                        u, v = sorted((self.block.block_id, nid))
                        is_u = (self.block.block_id == u)
                        factor = -1.0 if is_u else 1.0
                        for k, var in enumerate(vars_list):
                            if (u,v,k) in pi:
                                obj.add(var, factor * pi[u,v,k])

                    penalty_inputs = []
                    for nid, vars_list in self.boundary_vars.items():
                        u, v = sorted((self.block.block_id, nid))
                        cuts = active_cuts.get((u, v), [])
                        is_u = (self.block.block_id == u)
                        for cut_id, sig in cuts:
                            if cut_id in mu:
                                real_factor = -1.0 if is_u else 1.0
                                penalty_inputs.append((cut_id, sig, real_factor))
                        if penalty_inputs:
                            pen = self.strategy.apply_pricing_penalty(
                                self.model, vars_list,
                                [(c, s, f) for c,s,f in penalty_inputs],
                                mu
                            )
                            obj = obj + pen
                        penalty_inputs = []

                    self.model.setObjective(obj, gp.GRB.MAXIMIZE)
                    
                    with self.semaphore:
                        self.model.optimize()

                    if self.model.Status != gp.GRB.OPTIMAL:
                        self.out_q.put((self.p_idx, "RESULT", None))
                    else:
                        x_bounds = {}
                        w_sigs = {}
                        for nid, vars_list in self.boundary_vars.items():
                            vals = [int(round(v.X)) for v in vars_list]
                            x_bounds[nid] = vals
                            w_sigs[nid] = self.strategy.get_w_signature(vals)
                        
                        res = (self.model.ObjVal, self.block.local_objective_expr.getValue(), x_bounds, w_sigs)
                        self.out_q.put((self.p_idx, "RESULT", res))
                        
            except Exception as e:
                self.out_q.put((self.p_idx, "ERROR", None))
                print(f"PricingWorker {self.p_idx} Error: {e}")

        # Limpieza al destruir el hilo
        self.env.dispose()