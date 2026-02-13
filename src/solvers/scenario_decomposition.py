import time
import math # <--- NUEVO
import gurobipy as gp
import threading
import queue
import copy
from typing import List, Dict, Any
from ..blocks.base_block import AbstractBlock
from ..instance.topology import TopologyManager
from datetime import datetime

class ScenarioWorker(threading.Thread):
    def __init__(self, k, topology, center_block_copy, leaf_block, K, rho, boundary_indices, in_q, out_q, semaphore, num_threads): # <-- Agregar num_threads
        super().__init__()
        self.k = k
        self.topology = topology
        self.center_block = center_block_copy
        self.leaf_block = leaf_block
        self.K = K
        self.rho = rho
        self.boundary_indices = boundary_indices
        self.in_q = in_q
        self.out_q = out_q
        self.semaphore = semaphore
        self.num_threads = num_threads # <-- GUARDAR

        self.env = None
        self.model = None
        self.scenario_center_var_names = {}

    def run(self):
        # 1. Environment & Model initialization inside the worker thread
        self.env = gp.Env(empty=True)
        self.env.setParam("OutputFlag", 0)
        self.env.setParam("Threads", self.num_threads) # <--- AJUSTE DINÁMICO
        self.env.start()
        self.model = gp.Model(f"Scenario_{self.k}", env=self.env)

        # 2. Build Blocks
        self.center_block.build_model(parent_model=self.model, prefix="C")
        self.model.update()

        all_center_vars = self.center_block.vars
        self.scenario_center_var_names = {idx: all_center_vars[idx].VarName for idx in self.boundary_indices}

        self.leaf_block.build_model(parent_model=self.model, prefix="L")
        self.model.update()

        # 3. Coupling Constraints
        edge = self.topology.get_edge(self.center_block.block_id, self.leaf_block.block_id)
        c_vars = [self.model.getVarByName(self.scenario_center_var_names[i]) for i in edge.vars_u]
        l_vars = [self.leaf_block.vars[i] for i in edge.vars_v]

        for vc, vl in zip(c_vars, l_vars):
            self.model.addConstr(vc == vl, name=f"couple_{vc.VarName}_{vl.VarName}")

        # 4. Objective Definition
        total_obj = gp.LinExpr()
        if self.center_block.local_objective_expr:
            total_obj.add(self.center_block.local_objective_expr, 1.0 / self.K)
        if self.leaf_block.local_objective_expr:
            total_obj.add(self.leaf_block.local_objective_expr, 1.0)

        self.model.setObjective(total_obj, gp.GRB.MAXIMIZE)
        self.model.update()

        # 5. Event Loop
        while True:
            cmd, payload = self.in_q.get()

            try:
                if cmd == "STOP":
                    # FIX: Enviar ACK antes de romper el bucle para no bloquear al Main
                    self.out_q.put((self.k, "STOP_ACK", True))
                    break

                elif cmd == "SOLVE_LP":
                    r_model = self.model.relax()
                    r_model.Params.OutputFlag = 0
                    with self.semaphore:
                        r_model.optimize()
                    res = {}
                    if r_model.Status == gp.GRB.OPTIMAL:
                        for idx, name in self.scenario_center_var_names.items():
                            v = r_model.getVarByName(name)
                            res[idx] = v.X if v else 0.0
                    self.out_q.put((self.k, "LP", res))

                elif cmd == "APPLY_BIAS":
                    avg_x, lp_x_k = payload
                    current_obj = self.model.getObjective()
                    penalty_expr = gp.LinExpr()

                    for idx, name in self.scenario_center_var_names.items():
                        var = self.model.getVarByName(name)
                        x_val = lp_x_k.get(idx, 0.0)
                        x_avg = avg_x[idx]
                        mult = self.rho * (x_avg - x_val)
                        if abs(mult) > 1e-6:
                            penalty_expr.add(var, mult)

                    current_obj.add(penalty_expr)
                    self.model.setObjective(current_obj, gp.GRB.MAXIMIZE)
                    self.model.update()
                    self.out_q.put((self.k, "BIAS_DONE", True))

                elif cmd == "SOLVE_MIP":
                    with self.semaphore:
                        self.model.optimize()
                    if self.model.Status == gp.GRB.OPTIMAL:
                        x_sol = {}
                        for idx, name in self.scenario_center_var_names.items():
                            var = self.model.getVarByName(name)
                            x_sol[idx] = int(round(var.X))
                        self.out_q.put((self.k, "MIP", (self.model.ObjVal, x_sol)))
                    else:
                        self.out_q.put((self.k, "MIP", (-1e9, None)))

                elif cmd == "EVALUATE":
                    x_cand_items = payload
                    original_bounds = []
                    for idx, val in x_cand_items:
                        name = self.scenario_center_var_names[idx]
                        v = self.model.getVarByName(name)
                        original_bounds.append((v, v.LB, v.UB))
                        v.LB = val
                        v.UB = val

                    self.model.update()
                    with self.semaphore:
                        self.model.optimize()

                    val = 0.0
                    feasible = False
                    if self.model.Status == gp.GRB.OPTIMAL:
                        val = self.model.ObjVal
                        feasible = True

                    for v, lb, ub in original_bounds:
                        v.LB = lb
                        v.UB = ub
                    self.model.update()

                    self.out_q.put((self.k, "EVAL", (val, feasible)))

                elif cmd == "ADD_CUT":
                    x_cand = payload
                    lhs = gp.LinExpr()
                    for idx, val in x_cand.items():
                        name = self.scenario_center_var_names[idx]
                        var = self.model.getVarByName(name)
                        if val > 0.5:
                            lhs.addConstant(1.0)
                            lhs.add(var, -1.0)
                        else:
                            lhs.add(var, 1.0)
                    self.model.addConstr(lhs >= 1.0, name="NoGood")
                    self.model.update()
                    self.out_q.put((self.k, "CUT_DONE", True))

            except Exception as e:
                # Prevenir deadlocks si un worker lanza una excepción interna
                self.out_q.put((self.k, "ERROR", None))
                print(f"Worker {self.k} Error: {e}")

        self.env.dispose()

class ScenarioDecompositionSolver:
    def __init__(self, topology: TopologyManager, blocks: List[AbstractBlock], rho: float = 1.0):
        self.topology = topology
        self.blocks = blocks          
        self.center_block = blocks[0]
        self.leaf_blocks = blocks[1:]
        self.K = len(self.leaf_blocks)
        self.rho = rho
        
        # --- AJUSTE DINÁMICO DE HILOS Y WORKERS ---
        self.num_workers = min(len(self.blocks), 16)
        self.num_threads = max(1, math.floor(32 / self.num_workers))
        self.semaphore = threading.Semaphore(self.num_workers) # <--- ASIGNACIÓN DINÁMICA
        
        # --- 2. PROPAGACIÓN DE CONFLICTOS ---
        # Se ejecuta antes de crear cualquier worker o modelo de Gurobi
        if all(hasattr(b, 'inherit_conflicts') for b in self.blocks):
            self._propagate_conflicts()
            
        self.boundary_indices = set()
        for leaf in self.leaf_blocks:
            edge = self.topology.get_edge(self.center_block.block_id, leaf.block_id)
            self.boundary_indices.update(edge.vars_u)
        self.sorted_boundary_indices = sorted(list(self.boundary_indices))
        
        self.in_queues = [queue.Queue() for _ in range(self.K)]
        self.out_queue = queue.Queue()
        self.workers = []

        for k, leaf in enumerate(self.leaf_blocks):
            center_copy = copy.deepcopy(self.center_block)
            w = ScenarioWorker(
                k, self.topology, center_copy, leaf, self.K, self.rho, 
                self.sorted_boundary_indices, self.in_queues[k], self.out_queue,
                self.semaphore, self.num_threads # <-- PASAR A WORKER
            )
            w.start()
            self.workers.append(w)
            
        self._apply_lagrangian_bias()

    def _propagate_conflicts(self):
            print("Pre-procesamiento: Propagando conflictos Stable Set en SD...")
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

    def _broadcast_and_wait(self, cmd, payloads=None):
        if payloads is None:
            payloads = [None] * self.K
        for k in range(self.K):
            self.in_queues[k].put((cmd, payloads[k]))

        results = [None] * self.K
        for _ in range(self.K):
            k, _, data = self.out_queue.get()
            results[k] = data
        return results

    def _apply_lagrangian_bias(self):
        lp_results = self._broadcast_and_wait("SOLVE_LP")

        sum_x = {idx: 0.0 for idx in self.sorted_boundary_indices}
        for res in lp_results:
            if res:
                for idx, val in res.items():
                    sum_x[idx] += val

        avg_x = {idx: val / self.K for idx, val in sum_x.items()}

        payloads = [(avg_x, lp_results[k]) for k in range(self.K)]
        self._broadcast_and_wait("APPLY_BIAS", payloads)

    def solve(self, time_limit=300):
        start_time = time.time()
        best_lb = -float('inf')

        metrics = {
            "method": "ScenarioDecomp",
            "status": "Running",
            "total_time": 0.0,
            "primal_bound": -float('inf'),
            "dual_bound": float('inf'),
            "gap": 1.0,
            "iter": 0
        }

        evaluated_candidates = set()

        print(f"\\n--- Starting Scenario Decomposition (Actor Model Parallel) --- {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        try:
            while (time.time() - start_time) < time_limit:
                metrics["iter"] += 1

                mip_results = self._broadcast_and_wait("SOLVE_MIP")

                current_ub = 0.0
                candidates = []
                iteration_feasible = True

                for obj, x_sol in mip_results:
                    if x_sol is None:
                        iteration_feasible = False
                        break
                    current_ub += obj
                    candidates.append(x_sol)

                if not iteration_feasible:
                    if best_lb > -float('inf'):
                        metrics["status"] = "Optimal"
                        metrics["gap"] = 0.0
                        metrics["dual_bound"] = best_lb
                    else:
                        metrics["status"] = "Infeasible"
                    break

                metrics["dual_bound"] = current_ub

                if current_ub <= best_lb + 1e-4 or math.floor(current_ub + 1e-6) == math.floor(best_lb + 1e-6):
                    metrics["status"] = "Optimal"
                    metrics["gap"] = 0.0
                    metrics["dual_bound"] = best_lb
                    metrics["primal_bound"] = best_lb
                    print(f"  SD Iter {metrics['iter']}: UB {metrics['dual_bound']:.2f} <= LB {metrics['primal_bound']:.2f}, Converged.")
                    break

                unique_cands = []
                seen_sigs = set()
                for x in candidates:
                    sig = tuple(sorted(x.items()))
                    if sig not in seen_sigs:
                        seen_sigs.add(sig)
                        if sig not in evaluated_candidates:
                            unique_cands.append(x)

                if not unique_cands:
                    if abs(current_ub - best_lb) < 1e-4:
                        metrics["status"] = "Optimal"
                    else:
                        metrics["status"] = "Stalled"

                    elapsed = time.time() - start_time
                    if best_lb > -float('inf'):
                        metrics["gap"] = max(0.0, (metrics["dual_bound"] - best_lb) / abs(metrics["dual_bound"]))
                    print(f"  SD Iter {metrics['iter']}: UB {metrics['dual_bound']:.2f}, LB {metrics['primal_bound']:.2f}, Gap {metrics['gap']:.4%} (Elapsed: {elapsed:.2f}s)")
                    break

                for x_cand in unique_cands:
                    x_cand_items = list(x_cand.items())
                    payloads = [x_cand_items] * self.K
                    eval_results = self._broadcast_and_wait("EVALUATE", payloads)

                    total_val = 0.0
                    feasible = True
                    for val, is_feas in eval_results:
                        if not is_feas:
                            feasible = False
                            break
                        total_val += val

                    val_real = total_val if feasible else -1e9

                    if val_real > best_lb:
                        best_lb = val_real
                        metrics["primal_bound"] = best_lb

                    cut_payloads = [x_cand] * self.K
                    self._broadcast_and_wait("ADD_CUT", cut_payloads)
                    evaluated_candidates.add(tuple(sorted(x_cand.items())))

                if best_lb > -float('inf'):
                    if metrics["dual_bound"] <= best_lb + 1e-4:
                        metrics["status"] = "Optimal"
                        metrics["gap"] = 0.0
                        metrics["dual_bound"] = best_lb
                        metrics["primal_bound"] = best_lb
                        elapsed = time.time() - start_time
                        print(f"  SD Iter {metrics['iter']}: UB {metrics['dual_bound']:.2f} <= LB {metrics['primal_bound']:.2f}, Converged. (Elapsed: {elapsed:.2f}s)")
                        break

                if metrics["primal_bound"] > -float('inf'):
                    denom = abs(metrics["dual_bound"])
                    if denom < 1e-10: denom = 1.0
                    diff = metrics["dual_bound"] - metrics["primal_bound"]
                    metrics["gap"] = max(0.0, diff / denom)

                elapsed = time.time() - start_time
                print(f"  SD Iter {metrics['iter']}: UB {metrics['dual_bound']:.2f}, LB {metrics['primal_bound']:.2f}, Gap {metrics['gap']:.4%} (Elapsed: {elapsed:.2f}s)")

                if metrics["gap"] < 1e-4:
                    metrics["status"] = "Optimal"
                    break

            if (time.time() - start_time) >= time_limit and metrics["status"] == "Running":
                 metrics["status"] = "TimeLimit"

        finally:
            self._broadcast_and_wait("STOP")
            for w in self.workers:
                w.join()

        metrics["total_time"] = time.time() - start_time
        print(f"--- Finished Scenario Decomposition: {metrics['status']} ---")
        return metrics
