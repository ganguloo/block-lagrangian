import time
import gurobipy as gp
import concurrent.futures
from datetime import datetime
from typing import Dict, Any
from ..instance.topology import TopologyManager
from ..solver.master import MasterProblem
from ..solver.pricing import PricingSolver
from ..monolithic.solver import MonolithicSolver

class CRGManager:
    def __init__(self, blocks, topology, strategy):
        self.blocks = blocks
        self.topology = topology
        self.strategy = strategy
        self.master = MasterProblem(len(blocks))
        self.master.register_linear_constraints(topology)
        self.pricers = [PricingSolver(b, strategy, topology) for b in blocks]
        self.cut_registry = {}
        self.active_cuts_by_edge = {}
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=16)
        self.initial_dual_bound = float('inf')

    def _propagate_conflicts(self):
        print("Pre-procesamiento: Propagando conflictos Stable Set...")
        changed = True
        while changed:
            changed = False
            for (u_id, v_id), edge_info in self.topology.edges.items():
                blk_u = next(b for b in self.blocks if b.block_id == u_id)
                blk_v = next(b for b in self.blocks if b.block_id == v_id)
                if blk_v.inherit_conflicts(blk_u, edge_info.vars_v, edge_info.vars_u): changed = True
                if blk_u.inherit_conflicts(blk_v, edge_info.vars_u, edge_info.vars_v): changed = True

    def _solve_pricing_task(self, args):
        p_idx, alpha_i, pi, mu, cuts = args
        return p_idx, self.pricers[p_idx].solve(alpha_i, pi, mu, cuts)

    def _initialize_from_monolithic(self):
        print("Inicializando con Monolítico (10w)...")
        mono = MonolithicSolver(self.topology, self.blocks)
        mono.model.Params.OutputFlag = 0
        try:
            mono.build_and_solve(work_limit=10)
        except: pass

        if mono.model.SolCount > 0:
            print(f"[Init] Solución inicial encontrada: {mono.model.ObjVal}")
            self.primal_bound = mono.model.ObjVal
            initial_vals = {}
            for i in range(len(self.blocks)):
                initial_vals[i] = mono.get_block_solution(i)
            for pricer in self.pricers: pricer.rebuild_model()
            for i, pricer in enumerate(self.pricers):
                x_full = initial_vals[i]
                orig_lbs = {v: v.LB for v in pricer.block.vars.values()}
                orig_ubs = {v: v.UB for v in pricer.block.vars.values()}
                for idx, var in pricer.block.vars.items():
                    var.LB = x_full[idx]
                    var.UB = x_full[idx]
                pricer.model.update()
                rc, obj, x_b, w_s = pricer.solve(0.0, {}, {}, {})
                self.master.add_column(i, obj, x_b, w_s, {}, self.strategy)
                for var, lb in orig_lbs.items(): var.LB = lb
                for var, ub in orig_ubs.items(): var.UB = ub
                pricer.model.update()
        else:
            print("[Init] No se encontró solución.")
            self.primal_bound = -1e9

        print("[Init] Resolviendo relajación lineal monolítica para cota dual inicial...")
        try:
            m_copy = mono.model.copy()
            m_relax = m_copy.relax()
            m_relax.Params.OutputFlag = 0
            m_relax.optimize()
            if m_relax.Status == gp.GRB.OPTIMAL:
                self.initial_dual_bound = m_relax.ObjVal
                print(f"[Init] Cota Dual Inicial (LR): {self.initial_dual_bound}")
            else:
                self.initial_dual_bound = float('inf')
        except:
            self.initial_dual_bound = float('inf')

    def _check_integrality(self) -> bool:
        for b_id, cols in self.master.lambda_vars.items():
            for var in cols.values():
                val = var.X
                if val > 1e-5 and abs(val - 1.0) > 1e-5:
                    return False
        return True

    def _run_mip_heuristic(self, metrics):
        try:
            m_heur = self.master.model.copy()
            m_heur.Params.OutputFlag = 0
            m_heur.Params.TimeLimit = 5
            for v in m_heur.getVars():
                v.VType = gp.GRB.BINARY
            m_heur.optimize()
            if m_heur.SolCount > 0:
                if m_heur.ObjVal > metrics["primal_bound"]:
                    metrics["primal_bound"] = m_heur.ObjVal
        except: pass

    def run(self, time_limit=None) -> Dict[str, Any]:
        metrics = {
            "status": "Running",
            "total_time": 0.0,
            "time_master": 0.0,
            "time_pricing": 0.0,
            "iter_outer": 0,
            "iter_total_inner": 0,
            "cols_added": 0,
            "cuts_added": 0,
            "root_lp_val": None,
            "primal_bound": 0.0,
            "dual_bound": 0.0,
            "gap": 0.0
        }

        start_total = time.time()
        if all(hasattr(i, 'inherit_conflicts') for i in self.blocks): self._propagate_conflicts()
        for p in self.pricers: p.rebuild_model()
        self._initialize_from_monolithic()

        metrics["primal_bound"] = self.primal_bound
        metrics["dual_bound"] = self.initial_dual_bound

        while True:
            metrics["iter_outer"] += 1
            print(f"\n--- Iteración Externa {metrics['iter_outer']} --- {datetime.now()}")

            if time_limit and (time.time() - start_total) > time_limit:
                metrics["status"] = "TimeLimit"
                break

            inner_cols = 0
            stop_outer = False

            while True:
                metrics["iter_total_inner"] += 1
                t0 = time.time()
                self.master.solve()
                metrics["time_master"] += time.time() - t0

                if self.master.model.Status != gp.GRB.OPTIMAL:
                    metrics["status"] = "Master_Infeasible"
                    stop_outer = True
                    break

                current_obj = self.master.model.ObjVal

                is_int = self._check_integrality()
                star = "*" if is_int else ""
                if is_int:
                    if current_obj > metrics["primal_bound"]:
                        metrics["primal_bound"] = current_obj

                if time_limit and (time.time() - start_total) > time_limit:
                    metrics["status"] = "TimeLimit"
                    stop_outer = True
                    break

                alpha, pi, mu = self.master.get_duals()
                cols_added_iter = 0
                max_rc = 0.0

                t0 = time.time()
                tasks = [(i, alpha[i], pi, mu, self.active_cuts_by_edge) for i in range(len(self.pricers))]
                results = self.executor.map(self._solve_pricing_task, tasks)

                for i, res in results:
                    if res:
                        rc, obj, x_b, w_s = res
                        if rc > 1e-4:
                            added = self.master.add_column(i, obj, x_b, w_s, self.active_cuts_by_edge, self.strategy)
                            if added:
                                cols_added_iter += 1
                                max_rc = max(max_rc, rc)
                metrics["time_pricing"] += time.time() - t0
                metrics["cols_added"] += cols_added_iter
                inner_cols += cols_added_iter

                print(f"  Iter {metrics['iter_total_inner']}: Obj {current_obj:.4f} {star}, Time {(time.time()-start_total):.1f}s, Cols +{cols_added_iter}")

                if cols_added_iter == 0:
                    metrics["dual_bound"] = current_obj
                    break

            if metrics["iter_outer"] == 1 and metrics["root_lp_val"] is None:
                metrics["root_lp_val"] = metrics["dual_bound"]

            if metrics["dual_bound"] > 0 and metrics["primal_bound"] > -1e8:
                current_gap = abs(metrics["dual_bound"] - metrics["primal_bound"]) / abs(metrics["dual_bound"])
                if current_gap < 1e-4:
                    metrics["status"] = "Gap_Closed"
                    stop_outer = True

            if stop_outer:
                print(f"Fin Outer {metrics['iter_outer']} (Interrupted): Obj {metrics['dual_bound']:.4f}, Status: {metrics['status']}")
                break

            cuts_added_iter = 0
            w_sol = {}
            for b_id, cols in self.master.lambda_vars.items():
                for sig_tuple, var in cols.items():
                    if var.X > 1e-5:
                        w_sigs_list = sig_tuple[2]
                        for nid, sig in w_sigs_list:
                            u, v = sorted((b_id, nid))
                            if (u, v) not in w_sol: w_sol[u, v] = ({}, {})
                            target_dict = w_sol[u, v][0] if b_id == u else w_sol[u, v][1]
                            target_dict[sig] = target_dict.get(sig, 0.0) + var.X

            new_constraints = []
            for (u, v), (w_u, w_v) in w_sol.items():
                violations = self.strategy.separate(w_u, w_v)
                for sig in violations:
                    if (u, v, sig) not in self.cut_registry:
                        cut_name = f"cut_{u}_{v}_{len(self.cut_registry)}"
                        lhs = gp.LinExpr()
                        for b_id in [u, v]:
                            for sig_t, var in self.master.lambda_vars[b_id].items():
                                for n_chk, col_w_sig in sig_t[2]:
                                    if n_chk == (v if b_id == u else u):
                                        w_val = self.strategy.evaluate_cut(col_w_sig, sig)
                                        if abs(w_val) > 1e-6:
                                            coeff = w_val * (1.0 if b_id == u else -1.0)
                                            lhs.add(var, coeff)
                        constr = self.master.model.addConstr(lhs == 0.0, name=cut_name)
                        new_constraints.append((constr, u, v, sig))
                        cut_id = len(self.cut_registry)
                        self.cut_registry[u, v, sig] = cut_id
                        self.master.ctr_cuts[cut_id] = constr
                        if (u,v) not in self.active_cuts_by_edge: self.active_cuts_by_edge[u,v] = []
                        self.active_cuts_by_edge[u,v].append((cut_id, sig))
                        cuts_added_iter += 1

            metrics["cuts_added"] += cuts_added_iter

            if cuts_added_iter > 0:
                t0 = time.time()
                self.master.solve()
                metrics["time_master"] += time.time() - t0

            self._run_mip_heuristic(metrics)

            print(f"Fin Outer {metrics['iter_outer']}: Obj {metrics['dual_bound']:.4f}, Cuts +{cuts_added_iter}")

            if cuts_added_iter == 0 and inner_cols == 0:
                metrics["status"] = "Converged"
                break

        print(">>> Solving Final MIP (Heuristic)...")
        self.master.switch_to_binary()
        self.master.solve()
        if self.master.model.Status == gp.GRB.OPTIMAL:
             if self.master.model.ObjVal > metrics["primal_bound"]:
                 metrics["primal_bound"] = self.master.model.ObjVal
                 print(f"    New Primal Bound found: {metrics['primal_bound']}")

        metrics["total_time"] = time.time() - start_total
        if metrics["dual_bound"] > 0:
            metrics["gap"] = abs(metrics["dual_bound"] - metrics["primal_bound"]) / abs(metrics["dual_bound"])
        return metrics