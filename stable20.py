import argparse
import csv
import gc
import os
import platform
import datetime
import math
import traceback
import json
import contextlib
import concurrent.futures
import multiprocessing as mp
from typing import List, Dict, Any

# ==================== IMPORTS DE BLOQUES Y SOLVERS ====================
from src.blocks.stable_set import StableSetBlock
from src.blocks.dominating_set import DominatingSetBlock
from src.blocks.capacity_expansion import CapacityExpansionBlock
from src.instance.topology import TopologyManager
from src.strategies.m_lagrangian import MLagrangianStrategy
from src.strategies.v_lagrangian import VLagrangianStrategy
from src.strategies.exact_m_lagrangian import ExactMLagrangianStrategy
from src.strategies.reflected_m_lagrangian import ReflectedMLagrangianStrategy
from src.strategies.generalized_m_lagrangian import GeneralizedMLagrangianStrategy
from src.strategies.hybrid_m_lagrangian import HybridMLagrangianStrategy
from src.solver.manager import CRGManager
from src.monolithic.solver import MonolithicSolver
from src.solvers.integer_lshaped import IntegerLShapedSolver
from src.solvers.scenario_decomposition import ScenarioDecompositionSolver
import gurobipy as gp
from src.blocks.binary_qp import QPBlock

# ==================== CONFIGURATION ====================

OUTPUT_FILE = "stable20.csv"


INSTANCE_GRID = [
    {"problem": "stable_set", "n_blocks": 15, "n_nodes": 100, "n_edges": 500, "coupling": 20, "topo": "star"},
    {"problem": "stable_set", "n_blocks": 15, "n_nodes": 100, "n_edges": 500, "coupling": 20, "topo": "path"},
    {"problem": "stable_set", "n_blocks": 15, "n_nodes": 100, "n_edges": 500, "coupling": 20, "topo": "bintree"},
    {"problem": "stable_set", "n_blocks": 15, "n_nodes": 100, "n_edges": 500, "coupling": 20, "topo": "random_tree"},
    {"problem": "stable_set", "n_blocks": 15, "n_nodes": 100, "n_edges": 500, "coupling": 20, "topo": "star", "stochastic": True},
    {"problem": "stable_set", "n_blocks": 15, "n_nodes": 100, "n_edges": 500, "coupling": 20, "topo": "bintree", "stochastic": True},
]

SEEDS = [i for i in range(5)]

SOLVER_CONFIGS = [
#    {"name": "Monolithic", "type": "mono", "time_limit": 1800},
    {"name": "CRG_VLag_f100", "type": "crg", "class": VLagrangianStrategy, "args": {"factor":1.0}, "time_limit": 1800},
    {"name": "CRG_ExactMLag_f050", "type": "crg", "class": ExactMLagrangianStrategy, "args": {"factor":0.5}, "time_limit": 1800},
    {"name": "CRG_HybridMLag_f100_f050_outer5", "type": "crg", "class": HybridMLagrangianStrategy, "args": {"v_factor":1.0, "m_factor":0.5, "max_outer_iters":5}, "time_limit": 1800},
    {"name": "CRG_ReflectMLag_f050", "type": "crg", "class": ReflectedMLagrangianStrategy, "args": {"factor":0.5}, "time_limit": 1800},
    {"name": "CRG_GeneralMLag_f050", "type": "crg", "class": GeneralizedMLagrangianStrategy, "args": {"factor":0.5}, "time_limit": 1800},
    {"name": "IntegerLShaped", "type": "lshaped", "time_limit": 1800},
    {"name": "ScenarioDecomp", "type": "scenario", "time_limit": 1800},
]
# ========================================================

def get_completed_runs(parallel=None, workers=None, threads=None):
    completed = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r') as f:
            try:
                reader = csv.DictReader(f)
                for row in reader:
                    key = (
                        row.get("problem", "unknown"),
                        row["topo"],
                        int(row["n_blocks"]),
                        int(row["n_nodes"]),
                        int(row.get("n_edges", 0)),
                        int(row["coupling"]),
                        row["stochastic"],
                        int(row["seed"]),
                        row["solver"]
                    )
                    if parallel is not None and workers is not None and threads is not None:
                        key += (
                            int(row.get("parallel", -1)),
                            int(row.get("workers", -1)),
                            int(row.get("threads", -1)),
                        )
                    completed.add(key)
            except: pass
    return completed

class AutoFlushFile:
    """ Wrapper que fuerza a que cada print se escriba al disco duro inmediatamente """
    def __init__(self, f):
        self.f = f
    def write(self, x):
        self.f.write(x)
        self.f.flush()
    def flush(self):
        self.f.flush()

def run_single_experiment(inst_conf, seed, solver_conf, parallel, workers, threads, logdir):
    problem_type = inst_conf["problem"]
    n_blocks = inst_conf.get("n_blocks", 0)
    n_nodes = inst_conf.get("n_nodes", 0)
    coupling = inst_conf.get("coupling", 0)
    topo_type = inst_conf["topo"]
    n_edges = inst_conf.get("n_edges", 0)
    is_stochastic = inst_conf.get("stochastic", False)

    if problem_type in ["matching", "stable_set", "dominating_set"]:
        max_possible = (n_nodes * (n_nodes - 1)) // 2
        if n_edges > max_possible:
            n_edges = max_possible

    row = {
        "timestamp": datetime.datetime.now().isoformat(),
        "host": platform.node(),
        "cpu": platform.processor(),
        "problem": problem_type,
        "topo": topo_type,
        "n_blocks": n_blocks,
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "coupling": coupling,
        "stochastic": is_stochastic,
        "seed": seed,
        "solver": solver_conf["name"],
        "parallel": parallel,
        "workers": workers,
        "threads": threads,
        "thread_budget_nominal": parallel * workers * threads,
        "thread_budget_conservative": parallel * (workers + 1) * threads,
    }

    # 1. CONSTRUCCIÓN DEL NOMBRE DE ARCHIVO
    safe_solver_name = solver_conf["name"].replace(" ", "_").replace("/", "_")
    log_filename = f"{problem_type}_{topo_type}_b{n_blocks}_n{n_nodes}_m{n_edges}_c{coupling}_stoch{is_stochastic}_s{seed}_{safe_solver_name}.log"
    log_filepath = os.path.join(logdir, log_filename)

    # 2. TRUNCAR EL ARCHIVO Y ESCRIBIR EL ENCABEZADO DE PYTHON (Para evitar que Gurobi lo pise)
    with open(log_filepath, 'w') as f:
        f.write("============================================================\n")
        f.write(f"STARTING JOB: {log_filename}\n")
        f.write(f"Timestamp: {row['timestamp']}\n")
        f.write(f"Parallel experiments: {parallel}\n")
        f.write(f"Subproblem workers per experiment: {workers}\n")
        f.write(f"Gurobi threads per subproblem/master: {threads}\n")
        f.write(f"Monolithic Gurobi threads: {workers * threads}\n")
        f.write(f"Nominal total thread budget: {parallel * workers * threads}\n")
        f.write(f"Conservative total thread budget: {parallel * (workers + 1) * threads}\n")
        f.write("============================================================\n\n")
        f.flush()

    # 3. REDIRECCIONAR GUROBI DIRECTO AL ARCHIVO
    gp.setParam('OutputFlag', 1)        # Habilitar output
    gp.setParam('LogToConsole', 0)      # Apagarlo en la consola global
    gp.setParam('LogFile', log_filepath)# Escribir el log de Gurobi aquí

    # Fallback for any model that still uses Gurobi's default environment.
    gp.setParam('Threads', max(1, workers * threads))

# 4. REDIRECCIONAR PYTHON PRINTS (usando el AutoFlushFile)
    with open(log_filepath, 'a') as log_file:
        flushed_log = AutoFlushFile(log_file)
        with contextlib.redirect_stdout(flushed_log), contextlib.redirect_stderr(flushed_log):
            try:
                blocks = []
                block_sizes = []


                for i in range(n_blocks):
                    obj_factor = 1.0
                    if is_stochastic:
                        if topo_type == "star":
                            obj_factor = n_blocks - 1 if i == 0 else 1.0
                        elif topo_type == "bintree":
                            stages = int(math.log2(n_blocks + 1))
                            t = math.floor(math.log2(i + 1)) + 1
                            obj_factor = 2 ** (stages - t)

                    if problem_type == "stable_set":
                        b = StableSetBlock(i, n_nodes, num_edges=n_edges, seed=seed+i, obj_factor=obj_factor)
                        block_sizes.append(n_nodes)
                    elif problem_type == "dominating_set":
                        b = DominatingSetBlock(i, n_nodes, num_edges=n_edges, seed=seed+i, obj_factor=obj_factor)
                        block_sizes.append(n_nodes)
                    elif problem_type == "capacity_expansion":
                        b = CapacityExpansionBlock(i, num_facilities=coupling, num_clients=n_nodes, seed=seed+i, obj_factor=obj_factor)
                        block_sizes.append(2 * coupling)
                    elif problem_type == "boxqp":
                        b = QPBlock(
                            block_id=i,
                            n_nodes=n_nodes,
                            num_edges=n_edges,
                            bias=inst_conf.get("bias", 0.0),
                            seed=seed+i,
                            linearize=inst_conf.get("linearize", True)
                        )
                        block_sizes.append(n_nodes)

                    blocks.append(b)

                    topology = TopologyManager(block_sizes)

                    if topo_type == "path":
                        topology.create_path(coupling)
                    elif topo_type == "star":
                        topology.create_star(0, coupling)
                    elif topo_type == "bintree":
                        topology.create_bintree(coupling)
                    elif topo_type == "random_tree":
                        topology.create_random_tree(coupling, seed=seed)

                # ----- EJECUCIÓN DEL SOLVER -----
                if solver_conf["type"] == "mono":
                    solver = MonolithicSolver(topology, blocks, threads=workers * threads)
                    solver.model.Params.OutputFlag = 1
                    solver.model.Params.LogToConsole = 0

                    res = solver.build_and_solve(time_limit=solver_conf["time_limit"])
                    row.update({
                        "status": res["status"],
                        "total_time": res["total_time"],
                        "primal_bound": res["primal_bound"],
                        "dual_bound": res["dual_bound"],
                        "gap": res["gap"] * 100,
                        "root_lp": res["root_lp_val"],
                        "root_lp_presolved": res["root_lp_presolved_val"],
                        "node_count": res["node_count"]
                    })

                elif solver_conf["type"] == "crg":
                    strategy_args = solver_conf.get("args", {}).copy()
                    strategy_args["threads"] = workers * threads
                    strategy = solver_conf["class"](**strategy_args)

                    manager = CRGManager(blocks, topology, strategy, num_workers=workers, threads=threads)
                    res = manager.run(time_limit=solver_conf["time_limit"])
                    row.update({
                        "status": res["status"],
                        "total_time": res["total_time"],
                        "primal_bound": res["primal_bound"],
                        "dual_bound": res["dual_bound"],
                        "gap": res["gap"] * 100,
                        "root_lp": res["root_lp_val"],
                        "iter_outer": res["iter_outer"],
                        "iter_inner": res["iter_total_inner"],
                        "cols": res["cols_added"],
                        "cuts": res["cuts_added"],
                        "t_master": res["time_master"],
                        "t_pricing": res["time_pricing"],
                        "avg_t_inner": res["time_pricing"]/max(res["iter_total_inner"],1),
                        "cut_col_ratio": res["cuts_added"]/max(res["cols_added"],1),
                        "t_pricing_seq": res.get("time_pricing_seq", 0.0)
                    })

                    if "cut_history" in res and res["cut_history"]:
                        cuts_filename = log_filename.replace(".log", "_cuts.json")
                        cuts_filepath = os.path.join(logdir, cuts_filename)
                        try:
                            with open(cuts_filepath, 'w', encoding='utf-8') as f:
                                json.dump(res["cut_history"], f, indent=2)
                        except Exception as e:
                            print(f"Advertencia: No se pudo guardar el historial de cortes: {e}")

                elif solver_conf["type"] == "lshaped":
                    solver = IntegerLShapedSolver(topology, blocks, num_workers=workers, threads=threads)
                    res = solver.solve(time_limit=solver_conf["time_limit"])
                    row.update({
                        "status": res["status"],
                        "total_time": res["total_time"],
                        "primal_bound": res["primal_bound"],
                        "dual_bound": res["dual_bound"],
                        "gap": res["gap"] * 100,
                        "node_count": res["node_count"]
                    })

                elif solver_conf["type"] == "scenario":
                    solver = ScenarioDecompositionSolver(topology, blocks, num_workers=workers, threads=threads)
                    res = solver.solve(time_limit=solver_conf["time_limit"])
                    row.update({
                        "status": res["status"],
                        "total_time": res["total_time"],
                        "primal_bound": res["primal_bound"],
                        "dual_bound": res["dual_bound"],
                        "gap": res["gap"] * 100,
                        "iter_outer": res["iter"]
                    })

            except Exception as e:
                err_msg = traceback.format_exc()
                row["status"] = f"Error"
                row["total_time"] = 0.0
                print(f"\n[CRASH ERROR]\n{err_msg}", flush=True)

            finally:
                if 'blocks' in locals(): del blocks
                if 'topology' in locals(): del topology
                gc.collect()

    # Cerrar el archivo de Gurobi liberando el lock para el siguiente proceso
    gp.setParam('LogFile', '')

    return row

def main():
    parser = argparse.ArgumentParser(description="Unified Benchmark Runner")
    parser.add_argument("--parallel", type=int, required=True, help="Número de experimentos ejecutados en paralelo.")
    parser.add_argument("--workers", type=int, required=True, help="Número máximo de subproblemas simultáneos dentro de cada experimento.")
    parser.add_argument("--threads", type=int, required=True, help="Número de hilos de Gurobi por subproblema y por problema maestro.")
    parser.add_argument("--logdir", type=str, default="logs", help="Directorio donde se guardará la salida individual de cada experimento.")
    args = parser.parse_args()

    if args.parallel < 1:
        raise ValueError("--parallel must be at least 1")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if args.threads < 1:
        raise ValueError("--threads must be at least 1")

    os.makedirs(args.logdir, exist_ok=True)

    completed_runs = get_completed_runs(args.parallel, args.workers, args.threads)
    pending_tasks = []

    for inst_conf in INSTANCE_GRID:
        for seed in SEEDS:
            problem_type = inst_conf["problem"]
            n_blocks = inst_conf.get("n_blocks", 0)
            n_nodes = inst_conf.get("n_nodes", 0)
            coupling = inst_conf.get("coupling", 0)
            topo_type = inst_conf["topo"]
            n_edges = inst_conf.get("n_edges", 0)
            is_stochastic = inst_conf.get("stochastic", False)

            if problem_type in ["matching", "stable_set", "dominating_set"]:
                max_possible = (n_nodes * (n_nodes - 1)) // 2
                if n_edges > max_possible:
                    n_edges = max_possible

            for solver_conf in SOLVER_CONFIGS:
                run_key = (problem_type, topo_type, n_blocks, n_nodes, n_edges, coupling, str(is_stochastic), seed, solver_conf["name"], args.parallel, args.workers, args.threads)

                if run_key in completed_runs:
                    continue

                if (solver_conf["type"] == "lshaped" or solver_conf["type"] == "scenario") and topo_type != "star":
                    continue

                pending_tasks.append((inst_conf, seed, solver_conf))

    print(f"Total pending tasks: {len(pending_tasks)}", flush=True)
    if len(pending_tasks) == 0:
        print("All experiments completed!", flush=True)
        return

    fieldnames = [
        "timestamp", "host", "cpu", "problem", "topo", "n_blocks", "n_nodes", "n_edges", "coupling", "stochastic",
        "seed", "solver", "parallel", "workers", "threads", "thread_budget_nominal", "thread_budget_conservative",
        "status", "total_time", "primal_bound", "dual_bound", "gap",
        "root_lp", "root_lp_presolved", "node_count", "iter_outer", "iter_inner", "cols", "cuts",
        "t_master", "t_pricing", "avg_t_inner", "cut_col_ratio", "t_pricing_seq"
    ]
    file_exists = os.path.exists(OUTPUT_FILE)

    with open(OUTPUT_FILE, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
            f.flush()

        print(
            f"Starting Benchmark Suite on {platform.node()} with "
            f"parallel={args.parallel}, workers={args.workers}, threads={args.threads} "
            f"(nominal thread budget={args.parallel * args.workers * args.threads})",
            flush=True,
        )
        print(f"Logs will be saved to directory: {args.logdir}/", flush=True)
        completed_count = 0

        try:
            ctx = mp.get_context("forkserver")
        except ValueError:
            ctx = mp.get_context("spawn")

        with concurrent.futures.ProcessPoolExecutor(max_workers=args.parallel, mp_context=ctx) as executor:
            future_to_task = {
                executor.submit(
                    run_single_experiment,
                    task[0],
                    task[1],
                    task[2],
                    args.parallel,
                    args.workers,
                    args.threads,
                    args.logdir,
                ): task
                for task in pending_tasks
            }

            for future in concurrent.futures.as_completed(future_to_task):
                task_args = future_to_task[future]
                inst_info, seed_info, solver_info = task_args

                try:
                    row_result = future.result()
                    writer.writerow(row_result)
                    f.flush()

                    completed_count += 1
                    status = row_result.get('status', 'Unknown')
                    time_taken = row_result.get('total_time', 0)
                    print(f"[{completed_count}/{len(pending_tasks)}] DATE {datetime.datetime.now().isoformat()} DONE: {solver_info['name']} | Seed {seed_info} | {inst_info['topo']} -> Status: {status} ({time_taken:.1f}s)", flush=True)

                except Exception as exc:
                    print(f"Task generated an exception: {exc}", flush=True)

    print(f"\nBenchmark Finished. Results saved to {OUTPUT_FILE}", flush=True)

if __name__ == "__main__":
    main()
