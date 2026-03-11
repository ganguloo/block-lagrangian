import argparse
import csv
import gc
import os
import platform
import datetime
import math
import traceback
import contextlib
import concurrent.futures
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
from src.solver.manager import CRGManager
from src.monolithic.solver import MonolithicSolver
from src.solvers.integer_lshaped import IntegerLShapedSolver
from src.solvers.scenario_decomposition import ScenarioDecompositionSolver
import gurobipy as gp

# ==================== CONFIGURATION ====================
OUTPUT_FILE = "benchmark_results_n100_m4500_b15.csv"

INSTANCE_GRID = [
    {"problem": "stable_set", "n_blocks": 15, "n_nodes": 100, "n_edges": 4500, "coupling": 20, "topo": "star"},
    {"problem": "stable_set", "n_blocks": 15, "n_nodes": 100, "n_edges": 4500, "coupling": 20, "topo": "path"},
    {"problem": "stable_set", "n_blocks": 15, "n_nodes": 100, "n_edges": 4500, "coupling": 20, "topo": "bintree"},
    {"problem": "stable_set", "n_blocks": 15, "n_nodes": 100, "n_edges": 4500, "coupling": 30, "topo": "star"},
    {"problem": "stable_set", "n_blocks": 15, "n_nodes": 100, "n_edges": 4500, "coupling": 30, "topo": "path"},
    {"problem": "stable_set", "n_blocks": 15, "n_nodes": 100, "n_edges": 4500, "coupling": 30, "topo": "bintree"},
    {"problem": "stable_set", "n_blocks": 15, "n_nodes": 100, "n_edges": 4500, "coupling": 40, "topo": "star"},
    {"problem": "stable_set", "n_blocks": 15, "n_nodes": 100, "n_edges": 4500, "coupling": 40, "topo": "path"},
    {"problem": "stable_set", "n_blocks": 15, "n_nodes": 100, "n_edges": 4500, "coupling": 40, "topo": "bintree"},
    
    {"problem": "dominating_set", "n_blocks": 15, "n_nodes": 100, "n_edges": 4500, "coupling": 20, "topo": "star"},
    {"problem": "dominating_set", "n_blocks": 15, "n_nodes": 100, "n_edges": 4500, "coupling": 20, "topo": "path"},
    {"problem": "dominating_set", "n_blocks": 15, "n_nodes": 100, "n_edges": 4500, "coupling": 20, "topo": "bintree"},
    {"problem": "dominating_set", "n_blocks": 15, "n_nodes": 100, "n_edges": 4500, "coupling": 30, "topo": "star"},
    {"problem": "dominating_set", "n_blocks": 15, "n_nodes": 100, "n_edges": 4500, "coupling": 30, "topo": "path"},
    {"problem": "dominating_set", "n_blocks": 15, "n_nodes": 100, "n_edges": 4500, "coupling": 30, "topo": "bintree"},
    {"problem": "dominating_set", "n_blocks": 15, "n_nodes": 100, "n_edges": 4500, "coupling": 40, "topo": "star"},
    {"problem": "dominating_set", "n_blocks": 15, "n_nodes": 100, "n_edges": 4500, "coupling": 40, "topo": "path"},
    {"problem": "dominating_set", "n_blocks": 15, "n_nodes": 100, "n_edges": 4500, "coupling": 40, "topo": "bintree"},
]

SEEDS = [i for i in range(5)]

SOLVER_CONFIGS = [
    #{"name": "Monolithic", "type": "mono", "time_limit": 1800},
    {"name": "CRG_VLag", "type": "crg", "class": VLagrangianStrategy, "args": {}, "time_limit": 1800},
    {"name": "CRG_ExactMLag", "type": "crg", "class": ExactMLagrangianStrategy, "args": {}, "time_limit": 1800},
    {"name": "CRG_ReflectMLag", "type": "crg", "class": ReflectedMLagrangianStrategy, "args": {}, "time_limit": 1800},
    #{"name": "CRG_MLag-2-tol-6", "type": "crg", "class": MLagrangianStrategy, "args": {"tol":1e-6}, "time_limit": 1800},
    #{"name": "CRG_MLag-3", "type": "crg", "class": MLagrangianStrategy, "args": {"maxdeg":3}, "time_limit": 1800},
    {"name": "IntegerLShaped", "type": "lshaped", "time_limit": 1800},
    {"name": "ScenarioDecomp", "type": "scenario", "time_limit": 1800},
]
# ========================================================

def get_completed_runs():
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

def run_single_experiment(inst_conf, seed, solver_conf, single_threaded, logdir):
    problem_type = inst_conf["problem"]
    n_blocks = inst_conf["n_blocks"]
    n_nodes = inst_conf["n_nodes"]
    coupling = inst_conf["coupling"]
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
        "solver": solver_conf["name"]
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
        f.write(f"Single Threaded Mode: {single_threaded}\n")
        f.write("============================================================\n\n")
        f.flush()

    # 3. REDIRECCIONAR GUROBI DIRECTO AL ARCHIVO
    gp.setParam('OutputFlag', 1)        # Habilitar output
    gp.setParam('LogToConsole', 0)      # Apagarlo en la consola global
    gp.setParam('LogFile', log_filepath)# Escribir el log de Gurobi aquí
    
    if single_threaded:
        gp.setParam('Threads', 1)
    else:
        gp.setParam('Threads', 0)

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
                    
                    blocks.append(b)

                topology = TopologyManager(block_sizes)

                if topo_type == "path":
                    topology.create_path(coupling)
                elif topo_type == "star":
                    topology.create_star(0, coupling)
                elif topo_type == "bintree":
                    topology.create_bintree(coupling)

                # ----- EJECUCIÓN DEL SOLVER -----
                if solver_conf["type"] == "mono":
                    solver = MonolithicSolver(topology, blocks, single_threaded=single_threaded) 
                    solver.model.Params.OutputFlag = 1
                    solver.model.Params.LogToConsole = 0

                    res = solver.build_and_solve(time_limit=solver_conf["time_limit"])
                    row.update({
                        "status": res["status"],
                        "total_time": res["total_time"],
                        "primal_bound": res["primal_bound"],
                        "dual_bound": res["dual_bound"],
                        "gap": res["gap"],
                        "root_lp": res["root_lp_val"],
                        "root_lp_presolved": res["root_lp_presolved_val"],
                        "node_count": res["node_count"]
                    })

                elif solver_conf["type"] == "crg":
                    strategy_args = solver_conf.get("args", {}).copy()
                    strategy_args["single_threaded"] = single_threaded 
                    strategy = solver_conf["class"](**strategy_args)
                    
                    manager = CRGManager(blocks, topology, strategy, single_threaded=single_threaded) 
                    res = manager.run(time_limit=solver_conf["time_limit"])
                    row.update({
                        "status": res["status"],
                        "total_time": res["total_time"],
                        "primal_bound": res["primal_bound"],
                        "dual_bound": res["dual_bound"],
                        "gap": res["gap"],
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

                elif solver_conf["type"] == "lshaped":
                    solver = IntegerLShapedSolver(topology, blocks, single_threaded=single_threaded)
                    res = solver.solve(time_limit=solver_conf["time_limit"])
                    row.update({
                        "status": res["status"],
                        "total_time": res["total_time"],
                        "primal_bound": res["primal_bound"],
                        "dual_bound": res["dual_bound"],
                        "gap": res["gap"],
                        "node_count": res["node_count"]
                    })

                elif solver_conf["type"] == "scenario":
                    solver = ScenarioDecompositionSolver(topology, blocks, single_threaded=single_threaded)
                    res = solver.solve(time_limit=solver_conf["time_limit"])
                    row.update({
                        "status": res["status"],
                        "total_time": res["total_time"],
                        "primal_bound": res["primal_bound"],
                        "dual_bound": res["dual_bound"],
                        "gap": res["gap"],
                        "iter_outer": res["iter"]
                    })

            except Exception as e:
                err_msg = traceback.format_exc()
                row["status"] = f"Error"
                row["total_time"] = 0.0
                print(f"\n[CRASH ERROR]\n{err_msg}")

            finally:
                if 'blocks' in locals(): del blocks
                if 'topology' in locals(): del topology
                gc.collect()

    # Cerrar el archivo de Gurobi liberando el lock para el siguiente proceso
    gp.setParam('LogFile', '')
    
    return row

def main():
    parser = argparse.ArgumentParser(description="Unified Benchmark Runner")
    parser.add_argument("--workers", type=int, default=None, help="Número de workers en paralelo. Si no se entrega, se ejecuta secuencial.")
    parser.add_argument("--logdir", type=str, default="logs", help="Directorio donde se guardará la salida individual de cada experimento.")
    args = parser.parse_args()

    os.makedirs(args.logdir, exist_ok=True)

    completed_runs = get_completed_runs()
    pending_tasks = []
    
    for inst_conf in INSTANCE_GRID:
        for seed in SEEDS:
            problem_type = inst_conf["problem"]
            n_blocks = inst_conf["n_blocks"]
            n_nodes = inst_conf["n_nodes"]
            coupling = inst_conf["coupling"]
            topo_type = inst_conf["topo"]
            n_edges = inst_conf.get("n_edges", 0)
            is_stochastic = inst_conf.get("stochastic", False)

            if problem_type in ["matching", "stable_set", "dominating_set"]:
                max_possible = (n_nodes * (n_nodes - 1)) // 2
                if n_edges > max_possible:
                    n_edges = max_possible

            for solver_conf in SOLVER_CONFIGS:
                run_key = (problem_type, topo_type, n_blocks, n_nodes, n_edges, coupling, str(is_stochastic), seed, solver_conf["name"])

                if run_key in completed_runs:
                    continue

                if (solver_conf["type"] == "lshaped" or solver_conf["type"] == "scenario") and topo_type != "star":
                    continue
                
                pending_tasks.append((inst_conf, seed, solver_conf))

    print(f"Total pending tasks: {len(pending_tasks)}")
    if len(pending_tasks) == 0:
        print("All experiments completed!")
        return

    fieldnames = [
        "timestamp", "host", "cpu", "problem", "topo", "n_blocks", "n_nodes", "n_edges", "coupling", "stochastic",
        "seed", "solver", "status", "total_time", "primal_bound", "dual_bound", "gap",
        "root_lp", "root_lp_presolved", "node_count", "iter_outer", "iter_inner", "cols", "cuts",
        "t_master", "t_pricing", "avg_t_inner", "cut_col_ratio", "t_pricing_seq"
    ]
    file_exists = os.path.exists(OUTPUT_FILE)

    with open(OUTPUT_FILE, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
            f.flush()

        # ----- MODO SECUENCIAL -----
        if args.workers is None:
            print(f"Starting SEQUENTIAL Benchmark Suite on {platform.node()}")
            print(f"Logs will be saved to directory: {args.logdir}/")
            completed_count = 0
            
            for task in pending_tasks:
                inst_conf, seed, solver_conf = task
                row_result = run_single_experiment(inst_conf, seed, solver_conf, single_threaded=False, logdir=args.logdir)
                
                writer.writerow(row_result)
                f.flush()
                
                completed_count += 1
                status = row_result.get('status', 'Unknown')
                time_taken = row_result.get('total_time', 0)
                print(f"[{completed_count}/{len(pending_tasks)}] DONE: {solver_conf['name']} | Seed {seed} | {inst_conf['topo']} -> Status: {status} ({time_taken:.1f}s)")

        # ----- MODO PARALELO -----
        else:
            print(f"Starting PARALLEL Benchmark Suite on {platform.node()} with {args.workers} workers")
            print(f"Logs will be saved to directory: {args.logdir}/")
            completed_count = 0
            
            with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
                future_to_task = {
                    executor.submit(run_single_experiment, task[0], task[1], task[2], True, args.logdir): task 
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
                        print(f"[{completed_count}/{len(pending_tasks)}] DONE: {solver_info['name']} | Seed {seed_info} | {inst_info['topo']} -> Status: {status} ({time_taken:.1f}s)")
                        
                    except Exception as exc:
                        print(f"Task generated an exception: {exc}")

    print(f"\nBenchmark Finished. Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()