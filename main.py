
import csv
import gc
import os
import platform
import datetime
from typing import List, Dict, Any
from src.blocks.stable_set import StableSetBlock
from src.blocks.matching import MatchingBlock
from src.instance.topology import TopologyManager
from src.strategies.m_lagrangian import MLagrangianStrategy
from src.strategies.v_lagrangian import VLagrangianStrategy
from src.solver.manager import CRGManager
from src.monolithic.solver import MonolithicSolver
from src.solvers.integer_lshaped import IntegerLShapedSolver
from src.solvers.scenario_decomposition import ScenarioDecompositionSolver

# ==================== CONFIGURATION ====================
OUTPUT_FILE = "benchmark_results.csv"

INSTANCE_GRID = [
    {"problem": "stable_set", "n_blocks": 15, "n_nodes": 100, "n_edges": 0, "coupling": 20, "topo": "star"},
    {"problem": "stable_set", "n_blocks": 15, "n_nodes": 100, "n_edges": 0, "coupling": 30, "topo": "star"},
    {"problem": "stable_set", "n_blocks": 15, "n_nodes": 100, "n_edges": 0, "coupling": 40, "topo": "star"},
#    {"problem": "matching", "n_blocks": 5, "n_nodes": 50, "n_edges": 100, "coupling": 10, "topo": "star"},
#    {"problem": "matching", "n_blocks": 31, "n_nodes": 20, "n_edges": 200, "coupling": 20, "topo": "bintree"},
]

SEEDS = [i for i in range(5)]

SOLVER_CONFIGS = [
    {"name": "Monolithic", "type": "mono", "time_limit": 900},
    {"name": "CRG_VLag", "type": "crg", "class": VLagrangianStrategy, "args": {}, "time_limit": 900},
    {"name": "CRG_MLag_maxdeg2", "type": "crg", "class": MLagrangianStrategy, "args": {}, "time_limit": 900},
    {"name": "CRG_MLag_maxdeg3", "type": "crg", "class": MLagrangianStrategy, "args": {"maxdeg":3}, "time_limit": 900},
    {"name": "IntegerLShaped", "type": "lshaped", "time_limit": 900},
    {"name": "ScenarioDecomp", "type": "scenario", "time_limit": 900},
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
                        int(row["seed"]),
                        row["solver"]
                    )
                    completed.add(key)
            except: pass
    return completed

def run_experiment():
    print(f"Starting Benchmark Suite on {platform.node()}")

    completed_runs = get_completed_runs()

    fieldnames = [
        "timestamp", "host", "cpu", "problem", "topo", "n_blocks", "n_nodes", "n_edges", "coupling",
        "seed", "solver", "status", "total_time", "primal_bound", "dual_bound", "gap",
        "root_lp", "root_lp_presolved", "node_count", "iter_outer", "iter_inner", "cols", "cuts",
        "t_master", "t_pricing"
    ]

    file_exists = os.path.exists(OUTPUT_FILE)

    with open(OUTPUT_FILE, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        for inst_conf in INSTANCE_GRID:
            for seed in SEEDS:
                problem_type = inst_conf["problem"]
                n_blocks = inst_conf["n_blocks"]
                n_nodes = inst_conf["n_nodes"]
                coupling = inst_conf["coupling"]
                topo_type = inst_conf["topo"]
                n_edges = inst_conf.get("n_edges", 0)

                if problem_type == "matching":
                    max_possible = (n_nodes * (n_nodes - 1)) // 2
                    if n_edges > max_possible:
                        n_edges = max_possible

                print(f"\n>>> Processing Instance: {problem_type}, {topo_type}, {n_blocks} blocks, {n_nodes} nodes, {n_edges} edges, Seed {seed}")

                for solver_conf in SOLVER_CONFIGS:
                    run_key = (problem_type, topo_type, n_blocks, n_nodes, n_edges, coupling, seed, solver_conf["name"])

                    if run_key in completed_runs:
                        print(f"  > Skipping {solver_conf['name']} (Already done)")
                        continue

                    if (solver_conf["type"] == "lshaped" or solver_conf["type"] == "scenario") and topo_type != "star":
                        print(f"  > Skipping {solver_conf['name']} (Topology not supported)")
                        continue

                    print(f"  > Running {solver_conf['name']}...")
                    gc.collect()

                    blocks = []
                    block_sizes = []

                    for i in range(n_blocks):
                        if problem_type == "stable_set":
                            b = StableSetBlock(i, n_nodes, seed=seed+i)
                            blocks.append(b)
                            block_sizes.append(n_nodes)
                        elif problem_type == "matching":
                            b = MatchingBlock(i, n_nodes, n_edges, seed=seed+i)
                            blocks.append(b)
                            block_sizes.append(b.num_edges)

                    topology = TopologyManager(block_sizes)

                    if topo_type == "path":
                        topology.create_path(coupling)
                    elif topo_type == "star":
                        topology.create_star(0, coupling)
                    elif topo_type == "bintree":
                        topology.create_bintree(coupling)

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
                        "seed": seed,
                        "solver": solver_conf["name"]
                    }

                    try:
                        if solver_conf["type"] == "mono":
                            solver = MonolithicSolver(topology, blocks)
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
                            strategy = solver_conf["class"](**solver_conf["args"])
                            manager = CRGManager(blocks, topology, strategy)
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
                                "t_pricing": res["time_pricing"]
                            })

                        elif solver_conf["type"] == "lshaped":
                            solver = IntegerLShapedSolver(topology, blocks)
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
                            solver = ScenarioDecompositionSolver(topology, blocks)
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
                        print(f"    !!! ERROR: {e}")
                        import traceback
                        traceback.print_exc()
                        row["status"] = "Error"

                    writer.writerow(row)
                    f.flush()

    print(f"\nBenchmark Finished. Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    run_experiment()
