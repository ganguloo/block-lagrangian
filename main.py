
import csv
import gc
import os
import platform
import datetime
from typing import List, Dict, Any
from src.blocks.stable_set import StableSetBlock
from src.instance.topology import TopologyManager
from src.strategies.m_lagrangian import MLagrangianStrategy
from src.strategies.v_lagrangian import VLagrangianStrategy
from src.solver.manager import CRGManager
from src.monolithic.solver import MonolithicSolver

# ==================== CONFIGURATION ====================
OUTPUT_FILE = "benchmark_results.csv"

INSTANCE_GRID = [
    {"n_blocks": 15, "n_nodes": 100, "coupling": 20, "topo": "path"},
    {"n_blocks": 15, "n_nodes": 100, "coupling": 30, "topo": "path"},
    {"n_blocks": 15, "n_nodes": 100, "coupling": 40, "topo": "path"},
    {"n_blocks": 15, "n_nodes": 100, "coupling": 20, "topo": "bintree"},
    {"n_blocks": 15, "n_nodes": 100, "coupling": 30, "topo": "bintree"},
    {"n_blocks": 15, "n_nodes": 100, "coupling": 40, "topo": "bintree"},
    {"n_blocks": 15, "n_nodes": 100, "coupling": 20, "topo": "star"},
    {"n_blocks": 15, "n_nodes": 100, "coupling": 30, "topo": "star"},
    {"n_blocks": 15, "n_nodes": 100, "coupling": 40, "topo": "star"},
]

SEEDS = [s for s in range(5)]

SOLVER_CONFIGS = [
    {"name": "Monolithic", "type": "mono", "time_limit": 900},
    {"name": "CRG_VLag", "type": "crg", "class": VLagrangianStrategy, "args": {}, "time_limit": 900},
    {"name": "CRG_MLag_maxdeg2", "type": "crg", "class": MLagrangianStrategy, "args": {"maxdeg": 2}, "time_limit": 900},
    {"name": "CRG_MLag_maxdeg3", "type": "crg", "class": MLagrangianStrategy, "args": {"maxdeg": 3}, "time_limit": 900},
]
# ========================================================

def get_completed_runs():
    completed = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r') as f:
            try:
                reader = csv.DictReader(f)
                for row in reader:
                    key = (row["topo"], int(row["n_blocks"]), int(row["n_nodes"]), int(row["coupling"]), int(row["seed"]), row["solver"])
                    completed.add(key)
            except: pass
    return completed

def run_experiment():
    print(f"Starting Benchmark Suite on {platform.node()}")

    completed_runs = get_completed_runs()
    print(f"Found {len(completed_runs)} completed runs. Resuming...")

    fieldnames = [
        "timestamp", "host", "cpu", "topo", "n_blocks", "n_nodes", "coupling",
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
                print(f"\n>>> Generating Instance: {inst_conf}, Seed {seed}")
                n_blocks = inst_conf["n_blocks"]
                n_nodes = inst_conf["n_nodes"]

                for solver_conf in SOLVER_CONFIGS:
                    run_key = (inst_conf["topo"], n_blocks, n_nodes, inst_conf["coupling"], seed, solver_conf["name"])
                    if run_key in completed_runs:
                        print(f"  > Skipping {solver_conf['name']} (Already done)")
                        continue

                    print(f"  > Running {solver_conf['name']}...")
                    gc.collect()

                    topology = TopologyManager(n_blocks)
                    blocks = [StableSetBlock(i, n_nodes, seed=seed+i) for i in range(n_blocks)]

                    if inst_conf["topo"] == "path":
                        topology.create_path(n_nodes, inst_conf["coupling"])
                    elif inst_conf["topo"] == "star":
                        topology.create_star(0, n_nodes, inst_conf["coupling"])
                    elif inst_conf["topo"] == "bintree":
                        topology.create_bintree(n_nodes, inst_conf["coupling"])

                    row = {
                        "timestamp": datetime.datetime.now().isoformat(),
                        "host": platform.node(),
                        "cpu": platform.processor(),
                        "topo": inst_conf["topo"],
                        "n_blocks": n_blocks,
                        "n_nodes": n_nodes,
                        "coupling": inst_conf["coupling"],
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
