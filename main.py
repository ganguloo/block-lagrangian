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

# ==================== CONFIGURATION ====================
OUTPUT_FILE = "benchmark_results.csv"

# Grid actualizado con atributo "problem" y parámetros específicos
INSTANCE_GRID = [
    # Instancias de Stable Set (requieren 'n_nodes' y 'density' implícita o default)
    {"problem": "stable_set", "n_blocks": 5, "n_nodes": 50, "coupling": 10, "topo": "path"},
    
    # Instancias de Matching (requieren 'n_nodes' y 'n_edges')
    {"problem": "matching", "n_blocks": 5, "n_nodes": 50, "n_edges": 100, "coupling": 10, "topo": "path"},
    {"problem": "matching", "n_blocks": 5, "n_nodes": 50, "n_edges": 100, "coupling": 10, "topo": "star"},
]

SEEDS = [0, 1]

SOLVER_CONFIGS = [
    {"name": "Monolithic", "type": "mono", "time_limit": 300},
    {"name": "CRG_VLag", "type": "crg", "class": VLagrangianStrategy, "args": {}, "time_limit": 300},
]
# ========================================================

def get_completed_runs():
    completed = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r') as f:
            try:
                reader = csv.DictReader(f)
                for row in reader:
                    # La clave ahora debe incluir el problema para diferenciar
                    key = (row.get("problem", "unknown"), row["topo"], int(row["n_blocks"]), row["seed"], row["solver"])
                    completed.add(key)
            except: pass
    return completed

def run_experiment():
    print(f"Starting Benchmark Suite on {platform.node()}")
    
    completed_runs = get_completed_runs()
    
    # Actualizamos campos del CSV para incluir 'problem' y 'n_edges'
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
                print(f"\n>>> Generating Instance: {inst_conf}, Seed {seed}")
                
                problem_type = inst_conf["problem"]
                n_blocks = inst_conf["n_blocks"]
                n_nodes = inst_conf["n_nodes"]
                coupling = inst_conf["coupling"]
                topo_type = inst_conf["topo"]
                
                # Parámetros específicos por problema
                if problem_type == "matching":
                    n_edges = inst_conf["n_edges"]
                    # En Matching, las variables son aristas. El tamaño del bloque es n_edges.
                    block_var_size = n_edges 
                else:
                    n_edges = 0 # No aplica o es derivado
                    # En Stable Set, las variables son nodos.
                    block_var_size = n_nodes

                for solver_conf in SOLVER_CONFIGS:
                    run_key = (problem_type, topo_type, n_blocks, seed, solver_conf["name"])
                    if run_key in completed_runs:
                        print(f"  > Skipping {solver_conf['name']} (Already done)")
                        continue

                    print(f"  > Running {solver_conf['name']}...")
                    gc.collect() 
                    
                    # 1. Configurar Topología basada en el tamaño de variables (edges o nodes)
                    block_sizes = [block_var_size] * n_blocks
                    topology = TopologyManager(block_sizes)
                    
                    if topo_type == "path":
                        topology.create_path(coupling)
                    elif topo_type == "star":
                        topology.create_star(0, coupling)
                    elif topo_type == "bintree":
                        topology.create_bintree(coupling)

                    # 2. Crear Bloques según el problema
                    blocks = []
                    for i in range(n_blocks):
                        if problem_type == "stable_set":
                            # Stable Set usa density (default en clase si no se pasa)
                            blocks.append(StableSetBlock(i, n_nodes, seed=seed+i))
                        elif problem_type == "matching":
                            blocks.append(MatchingBlock(i, n_nodes, n_edges, seed=seed+i))

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

                    except Exception as e:
                        print(f"    !!! ERROR: {e}")
                        import traceback
                        traceback.print_exc()
                        row["status"] = "Error"
                    
                    writer.writerow(row)
                    f.flush()

    print(f"\\nBenchmark Finished. Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    run_experiment()