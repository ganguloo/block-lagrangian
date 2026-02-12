import numpy as np
import gurobipy as gp
from src.instance.topology import TopologyManager
from src.blocks.dominating_set import DominatingSetBlock
from src.monolithic.solver import MonolithicSolver
from src.solver.manager import CRGManager
from src.strategies.v_lagrangian import VLagrangianStrategy

def generate_global_graph(total_nodes, density, seed):
    """
    Genera un grafo aleatorio global (Erdos-Renyi).
    Retorna lista de todas las aristas (u, v) con u < v.
    """
    rng = np.random.default_rng(seed)
    edges = []
    for i in range(total_nodes):
        for j in range(i + 1, total_nodes):
            if rng.random() < density:
                edges.append((i, j))
    return edges

def get_subgraph_edges(nodes, global_edges):
    """Filtra las aristas para quedarse solo con las inducidas por el conjunto de nodos."""
    node_set = set(nodes)
    sub_edges = []
    for u, v in global_edges:
        if u in node_set and v in node_set:
            sub_edges.append((u, v))
    return sub_edges

def get_node_sets(topology_type, num_blocks, block_size, overlap):
    """Deduce los conjuntos de nodos globales para cada bloque."""
    block_nodes = {}
    current_new_node = 0
    
    # Bloque 0
    block_nodes[0] = list(range(current_new_node, current_new_node + block_size))
    current_new_node += block_size
    
    def get_last_k(b_idx): return block_nodes[b_idx][-overlap:]
    
    if topology_type == "Path":
        for i in range(1, num_blocks):
            prev_shared = get_last_k(i - 1)
            num_new = block_size - overlap
            new_nodes = list(range(current_new_node, current_new_node + num_new))
            current_new_node += num_new
            block_nodes[i] = prev_shared + new_nodes

    elif topology_type == "Star":
        center_shared = get_last_k(0)
        for i in range(1, num_blocks):
            num_new = block_size - overlap
            new_nodes = list(range(current_new_node, current_new_node + num_new))
            current_new_node += num_new
            block_nodes[i] = center_shared + new_nodes

    elif topology_type == "BinTree":
        for i in range(num_blocks):
            c1, c2 = 2 * i + 1, 2 * i + 2
            if c1 < num_blocks:
                parent_shared = get_last_k(i)
                num_new = block_size - overlap
                new_nodes = list(range(current_new_node, current_new_node + num_new))
                current_new_node += num_new
                block_nodes[c1] = parent_shared + new_nodes
            if c2 < num_blocks:
                parent_shared = get_last_k(i)
                num_new = block_size - overlap
                new_nodes = list(range(current_new_node, current_new_node + num_new))
                current_new_node += num_new
                block_nodes[c2] = parent_shared + new_nodes

    return block_nodes, current_new_node

def run_dominating_experiment():
    # --- Parámetros ---
    NUM_BLOCKS = 7
    NODES_PER_BLOCK = 80
    COMMON_NODES = 20
    DENSITY = 0.2  # Densidad media-baja para que el DS no sea trivial (1 nodo)
    
    TOPOLOGIES = ["Path", "Star", "BinTree"]
    
    for topo_type in TOPOLOGIES:
        print(f"\n==================================================")
        print(f"Min Cardinality Dominating Set: {topo_type}")
        print(f"Blocks: {NUM_BLOCKS}, Size: {NODES_PER_BLOCK}, Overlap: {COMMON_NODES}")
        print(f"==================================================")

        # 1. Deducir Nodos y Generar Grafo Global
        block_nodes_map, total_nodes = get_node_sets(topo_type, NUM_BLOCKS, NODES_PER_BLOCK, COMMON_NODES)
        print(f"Total Unique Nodes: {total_nodes}")
        
        global_edges = generate_global_graph(total_nodes, DENSITY, seed=42)
        print(f"Global Edges Generated: {len(global_edges)}")

        # 2. Crear Bloques (Subgrafos Inducidos)
        blocks = []
        for i in range(NUM_BLOCKS):
            nodes = block_nodes_map[i]
            local_edges = get_subgraph_edges(nodes, global_edges)
            
            blk = DominatingSetBlock(i, nodes, local_edges)
            blocks.append(blk)

        # 3. Construir Topología (Acoplamiento de Nodos)
        topology = TopologyManager(block_sizes=0)
        
        def try_couple(u_idx, v_idx):
            # Intersección de NODOS
            nodes_u = set(block_nodes_map[u_idx])
            nodes_v = set(block_nodes_map[v_idx])
            shared_nodes = list(nodes_u.intersection(nodes_v))
            
            if shared_nodes:
                # Acoplamos las variables x_u para u en shared_nodes
                # En DominatingSetBlock, get_vars_by_index recibe lista de int (IDs de nodos)
                topology.add_coupling(u_idx, v_idx, shared_nodes, shared_nodes)
                print(f"  Link {u_idx}-{v_idx}: {len(shared_nodes)} nodos acoplados.")

        if topo_type == "Path":
            for i in range(NUM_BLOCKS - 1): try_couple(i, i+1)
        elif topo_type == "Star":
            for i in range(1, NUM_BLOCKS): try_couple(0, i)
        elif topo_type == "BinTree":
            for i in range(NUM_BLOCKS):
                c1, c2 = 2*i + 1, 2*i + 2
                if c1 < NUM_BLOCKS: try_couple(i, c1)
                if c2 < NUM_BLOCKS: try_couple(i, c2)

        # 4. Solvers
        
        # A. Monolítico
        print("\n> [1/2] Monolithic...")
        mono = MonolithicSolver(topology, blocks)
        res_mono = mono.build_and_solve(time_limit=900)
        # El objetivo será negativo (ej. -15)
        true_cardinality = -res_mono['primal_bound'] if res_mono['primal_bound'] > -float('inf') else float('inf')
        print(f"Monolithic: Status={res_mono['status']}, Obj={res_mono['primal_bound']} (Card: {true_cardinality})")

        # B. CRG
        print("\n> [2/2] CRG (V-Lagrangian)...")
        strategy = VLagrangianStrategy()
        crg = CRGManager(blocks, topology, strategy)
        res_crg = crg.run(time_limit=900)
        
        final_card = -res_crg['primal_bound'] if res_crg['primal_bound'] > -float('inf') else float('inf')
        
        print(f"\nResults {topo_type}:")
        print(f"  Dual Bound (Relaxed): {res_crg['dual_bound']:.4f}")
        print(f"  Primal Bound (Best):  {res_crg['primal_bound']:.4f} (Card: {final_card})")
        print(f"  Gap:                  {res_crg['gap']:.4%}")

if __name__ == "__main__":
    run_dominating_experiment()