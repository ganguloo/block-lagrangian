import numpy as np
import gurobipy as gp
from src.instance.topology import TopologyManager
from src.blocks.general_matching import GeneralMatchingBlock
from src.monolithic.solver import MonolithicSolver
from src.solver.manager import CRGManager
from src.strategies.v_lagrangian import VLagrangianStrategy

def generate_random_undirected_graph(nodes, density, seed):
    """
    Genera un grafo aleatorio no dirigido sobre un conjunto de nodos.
    Retorna lista de aristas (u, v) con u < v.
    """
    rng = np.random.default_rng(seed)
    edges = []
    
    node_list = sorted(list(nodes))
    n = len(node_list)
    
    for i in range(n):
        for j in range(i + 1, n): # Solo j > i para no dirigidos
            if rng.random() < density:
                u, v = node_list[i], node_list[j]
                edges.append((u, v))
                
    return edges

def get_node_sets(topology_type, num_blocks, block_size, overlap):
    """
    Deduce los conjuntos de nodos globales para cada bloque.
    Misma lógica que en ASP/LOP.
    """
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

def run_matching_experiment():
    # --- Parámetros ---
    NUM_BLOCKS = 15
    NODES_PER_BLOCK = 100
    COMMON_NODES = 20
    DENSITY = 0.6
    
    TOPOLOGIES = ["Path", "Star", "BinTree"]
    
    for topo_type in TOPOLOGIES:
        print(f"\n==================================================")
        print(f"Max Cardinality Matching: {topo_type}")
        print(f"Blocks: {NUM_BLOCKS}, Size: {NODES_PER_BLOCK}, Overlap: {COMMON_NODES}")
        print(f"==================================================")

        # 1. Deducir Nodos Globales
        block_nodes_map, total_nodes = get_node_sets(topo_type, NUM_BLOCKS, NODES_PER_BLOCK, COMMON_NODES)
        print(f"Total Nodes: {total_nodes}")

        # 2. Generar Bloques
        blocks = []
        block_edges_map = {} # Para intersección rápida
        
        for i in range(NUM_BLOCKS):
            nodes = block_nodes_map[i]
            edges = generate_random_undirected_graph(nodes, density=DENSITY, seed=42+i)
            
            blk = GeneralMatchingBlock(i, nodes, edges)
            blocks.append(blk)
            block_edges_map[i] = set(edges)

        # 3. Construir Topología
        topology = TopologyManager(block_sizes=0)
        
        def try_couple(u_idx, v_idx):
            # Intersección de ARISTAS
            # Solo acoplamos si la arista existe en AMBOS bloques
            shared_edges = list(block_edges_map[u_idx].intersection(block_edges_map[v_idx]))
            
            if shared_edges:
                topology.add_coupling(u_idx, v_idx, shared_edges, shared_edges)
                print(f"  Link {u_idx}-{v_idx}: {len(shared_edges)} aristas compartidas.")
            else:
                print(f"  Link {u_idx}-{v_idx}: 0 aristas compartidas.")

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
        res_mono = mono.build_and_solve(time_limit=60)
        print(f"Monolithic: Status={res_mono['status']}, Obj={res_mono['primal_bound']}")

        # B. CRG
        print("\n> [2/2] CRG (V-Lagrangian)...")
        strategy = VLagrangianStrategy()
        crg = CRGManager(blocks, topology, strategy)
        res_crg = crg.run(time_limit=180)
        
        print(f"\nResults {topo_type}:")
        print(f"  Dual Bound:   {res_crg['dual_bound']:.4f}")
        print(f"  Primal Bound: {res_crg['primal_bound']:.4f}")
        print(f"  Gap:          {res_crg['gap']:.4%}")

if __name__ == "__main__":
    run_matching_experiment()