import numpy as np
import gurobipy as gp
from src.instance.topology import TopologyManager
from src.blocks.asp import AcyclicSubgraphBlock
from src.monolithic.solver import MonolithicSolver
from src.solver.manager import CRGManager
from src.strategies.v_lagrangian import VLagrangianStrategy

def generate_random_graph_block(nodes, density, seed):
    """Genera un grafo aleatorio sobre un conjunto de nodos dados."""
    rng = np.random.default_rng(seed)
    edges = []
    weights = {}
    
    node_list = sorted(list(nodes))
    n = len(node_list)
    
    # Generar arcos posibles (i != j)
    for i in range(n):
        for j in range(n):
            if i == j: continue
            
            if rng.random() < density:
                u, v = node_list[i], node_list[j]
                edges.append((u, v))
                # Peso entero positivo [1, 100]
                weights[(u, v)] = rng.integers(1, 101)
                
    return edges, weights

def get_node_sets(topology_type, num_blocks, block_size, overlap):
    """
    Deduce los conjuntos de nodos globales para cada bloque según la topología.
    """
    block_nodes = {}
    current_new_node = 0
    
    # Bloque 0 siempre inicia
    block_nodes[0] = list(range(current_new_node, current_new_node + block_size))
    current_new_node += block_size
    
    # Helper: Obtener los últimos K nodos de un bloque
    def get_last_k(b_idx):
        return block_nodes[b_idx][-overlap:]
    
    # Helper: Obtener los primeros K nodos de un bloque (para Star consistente)
    def get_first_k(b_idx):
        return block_nodes[b_idx][:overlap]

    if topology_type == "Path":
        # 0 -> 1 -> 2 ...
        # Overlap: Últimos K de (i-1) == Primeros K de (i)
        for i in range(1, num_blocks):
            prev_shared = get_last_k(i - 1)
            num_new = block_size - overlap
            new_nodes = list(range(current_new_node, current_new_node + num_new))
            current_new_node += num_new
            block_nodes[i] = prev_shared + new_nodes

    elif topology_type == "Star":
        # Bloque 0 es el Centro.
        # Todos los bloques hojas (1..N) comparten sus PRIMEROS K nodos
        # con los ÚLTIMOS K nodos del Centro (puerto común).
        # (Opcional: Podría ser los primeros K del centro, aquí usamos últimos K del centro)
        center_shared = get_last_k(0)
        
        for i in range(1, num_blocks):
            num_new = block_size - overlap
            new_nodes = list(range(current_new_node, current_new_node + num_new))
            current_new_node += num_new
            # Construcción: [Shared] + [New]
            block_nodes[i] = center_shared + new_nodes

    elif topology_type == "BinTree":
        # i -> hijos 2i+1, 2i+2
        # Padre comparte sus últimos K con los primeros K de ambos hijos.
        for i in range(num_blocks):
            c1, c2 = 2 * i + 1, 2 * i + 2
            
            # Hijo Izquierdo
            if c1 < num_blocks:
                parent_shared = get_last_k(i)
                num_new = block_size - overlap
                new_nodes = list(range(current_new_node, current_new_node + num_new))
                current_new_node += num_new
                block_nodes[c1] = parent_shared + new_nodes
            
            # Hijo Derecho
            if c2 < num_blocks:
                parent_shared = get_last_k(i)
                num_new = block_size - overlap
                new_nodes = list(range(current_new_node, current_new_node + num_new))
                current_new_node += num_new
                block_nodes[c2] = parent_shared + new_nodes

    return block_nodes, current_new_node

def run_asp_experiment():
    # --- Parámetros Solicitados ---
    NUM_BLOCKS = 15
    NODES_PER_BLOCK = 20
    COMMON_NODES = 7
    DENSITY = 0.5
    
    TOPOLOGIES = ["Path", "Star", "BinTree"]
    
    for topo_type in TOPOLOGIES:
        print(f"\n==================================================")
        print(f"ASP Experiment: {topo_type}")
        print(f"Blocks: {NUM_BLOCKS}, Size: {NODES_PER_BLOCK}, Overlap: {COMMON_NODES}")
        print(f"==================================================")

        # 1. Deducir Nodos Globales
        block_nodes_map, total_nodes = get_node_sets(topo_type, NUM_BLOCKS, NODES_PER_BLOCK, COMMON_NODES)
        print(f"Total Unique Nodes Deduced: {total_nodes}")

        # 2. Generar Bloques (Grafos Locales)
        blocks = []
        block_info = {} # Guardar edges para intersección rápida
        
        for i in range(NUM_BLOCKS):
            nodes = block_nodes_map[i]
            # Semilla única por bloque para grafos distintos
            edges, weights = generate_random_graph_block(nodes, density=DENSITY, seed=42+i)
            
            blk = AcyclicSubgraphBlock(i, nodes, edges, weights)
            blocks.append(blk)
            
            block_info[i] = {"nodes": set(nodes), "edges": set(edges)}

        # 3. Construir Topología con Acoplamiento Condicional
        topology = TopologyManager(block_sizes=0)
        
        def try_couple(u_idx, v_idx):
            """Intenta acoplar arcos comunes entre bloques u y v."""
            # Intersección de nodos
            shared_nodes = block_info[u_idx]["nodes"].intersection(block_info[v_idx]["nodes"])
            common_arcs = []
            
            # Fuerza bruta sobre pares de nodos compartidos (son pocos, ~Overlap^2)
            s_list = list(shared_nodes)
            for n1 in s_list:
                for n2 in s_list:
                    if n1 == n2: continue
                    arc = (n1, n2)
                    
                    # CONDICIÓN CLAVE: El arco debe existir en AMBOS grafos
                    in_u = arc in block_info[u_idx]["edges"]
                    in_v = arc in block_info[v_idx]["edges"]
                    
                    if in_u and in_v:
                        common_arcs.append(arc)
            
            if common_arcs:
                topology.add_coupling(u_idx, v_idx, common_arcs, common_arcs)
                print(f"  Link {u_idx}-{v_idx}: {len(common_arcs)} arcos acoplados.")

        # Definir conexiones según la topología abstracta
        if topo_type == "Path":
            for i in range(NUM_BLOCKS - 1):
                try_couple(i, i+1)
                
        elif topo_type == "Star":
            for i in range(1, NUM_BLOCKS):
                try_couple(0, i) # Centro (0) con Hojas (i)
                
        elif topo_type == "BinTree":
            for i in range(NUM_BLOCKS):
                c1, c2 = 2*i + 1, 2*i + 2
                if c1 < NUM_BLOCKS: try_couple(i, c1)
                if c2 < NUM_BLOCKS: try_couple(i, c2)

        # 4. Solvers
        
        # A. Monolítico
        print("\n> [1/2] Monolithic Solver...")
        mono = MonolithicSolver(topology, blocks)
        res_mono = mono.build_and_solve(time_limit=60)
        print(f"Monolithic: Status={res_mono['status']}, Obj={res_mono['primal_bound']}")

        # B. CRG
        print("\n> [2/2] CRG Solver (V-Lagrangian)...")
        strategy = VLagrangianStrategy()
        crg = CRGManager(blocks, topology, strategy)
        res_crg = crg.run(time_limit=180)
        
        print(f"\nResults {topo_type}:")
        print(f"  Dual Bound:   {res_crg['dual_bound']:.4f}")
        print(f"  Primal Bound: {res_crg['primal_bound']:.4f}")
        print(f"  Gap:          {res_crg['gap']:.4%}")

if __name__ == "__main__":
    run_asp_experiment()