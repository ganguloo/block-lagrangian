import sys
import numpy as np
import gurobipy as gp
from src.instance.topology import TopologyManager
from src.blocks.lop import LinearOrderingBlock
from src.monolithic.solver import MonolithicSolver
from src.solver.manager import CRGManager
from src.strategies.v_lagrangian import VLagrangianStrategy

def generate_lop_weights(num_nodes, seed=42):
    """Generates a random preference matrix W where w_ii = 0."""
    rng = np.random.default_rng(seed)
    weights = rng.integers(0, 101, size=(num_nodes, num_nodes))
    np.fill_diagonal(weights, 0)
    return weights.tolist()

def get_shared_arcs(nodes_u, nodes_v):
    """
    Returns list of all directed arcs (x, y) where x, y are in the intersection of nodes_u and nodes_v.
    Includes both (x, y) and (y, x) for x != y.
    """
    shared_nodes = sorted(list(set(nodes_u).intersection(set(nodes_v))))
    arcs = []
    for i in range(len(shared_nodes)):
        for j in range(len(shared_nodes)):
            if i == j: continue
            u, v = shared_nodes[i], shared_nodes[j]
            arcs.append((u, v))
    return arcs

def build_lop_instance(topology_type, block_size, overlap, num_blocks):
    """
    Constructs blocks and topology based on the type.
    Coupling rule: Last 'overlap' nodes of block A = First 'overlap' nodes of block B.
    """
    
    # 1. Determine Global Node Ranges for each Block
    block_nodes = {}
    current_new_node = 0
    
    # Block 0 always starts at 0
    # Range: [0, block_size)
    block_nodes[0] = list(range(current_new_node, current_new_node + block_size))
    current_new_node += block_size
    
    # Helper to get the "Last K" nodes of a block
    def get_last_k(b_idx):
        nodes = block_nodes[b_idx]
        return nodes[-overlap:]

    if topology_type == "Path":
        # 0 -> 1 -> 2 ...
        for i in range(1, num_blocks):
            prev_shared = get_last_k(i - 1)
            # New unique nodes needed: block_size - overlap
            num_new = block_size - overlap
            new_nodes = list(range(current_new_node, current_new_node + num_new))
            current_new_node += num_new
            
            block_nodes[i] = prev_shared + new_nodes

    elif topology_type == "Star":
        # 0 is Center. 1..N are Leaves.
        # Center connects to all leaves.
        # Rule: Center's last K nodes are shared with Leaf's first K nodes.
        # Assumption: All leaves attach to the SAME "port" (last K) of Center.
        center_shared = get_last_k(0)
        
        for i in range(1, num_blocks):
            num_new = block_size - overlap
            new_nodes = list(range(current_new_node, current_new_node + num_new))
            current_new_node += num_new
            
            block_nodes[i] = center_shared + new_nodes

    elif topology_type == "BinTree":
        # 0 -> 1, 2.  1 -> 3, 4. etc.
        # i -> 2i+1, 2i+2
        # Parent shares its last K nodes with both children.
        for i in range(num_blocks):
            # Children indices
            c1 = 2 * i + 1
            c2 = 2 * i + 2
            
            if c1 < num_blocks:
                parent_shared = get_last_k(i)
                num_new = block_size - overlap
                new_nodes = list(range(current_new_node, current_new_node + num_new))
                current_new_node += num_new
                block_nodes[c1] = parent_shared + new_nodes
                
            if c2 < num_blocks:
                # Same parent, same shared nodes, DIFFERENT new nodes
                parent_shared = get_last_k(i)
                num_new = block_size - overlap
                new_nodes = list(range(current_new_node, current_new_node + num_new))
                current_new_node += num_new
                block_nodes[c2] = parent_shared + new_nodes

    total_unique_nodes = current_new_node
    print(f"Topology: {topology_type}, Blocks: {num_blocks}, Total Unique Nodes: {total_unique_nodes}")

    # 2. Generate Weights
    weights = generate_lop_weights(total_unique_nodes)

    # 3. Create Blocks
    blocks = []
    for i in range(num_blocks):
        blk = LinearOrderingBlock(i, block_nodes[i], weights)
        blocks.append(blk)

    # 4. Build Topology Coupling
    topology = TopologyManager(block_sizes=0) # Dummy size
    
    if topology_type == "Path":
        for i in range(num_blocks - 1):
            u, v = i, i+1
            arcs = get_shared_arcs(block_nodes[u], block_nodes[v])
            if arcs:
                topology.add_coupling(u, v, arcs, arcs)
                
    elif topology_type == "Star":
        for i in range(1, num_blocks):
            # Center (0) to Leaf (i)
            arcs = get_shared_arcs(block_nodes[0], block_nodes[i])
            if arcs:
                topology.add_coupling(0, i, arcs, arcs)

    elif topology_type == "BinTree":
        for i in range(num_blocks):
            c1, c2 = 2*i + 1, 2*i + 2
            if c1 < num_blocks:
                arcs = get_shared_arcs(block_nodes[i], block_nodes[c1])
                if arcs: topology.add_coupling(i, c1, arcs, arcs)
            if c2 < num_blocks:
                arcs = get_shared_arcs(block_nodes[i], block_nodes[c2])
                if arcs: topology.add_coupling(i, c2, arcs, arcs)

    return blocks, topology

def run_experiment():
    # Settings
    BLOCK_SIZE = 15
    OVERLAP = 5
    NUM_BLOCKS = 5
    TOPOLOGIES = ["Path", "Star", "BinTree"]

    for topo_type in TOPOLOGIES:
        print(f"\n========================================")
        print(f"Running LOP Experiment: {topo_type}")
        print(f"========================================")
        
        blocks, topology = build_lop_instance(topo_type, BLOCK_SIZE, OVERLAP, NUM_BLOCKS)
        
        # 1. Monolithic
        print("\n> [1/2] Monolithic Solver...")
        mono = MonolithicSolver(topology, blocks)
        res_mono = mono.build_and_solve(time_limit=60)
        print(f"Monolithic: Status={res_mono['status']}, Obj={res_mono['primal_bound']}")

        # 2. CRG
        print("\n> [2/2] CRG Solver...")
        # VLagrangian is suitable for equality constraints x_uv^A - x_uv^B = 0
        strategy = VLagrangianStrategy()
        crg = CRGManager(blocks, topology, strategy)
        
        # Ensure your src/solver/manager.py has the fixes for negative/zero bounds if weights are neg
        # LOP weights here are positive [0, 100], so standard logic is fine.
        res_crg = crg.run(time_limit=180)
        
        print(f"\nResults {topo_type}:")
        print(f"  Dual Bound:   {res_crg['dual_bound']:.4f}")
        print(f"  Primal Bound: {res_crg['primal_bound']:.4f}")
        print(f"  Gap:          {res_crg['gap']:.4%}")

if __name__ == "__main__":
    run_experiment()