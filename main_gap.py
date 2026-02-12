import sys
import gurobipy as gp
from src.instance.topology import TopologyManager
from src.blocks.assignment import AssignmentBlock
from src.monolithic.solver import MonolithicSolver
from src.solver.manager import CRGManager
from src.strategies.v_lagrangian import VLagrangianStrategy
from src.instance.gap_generator import generate_gap_instance 

def run_gap_experiment():
    # --- Configuración ---
    M = 20     # Máquinas
    N = 60   # Trabajos (Max 100)
    SEED = 42
    TYPE = "D" # Probar 'D' (Constant Cap) o 'E' (Variable Cap)
    
    print(f"Generando GAP Tipo {TYPE} (Chu & Beasley Original Invertido) con {M} máquinas y {N} trabajos...")
    
    # 1. Generar Instancia
    # Esto retornará beneficios negativos.
    capacities, weights_matrix, profits_matrix = generate_gap_instance(M, N, SEED, instance_type=TYPE)

    print(f"  Ejemplo de beneficio p_00: {profits_matrix[0][0]} (Debería ser negativo)")

    # 2. Crear Bloques
    blocks = []
    for i in range(M):
        is_first = (i == 0)
        is_last = (i == M - 1)
        
        blk = AssignmentBlock(
            block_id=i,
            capacity=capacities[i],
            weights=weights_matrix[i],
            profits=profits_matrix[i],
            num_jobs=N,
            is_first=is_first,
            is_last=is_last
        )
        blocks.append(blk)

    # 3. Topología (Z -> Y)
    topology = TopologyManager(block_sizes=2*N) 
    indices_out_z = list(range(N, 2*N)) 
    indices_in_y  = list(range(0, N))   

    for i in range(M - 1):
        u, v = i, i+1
        topology.add_coupling(u, v, indices_out_z, indices_in_y)

    print("Topología construida.")

    # 4. Monolítico
    print("\n> [1/2] Ejecutando Monolítico...")
    mono = MonolithicSolver(topology, blocks)
    # Importante: El Monolítico debe poder manejar objetivos negativos (Gurobi lo hace nativo)
    res_mono = mono.build_and_solve(time_limit=60)
    print(f"Monolítico: Status={res_mono['status']}, Obj={res_mono['primal_bound']}")

    # 5. CRG (Column Generation)
    print("\n> [2/2] Ejecutando CRG...")
    strategy = VLagrangianStrategy() 
    crg = CRGManager(blocks, topology, strategy)
    
    # Asegúrate de que src/solver/manager.py tenga los fixes de -float('inf')
    res_crg = crg.run(time_limit=180) 
    
    print(f"\nResultados Finales GAP Tipo {TYPE}:")
    print(f"  CRG Dual Bound:   {res_crg['dual_bound']:.4f}")
    print(f"  CRG Primal Bound: {res_crg['primal_bound']:.4f}")
    print(f"  Gap Final:        {res_crg['gap']:.4%}")

if __name__ == "__main__":
    run_gap_experiment()