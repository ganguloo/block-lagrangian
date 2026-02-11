import sys
import gurobipy as gp
from src.instance.topology import TopologyManager
from src.blocks.assignment import AssignmentBlock  # <--- Importar nuevo bloque
from src.monolithic.solver import MonolithicSolver
from src.solver.manager import CRGManager
from src.strategies.v_lagrangian import VLagrangianStrategy
from src.instance.gap_generator import generate_gap_instance

def run_gap_experiment():
    # 1. Parámetros
    M = 5   # Máquinas (Bloques)
    N = 20  # Trabajos (Tamaño del acople)
    SEED = 42
    
    print(f"Generando GAP con {M} máquinas y {N} trabajos...")
    
    # 2. Generar Instancia (Copia la función generate_gap_instance aquí o impórtala)
    capacities, weights_matrix, profits_matrix = generate_gap_instance(M, N, SEED)

    # 3. Crear Bloques
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

    # 4. Definir Topología de Camino (Path) Estricta
    # Cada bloque expone 2*N variables:
    #   [0...N-1] -> Entrada (y)
    #   [N...2N-1] -> Salida (z)
    # Conectamos z del bloque i con y del bloque i+1
    
    topology = TopologyManager(block_sizes=2*N) 
    
    coupling_size = N
    indices_out_z = list(range(N, 2*N)) # Salida del anterior
    indices_in_y  = list(range(0, N))   # Entrada del siguiente

    for i in range(M - 1):
        u, v = i, i+1
        # IMPORTANTE: topology.add_coupling(u, v, vars_u, vars_v)
        topology.add_coupling(u, v, indices_out_z, indices_in_y)

    print("Topología GAP construida: Cadena de máquinas.")

    # 5. Resolver con Monolítico (para validar)
    print("\n> Ejecutando Monolítico...")
    mono = MonolithicSolver(topology, blocks)
    res_mono = mono.build_and_solve(time_limit=60)
    print(f"Monolítico: {res_mono['status']}, Obj: {res_mono['primal_bound']}")

    # 6. Resolver con CRG
    print("\n> Ejecutando CRG...")
    # Usamos VLagrangianStrategy porque el acople es igualdad binaria (z - y = 0)
    # y queremos separar cortes sobre combinaciones de valores.
    strategy = VLagrangianStrategy() 
    
    crg = CRGManager(blocks, topology, strategy)
    res_crg = crg.run(time_limit=120)
    print(f"CRG: {res_crg['status']}, Dual Bound: {res_crg['dual_bound']}, Primal: {res_crg['primal_bound']}")

if __name__ == "__main__":
    # Necesitas definir generate_gap_instance o importarla
    # ... (pegar generate_gap_instance aquí si es un solo archivo)
    run_gap_experiment()