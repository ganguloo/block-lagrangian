import numpy as np

def generate_gap_instance(num_machines, num_jobs, seed=42, instance_type="D"):
    """
    Genera instancias GAP siguiendo la lógica original de Chu & Beasley (OR-Library),
    pero invirtiendo el signo de la función objetivo para Maximización.
    
    Tipos:
    - "D": Inverse Correlated, Capacidades Constantes. (Originalmente Minimización)
    - "E": Inverse Correlated, Capacidades Variables. (Originalmente Minimización)
    
    Salida:
    - Beneficios negativos (p_ij = -c_ij).
    """
    # Restricción solicitada: máximo 100 trabajos
    if num_jobs > 100:
        print(f"[*] Advertencia: Limitando trabajos a 100 (solicitado {num_jobs}).")
        num_jobs = 100

    rng = np.random.default_rng(seed)
    
    # 1. Pesos (Weights): Lógica Original Chu & Beasley [5, 25]
    # (Nota: El rango [1, 100] es para instancias "Large Scale", pero las D/E clásicas son [5, 25])
    weights = rng.integers(5, 26, size=(num_machines, num_jobs))
    
    profits = np.zeros((num_machines, num_jobs))
    
    # 2. Beneficios: Invertir signo del costo original
    if instance_type in ["D", "E"]:
        # Costo original (Minimización): c_ij = 111 - w_ij + U[-10, 10]
        # Inversión para Maximización:   p_ij = -c_ij
        noise = rng.integers(-10, 11, size=(num_machines, num_jobs))
        costs = 111 - weights + noise
        profits = -costs  # Valores negativos (aprox -100)
        
    elif instance_type == "C":
        # Correlacionada positiva simple (p ~ w)
        profits = weights + rng.integers(-2, 6, size=(num_machines, num_jobs))
        
    else:
        # Uncorrelated
        profits = rng.integers(10, 31, size=(num_machines, num_jobs))

    # 3. Capacidades (Tightness = 0.8)
    total_weight_sum = np.sum(weights)
    avg_capacity = int(0.1 * total_weight_sum / num_machines)
    
    if instance_type == "E":
        # Tipo E: Capacidades variables
        # Variación típica: +/- 20% alrededor del promedio, manteniendo la tensión
        # O simplemente aleatorio centrado. Usaremos distribución uniforme [0.7, 1.0] 
        # para generar asimetría como suele aparecer en instancias difíciles tipo E.
        capacities = [int(avg_capacity * rng.uniform(0.7, 1.05)) for _ in range(num_machines)]
    else:
        # Tipo D (y otros): Capacidades constantes
        capacities = [avg_capacity for _ in range(num_machines)]
    
    return list(capacities), weights.tolist(), profits.tolist()