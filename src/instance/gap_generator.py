import numpy as np

def generate_gap_instance(num_machines, num_jobs, seed=42, correlation="high"):
    """
    Genera una instancia de Generalized Assignment Problem (Maximización).
    
    Retorna:
        capacities: Lista [M]
        weights: Matriz [M][N] (pesos w_{ij})
        profits: Matriz [M][N] (beneficios p_{ij})
    """
    rng = np.random.default_rng(seed)
    
    # Pesos w_{ij} ~ U[5, 25]
    weights = rng.integers(5, 26, size=(num_machines, num_jobs))
    
    profits = np.zeros((num_machines, num_jobs))
    
    if correlation == "high":
        # Beneficios correlacionados con el peso: p_{ij} = w_{ij} + U[-2, 5]
        # Mayor peso -> Mayor beneficio (típico knapsack difícil)
        for i in range(num_machines):
            for j in range(num_jobs):
                profits[i, j] = weights[i, j] + rng.integers(-2, 6)
    else:
        # Beneficios independientes ~ U[10, 30]
        profits = rng.integers(10, 31, size=(num_machines, num_jobs))
    
    # Asegurar beneficios positivos
    profits = np.maximum(profits, 1)

    # Capacidades: Ajustadas para que la instancia sea "ajustada" (tight)
    # C_i = 0.8 * (Sum total pesos) / M
    total_weight = np.sum(weights)
    avg_capacity = int(0.8 * total_weight / num_machines)
    
    # Variar ligeramente las capacidades entre máquinas
    capacities = [int(avg_capacity * rng.uniform(0.8, 1.2)) for _ in range(num_machines)]
    
    return list(capacities), weights.tolist(), profits.tolist()