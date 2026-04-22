import random
import networkx as nx
from networkx.algorithms.approximation.treewidth import treewidth_min_fill_in, treewidth_min_degree
from networkx.generators.trees import random_unlabeled_tree
from collections import defaultdict

def generate_and_decompose_boxqp(n: int, num_edges: int, bias: float = 0.0, seed: int = 42):
    """
    1. Genera un árbol base aleatorio (conexo).
    2. Agrega arcos adicionales o elementos en la diagonal (D) hasta alcanzar num_edges.
    3. Genera pesos Q (arcos y diagonal) y c (lineales).
    4. Aplica treewidth_min_fill_in y distribuye los pesos en los bloques.
    """
    # --- Validaciones iniciales ---
    if num_edges < n - 1:
        raise ValueError(f"num_edges ({num_edges}) debe ser al menos n - 1 para mantener el árbol conexo.")
    max_posibles = n * (n + 1) // 2
    if num_edges > max_posibles:
        raise ValueError(f"num_edges ({num_edges}) excede el máximo de pares i <= j posibles ({max_posibles}).")

    rng = random.Random(seed)
    
    # =========================================================================
    # 1. GENERAR ÁRBOL BASE
    # =========================================================================
    # random_unlabeled_tree devuelve un grafo. Convertimos las etiquetas a enteros 0..n-1
    G_base = random_unlabeled_tree(n, seed=seed)
    G = nx.convert_node_labels_to_integers(G_base)
    
    # Extraer las aristas del árbol asegurando que (i, j) cumpla i < j
    tree_edges = set()
    for u, v in G.edges():
        tree_edges.add((min(u, v), max(u, v)))
        
    # =========================================================================
    # 2. SELECCIONAR ARCOS ADICIONALES Y DIAGONAL (D)
    # =========================================================================
    # Identificar todos los pares (i, j) con i <= j que NO están en el árbol
    pares_disponibles = []
    for i in range(n):
        for j in range(i, n):
            if (i, j) not in tree_edges:
                pares_disponibles.append((i, j))
                
    # Muestrear (num_edges - n + 1) pares de los disponibles
    # Nota: un árbol de n nodos siempre tiene (n - 1) aristas.
    pares_agregados = rng.sample(pares_disponibles, num_edges - n + 1)
    
    D = set() # Almacenará los i donde i == j
    for i, j in pares_agregados:
        if i != j:
            G.add_edge(i, j)
        else:
            D.add(i)
            
    # =========================================================================
    # 3. GENERAR PESOS (c y Q)
    # =========================================================================
    # Términos puramente lineales (opcional en BoxQP, pero mantenemos la estructura)
    c = {i: rng.randint(-50 + bias, 50 + bias) for i in range(n)}
    Q = {}
    
    # Pesos para las aristas en G (i != j)
    for u, v in G.edges():
        i, j = min(u, v), max(u, v)
        Q[(i, j)] = rng.randint(-50 + bias, 50 + bias)
        
    # Pesos para la diagonal D (i == j)
    for i in D:
        Q[(i, i)] = rng.randint(-50 + bias, 50 + bias)
        
    # =========================================================================
    # 4. TREE DECOMPOSITION Y REPARTO
    # =========================================================================
    #tw, tree_decomp = treewidth_min_fill_in(G)
    tw, tree_decomp = treewidth_min_degree(G)
    
    bags = list(tree_decomp.nodes())
    bag_to_id = {bag: idx for idx, bag in enumerate(bags)}
    
    topology_edges = []
    for u, v in tree_decomp.edges():
        topology_edges.append((bag_to_id[u], bag_to_id[v]))
        
    K = len(bags)
    local_Q = {k: {} for k in range(K)}
    local_c = {k: defaultdict(float) for k in range(K)}
    
    # Repartir términos lineales (c)
    for i in range(n):
        containing_bags = [k for bag, k in bag_to_id.items() if i in bag]
        if containing_bags:
            split_val = c[i] / len(containing_bags)
            for k in containing_bags:
                local_c[k][i] += split_val
                
    # Repartir términos de la matriz Q (incluye arcos y diagonal)
    for (i, j), val in Q.items():
        # ¡Magia aquí! Si i == j, la condición 'i in bag and j in bag' 
        # se evalúa correctamente a 'i in bag', repartiendo la diagonal D de forma impecable.
        containing_bags = [k for bag, k in bag_to_id.items() if i in bag and j in bag]
        if containing_bags:
            split_val = val / len(containing_bags)
            for k in containing_bags:
                local_Q[k][(i, j)] = split_val
                
    return {
        "n": n,
        "num_edges": num_edges,
        "num_blocks": K,
        "treewidth": tw,
        "bags": [list(bag) for bag in bags],
        "topology_edges": topology_edges,
        "local_Q": local_Q,
        "local_c": local_c
    }