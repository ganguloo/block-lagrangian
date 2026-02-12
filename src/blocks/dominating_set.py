import gurobipy as gp
import numpy as np
from typing import List, Tuple, Optional
from .base_block import AbstractBlock

class DominatingSetBlock(AbstractBlock):
    def __init__(self, block_id: int, num_nodes: int = 0, density: float = 0.2, seed: int = 42, 
                 global_nodes: Optional[List[int]] = None, edges: Optional[List[Tuple[int, int]]] = None):
        """
        Puede inicializarse de dos formas:
        1. Auto-generación (Benchmark): Pasar num_nodes, density, seed.
        2. Explícito (Experimentos): Pasar global_nodes y edges.
        """
        super().__init__(block_id, name=f"DomSet_{block_id}")
        
        if edges is not None:
            # Modo Explícito
            self.global_nodes = sorted(list(set(global_nodes)))
            self.edges = edges
            self.num_nodes = len(self.global_nodes)
        else:
            # Modo Auto-generación (Compatible con main.py)
            self.num_nodes = num_nodes
            self.global_nodes = list(range(num_nodes)) # Índices locales 0..N-1
            self.edges = self._generate_graph(num_nodes, density, seed)
            
        # Construir lista de adyacencia local
        self.adj = {u: [] for u in self.global_nodes}
        for u, v in self.edges:
            if u in self.adj: self.adj[u].append(v)
            if v in self.adj: self.adj[v].append(u)

    def _generate_graph(self, n, density, seed):
        rng = np.random.default_rng(seed)
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < density:
                    edges.append((i, j))
        return edges

    def build_model(self, parent_model: gp.Model = None, prefix: str = None):
        if parent_model:
            self.model = parent_model
            pfx = prefix if prefix else f"B{self.block_id}"
        else:
            self.model = gp.Model(self.name)
            self.model.Params.OutputFlag = 0
            pfx = "ds"

        self.vars = {} 

        # 1. Variables x_u = 1 si el nodo u está en el Dominating Set
        for u in self.global_nodes:
            var = self.model.addVar(vtype=gp.GRB.BINARY, name=f"{pfx}_x_{u}")
            self.vars[u] = var

        # 2. Restricciones de Dominación
        # x_u + sum(x_v for v in N(u)) >= 1
        for u in self.global_nodes:
            neighbors = self.adj[u]
            expr = gp.LinExpr()
            expr.add(self.vars[u], 1.0)
            for v in neighbors:
                expr.add(self.vars[v], 1.0)
            
            self.model.addConstr(expr >= 1, name=f"dom_{u}")

        # 3. Función Objetivo
        # Minimizar Cardinalidad <=> Maximizar Sum(-1 * x_u)
        self.local_objective_expr = gp.LinExpr()
        for var in self.vars.values():
            self.local_objective_expr.add(var, -1.0)

        if parent_model is None:
            self.model.setObjective(self.local_objective_expr, gp.GRB.MAXIMIZE)
            self.model.update()

    def get_vars_by_index(self, indices: List[int]) -> List[gp.Var]:
        """
        Retorna las variables x_u correspondientes a los nodos solicitados.
        """
        return [self.vars[u] for u in indices if u in self.vars]