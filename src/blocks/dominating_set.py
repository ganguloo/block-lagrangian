import gurobipy as gp
from typing import List, Tuple
from .base_block import AbstractBlock

class DominatingSetBlock(AbstractBlock):
    def __init__(self, block_id: int, global_nodes: List[int], edges: List[Tuple[int, int]]):
        """
        block_id: ID del bloque.
        global_nodes: Lista de IDs globales de nodos que pertenecen a este bloque.
        edges: Lista de aristas (u, v) del subgrafo inducido por global_nodes.
        """
        super().__init__(block_id, name=f"DomSet_{block_id}")
        self.global_nodes = sorted(list(set(global_nodes)))
        self.edges = edges
        
        # Construir lista de adyacencia local
        self.adj = {u: [] for u in self.global_nodes}
        for u, v in self.edges:
            if u in self.adj: self.adj[u].append(v)
            if v in self.adj: self.adj[v].append(u)

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
        # Para CADA nodo u en el bloque, debe estar dominado por él mismo o un vecino.
        # x_u + sum(x_v for v in N(u)) >= 1
        for u in self.global_nodes:
            neighbors = self.adj[u]
            # Nota: Solo consideramos vecinos que están DENTRO de este bloque.
            # Esto es una aproximación válida si la topología de bloques cubre el grafo
            # de manera densa, o si asumimos que la dominación debe satisfacerse localmente.
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