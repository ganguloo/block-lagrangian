import gurobipy as gp
from typing import List, Tuple
from .base_block import AbstractBlock

class GeneralMatchingBlock(AbstractBlock):
    def __init__(self, block_id: int, global_nodes: List[int], edges: List[Tuple[int, int]]):
        """
        block_id: ID del bloque.
        global_nodes: Lista de IDs globales de nodos en este bloque.
        edges: Lista de tuplas (u, v) representando las aristas EXISTENTES.
               Se asume u < v.
        """
        super().__init__(block_id, name=f"Matching_{block_id}")
        self.global_nodes = sorted(list(set(global_nodes)))
        self.edges = edges
        # Crear mapa de adyacencia para facilitar restricciones
        self.adj = {u: [] for u in self.global_nodes}
        for u, v in self.edges:
            # Asumiendo u < v, agregamos a ambos para el grado
            if u in self.adj: self.adj[u].append((u, v))
            if v in self.adj: self.adj[v].append((u, v))

    def build_model(self, parent_model: gp.Model = None, prefix: str = None):
        if parent_model:
            self.model = parent_model
            pfx = prefix if prefix else f"B{self.block_id}"
        else:
            self.model = gp.Model(self.name)
            self.model.Params.OutputFlag = 0
            pfx = "match"

        self.vars = {} 

        # 1. Variables de Arista x_uv
        for u, v in self.edges:
            var = self.model.addVar(vtype=gp.GRB.BINARY, name=f"{pfx}_x_{u}_{v}")
            self.vars[(u, v)] = var

        # 2. Restricciones de Matching (Grado <= 1)
        for node in self.global_nodes:
            incident_edges = self.adj[node]
            if incident_edges:
                self.model.addConstr(
                    gp.quicksum(self.vars[edge] for edge in incident_edges) <= 1,
                    name=f"deg_{node}"
                )

        # 3. Función Objetivo: Cardinalidad Máxima (Suma de x_e)
        self.local_objective_expr = gp.LinExpr()
        for var in self.vars.values():
            self.local_objective_expr.add(var, 1.0)

        if parent_model is None:
            self.model.setObjective(self.local_objective_expr, gp.GRB.MAXIMIZE)
            self.model.update()

    def get_vars_by_index(self, indices: List[Tuple[int, int]]) -> List[gp.Var]:
        """
        Retorna las variables x_uv correspondientes a las aristas solicitadas.
        Solo retorna si la arista existe en este bloque.
        """
        return [self.vars[uv] for uv in indices if uv in self.vars]