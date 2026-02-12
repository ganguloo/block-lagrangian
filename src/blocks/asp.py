import gurobipy as gp
from typing import List, Tuple, Dict
from .base_block import AbstractBlock

class AcyclicSubgraphBlock(AbstractBlock):
    def __init__(self, block_id: int, global_nodes: List[int], edges: List[Tuple[int, int]], weights: Dict[Tuple[int, int], float]):
        """
        block_id: ID del bloque.
        global_nodes: Lista de IDs globales de nodos en este bloque.
        edges: Lista de tuplas (u, v) representando los arcos EXISTENTES en este grafo local.
        weights: Diccionario {(u, v): peso}.
        """
        super().__init__(block_id, name=f"ASP_{block_id}")
        self.global_nodes = sorted(list(set(global_nodes)))
        self.edges = edges
        self.weights = weights
        self.n_local = len(self.global_nodes)

    def build_model(self, parent_model: gp.Model = None, prefix: str = None):
        if parent_model:
            self.model = parent_model
            pfx = prefix if prefix else f"B{self.block_id}"
        else:
            self.model = gp.Model(self.name)
            self.model.Params.OutputFlag = 0
            pfx = "asp"

        self.vars = {}      # x_uv variables (arcos seleccionados)
        self.potentials = {} # u_i variables (orden topológico)

        # 1. Variables de Potencial u_i
        # Representan la posición en el orden topológico. Rango [0, N].
        # Usamos IDs globales en el nombre para facilitar debug.
        for node in self.global_nodes:
            self.potentials[node] = self.model.addVar(
                lb=0, ub=self.n_local, vtype=gp.GRB.CONTINUOUS, name=f"{pfx}_u_{node}"
            )

        # 2. Variables de Arco x_uv (Solo para arcos existentes en este grafo)
        for u, v in self.edges:
            var = self.model.addVar(vtype=gp.GRB.BINARY, name=f"{pfx}_x_{u}_{v}")
            self.vars[(u, v)] = var

        # 3. Restricciones MTZ (Miller-Tucker-Zemlin) para eliminar ciclos
        # Si x_uv = 1 => u_v >= u_u + 1
        # Formulación Big-M: u_u - u_v + N * x_uv <= N - 1
        BigM = self.n_local
        
        for u, v in self.edges:
            x_var = self.vars[(u, v)]
            u_start = self.potentials[u]
            u_end = self.potentials[v]
            
            # Restricción: Potencial de v debe ser mayor que u si existe arco
            self.model.addConstr(
                u_start - u_end + BigM * x_var <= BigM - 1,
                name=f"mtz_{u}_{v}"
            )

        # 4. Función Objetivo: Maximizar peso de arcos seleccionados
        self.local_objective_expr = gp.LinExpr()
        for (u, v), var in self.vars.items():
            w = self.weights.get((u, v), 0.0)
            self.local_objective_expr.add(var, w)

        if parent_model is None:
            self.model.setObjective(self.local_objective_expr, gp.GRB.MAXIMIZE)
            self.model.update()

    def get_vars_by_index(self, indices: List[Tuple[int, int]]) -> List[gp.Var]:
        """
        Retorna las variables x_uv correspondientes a los arcos solicitados.
        IMPORTANTE: Filtramos los índices. Si el arco no existe en este bloque,
        no retornamos nada (o podríamos manejarlo según requiera el solver, 
        pero el TopologyManager en main ya debería haber filtrado).
        """
        return [self.vars[uv] for uv in indices if uv in self.vars]