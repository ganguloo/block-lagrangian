import gurobipy as gp
from typing import List
from .base_block import AbstractBlock

class AssignmentBlock(AbstractBlock):
    def __init__(self, block_id: int, capacity: int, weights: List[int], profits: List[int], 
                 num_jobs: int, is_first: bool = False, is_last: bool = False):
        """
        Representa una máquina en el problema GAP reformulado como flujo.
        """
        super().__init__(block_id, name=f"Machine_{block_id}")
        self.capacity = capacity
        self.weights = weights      # Pesos de los trabajos en ESTA máquina
        self.profits = profits      # Beneficios de los trabajos en ESTA máquina
        self.num_jobs = num_jobs
        self.is_first = is_first
        self.is_last = is_last

    def build_model(self, parent_model: gp.Model = None, prefix: str = None):
        if parent_model:
            self.model = parent_model
            pfx = prefix if prefix else f"B{self.block_id}"
        else:
            self.model = gp.Model(self.name)
            self.model.Params.OutputFlag = 0
            pfx = "x"

        self.vars = {} 
        
        # --- Variables ---
        # x[j]: 1 si esta máquina procesa el trabajo j (Decisión Local)
        x = self.model.addVars(self.num_jobs, vtype=gp.GRB.BINARY, name=f"{pfx}_do")
        
        # y[j]: 1 si el trabajo j YA fue procesado por máquinas anteriores (Estado Entrada)
        y = self.model.addVars(self.num_jobs, vtype=gp.GRB.BINARY, name=f"{pfx}_in")
        
        # z[j]: 1 si el trabajo j está procesado al terminar esta etapa (Estado Salida)
        z = self.model.addVars(self.num_jobs, vtype=gp.GRB.BINARY, name=f"{pfx}_out")

        # Mapeo para TopologyManager:
        # Índices 0..N-1 -> Entrada (y)
        # Índices N..2N-1 -> Salida (z)
        for j in range(self.num_jobs):
            self.vars[j] = y[j]
            self.vars[self.num_jobs + j] = z[j]

        # --- Restricciones ---

        # 1. Conservación de Flujo / Estado: z = y + x
        # Esto implícitamente fuerza que si y=1, x debe ser 0 (porque z es binario)
        for j in range(self.num_jobs):
            self.model.addConstr(z[j] == y[j] + x[j], name=f"{pfx}_flow_{j}")

        # 2. Capacidad (Knapsack Local)
        self.model.addConstr(
            gp.quicksum(self.weights[j] * x[j] for j in range(self.num_jobs)) <= self.capacity,
            name=f"{pfx}_capacity"
        )

        # 3. Condiciones de Borde
        if self.is_first:
            # Al inicio (Máquina 0), nada viene hecho
            for j in range(self.num_jobs):
                self.model.addConstr(y[j] == 0, name=f"{pfx}_bc_start")
        
        if self.is_last:
            # Al final (Última Máquina), todo debe estar hecho
            for j in range(self.num_jobs):
                self.model.addConstr(z[j] == 1, name=f"{pfx}_bc_end")

        # --- Función Objetivo Local (Maximización) ---
        self.local_objective_expr = gp.LinExpr()
        for j in range(self.num_jobs):
            self.local_objective_expr.add(x[j], self.profits[j])

        if parent_model is None:
            self.model.setObjective(self.local_objective_expr, gp.GRB.MAXIMIZE)
            self.model.update()

    def get_vars_by_index(self, indices: List[int]) -> List[gp.Var]:
        return [self.vars[i] for i in indices]