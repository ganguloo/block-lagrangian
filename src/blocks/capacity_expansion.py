import gurobipy as gp
import random
import math
from typing import List
from .base_block import AbstractBlock

class CapacityExpansionBlock(AbstractBlock):
    def __init__(self, block_id: int, num_facilities: int, num_clients: int, 
                 seed: int = 42, obj_factor: float = 1.0):
        """
        Bloque para el Multi-Period Facility Location / Capacity Expansion Problem.
        
        :param num_facilities: Número de instalaciones candidatas (|F|)
        :param num_clients: Número de clientes a atender (|C|)
        """
        super().__init__(block_id, name=f"CapExp_{block_id}", obj_factor=obj_factor)
        self.num_facilities = num_facilities
        self.num_clients = num_clients
        self.seed = seed
        
        # Generar datos del problema
        self._generate_data()

    def _generate_data(self):
        # Usamos una semilla basada en el ID del bloque para que los clientes/demandas
        # puedan variar por período, pero si quieres que las instalaciones sean estáticas
        # podrías usar una semilla fija para las coordenadas de las instalaciones.
        rng = random.Random(self.seed)
        
        # Costos de las instalaciones (Enteros)
        self.fac_costs_open = [rng.randint(500, 1000) for _ in range(self.num_facilities)]
        self.fac_costs_close = [rng.randint(100, 300) for _ in range(self.num_facilities)]
        self.fac_costs_oper = [rng.randint(50, 150) for _ in range(self.num_facilities)]
        
        # Coordenadas espaciales para calcular transporte
        fac_coords = [(rng.uniform(0, 100), rng.uniform(0, 100)) for _ in range(self.num_facilities)]
        client_coords = [(rng.uniform(0, 100), rng.uniform(0, 100)) for _ in range(self.num_clients)]
        
        # Costos de transporte (Distancia euclidiana ponderada y redondeada)
        self.transp_costs = {}
        for i in range(self.num_facilities):
            for j in range(self.num_clients):
                dist = math.hypot(fac_coords[i][0] - client_coords[j][0], 
                                  fac_coords[i][1] - client_coords[j][1])
                # Añadimos un factor aleatorio de costo por ruta
                self.transp_costs[(i, j)] = int(round(dist * rng.uniform(1.0, 2.0)))

    def build_model(self, parent_model: gp.Model = None, prefix: str = None):
        if parent_model:
            m = parent_model
            pfx = prefix if prefix else f"B{self.block_id}"
        else:
            m = gp.Model(self.name)
            m.Params.OutputFlag = 0
            pfx = "ce"
        
        # --- Variables ---
        self.y_in = m.addVars(self.num_facilities, vtype=gp.GRB.BINARY, name=f"{pfx}Y_in")
        self.y_out = m.addVars(self.num_facilities, vtype=gp.GRB.BINARY, name=f"{pfx}Y_out")
        self.z = m.addVars(self.num_facilities, vtype=gp.GRB.BINARY, name=f"{pfx}z_open")
        self.w = m.addVars(self.num_facilities, vtype=gp.GRB.BINARY, name=f"{pfx}w_close")
        self.x = m.addVars(self.num_facilities, self.num_clients, vtype=gp.GRB.BINARY, name=f"{pfx}x_assign")
        
        # --- Función Objetivo (Maximizar el negativo de los costos) ---
        self.local_objective_expr = gp.LinExpr()
        
        for i in range(self.num_facilities):
            self.local_objective_expr.add(self.z[i], -self.fac_costs_open[i] * self.obj_factor)
            self.local_objective_expr.add(self.w[i], -self.fac_costs_close[i] * self.obj_factor)
            self.local_objective_expr.add(self.y_out[i], -self.fac_costs_oper[i] * self.obj_factor)
            
            for j in range(self.num_clients):
                self.local_objective_expr.add(self.x[i, j], -self.transp_costs[(i, j)] * self.obj_factor)
        
        if not parent_model:
            m.setObjective(self.local_objective_expr, gp.GRB.MAXIMIZE)
            
        # --- Restricciones ---
        # 1. Satisfacción de la demanda (cada cliente debe ser atendido)
        for j in range(self.num_clients):
            m.addConstr(gp.quicksum(self.x[i, j] for i in range(self.num_facilities)) == 1, name=f"{pfx}dem_{j}")
            
        # 2. Lógica de capacidad (sólo atiendo si la instalación termina abierta)
        for i in range(self.num_facilities):
            for j in range(self.num_clients):
                m.addConstr(self.x[i, j] <= self.y_out[i], name=f"{pfx}logic_{i}_{j}")
                
        # 3. Transición Markoviana (Ecuación de balance de estado)
        for i in range(self.num_facilities):
            m.addConstr(self.y_out[i] == self.y_in[i] + self.z[i] - self.w[i], name=f"{pfx}trans_{i}")
            
        # 4. Prevención de apertura y cierre simultáneo (redundante pero ayuda al LP)
        for i in range(self.num_facilities):
            m.addConstr(self.z[i] + self.w[i] <= 1, name=f"{pfx}mutex_{i}")
            
        # 5. Condición Inicial (Período 0)
        # Asumimos que al inicio del horizonte de planificación todas las instalaciones están cerradas.
        if self.block_id == 0:
            for i in range(self.num_facilities):
                m.addConstr(self.y_in[i] == 0, name=f"{pfx}init_{i}")
                
        # --- Mapeo de Interfaz para la Topología ---
        # Registramos IN en los índices 0 a |F|-1
        # Registramos OUT en los índices |F| a 2|F|-1
        for i in range(self.num_facilities):
            self.vars[i] = self.y_in[i]
            self.vars[i + self.num_facilities] = self.y_out[i]
            
        self.model = m
        self.model.update()

    def get_vars_by_index(self, indices: List[int]) -> List[gp.Var]:
        return [self.vars[idx] for idx in indices]