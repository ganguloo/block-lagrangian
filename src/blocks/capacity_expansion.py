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
        rng = random.Random(self.seed)
        
        # --- NUEVO: Demandas de clientes ---
        self.client_demands = [rng.randint(10, 50) for _ in range(self.num_clients)]
        total_demand = sum(self.client_demands)
        
        self.fac_costs_open = []
        self.fac_costs_close = []
        self.fac_costs_oper = []
        self.capacities = [] # --- NUEVO: Capacidades ---
        transport_multipliers = []
        
        # 1. Generación de Costos Correlacionados y Capacidades
        for _ in range(self.num_facilities):
            f_i = rng.random() # factor de eficiencia
            n_open = rng.uniform(0.9, 1.1)
            n_close = rng.uniform(0.9, 1.1)
            n_oper = rng.uniform(0.9, 1.1)
            
            open_c = int(round((500 + 500 * f_i) * n_open))
            close_c = int(round((100 + 200 * f_i) * n_close))
            oper_c = int(round((150 - 100 * f_i) * n_oper))
            t_mult = 1.0 + 2.0 * (1.0 - f_i)
            
            # --- NUEVO: La capacidad también se relaciona con la eficiencia ---
            # Instalaciones más eficientes son más grandes. 
            # Hacemos que cada instalación cubra entre el 10% y el 30% de la demanda total.
            # Esto obliga a abrir entre 4 y 10 instalaciones forzosamente.
            base_cap = total_demand * rng.uniform(0.10, 0.30)
            cap = int(round(base_cap * (0.5 + 0.5 * f_i))) 
            
            self.fac_costs_open.append(open_c)
            self.fac_costs_close.append(close_c)
            self.fac_costs_oper.append(oper_c)
            self.capacities.append(cap)
            transport_multipliers.append(t_mult)
        
        # 2. Coordenadas espaciales
        fac_coords = [(rng.uniform(0, 100), rng.uniform(0, 100)) for _ in range(self.num_facilities)]
        client_coords = [(rng.uniform(0, 100), rng.uniform(0, 100)) for _ in range(self.num_clients)]
        
        # 3. Costos de transporte correlacionados
        self.transp_costs = {}
        for i in range(self.num_facilities):
            for j in range(self.num_clients):
                dist = math.hypot(fac_coords[i][0] - client_coords[j][0], 
                                  fac_coords[i][1] - client_coords[j][1])
                noise_route = rng.uniform(0.8, 1.2)
                cost = int(round(dist * transport_multipliers[i] * noise_route))
                self.transp_costs[(i, j)] = cost
        
        # 2. Coordenadas espaciales (sin cambios)
        fac_coords = [(rng.uniform(0, 100), rng.uniform(0, 100)) for _ in range(self.num_facilities)]
        client_coords = [(rng.uniform(0, 100), rng.uniform(0, 100)) for _ in range(self.num_clients)]
        
        # 3. Costos de transporte correlacionados a la eficiencia de la instalación
        self.transp_costs = {}
        for i in range(self.num_facilities):
            for j in range(self.num_clients):
                dist = math.hypot(fac_coords[i][0] - client_coords[j][0], 
                                  fac_coords[i][1] - client_coords[j][1])
                
                # Ruido aleatorio específico por ruta (±20%) para que la topología importe
                noise_route = rng.uniform(0.8, 1.2)
                
                # Costo final: Distancia * Multiplicador de la Instalación * Ruido
                cost = int(round(dist * transport_multipliers[i] * noise_route))
                self.transp_costs[(i, j)] = cost

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
        # 1. Satisfacción de la demanda (Single-Sourcing: cada cliente va a 1 sola instalación)
        for j in range(self.num_clients):
            m.addConstr(gp.quicksum(self.x[i, j] for i in range(self.num_facilities)) == 1, name=f"{pfx}dem_{j}")
            
        # 2. NUEVO: Restricción de Capacidad (Knapsack constraint)
        for i in range(self.num_facilities):
            m.addConstr(
                gp.quicksum(self.client_demands[j] * self.x[i, j] for j in range(self.num_clients)) <= self.capacities[i] * self.y_out[i],
                name=f"{pfx}capacity_{i}"
            )
                
        # 3. MANTENER: Strong Inequalities (Vitales según el paper para la relajación LP)
        for i in range(self.num_facilities):
            for j in range(self.num_clients):
                m.addConstr(self.x[i, j] <= self.y_out[i], name=f"{pfx}strong_logic_{i}_{j}")
                
        # 4. Transición Markoviana (Ecuación de balance de estado)
        for i in range(self.num_facilities):
            m.addConstr(self.y_out[i] == self.y_in[i] + self.z[i] - self.w[i], name=f"{pfx}trans_{i}")
            
        # 5. Prevención de apertura y cierre simultáneo
        for i in range(self.num_facilities):
            m.addConstr(self.z[i] + self.w[i] <= 1, name=f"{pfx}mutex_{i}")
            
        # 6. Condición Inicial (Período 0)
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