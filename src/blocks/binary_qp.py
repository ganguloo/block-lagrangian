import random
import gurobipy as gp
from .base_block import AbstractBlock

class QPBlock(AbstractBlock):
    def __init__(self, block_id: int, n_nodes: int, num_edges: int, bias: float = 0.0, seed: int = 42, linearize: bool = True):
        super().__init__(block_id)
        self.n_nodes = n_nodes
        self.num_edges = num_edges  # Representa 'm' (términos cuadráticos locales)
        self.bias = bias
        self.seed = seed
        self.linearize = linearize
        
        # 1. Generar la instancia local con su propia semilla
        rng = random.Random(self.seed)
        
        # Pesos lineales c_i
        self.local_c = {i: rng.randint(-50 + self.bias, 50 + self.bias) for i in range(self.n_nodes)}
        
        # Seleccionar m pares únicos (i, j) para los términos cuadráticos Q_ij
        all_pairs = [(i, j) for i in range(self.n_nodes) for j in range(i, self.n_nodes)]
        if self.num_edges > len(all_pairs):
            raise ValueError(f"num_edges ({self.num_edges}) excede el máximo de pares posibles ({len(all_pairs)}).")
            
        chosen_pairs = rng.sample(all_pairs, self.num_edges)
        self.local_Q = { (i, j): rng.randint(-50 + self.bias, 50 + self.bias) for i, j in chosen_pairs }

    def build_model(self, parent_model: gp.Model = None, prefix: str = None):
        m = parent_model if parent_model else gp.Model(f"QPBlock_{self.block_id}")
        pfx = prefix + "_" if prefix else f"b{self.block_id}_"
        
        # 2. Instanciar variables
        self.x = {}
        for i in range(self.n_nodes):
            self.x[i] = m.addVar(vtype=gp.GRB.BINARY, name=f"{pfx}x_{i}")
            
        obj_expr = gp.QuadExpr()
        
        # 3. Añadir términos a la función objetivo
        for i, coef in self.local_c.items():
            obj_expr += coef * self.x[i]
            
        if self.linearize:
            for (i, j), coef in self.local_Q.items():
                y = m.addVar(vtype=gp.GRB.CONTINUOUS, lb=0.0, ub=1.0, name=f"{pfx}y_{i}_{j}")
                m.addConstr(y <= self.x[i], name=f"{pfx}mc1_{i}_{j}")
                m.addConstr(y <= self.x[j], name=f"{pfx}mc2_{i}_{j}")
                m.addConstr(y >= self.x[i] + self.x[j] - 1, name=f"{pfx}mc3_{i}_{j}")
                obj_expr += coef * y
        else:
            m.Params.NonConvex = 2
            for (i, j), coef in self.local_Q.items():
                obj_expr += coef * self.x[i] * self.x[j]
                
        m.setObjective(obj_expr, gp.GRB.MAXIMIZE)
        m.update()
        
        # 4. Mapeo para el Maestro de CRG
        self.vars = {i: self.x[i] for i in range(self.n_nodes)}
        self.local_objective_expr = obj_expr
        self.model = m

    def get_vars_by_index(self, indices):
        """Devuelve las variables de Gurobi para los índices solicitados en el acople"""
        return [self.vars[i] for i in indices]