
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import gurobipy as gp

class AbstractBlock(ABC):
    def __init__(self, block_id: int, name: str = ""):
        self.block_id = block_id
        self.name = name
        self.model: Optional[gp.Model] = None
        self.vars: Dict[int, gp.Var] = {}
        self.local_objective_expr: Optional[gp.LinExpr] = None

    @abstractmethod
    def build_model(self, parent_model: gp.Model = None, prefix: str = None):
        pass

    @abstractmethod
    def get_vars_by_index(self, indices: List[int]) -> List[gp.Var]:
        pass
